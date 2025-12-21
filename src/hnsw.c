#include "postgres.h"

#include <float.h>
#include <math.h>

#include "access/amapi.h"
#include "access/reloptions.h"
#include "commands/progress.h"
#include "commands/vacuum.h"
#include "hnsw.h"
#include "miscadmin.h"
#include "utils/float.h"
#include "utils/guc.h"
#include "utils/selfuncs.h"
#include "utils/spccache.h"

#if PG_VERSION_NUM < 150000
#define MarkGUCPrefixReserved(x) EmitWarningsOnPlaceholders(x)
#endif

static const struct config_enum_entry hnsw_iterative_scan_options[] = {
	{"off", HNSW_ITERATIVE_SCAN_OFF, false},
	{"relaxed_order", HNSW_ITERATIVE_SCAN_RELAXED, false},
	{"strict_order", HNSW_ITERATIVE_SCAN_STRICT, false},
	{NULL, 0, false}
};

int			hnsw_ef_search;
int			hnsw_iterative_scan;
int			hnsw_max_scan_tuples;
double		hnsw_scan_mem_multiplier;
int			hnsw_lock_tranche_id;
static relopt_kind hnsw_relopt_kind;

/*
 * Assign a tranche ID for our LWLocks. This only needs to be done by one
 * backend, as the tranche ID is remembered in shared memory.
 *
 * This shared memory area is very small, so we just allocate it from the
 * "slop" that PostgreSQL reserves for small allocations like this. If
 * this grows bigger, we should use a shmem_request_hook and
 * RequestAddinShmemSpace() to pre-reserve space for this.
 */
void
HnswInitLockTranche(void)
{
	int		   *tranche_ids;
	bool		found;

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
	tranche_ids = ShmemInitStruct("hnsw LWLock ids",
								  sizeof(int) * 1,
								  &found);
	if (!found)
		tranche_ids[0] = LWLockNewTrancheId();
	hnsw_lock_tranche_id = tranche_ids[0];
	LWLockRelease(AddinShmemInitLock);

	/* Per-backend registration of the tranche ID */
	LWLockRegisterTranche(hnsw_lock_tranche_id, "HnswBuild");
}

/*
 * Initialize index options and variables
 */
void
HnswInit(void)
{
	if (!process_shared_preload_libraries_in_progress)
		HnswInitLockTranche();

	hnsw_relopt_kind = add_reloption_kind();
	add_int_reloption(hnsw_relopt_kind, "m", "Max number of connections",
					  HNSW_DEFAULT_M, HNSW_MIN_M, HNSW_MAX_M, AccessExclusiveLock);
	add_int_reloption(hnsw_relopt_kind, "ef_construction", "Size of the dynamic candidate list for construction",
					  HNSW_DEFAULT_EF_CONSTRUCTION, HNSW_MIN_EF_CONSTRUCTION, HNSW_MAX_EF_CONSTRUCTION, AccessExclusiveLock);
	// todo dkx
	add_int_reloption(hnsw_relopt_kind, "m1", "Max number of connections (column 1)",
					  0, 0, HNSW_MAX_M, AccessExclusiveLock);
	add_int_reloption(hnsw_relopt_kind, "ef_construction1", "Size of the dynamic candidate list for construction (column 1)",
					  0, 0, HNSW_MAX_EF_CONSTRUCTION, AccessExclusiveLock);

	add_int_reloption(hnsw_relopt_kind, "m2", "Max number of connections (column 2)",
					  0, 0, HNSW_MAX_M, AccessExclusiveLock);
	add_int_reloption(hnsw_relopt_kind, "ef_construction2", "Size of the dynamic candidate list for construction (column 2)",
					  0, 0, HNSW_MAX_EF_CONSTRUCTION, AccessExclusiveLock);

	DefineCustomIntVariable("hnsw.ef_search", "Sets the size of the dynamic candidate list for search",
							"Valid range is 1..1000.", &hnsw_ef_search,
							HNSW_DEFAULT_EF_SEARCH, HNSW_MIN_EF_SEARCH, HNSW_MAX_EF_SEARCH, PGC_USERSET, 0, NULL, NULL, NULL);

	DefineCustomEnumVariable("hnsw.iterative_scan", "Sets the mode for iterative scans",
							 NULL, &hnsw_iterative_scan,
							 HNSW_ITERATIVE_SCAN_OFF, hnsw_iterative_scan_options, PGC_USERSET, 0, NULL, NULL, NULL);

	/* This is approximate and does not affect the initial scan */
	DefineCustomIntVariable("hnsw.max_scan_tuples", "Sets the max number of tuples to visit for iterative scans",
							NULL, &hnsw_max_scan_tuples,
							20000, 1, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);

	/* Same range as hash_mem_multiplier */
	DefineCustomRealVariable("hnsw.scan_mem_multiplier", "Sets the multiple of work_mem to use for iterative scans",
							 NULL, &hnsw_scan_mem_multiplier,
							 1, 1, 1000, PGC_USERSET, 0, NULL, NULL, NULL);

	MarkGUCPrefixReserved("hnsw");
}

/*
 * Get the name of index build phase
 */
static char *
hnswbuildphasename(int64 phasenum)
{
	switch (phasenum)
	{
		case PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE:
			return "initializing";
		case PROGRESS_HNSW_PHASE_LOAD:
			return "loading tuples";
		default:
			return NULL;
	}
}

/*
 * Estimate the cost of an index scan
 */
static void
hnswcostestimate(PlannerInfo *root, IndexPath *path, double loop_count,
				 Cost *indexStartupCost, Cost *indexTotalCost,
				 Selectivity *indexSelectivity, double *indexCorrelation,
				 double *indexPages)
{
	GenericCosts costs;
	int			m;
	double		ratio;
	double		startupPages;
	double		spc_seq_page_cost;
	Relation	index;

	/* Never use index without order */
	if (path->indexorderbys == NULL)
	{
		*indexStartupCost = get_float8_infinity();
		*indexTotalCost = get_float8_infinity();
		*indexSelectivity = 0;
		*indexCorrelation = 0;
		*indexPages = 0;
#if PG_VERSION_NUM >= 180000
		/* See "On disable_cost" thread on pgsql-hackers */
		path->path.disabled_nodes = 2;
#endif
		return;
	}

	MemSet(&costs, 0, sizeof(costs));

	genericcostestimate(root, path, loop_count, &costs);

	index = index_open(path->indexinfo->indexoid, NoLock);
	HnswGetMetaPageInfo(index, &m, NULL);
	index_close(index, NoLock);

	/*
	 * HNSW cost estimation follows a formula that accounts for the total
	 * number of tuples indexed combined with the parameters that most
	 * influence the duration of the index scan, namely: m - the number of
	 * tuples that are scanned in each step of the HNSW graph traversal
	 * ef_search - which influences the total number of steps taken at layer 0
	 *
	 * The source of the vector data can impact how many steps it takes to
	 * converge on the set of vectors to return to the executor. Currently, we
	 * use a hardcoded scaling factor (HNSWScanScalingFactor) to help
	 * influence that, but this could later become a configurable parameter
	 * based on the cost estimations.
	 *
	 * The tuple estimator formula is below:
	 *
	 * numIndexTuples = entryLevel * m + layer0TuplesMax * layer0Selectivity
	 *
	 * "entryLevel * m" represents the floor of tuples we need to scan to get
	 * to layer 0 (L0).
	 *
	 * "layer0TuplesMax" is the estimated total number of tuples we'd scan at
	 * L0 if we weren't discarding already visited tuples as part of the scan.
	 *
	 * "layer0Selectivity" estimates the percentage of tuples that are scanned
	 * at L0, accounting for previously visited tuples, multiplied by the
	 * "scalingFactor" (currently hardcoded).
	 */
	if (path->indexinfo->tuples > 0)
	{
		double		scalingFactor = 0.55;
		int			entryLevel = (int) (log(path->indexinfo->tuples) * HnswGetMl(m));
		int			layer0TuplesMax = HnswGetLayerM(m, 0) * hnsw_ef_search;
		double		layer0Selectivity = scalingFactor * log(path->indexinfo->tuples) / (log(m) * (1 + log(hnsw_ef_search)));

		ratio = (entryLevel * m + layer0TuplesMax * layer0Selectivity) / path->indexinfo->tuples;

		if (ratio > 1)
			ratio = 1;
	}
	else
		ratio = 1;

	get_tablespace_page_costs(path->indexinfo->reltablespace, NULL, &spc_seq_page_cost);

	/* Startup cost is cost before returning the first row */
	costs.indexStartupCost = costs.indexTotalCost * ratio;

	/* Adjust cost if needed since TOAST not included in seq scan cost */
	startupPages = costs.numIndexPages * ratio;
	if (startupPages > path->indexinfo->rel->pages && ratio < 0.5)
	{
		/* Change all page cost from random to sequential */
		costs.indexStartupCost -= startupPages * (costs.spc_random_page_cost - spc_seq_page_cost);

		/* Remove cost of extra pages */
		costs.indexStartupCost -= (startupPages - path->indexinfo->rel->pages) * spc_seq_page_cost;
	}

	*indexStartupCost = costs.indexStartupCost;
	*indexTotalCost = costs.indexTotalCost;
	*indexSelectivity = costs.indexSelectivity;
	*indexCorrelation = costs.indexCorrelation;
	*indexPages = costs.numIndexPages;
}

/*
 * Parse and validate the reloptions
 */
static bytea *
hnswoptions(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"m", RELOPT_TYPE_INT, offsetof(HnswOptions, m)},
		{"ef_construction", RELOPT_TYPE_INT, offsetof(HnswOptions, efConstruction)},

		// todo dkx
		{"m1", RELOPT_TYPE_INT, offsetof(HnswOptions, m1)},
		{"ef_construction1", RELOPT_TYPE_INT, offsetof(HnswOptions, efConstruction1)},
		{"m2", RELOPT_TYPE_INT, offsetof(HnswOptions, m2)},
		{"ef_construction2", RELOPT_TYPE_INT, offsetof(HnswOptions, efConstruction2)},
	};
	HnswOptions *opts = (HnswOptions *) build_reloptions(reloptions, validate,
													 hnsw_relopt_kind,
													 sizeof(HnswOptions),
													 tab, lengthof(tab));

	bool new_used =
		(opts->m1 > 0) || (opts->efConstruction1 > 0) ||
		(opts->m2 > 0) || (opts->efConstruction2 > 0);

	if (new_used)
	{
		/* 4 参数模式：要求四个都给齐 */
		if (opts->m1 <= 0 || opts->efConstruction1 <= 0 ||
			opts->m2 <= 0 || opts->efConstruction2 <= 0)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("hydex HNSW options: when using m1/ef_construction1/m2/ef_construction2, all four must be > 0")));
		}

		/* 这里通常选择：忽略 legacy m/efConstruction（保持兼容），或者你也可以在 validate 时更严格 */
	}
	else
	{
		/* 2 参数模式（旧逻辑）：把旧参数复制到两套上，后续统一用 m1/ef1/m2/ef2 */
		opts->m1 = opts->m;
		opts->efConstruction1 = opts->efConstruction;
		opts->m2 = opts->m;
		opts->efConstruction2 = opts->efConstruction;
	}

	return (bytea *) opts;
	// todo dkx


	// return (bytea *) build_reloptions(reloptions, validate,
	// 								  hnsw_relopt_kind,
	// 								  sizeof(HnswOptions),
	// 								  tab, lengthof(tab));
}

/*
 * Validate catalog entries for the specified operator class
 */
static bool
hnswvalidate(Oid opclassoid)
{
	return true;
}

/*
 * Define index handler
 *
 * See https://www.postgresql.org/docs/current/index-api.html
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hnswhandler);
Datum
hnswhandler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

	amroutine->amstrategies = 0;
	amroutine->amsupport = 3;
	amroutine->amoptsprocnum = 0;
	amroutine->amcanorder = false;
	amroutine->amcanorderbyop = true;
	amroutine->amcanbackward = false;	/* can change direction mid-scan */
	amroutine->amcanunique = false;
	amroutine->amcanmulticol = true;	// todo dkx
	amroutine->amoptionalkey = true;
	amroutine->amsearcharray = false;
	amroutine->amsearchnulls = false;
	amroutine->amstorage = false;
	amroutine->amclusterable = false;
	amroutine->ampredlocks = false;
	amroutine->amcanparallel = false;
#if PG_VERSION_NUM >= 170000
	amroutine->amcanbuildparallel = true;
#endif
	amroutine->amcaninclude = false;
	amroutine->amusemaintenanceworkmem = false; /* not used during VACUUM */
#if PG_VERSION_NUM >= 160000
	amroutine->amsummarizing = false;
#endif
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL;
	amroutine->amkeytype = InvalidOid;

	/* Interface functions */
	amroutine->ambuild = hnswbuild;			// todo dkx
	amroutine->ambuildempty = hnswbuildempty;
	amroutine->aminsert = hnswinsert;
#if PG_VERSION_NUM >= 170000
	amroutine->aminsertcleanup = NULL;
#endif
	amroutine->ambulkdelete = hnswbulkdelete;
	amroutine->amvacuumcleanup = hnswvacuumcleanup;
	amroutine->amcanreturn = NULL;
	amroutine->amcostestimate = hnswcostestimate;
	amroutine->amoptions = hnswoptions;		// todo dkx ok
	amroutine->amproperty = NULL;	/* TODO AMPROP_DISTANCE_ORDERABLE */
	amroutine->ambuildphasename = hnswbuildphasename;
	amroutine->amvalidate = hnswvalidate;				// todo dkx ok
#if PG_VERSION_NUM >= 140000
	amroutine->amadjustmembers = NULL;
#endif
	amroutine->ambeginscan = hnswbeginscan;
	amroutine->amrescan = hnswrescan;
	amroutine->amgettuple = hnswgettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = hnswendscan;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;

	/* Interface functions to support parallel index scans */
	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan = NULL;
	amroutine->amparallelrescan = NULL;

	PG_RETURN_POINTER(amroutine);
}
