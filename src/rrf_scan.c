/* src/rrf_scan.c */
#include "postgres.h"

#include <string.h>
#include <limits.h>

#include "nodes/parsenodes.h"
#include "access/genam.h"
#include "access/table.h"
#include "access/tableam.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "fmgr.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/pathnode.h"
#include "optimizer/planner.h"
#include "parser/parse_clause.h"
#include "utils/builtins.h"
#include "utils/hsearch.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "hnswtopk.h"

#include "optimizer/paths.h"
#include "optimizer/clauses.h"

#include "access/htup_details.h"
#include "utils/syscache.h"
#include "optimizer/tlist.h"
#include "catalog/pg_operator.h"
#include "utils/fmgroids.h"
#include "catalog/indexing.h"

#include "parser/parse_func.h"   /* FuncnameGetCandidates, FuncCandidateList */
#include "nodes/pg_list.h"       /* List, T_OidList */

#include "catalog/namespace.h"  /* FuncnameGetCandidates */


extern List *extract_actual_clauses(List *quals, bool pseudoconstant);

void VectorRrfInit(void);
static set_rel_pathlist_hook_type prev_set_rel_pathlist_hook = NULL;
static Oid rrf_func_oid = InvalidOid;
static bool rrf_oids_loaded = false;
static List *rrf_func_oids = NIL;

static bool rrf_func_oids_inited = false;


double *current_rrf_score = NULL;

/* ---------------- Result structs ---------------- */

typedef struct RRFResultItem
{
    ItemPointerData tid;
    float8 score;

    /* optional debug */
    int32  r1;
    int32  r2;
    float8 d1;
    float8 d2;
} RRFResultItem;


typedef struct RRFHashEntry
{
    uint64 key;
    ItemPointerData tid;
    int32  r1;
    int32  r2;
    float8 d1;
    float8 d2;
    float8 score;
} RRFHashEntry;

typedef struct VectorRRFScanState
{
    CustomScanState css;

    Relation heapRel;
    Relation indexRel;
    Snapshot snapshot;

    TupleTableSlot *heapSlot;   /* for table_tuple_fetch_row_version */

    int col1;   /* 0-based index key position */
    int col2;

    /* ExprState for parameters */
    ExprState *op1_state;
    ExprState *op2_state;
    ExprState *q1_state;
    ExprState *q2_state;
    ExprState *k_state;
    ExprState *w1_state;
    ExprState *w2_state;
    ExprState *cand1_state;
    ExprState *cand2_state;
    ExprState *limit_state;


    Oid   op1;
    Oid   op2;
    int32 k_int;
    float8 k;
    float8 w1;
    float8 w2;
    int32 cand1;
    int32 cand2;
    int32 limit;


    RRFResultItem *results;
    int nresults;
    int cursor;

    MemoryContext rrf_mcxt;

    /* heap fetch via index (handles HOT redirects) */
    IndexFetchTableData *fetch;
} VectorRRFScanState;

/* ---------------- CustomScan method decls ---------------- */

static Node *vector_rrf_create_scan_state(CustomScan *cscan);
static void vector_rrf_begin(CustomScanState *node, EState *estate, int eflags);
static TupleTableSlot *vector_rrf_exec(CustomScanState *node);
static void vector_rrf_end(CustomScanState *node);
static void vector_rrf_rescan(CustomScanState *node);
static void vector_rrf_explain(CustomScanState *node, List *ancestors, ExplainState *es);

static CustomScanMethods vector_rrf_scan_methods = {
    .CustomName = "VectorRRF",
    .CreateCustomScanState = vector_rrf_create_scan_state
};

static CustomExecMethods vector_rrf_exec_methods = {
    .CustomName = "VectorRRF",
    .BeginCustomScan = vector_rrf_begin,
    .ExecCustomScan = vector_rrf_exec,
    .EndCustomScan = vector_rrf_end,
    .ReScanCustomScan = vector_rrf_rescan,
    .ExplainCustomScan = vector_rrf_explain
};

static void
load_rrf_func_oids(void)
{
    MemoryContext oldcxt;
    List *fname;
    FuncCandidateList clist;

    if (rrf_func_oids_inited)
        return;

    if (rrf_func_oids != NIL && rrf_func_oids->type != T_OidList)
        rrf_func_oids = NIL;

    oldcxt = MemoryContextSwitchTo(TopMemoryContext);

    fname = list_make1(makeString("rrf"));

#if PG_VERSION_NUM >= 170000
    /*
     * PG17: FuncnameGetCandidates() 多了一个参数（通常是 missing_ok）
     * 这里传 true：找不到也别报错，返回空即可
     */
    clist = FuncnameGetCandidates(fname,
                                 -1,    /* nargs */
                                 NIL,   /* argnames */
                                 false, /* expand_variadic */
                                 false, /* expand_defaults */
                                 false, /* include_out_arguments */
                                 true   /* missing_ok */
                                 );
#else
    clist = FuncnameGetCandidates(fname,
                                 -1,    /* nargs */
                                 NIL,   /* argnames */
                                 false, /* expand_variadic */
                                 false, /* expand_defaults */
                                 false  /* include_out_arguments */
                                 );
#endif

    for (; clist != NULL; clist = clist->next)
        rrf_func_oids = lappend_oid(rrf_func_oids, clist->oid);

    rrf_func_oids_inited = true;

    MemoryContextSwitchTo(oldcxt);
}




static inline bool
rrf_funcid_is_rrf(Oid maybe_funcid)
{
    if (!OidIsValid(maybe_funcid))
        return false;

#ifdef OPEROID
    /*
     * 保险：如果它其实是 operator OID（你之前遇到过），直接返回 false。
     * （避免把 operator oid 误判成 func oid）
     */
    if (SearchSysCacheExists1(OPEROID, ObjectIdGetDatum(maybe_funcid)))
        return false;
#endif

    load_rrf_func_oids();


    if (rrf_func_oids == NIL)
        return false;


    return list_member_oid(rrf_func_oids, maybe_funcid);
}



static void
rrf_assert_expr_type(Expr *e, const char *what)
{
    Oid t = exprType((Node *) e);
    if (!OidIsValid(t))
        ereport(ERROR,
                (errmsg("rrf: %s has invalid type Oid=0, node=%s",
                        what, nodeToString(e))));
}

/* ---------------- CustomPath planning ---------------- */

static Plan *vector_rrf_plan_custom_path(PlannerInfo *root,
                                         RelOptInfo *rel,
                                         CustomPath *best_path,
                                         List *tlist,
                                         List *clauses,
                                         List *custom_plans);

static CustomPathMethods vector_rrf_path_methods = {
    .CustomName = "VectorRRF",
    .PlanCustomPath = vector_rrf_plan_custom_path
};


static Expr *
strip_expr_wrappers(Expr *e)
{
    for (;;)
    {
        if (e == NULL)
            return NULL;
        if (IsA(e, RelabelType))
            e = ((RelabelType *) e)->arg;
        else if (IsA(e, CoerceViaIO))
            e = ((CoerceViaIO *) e)->arg;
        else
            return e;
    }
}

static Expr *
make_int4_const(int32 v)
{
    return (Expr *) makeConst(INT4OID, -1, InvalidOid, sizeof(int32),
                              Int32GetDatum(v), false, true);
}

static Expr *
make_float8_const(float8 v)
{
    return (Expr *) makeConst(FLOAT8OID, -1, InvalidOid, sizeof(float8),
                              Float8GetDatum(v), false, true);
}

static uint64
tid_to_key(const ItemPointer tid)
{
    return (((uint64) ItemPointerGetBlockNumber(tid)) << 16) |
           (uint64) ItemPointerGetOffsetNumber(tid);
}

static int
cmp_rrf_item_desc(const void *a, const void *b)
{
    const RRFResultItem *x = (const RRFResultItem *) a;
    const RRFResultItem *y = (const RRFResultItem *) b;

    if (x->score > y->score) return -1;
    if (x->score < y->score) return  1;

    int rx = (x->r1 > 0) ? x->r1 : INT_MAX;
    int ry = (y->r1 > 0) ? y->r1 : INT_MAX;
    if (rx != ry) return (rx < ry) ? -1 : 1;

    rx = (x->r2 > 0) ? x->r2 : INT_MAX;
    ry = (y->r2 > 0) ? y->r2 : INT_MAX;
    if (rx != ry) return (rx < ry) ? -1 : 1;

    return 0;
}

static TargetEntry *
find_tle_by_sortgroupref(List *tlist, Index ref)
{
    ListCell *lc;
    foreach (lc, tlist)
    {
        TargetEntry *tle = (TargetEntry *) lfirst(lc);
        if (tle->ressortgroupref == ref)
            return tle;
    }
    return NULL;
}


static FuncExpr *
find_rrf_sort_expr(PlannerInfo *root, bool *is_desc_out)
{
    Query *q = root->parse;

    if (q->sortClause == NIL || list_length(q->sortClause) != 1)
        return NULL;

    SortGroupClause *sgc = linitial_node(SortGroupClause, q->sortClause);


    TargetEntry *tle = find_tle_by_sortgroupref(q->targetList, sgc->tleSortGroupRef);
    if (tle == NULL)
        return NULL;

    Expr *expr = strip_expr_wrappers((Expr *) tle->expr);
    if (!IsA(expr, FuncExpr))
        return NULL;

    /* MVP：只支持 DESC（你现在的查询就是 DESC） */
    *is_desc_out = true;

    return (FuncExpr *) expr;
}


/* pick index + compute key positions */
static bool
pick_index_for_vars(RelOptInfo *rel, Var *v1, Var *v2, Oid *indexoid_out, int *col1_out, int *col2_out)
{
    ListCell *lc;

    foreach (lc, rel->indexlist)
    {
        IndexOptInfo *idx = (IndexOptInfo *) lfirst(lc);
        if (idx->nkeycolumns < 2)
            continue;

        int c1 = -1, c2 = -1;
        for (int i = 0; i < idx->nkeycolumns; i++)
        {
            if (idx->indexkeys[i] == v1->varattno)
                c1 = i;
            if (idx->indexkeys[i] == v2->varattno)
                c2 = i;
        }

        if (c1 >= 0 && c2 >= 0)
        {
            *indexoid_out = idx->indexoid;
            *col1_out = c1;
            *col2_out = c2;
            return true;
        }
    }

    return false;
}

/* mutator: replace rrf(...) with Var(INDEX_VAR, score_resno) */
static Node *
replace_rrf_with_score_var_mutator(Node *node, void *ctx)
{
    int score_resno = *((int *) ctx);

    if (node == NULL)
        return NULL;

    if (IsA(node, FuncExpr))
    {
        FuncExpr *fe = (FuncExpr *) node;

        if (rrf_funcid_is_rrf(fe->funcid))
        {
            return (Node *) makeVar(INDEX_VAR, score_resno, FLOAT8OID, -1, InvalidOid, 0);
        }

    }

    return expression_tree_mutator(node, replace_rrf_with_score_var_mutator, ctx);
}

static List *
replace_rrf_in_tlist(List *tlist, int score_resno)
{
    List *out = NIL;
    ListCell *lc;

    foreach (lc, tlist)
    {
        TargetEntry *tle = (TargetEntry *) lfirst(lc);
        TargetEntry *ntle = copyObject(tle);

        int ctx = score_resno;
        ntle->expr = (Expr *) replace_rrf_with_score_var_mutator((Node *) ntle->expr, &ctx);

        out = lappend(out, ntle);
    }

    return out;
}


static List *
build_custom_scan_tlist_for_table(PlannerInfo *root, RelOptInfo *rel, int *score_resno_out)
{

    RangeTblEntry *rte = (RangeTblEntry *) list_nth(root->parse->rtable, rel->relid - 1);

    if (rte == NULL || rte->rtekind != RTE_RELATION)
        ereport(ERROR, (errmsg("VectorRRF expects a base table RTE")));

    Relation tableRel = table_open(rte->relid, NoLock);
    TupleDesc desc = RelationGetDescr(tableRel);
    int natts = desc->natts;

    List *scan_tlist = NIL;

    for (int attno = 1; attno <= natts; attno++)
    {
        Form_pg_attribute att = TupleDescAttr(desc, attno - 1);

        Expr *expr;
        char *colname;

        if (att->attisdropped || !OidIsValid(att->atttypid))
        {

            expr = (Expr *) makeNullConst(TEXTOID, -1, InvalidOid);
            colname = pstrdup("dropped");
        }
        else
        {
            Var *v = makeVar(rel->relid,
                             attno,
                             att->atttypid,
                             att->atttypmod,
                             att->attcollation,
                             0);
            expr = (Expr *) v;
            colname = pstrdup(NameStr(att->attname));
        }

        TargetEntry *tle = makeTargetEntry(expr, attno, colname, false);
        scan_tlist = lappend(scan_tlist, tle);
    }


    /* last column is score */
    int score_resno = natts + 1;
    TargetEntry *score_tle = makeTargetEntry(make_float8_const(0.0),
                                             score_resno,
                                             pstrdup("rrf_score"),
                                             false);
    scan_tlist = lappend(scan_tlist, score_tle);

    table_close(tableRel, NoLock);

    *score_resno_out = score_resno;
    return scan_tlist;
}



static void
vector_rrf_set_rel_pathlist_hook(PlannerInfo *root, RelOptInfo *rel, Index rti, RangeTblEntry *rte)
{

    if (prev_set_rel_pathlist_hook)
        prev_set_rel_pathlist_hook(root, rel, rti, rte);


    if (rte->rtekind != RTE_RELATION)
        return;
    if (rel->reloptkind != RELOPT_BASEREL)
        return;

    bool is_desc = false;
    FuncExpr *fe = find_rrf_sort_expr(root, &is_desc);
    if (fe == NULL)
        return;

    if (!rrf_funcid_is_rrf(fe->funcid))
        return;

    if (!is_desc)
        return;


    int nargs = list_length(fe->args);
    if (nargs < 6)
        return;

    Expr *emb1 = strip_expr_wrappers((Expr *) list_nth(fe->args, 0));

    Expr *emb2 = strip_expr_wrappers((Expr *) list_nth(fe->args, 3));



    Expr *op1  = (Expr *) list_nth(fe->args, 1);
    Expr *q1   = (Expr *) list_nth(fe->args, 2);
    Expr *op2  = (Expr *) list_nth(fe->args, 4);
    Expr *q2   = (Expr *) list_nth(fe->args, 5);

    if (!IsA(emb1, Var) || !IsA(emb2, Var))
        return;


    rrf_assert_expr_type(op1, "op1");
    rrf_assert_expr_type(q1,  "q1");
    rrf_assert_expr_type(op2, "op2");
    rrf_assert_expr_type(q2,  "q2");

    Var *v1 = (Var *) emb1;
    Var *v2 = (Var *) emb2;


    if (v1->varno != rti || v2->varno != rti)
        return;

    Oid indexoid = InvalidOid;
    int col1 = -1, col2 = -1;
    if (!pick_index_for_vars(rel, v1, v2, &indexoid, &col1, &col2))
        return;


    Expr *k_expr     = (nargs >= 7)  ? (Expr *) list_nth(fe->args, 6)  : make_int4_const(60);
    Expr *w1_expr    = (nargs >= 8)  ? (Expr *) list_nth(fe->args, 7)  : make_float8_const(0.5);
    Expr *w2_expr    = (nargs >= 9)  ? (Expr *) list_nth(fe->args, 8)  : make_float8_const(0.5);
    Expr *cand1_expr = (nargs >= 10) ? (Expr *) list_nth(fe->args, 9)  : make_int4_const(200);
    Expr *cand2_expr = (nargs >= 11) ? (Expr *) list_nth(fe->args, 10) : make_int4_const(200);


    Expr *limit_expr = (root->parse->limitCount != NULL) ?
                       (Expr *) root->parse->limitCount :
                       make_int4_const(0);

    /* build CustomPath */
    CustomPath *cpath = makeNode(CustomPath);
    cpath->path.pathtype = T_CustomScan;
    cpath->path.parent = rel;
    cpath->path.param_info = NULL;
    cpath->path.rows = rel->rows;


    cpath->path.pathtarget = rel->reltarget;


    cpath->path.startup_cost = 0;
    cpath->path.total_cost = 1;


    cpath->path.pathkeys = root->sort_pathkeys;


    cpath->path.parallel_aware = false;
    cpath->path.parallel_safe = rel->consider_parallel;
    cpath->path.parallel_workers = 0;

    cpath->methods = &vector_rrf_path_methods;


    cpath->custom_private = NIL;
    cpath->custom_private = lappend(cpath->custom_private, makeInteger(indexoid));
    cpath->custom_private = lappend(cpath->custom_private, makeInteger(col1));
    cpath->custom_private = lappend(cpath->custom_private, makeInteger(col2));
    cpath->custom_private = lappend(cpath->custom_private, (Node *) copyObject(limit_expr));
    cpath->custom_private = lappend(cpath->custom_private, (Node *) copyObject(op1));
    cpath->custom_private = lappend(cpath->custom_private, (Node *) copyObject(op2));
    cpath->custom_private = lappend(cpath->custom_private, (Node *) copyObject(q1));
    cpath->custom_private = lappend(cpath->custom_private, (Node *) copyObject(q2));
    cpath->custom_private = lappend(cpath->custom_private, (Node *) copyObject(k_expr));
    cpath->custom_private = lappend(cpath->custom_private, (Node *) copyObject(w1_expr));
    cpath->custom_private = lappend(cpath->custom_private, (Node *) copyObject(w2_expr));
    cpath->custom_private = lappend(cpath->custom_private, (Node *) copyObject(cand1_expr));
    cpath->custom_private = lappend(cpath->custom_private, (Node *) copyObject(cand2_expr));

    add_path(rel, &cpath->path);
}



static Plan *
vector_rrf_plan_custom_path(PlannerInfo *root,
                            RelOptInfo *rel,
                            CustomPath *best_path,
                            List *tlist,
                            List *clauses,
                            List *custom_plans)
{
    CustomScan *cscan = makeNode(CustomScan);

    cscan->methods = &vector_rrf_scan_methods;
    cscan->custom_private = best_path->custom_private;
    cscan->custom_plans = NIL;
    cscan->custom_exprs = NIL;

    cscan->scan.scanrelid = rel->relid;


    int score_resno = 0;
    cscan->custom_scan_tlist = build_custom_scan_tlist_for_table(root, rel, &score_resno);


    List *rewritten_tlist = replace_rrf_in_tlist(tlist, score_resno);

    cscan->scan.plan.targetlist = rewritten_tlist;
    cscan->scan.plan.qual = extract_actual_clauses(clauses, false);

    return &cscan->scan.plan;
}



static Node *
vector_rrf_create_scan_state(CustomScan *cscan)
{
    VectorRRFScanState *st = palloc0(sizeof(*st));
    NodeSetTag(st, T_CustomScanState);
    st->css.methods = &vector_rrf_exec_methods;
    return (Node *) st;
}

static void
vector_rrf_eval_params(VectorRRFScanState *st)
{
    ExprContext *econtext = st->css.ss.ps.ps_ExprContext;

    bool isnull = false;

    Datum dop1 = ExecEvalExprSwitchContext(st->op1_state, econtext, &isnull);
    if (isnull)
        ereport(ERROR, (errmsg("op1 must not be NULL")));
    st->op1 = DatumGetObjectId(dop1);

    Datum dop2 = ExecEvalExprSwitchContext(st->op2_state, econtext, &isnull);
    if (isnull)
        ereport(ERROR, (errmsg("op2 must not be NULL")));
    st->op2 = DatumGetObjectId(dop2);

    Datum dk = ExecEvalExprSwitchContext(st->k_state, econtext, &isnull);
    if (isnull)
        ereport(ERROR, (errmsg("k must not be NULL")));
    st->k_int = DatumGetInt32(dk);
    if (st->k_int <= 0)
        ereport(ERROR, (errmsg("k must be > 0")));
    st->k = (float8) st->k_int;

    Datum dw1 = ExecEvalExprSwitchContext(st->w1_state, econtext, &isnull);
    if (isnull)
        ereport(ERROR, (errmsg("w1 must not be NULL")));
    st->w1 = DatumGetFloat8(dw1);

    Datum dw2 = ExecEvalExprSwitchContext(st->w2_state, econtext, &isnull);
    if (isnull)
        ereport(ERROR, (errmsg("w2 must not be NULL")));
    st->w2 = DatumGetFloat8(dw2);

    Datum dc1 = ExecEvalExprSwitchContext(st->cand1_state, econtext, &isnull);
    if (isnull)
        ereport(ERROR, (errmsg("cand1 must not be NULL")));
    st->cand1 = DatumGetInt32(dc1);

    Datum dc2 = ExecEvalExprSwitchContext(st->cand2_state, econtext, &isnull);
    if (isnull)
        ereport(ERROR, (errmsg("cand2 must not be NULL")));
    st->cand2 = DatumGetInt32(dc2);

    Datum dlim = ExecEvalExprSwitchContext(st->limit_state, econtext, &isnull);
    st->limit = isnull ? 0 : DatumGetInt32(dlim);
}

static void
vector_rrf_prepare_results(VectorRRFScanState *st)
{
    MemoryContext old = MemoryContextSwitchTo(st->rrf_mcxt);


    vector_rrf_eval_params(st);


    ExprContext *econtext = st->css.ss.ps.ps_ExprContext;
    bool isnull1 = false, isnull2 = false;
    Datum q1 = ExecEvalExprSwitchContext(st->q1_state, econtext, &isnull1);
    Datum q2 = ExecEvalExprSwitchContext(st->q2_state, econtext, &isnull2);
    if (isnull1 || isnull2)
        ereport(ERROR, (errmsg("rrf query vector must not be NULL")));

    if (st->cand1 <= 0 || st->cand2 <= 0)
        ereport(ERROR, (errmsg("cand1/cand2 must be > 0")));


    HnswTopKItem *list1 = palloc0(sizeof(HnswTopKItem) * st->cand1);
    HnswTopKItem *list2 = palloc0(sizeof(HnswTopKItem) * st->cand2);
    int n1 = 0, n2 = 0;

    HnswTopKForColumn(st->heapRel, st->indexRel, st->col1, st->op1, q1,
                      st->cand1, list1, &n1);
    HnswTopKForColumn(st->heapRel, st->indexRel, st->col2, st->op2, q2,
                      st->cand2, list2, &n2);




    HASHCTL ctl;
    memset(&ctl, 0, sizeof(ctl));
    ctl.keysize = sizeof(uint64);
    ctl.entrysize = sizeof(RRFHashEntry);

    HTAB *ht = hash_create("rrf_hash", Max(n1 + n2, 16), &ctl, HASH_ELEM | HASH_BLOBS);


    for (int i = 0; i < n1; i++)
    {
        uint64 key = tid_to_key(&list1[i].tid);
        bool found;
        RRFHashEntry *e = hash_search(ht, &key, HASH_ENTER, &found);
        if (!found)
        {
            e->key = key;
            e->tid = list1[i].tid;
            e->r1 = e->r2 = 0;
            e->d1 = e->d2 = 0;
            e->score = 0;
        }
        e->r1 = i + 1;
        e->d1 = list1[i].distance;
        e->score += st->w1 / (st->k + (float8)(i + 1));
    }


    for (int i = 0; i < n2; i++)
    {
        uint64 key = tid_to_key(&list2[i].tid);
        bool found;
        RRFHashEntry *e = hash_search(ht, &key, HASH_ENTER, &found);
        if (!found)
        {
            e->key = key;
            e->tid = list2[i].tid;
            e->r1 = e->r2 = 0;
            e->d1 = e->d2 = 0;
            e->score = 0;
        }
        e->r2 = i + 1;
        e->d2 = list2[i].distance;
        e->score += st->w2 / (st->k + (float8)(i + 1));
    }


    int cap = hash_get_num_entries(ht);
    RRFResultItem *arr = palloc0(sizeof(RRFResultItem) * cap);

    HASH_SEQ_STATUS seq;
    hash_seq_init(&seq, ht);

    int n = 0;
    for (;;)
    {
        RRFHashEntry *e = (RRFHashEntry *) hash_seq_search(&seq);
        if (!e) break;

        arr[n].tid = e->tid;
        arr[n].score = e->score;
        arr[n].r1 = e->r1;
        arr[n].r2 = e->r2;
        arr[n].d1 = e->d1;
        arr[n].d2 = e->d2;
        n++;
    }


    qsort(arr, n, sizeof(RRFResultItem), cmp_rrf_item_desc);

    if (st->limit > 0 && n > st->limit)
        n = st->limit;

    st->results = arr;
    st->nresults = n;
    st->cursor = 0;

    MemoryContextSwitchTo(old);
}

static void
vector_rrf_begin(CustomScanState *node, EState *estate, int eflags)
{
    VectorRRFScanState *st = (VectorRRFScanState *) node;
    CustomScan *cscan = (CustomScan *) node->ss.ps.plan;

    (void) eflags;

    st->snapshot = estate->es_snapshot;


    st->heapRel = node->ss.ss_currentRelation;
    if (st->heapRel == NULL)
        ereport(ERROR, (errmsg("VectorRRF requires ss_currentRelation for base table scan")));


    List *priv = cscan->custom_private;
    Oid indexoid = intVal((Node *) list_nth(priv, 0));
    st->col1 = intVal((Node *) list_nth(priv, 1));
    st->col2 = intVal((Node *) list_nth(priv, 2));

    st->indexRel = index_open(indexoid, AccessShareLock);


    Expr *limit_expr = (Expr *) list_nth(priv, 3);
    Expr *op1_expr    = (Expr *) list_nth(priv, 4);
    Expr *op2_expr    = (Expr *) list_nth(priv, 5);
    Expr *q1_expr     = (Expr *) list_nth(priv, 6);
    Expr *q2_expr     = (Expr *) list_nth(priv, 7);
    Expr *k_expr      = (Expr *) list_nth(priv, 8);
    Expr *w1_expr     = (Expr *) list_nth(priv, 9);
    Expr *w2_expr     = (Expr *) list_nth(priv, 10);
    Expr *cand1_expr  = (Expr *) list_nth(priv, 11);
    Expr *cand2_expr  = (Expr *) list_nth(priv, 12);

    st->limit_state = ExecInitExpr(limit_expr, (PlanState *) node);
    st->op1_state   = ExecInitExpr(op1_expr, (PlanState *) node);
    st->op2_state   = ExecInitExpr(op2_expr, (PlanState *) node);
    st->q1_state    = ExecInitExpr(q1_expr, (PlanState *) node);
    st->q2_state    = ExecInitExpr(q2_expr, (PlanState *) node);
    st->k_state     = ExecInitExpr(k_expr, (PlanState *) node);
    st->w1_state    = ExecInitExpr(w1_expr, (PlanState *) node);
    st->w2_state    = ExecInitExpr(w2_expr, (PlanState *) node);
    st->cand1_state = ExecInitExpr(cand1_expr, (PlanState *) node);
    st->cand2_state = ExecInitExpr(cand2_expr, (PlanState *) node);


    st->rrf_mcxt = AllocSetContextCreate(estate->es_query_cxt,
                                         "vector_rrf_mcxt",
                                         ALLOCSET_DEFAULT_SIZES);


    st->heapSlot = ExecInitExtraTupleSlot(estate,
                                          RelationGetDescr(st->heapRel),
                                          &TTSOpsBufferHeapTuple);


    st->fetch = table_index_fetch_begin(st->heapRel);

    /* compute results */
    MemoryContextReset(st->rrf_mcxt);
    vector_rrf_prepare_results(st);
}

static TupleTableSlot *
vector_rrf_exec(CustomScanState *node)
{
    VectorRRFScanState *st = (VectorRRFScanState *) node;
    ExprContext *econtext = node->ss.ps.ps_ExprContext;

    TupleTableSlot *scanSlot   = node->ss.ss_ScanTupleSlot;
    TupleTableSlot *resultSlot = node->ss.ps.ps_ResultTupleSlot;

    while (st->cursor < st->nresults)
    {
        int cur = st->cursor;              /* 当前要处理的下标 */
        RRFResultItem *it = &st->results[st->cursor++];

        /* --- DEBUG: cursor / tid / score --- */
        if (!ItemPointerIsValid(&it->tid))
        {
            elog(WARNING,
                 "VectorRRF exec: invalid TID at cursor=%d/%d (score=%.6f)",
                 cur, st->nresults, it->score);
            continue;
        }

        elog(WARNING,
             "VectorRRF exec: cursor=%d/%d tid=%u/%u score=%.6f",
             cur, st->nresults,
             ItemPointerGetBlockNumber(&it->tid),
             ItemPointerGetOffsetNumber(&it->tid),
             it->score);

        ExecClearTuple(st->heapSlot);



        bool ok = false;
        bool call_again = false;
        bool all_dead = false;


        do {
            call_again = false;
            ExecClearTuple(st->heapSlot);

#if PG_VERSION_NUM >= 160000
            ok = table_index_fetch_tuple(st->fetch,
                                         &it->tid,
                                         st->snapshot,
                                         st->heapSlot,
                                         &call_again,
                                         &all_dead);
#else
            /* older versions: signature may differ; keep for safety */
            ok = table_index_fetch_tuple(st->fetch,
                                         &it->tid,
                                         st->snapshot,
                                         st->heapSlot,
                                         &call_again);
#endif
        } while (call_again);

        elog(WARNING,
             "VectorRRF exec: index_fetch tid=%u/%u ok=%d all_dead=%d",
             ItemPointerGetBlockNumber(&it->tid),
             ItemPointerGetOffsetNumber(&it->tid),
             (int) ok,
             (int) all_dead);

        if (!ok)
            continue;



        current_rrf_score = &it->score;

        elog(WARNING, "VectorRRF exec: set global ptr for cursor=%d score=%.6f", st->cursor, it->score);


        ExecClearTuple(scanSlot);


        slot_getallattrs(st->heapSlot);

        int natts_heap = st->heapSlot->tts_tupleDescriptor->natts;
        int natts_scan = scanSlot->tts_tupleDescriptor->natts;

        elog(WARNING,
             "VectorRRF exec: natts_heap=%d natts_scan=%d (expect scan=heap+1)",
             natts_heap, natts_scan);


        if (natts_scan != natts_heap + 1)
            ereport(ERROR,
                    (errmsg("VectorRRF scan slot natts=%d does not match heap natts+1=%d",
                            natts_scan, natts_heap + 1)));

        for (int i = 0; i < natts_heap; i++)
        {
            scanSlot->tts_values[i] = st->heapSlot->tts_values[i];
            scanSlot->tts_isnull[i] = st->heapSlot->tts_isnull[i];
        }

        scanSlot->tts_values[natts_scan - 1] = Float8GetDatum(it->score);
        scanSlot->tts_isnull[natts_scan - 1] = false;

        ExecStoreVirtualTuple(scanSlot);


        econtext->ecxt_scantuple = scanSlot;
        if (node->ss.ps.qual && !ExecQual(node->ss.ps.qual, econtext))
        {
            elog(WARNING,
                 "VectorRRF exec: qual filtered tid=%u/%u",
                 ItemPointerGetBlockNumber(&it->tid),
                 ItemPointerGetOffsetNumber(&it->tid));
            continue;
        }

        if (node->ss.ps.ps_ProjInfo)
        {


            TupleTableSlot *out = ExecProject(node->ss.ps.ps_ProjInfo);


            elog(WARNING, "VectorRRF exec: passed score %.6f via global ptr", it->score);
            elog(WARNING,
                 "VectorRRF exec: returned projected tuple for tid=%u/%u",
                 ItemPointerGetBlockNumber(&it->tid),
                 ItemPointerGetOffsetNumber(&it->tid));

            return out;
        }


        ExecCopySlot(resultSlot, scanSlot);
        elog(WARNING,
             "VectorRRF exec: returned tuple (no projection) for tid=%u/%u",
             ItemPointerGetBlockNumber(&it->tid),
             ItemPointerGetOffsetNumber(&it->tid));
        return resultSlot;
    }

    elog(WARNING, "VectorRRF exec: EOF cursor=%d nresults=%d", st->cursor, st->nresults);
    return NULL;
}


static void
vector_rrf_end(CustomScanState *node)
{

    current_rrf_score = NULL;
    VectorRRFScanState *st = (VectorRRFScanState *) node;

    if (st->fetch)
        table_index_fetch_end(st->fetch);
    st->fetch = NULL;

    if (st->indexRel)
        index_close(st->indexRel, AccessShareLock);


}

static void
vector_rrf_rescan(CustomScanState *node)
{
    VectorRRFScanState *st = (VectorRRFScanState *) node;


    MemoryContextReset(st->rrf_mcxt);
    vector_rrf_prepare_results(st);
}

static void
vector_rrf_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
    VectorRRFScanState *st = (VectorRRFScanState *) node;
    CustomScan *cscan = (CustomScan *) node->ss.ps.plan;
    List *priv = cscan->custom_private;

    ExplainPropertyText("VectorRRF", "enabled", es);

    if (list_length(priv) >= 3)
    {
        ExplainPropertyInteger("index_oid", NULL, intVal((Node *) list_nth(priv, 0)), es);
        ExplainPropertyInteger("col1", NULL, intVal((Node *) list_nth(priv, 1)), es);
        ExplainPropertyInteger("col2", NULL, intVal((Node *) list_nth(priv, 2)), es);
    }


    ExplainPropertyInteger("k", NULL, st->k_int, es);
    ExplainPropertyFloat("w1", NULL, st->w1, 3, es);
    ExplainPropertyFloat("w2", NULL, st->w2, 3, es);
    ExplainPropertyInteger("cand1", NULL, st->cand1, es);
    ExplainPropertyInteger("cand2", NULL, st->cand2, es);
    ExplainPropertyInteger("limit", NULL, st->limit, es);
    ExplainPropertyInteger("op1_oid", NULL, st->op1, es);
    ExplainPropertyInteger("op2_oid", NULL, st->op2, es);

    (void) ancestors;
}



void
VectorRrfInit(void)
{
    RegisterCustomScanMethods(&vector_rrf_scan_methods);

    prev_set_rel_pathlist_hook = set_rel_pathlist_hook;
    set_rel_pathlist_hook = vector_rrf_set_rel_pathlist_hook;
}
