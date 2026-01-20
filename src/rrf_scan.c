/* src/rrf_scan.c */
#include "postgres.h"

#include <string.h>
#include <limits.h>

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
#include "nodes/parsenodes.h"
extern List *extract_actual_clauses(List *quals, bool pseudoconstant);

void VectorRrfInit(void);
static set_rel_pathlist_hook_type prev_set_rel_pathlist_hook = NULL;

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

/* hash entry 用 tid 做 key */
typedef struct RRFHashEntry
{
    uint64 key;            /* (block<<16)|offset */
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

    /* from planner (custom_private) */
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

    /* evaluated parameters */
    Oid   op1;
    Oid   op2;
    int32 k_int;
    float8 k;
    float8 w1;
    float8 w2;
    int32 cand1;
    int32 cand2;
    int32 limit;

    /* precomputed results */
    RRFResultItem *results;
    int nresults;
    int cursor;

    MemoryContext rrf_mcxt;
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

/* ---------- Helpers: find rrf() in ORDER BY ---------- */

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

/* 不用 get_sortgroupclause_tle，手工按 tleSortGroupRef 匹配 */
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

/* Find rrf() in ORDER BY target entry (MVP: single sort key, DESC only) */
static FuncExpr *
find_rrf_sort_expr(PlannerInfo *root, bool *is_desc_out)
{
    Query *q = root->parse;

    if (q->sortClause == NIL || list_length(q->sortClause) != 1)
        return NULL;

    SortGroupClause *sgc = (SortGroupClause *) linitial(q->sortClause);

    TargetEntry *tle = find_tle_by_sortgroupref(q->targetList, sgc->tleSortGroupRef);
    if (tle == NULL)
        return NULL;

    Expr *expr = strip_expr_wrappers((Expr *) tle->expr);
    if (!IsA(expr, FuncExpr))
        return NULL;

    /* PG17 有 reverse_sort：true 表示 DESC */
    // *is_desc_out = sgc->reverse_sort;
    // *is_desc_out = false;  // 默认 ASC
    (void) sgc->sortop;
    (void) sgc->nulls_first;
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
        const char *fname = get_func_name(fe->funcid);

        if (fname && strcmp(fname, "rrf") == 0)
        {
            /* use INDEX_VAR to reference custom scan tuple attribute */
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

/* Build custom_scan_tlist = all table columns + rrf_score placeholder */
static List *
build_custom_scan_tlist_for_table(PlannerInfo *root, RelOptInfo *rel, int *score_resno_out)
{
    /* rel->relid 是 rtable 的 1-based 索引 */
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

        Var *v = makeVar(rel->relid,
                         attno,
                         att->atttypid,
                         att->atttypmod,
                         att->attcollation,
                         0);

        TargetEntry *tle = makeTargetEntry((Expr *) v, attno,
                                           pstrdup(NameStr(att->attname)),
                                           false);
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

/* ---------------- Hook: add CustomPath ---------------- */

static void
vector_rrf_set_rel_pathlist_hook(PlannerInfo *root, RelOptInfo *rel, Index rti, RangeTblEntry *rte)
{
    /* 先让原 hook 跑 */
    if (prev_set_rel_pathlist_hook)
        prev_set_rel_pathlist_hook(root, rel, rti, rte);

    /* 只处理简单 base relation */
    if (rte->rtekind != RTE_RELATION)
        return;
    if (rel->reloptkind != RELOPT_BASEREL)
        return;

    bool is_desc = false;
    FuncExpr *fe = find_rrf_sort_expr(root, &is_desc);
    if (fe == NULL)
        return;

    const char *fname = get_func_name(fe->funcid);
    if (!fname || strcmp(fname, "rrf") != 0)
        return;

    if (!is_desc)
        return; /* RRF score only supports DESC in MVP */

    /*
     * New signature:
     * rrf(emb1, op1, q1, emb2, op2, q2, k?, w1?, w2?, cand1?, cand2?)
     */
    int nargs = list_length(fe->args);
    if (nargs < 6)
        return;

    Expr *emb1 = strip_expr_wrappers((Expr *) list_nth(fe->args, 0));
    Expr *op1  = strip_expr_wrappers((Expr *) list_nth(fe->args, 1));
    Expr *q1   = strip_expr_wrappers((Expr *) list_nth(fe->args, 2));
    Expr *emb2 = strip_expr_wrappers((Expr *) list_nth(fe->args, 3));
    Expr *op2  = strip_expr_wrappers((Expr *) list_nth(fe->args, 4));
    Expr *q2   = strip_expr_wrappers((Expr *) list_nth(fe->args, 5));

    if (!IsA(emb1, Var) || !IsA(emb2, Var))
        return;

    Var *v1 = (Var *) emb1;
    Var *v2 = (Var *) emb2;

    /* 两个列必须来自同一张表（同一个 rti） */
    if (v1->varno != rti || v2->varno != rti)
        return;

    Oid indexoid = InvalidOid;
    int col1 = -1, col2 = -1;
    if (!pick_index_for_vars(rel, v1, v2, &indexoid, &col1, &col2))
        return;

    /* optional args with defaults */
    Expr *k_expr     = (nargs >= 7)  ? (Expr *) list_nth(fe->args, 6)  : make_int4_const(60);
    Expr *w1_expr    = (nargs >= 8)  ? (Expr *) list_nth(fe->args, 7)  : make_float8_const(0.5);
    Expr *w2_expr    = (nargs >= 9)  ? (Expr *) list_nth(fe->args, 8)  : make_float8_const(0.5);
    Expr *cand1_expr = (nargs >= 10) ? (Expr *) list_nth(fe->args, 9)  : make_int4_const(200);
    Expr *cand2_expr = (nargs >= 11) ? (Expr *) list_nth(fe->args, 10) : make_int4_const(200);

    /* LIMIT expr */
    Expr *limit_expr = (root->parse->limitCount != NULL) ?
                       (Expr *) root->parse->limitCount :
                       make_int4_const(0);

    /* build CustomPath */
    CustomPath *cpath = makeNode(CustomPath);
    cpath->path.pathtype = T_CustomScan;
    cpath->path.parent = rel;
    cpath->path.param_info = NULL;
    cpath->path.rows = rel->rows;

    /* MVP：为了先让 planner 选中它，把 cost 压低；后面再做真实 cost 估算 */
    cpath->path.startup_cost = 0;
    cpath->path.total_cost = 1;

    /* 告诉 planner：我能提供 query 的排序（避免额外 Sort） */
    cpath->path.pathkeys = root->query_pathkeys;

    cpath->methods = &vector_rrf_path_methods;

    /* custom_private layout:
     * 0 indexoid
     * 1 col1
     * 2 col2
     * 3 limit_expr
     * 4 op1_expr
     * 5 op2_expr
     * 6 q1_expr
     * 7 q2_expr
     * 8 k_expr
     * 9 w1_expr
     * 10 w2_expr
     * 11 cand1_expr
     * 12 cand2_expr
     */
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

/* ---------- PlanCustomPath: CustomPath -> CustomScan ---------- */

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

    /* build scan tuple (all columns + score) */
    int score_resno = 0;
    cscan->custom_scan_tlist = build_custom_scan_tlist_for_table(root, rel, &score_resno);

    /* rewrite rrf(...) in this plan node targetlist to Var(INDEX_VAR, score_resno) */
    List *rewritten_tlist = replace_rrf_in_tlist(tlist, score_resno);

    cscan->scan.plan.targetlist = rewritten_tlist;
    cscan->scan.plan.qual = extract_actual_clauses(clauses, false);

    return &cscan->scan.plan;
}

/* ---------- Executor skeleton (MVP: error) ---------- */

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

    /* evaluate params (op/k/w/cand/limit) */
    vector_rrf_eval_params(st);

    /* Eval q1/q2 */
    ExprContext *econtext = st->css.ss.ps.ps_ExprContext;
    bool isnull1 = false, isnull2 = false;
    Datum q1 = ExecEvalExprSwitchContext(st->q1_state, econtext, &isnull1);
    Datum q2 = ExecEvalExprSwitchContext(st->q2_state, econtext, &isnull2);
    if (isnull1 || isnull2)
        ereport(ERROR, (errmsg("rrf query vector must not be NULL")));

    if (st->cand1 <= 0 || st->cand2 <= 0)
        ereport(ERROR, (errmsg("cand1/cand2 must be > 0")));

    /* topK arrays */
    HnswTopKItem *list1 = palloc0(sizeof(HnswTopKItem) * st->cand1);
    HnswTopKItem *list2 = palloc0(sizeof(HnswTopKItem) * st->cand2);
    int n1 = 0, n2 = 0;

    /* two topK */
    HnswTopKForColumn(st->heapRel, st->indexRel, st->col1, st->op1, q1,
                      st->cand1, list1, &n1);
    HnswTopKForColumn(st->heapRel, st->indexRel, st->col2, st->op2, q2,
                      st->cand2, list2, &n2);

    /* merge by tid */
    HASHCTL ctl;
    memset(&ctl, 0, sizeof(ctl));
    ctl.keysize = sizeof(uint64);
    ctl.entrysize = sizeof(RRFHashEntry);
    HTAB *ht = hash_create("rrf_hash", n1 + n2, &ctl, HASH_ELEM);

    /* 路1 */
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

    /* 路2 */
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

    /* 5) hash → array */
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

    /* 6) sort + 截断 top LIMIT */
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

    /* base rel */
    st->heapRel = node->ss.ss_currentRelation;
    if (st->heapRel == NULL)
        ereport(ERROR, (errmsg("VectorRRF requires ss_currentRelation for base table scan")));

    /* open index */
    List *priv = cscan->custom_private;
    Oid indexoid = intVal((Node *) list_nth(priv, 0));
    st->col1 = intVal((Node *) list_nth(priv, 1));
    st->col2 = intVal((Node *) list_nth(priv, 2));

    st->indexRel = index_open(indexoid, AccessShareLock);

    /* expr nodes from custom_private */
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

    /* memory context for precomputed results */
    st->rrf_mcxt = AllocSetContextCreate(estate->es_query_cxt,
                                         "vector_rrf_mcxt",
                                         ALLOCSET_DEFAULT_SIZES);

    /* heap fetch slot */
    st->heapSlot = ExecInitExtraTupleSlot(estate,
                                          RelationGetDescr(st->heapRel),
                                          &TTSOpsHeapTuple);

    /* compute results */
    MemoryContextReset(st->rrf_mcxt);
    vector_rrf_prepare_results(st);
}

static TupleTableSlot *
vector_rrf_exec(CustomScanState *node)
{
    VectorRRFScanState *st = (VectorRRFScanState *) node;
    ExprContext *econtext = node->ss.ps.ps_ExprContext;

    TupleTableSlot *scanSlot = node->ss.ss_ScanTupleSlot;
    TupleTableSlot *resultSlot = node->ss.ps.ps_ResultTupleSlot;

    while (st->cursor < st->nresults)
    {
        RRFResultItem *it = &st->results[st->cursor++];

        ExecClearTuple(st->heapSlot);

        /* 1) 抓 heap tuple */
        if (!table_tuple_fetch_row_version(st->heapRel, &it->tid,
                                           st->snapshot, st->heapSlot))
            continue;

        /* fill scanSlot = all heap cols + score (last) */
        ExecClearTuple(scanSlot);
        slot_getallattrs(st->heapSlot);

        int natts_heap = st->heapSlot->tts_tupleDescriptor->natts;
        int natts_scan = scanSlot->tts_tupleDescriptor->natts;

        /* expect scan = heap + 1 */
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

        /* qual */
        econtext->ecxt_scantuple = scanSlot;
        if (node->ss.ps.qual && !ExecQual(node->ss.ps.qual, econtext))
            continue;

        /* projection: fill ps_ResultTupleSlot and return */
        if (node->ss.ps.ps_ProjInfo)
            return ExecProject(node->ss.ps.ps_ProjInfo);

        ExecCopySlot(resultSlot, scanSlot);
        return resultSlot;
    }

    return NULL;
}

static void
vector_rrf_end(CustomScanState *node)
{
    VectorRRFScanState *st = (VectorRRFScanState *) node;

    if (st->indexRel)
        index_close(st->indexRel, AccessShareLock);

    /* rrf_mcxt 挂在 es_query_cxt 下，一般不手动删也行；想严谨可删 */
}

static void
vector_rrf_rescan(CustomScanState *node)
{
    VectorRRFScanState *st = (VectorRRFScanState *) node;

    /* if params may change, recompute */
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

    /* executor-time evaluated params */
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

/* ---------------- Extension init ---------------- */

void
VectorRrfInit(void)
{
    RegisterCustomScanMethods(&vector_rrf_scan_methods); /* 注册 CustomScan methods（必须） */

    prev_set_rel_pathlist_hook = set_rel_pathlist_hook; /* 安装 hook */
    set_rel_pathlist_hook = vector_rrf_set_rel_pathlist_hook;
}
