//
// Created by kaixu on 2026/1/20.
//
/* src/rrf_scan.c */
#include "postgres.h"

#include "access/htup_details.h"
#include "catalog/pg_proc.h"
#include "executor/executor.h"
#include "fmgr.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/pathnode.h"
#include "optimizer/planner.h"
#include "parser/parse_clause.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"

static set_rel_pathlist_hook_type prev_set_rel_pathlist_hook = NULL;

/* ---------- CustomScan method declarations ---------- */

typedef struct VectorRRFScanState
{
    CustomScanState css;
} VectorRRFScanState;

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

/* ---------- CustomPath planning ---------- */

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

static FuncExpr *
find_rrf_sort_expr(PlannerInfo *root, bool *is_desc_out)
{
    Query *q = root->parse;
    ListCell *lc;

    if (q->sortClause == NIL)
        return NULL;

    /* MVP：只支持单一排序键（ORDER BY s1 DESC），多键先不处理 */
    if (list_length(q->sortClause) != 1)
        return NULL;

    foreach(lc, q->sortClause)
    {
        SortGroupClause *sgc = (SortGroupClause *) lfirst(lc);
        TargetEntry *tle;
        Expr *expr;
        Oid sortop;

        /* sortClause 引用 targetlist (ORDER BY s1) 时，这里能拿到 rrf(...) 那个 TLE */
        tle = get_sortgroupclause_tle(sgc, q->targetList);
        if (tle == NULL)
            return NULL;

        expr = (Expr *) tle->expr;
        expr = strip_expr_wrappers(expr);
        if (!IsA(expr, FuncExpr))
            return NULL;

        /* 判断是否 DESC：通过 sort operator 的策略比较复杂，这里用最简单可工作的办法：
           只要用户写 DESC，我们就把 is_desc_out 设 true（parse 阶段已把 sortop 设置好了）。
           MVP 阶段直接假设：你只用 DESC。 */
        sortop = sgc->sortop;
        (void) sortop;
        *is_desc_out = true;

        return (FuncExpr *) expr;
    }

    return NULL;
}

/* ---------- Helpers: pick a multicol index containing emb1 & emb2 ---------- */

static Oid
pick_index_for_vars(RelOptInfo *rel, Index rti, Var *v1, Var *v2)
{
    ListCell *lc;

    foreach(lc, rel->indexlist)
    {
        IndexOptInfo *idx = (IndexOptInfo *) lfirst(lc);

        if (idx->nkeycolumns < 2)
            continue;

        /* 只要 indexkeys 覆盖这两个列（顺序不强制，MVP 先放宽） */
        bool has1 = false, has2 = false;
        for (int i = 0; i < idx->nkeycolumns; i++)
        {
            if (idx->indexkeys[i] == v1->varattno)
                has1 = true;
            if (idx->indexkeys[i] == v2->varattno)
                has2 = true;
        }
        if (has1 && has2)
            return idx->indexoid;
    }

    return InvalidOid;
}

/* ---------- Hook: add CustomPath when ORDER BY rrf(...) ---------- */

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

    /* 确认函数名叫 rrf（避免误匹配别的 FuncExpr） */
    if (strcmp(get_func_name(fe->funcid), "rrf") != 0)
        return;

    /* rrf(emb1,q1,emb2,q2,...)：取出 emb1/emb2 两个 Var */
    if (list_length(fe->args) < 4)
        return;

    Expr *a0 = strip_expr_wrappers((Expr *) linitial(fe->args));
    Expr *a2 = strip_expr_wrappers((Expr *) lthird(fe->args));

    if (!IsA(a0, Var) || !IsA(a2, Var))
        return;

    Var *v1 = (Var *) a0;
    Var *v2 = (Var *) a2;

    /* 两个列必须来自同一张表（同一个 rti） */
    if (v1->varno != rti || v2->varno != rti)
        return;

    Oid indexoid = pick_index_for_vars(rel, rti, v1, v2);
    if (!OidIsValid(indexoid))
        return;

    /* 构造 CustomPath */
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

    /* 把执行所需信息塞进 custom_private（必须是可序列化 Node） */
    cpath->custom_private = list_make3(
        makeInteger(indexoid),
        makeInteger(is_desc ? 1 : 0),
        (Node *) copyObject(fe)  /* 把整个 rrf(...) FuncExpr 带到执行器 */
    );

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

    cscan->scan.plan.targetlist = tlist;
    cscan->scan.plan.qual = extract_actual_clauses(clauses, false);
    cscan->scan.scanrelid = rel->relid;

    /* 输出 targetlist（包含 rrf(...) AS s1 那一列） */
    cscan->custom_scan_tlist = tlist;

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
vector_rrf_begin(CustomScanState *node, EState *estate, int eflags)
{
    /* 第 3 步再实现：两路 topK + RRF merge */
    (void) node; (void) estate; (void) eflags;
}

static TupleTableSlot *
vector_rrf_exec(CustomScanState *node)
{
    ereport(ERROR,
            (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
             errmsg("VectorRRF CustomScan executor not implemented yet")));
    return NULL;
}

static void
vector_rrf_end(CustomScanState *node)
{
    (void) node;
}

static void
vector_rrf_rescan(CustomScanState *node)
{
    (void) node;
}

static void
vector_rrf_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
    CustomScan *cscan = (CustomScan *) node->ss.ps.plan;
    List *priv = cscan->custom_private;

    ExplainPropertyText("VectorRRF", "enabled", es);

    if (list_length(priv) >= 2)
    {
        int indexoid = intVal((Node *) linitial(priv));
        int is_desc = intVal((Node *) lsecond(priv));
        ExplainPropertyInteger("index_oid", NULL, indexoid, es);
        ExplainPropertyBool("order_desc", NULL, is_desc != 0, es);
    }

    (void) ancestors;
}

/* ---------- Extension init ---------- */

void
VectorRrfInit(void)
{
    RegisterCustomScanMethods(&vector_rrf_scan_methods); /* 注册 CustomScan methods（必须） */

    prev_set_rel_pathlist_hook = set_rel_pathlist_hook; /* 安装 hook */
    set_rel_pathlist_hook = vector_rrf_set_rel_pathlist_hook;
}