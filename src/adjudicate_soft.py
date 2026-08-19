#!/usr/bin/env python3
"""
src/adjudicate_soft.py — T1-1/T1-2「组级软排序 + 平票门控」裁决升级（纯 CPU）。

动机（R³-SQL / CHASE-SQL 对 MI-VAV 最大组胜出的批评）：大的错误签名组会压过
小的正确组。本脚本在 adjudicate_pool 的同一批 B1 候选池（1034 题 × 32 候选，
sft_phase1/sft_v2 双模型，全部 parse_success）上离线重算 5 个新裁决臂 + 基线，
全部走官方语义判定（exec match：postprocess → remove_distinct → replace_cur_year
→ result_eq，全实例等价）：

  arm_vav_multi_all      基线 = adjudicate_pool 的 choose_group_vav 最大组胜出
                         （both 池，74.3%），用于 fixed/broken 对比
  arm_soft_cross_w1/2/3  组评分 = 组大小 + w_cross × 组内模型数(1|2)，平票取
                         向量字符串最大；w_cross ∈ {1,2,3} 三档全部如实报，
                         不调参选最优
  arm_soft_ladder        确定性阶梯：① 双模型组优先 ② 组大小 ③ 向量字符串最大
  arm_gated_structural   平票门控：top1 与 top2 组大小差 ≤1 或 top1 组大小 <2 时
                         触发二次裁决（① 双模型组 ② 组大小 ③ JOIN 数更少 ④ 向量
                         字符串最大）；非触发题沿用最大组规则。JOIN 数用 sqlglot
                         解析 AST 数 exp.Join 节点；sqlglot 缺失或解析失败退化
                         正则数 \\bJOIN\\b 单词（方法记录于 summary/items）。

所有臂都在 both 池（全部 32 候选）上裁决。复用（import，不复制重写）
adjudicate_pool 的执行引擎/去重/签名/空零组过滤/官方判定函数；分组循环
（build_groups）逐行镜像 adjudicate_pool._arm_vav 的入组语义，额外收集组内
模型集合与代表 SQL（供软排序臂与判别力验证使用）。

新增 group_reps 阶段（默认开启，--no-group-correctness 关闭）：对每个签名组的
代表 SQL 做官方判定，用于验证本方法的前提假设——双模型组的正确率显著高于
单模型组（跨模型信号的判别力）。

输出
  outputs/adjudicate_soft/summary.json（各臂准确率 + winner 来源 + vs 基线
    fixed/broken + 门控触发统计 + 双模型组判别力验证）
  outputs/adjudicate_soft/items_<arm>.json（6 个，predicted_sql 与
    scripts/eval_official.sh 兼容；空胜者写 "SELECT 1" 不跳过）
  outputs/adjudicate_soft/group_level_correctness.json（组级判定明细）

用法
  # HPC 全量（默认 1034 题、全部实例）：
  python src/adjudicate_soft.py
  # 冒烟：--limit 30；实例数上限策略：--max-instances 8
"""

import argparse
import json
import re
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import adjudicate_pool as AP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ITEMS = PROJECT_ROOT / "outputs" / "eval_pool_b1" / "items.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "adjudicate_soft"
DEFAULT_SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"

POOL = "both"  # 新臂全部在双模型全池上裁决（与基线 both 池同口径）
BASELINE_ARM = "arm_vav_multi_all"
SOFT_CROSS_WS = (1, 2, 3)
SOFT_CROSS_ARMS = [f"arm_soft_cross_w{w}" for w in SOFT_CROSS_WS]
NEW_ARMS = SOFT_CROSS_ARMS + ["arm_soft_ladder", "arm_gated_structural"]
ALL_ARMS = [BASELINE_ARM] + NEW_ARMS

# ---- JOIN 计数（sqlglot 优先，正则退化）----
try:
    import sqlglot  # type: ignore
    from sqlglot import exp  # type: ignore

    _HAS_SQLGLOT = True
except ImportError:  # 退化：正则（记录于 summary.meta.join_counter_method）
    sqlglot = None
    exp = None
    _HAS_SQLGLOT = False

_JOIN_RE = re.compile(r"\bJOIN\b", flags=re.IGNORECASE)


def _sqlglot_join_count(sql: str) -> Optional[int]:
    """sqlglot AST 数 exp.Join 节点；解析失败返回 None（调用方退化正则）。"""
    if not _HAS_SQLGLOT:
        return None
    try:
        tree = sqlglot.parse_one(sql or "", read="sqlite")
    except Exception:
        return None
    join_cls = getattr(exp, "Join", None)
    if join_cls is None:
        return sum(1 for n in tree.walk() if getattr(n, "key", None) == "join")
    return len([n for n in tree.find_all(join_cls)])


def count_joins(sql: Any) -> Tuple[int, str]:
    """返回 (JOIN 数, 计数方法)。sqlglot 可用 → AST 数 exp.Join；缺失/解析失败 →
    正则数 \\bJOIN\\b 单词（每条 JOIN 子句恰好一个 JOIN 关键字，INNER/LEFT/RIGHT/
    FULL/CROSS/NATURAL 均覆盖）。已知局限：UNION 分支的 JOIN 依赖 sqlglot 解析
    范围，不保证逐分支覆盖（记录为风险点）。"""
    text = (sql or "")
    n = _sqlglot_join_count(text)
    if n is not None:
        return n, "sqlglot"
    return len(_JOIN_RE.findall(text)), "regex"


# ===================================================================
# 分组与组级软排序（分组循环逐行镜像 adjudicate_pool._arm_vav）
# ===================================================================


def build_groups(entries: List[Dict[str, Any]], sigs_per_entry: List[List[str]],
                 votes: Dict[int, int], n_instances: int) -> Tuple[Dict, int, int]:
    """镜像 adjudicate_pool._arm_vav 的入组循环：只 SUCCESS 候选入组（签名向量
    含任一 ERROR 分量的候选整条排除）；size = 池内票数加权（去重后按票数计）。
    额外收集 groups[key]["models"]（组内模型并集）与 members（去重组下标）。
    返回 (groups, grouped 票数, excluded 票数)。"""
    groups: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    grouped = 0
    excluded = 0
    for ei, cnt in sorted(votes.items()):
        sigs = sigs_per_entry[ei][:n_instances]
        if any(s == AP.ERROR_SIG for s in sigs):
            excluded += cnt
            continue
        grouped += cnt
        key = tuple(sigs)
        g = groups.setdefault(key, {"size": 0, "members": [], "models": set()})
        g["size"] += cnt
        g["members"].append(ei)
        for m in entries[ei]["models"]:
            if m:
                g["models"].add(m)
    return groups, grouped, excluded


def rankable_groups(groups: Dict) -> List[Tuple[Tuple[str, ...], Dict[str, Any]]]:
    """先硬跳过空组/全零组（与 AP.choose_group_vav 一致）；全被过滤 → 退回全组。
    空组 dict → []（调用方走 NO_RESULTS → fallback_maj）。"""
    if not groups:
        return []
    filtered = [
        (k, m) for k, m in groups.items()
        if not AP.vector_is_empty(k) and not AP.vector_is_all_zero(k)
    ]
    return filtered if filtered else list(groups.items())


def _group_rep(entries: List[Dict[str, Any]], g: Dict[str, Any]) -> Dict[str, Any]:
    """组内代表：min_sample_idx 最小、再 key 最小（与 _arm_vav 胜者选择一致）。"""
    best = min(g["members"], key=lambda ei: (entries[ei]["min_sample_idx"], entries[ei]["key"]))
    return entries[best]


def _joins_cached(rep: Dict[str, Any],
                  cache: Dict[str, Tuple[int, str]]) -> Tuple[int, str]:
    """按归一化 SQL 文本缓存 JOIN 计数（跨题同文本共享）。"""
    key = rep["key"]
    v = cache.get(key)
    if v is None:
        v = count_joins(rep["sql_text"])
        cache[key] = v
    return v


def _base_record(entries: List[Dict[str, Any]], chosen_key: Tuple[str, ...],
                 chosen_g: Dict[str, Any], source: str, n_used: int, grouped: int,
                 excluded: int, joins_cache: Dict[str, Tuple[int, str]]) -> Dict[str, Any]:
    """组胜者 → 记录（字段与 adjudicate_pool 胜者记录兼容 + 软排序扩展字段）。"""
    rep = _group_rep(entries, chosen_g)
    n_joins, jc = _joins_cached(rep, joins_cache)
    return {
        "source": source, "text": rep["sql_text"], "votes": chosen_g["size"],
        "group_key": str(chosen_key), "group_size": chosen_g["size"],
        "instances_used": n_used, "vav_grouped": grouped, "vav_excluded": excluded,
        "winner_models": len(chosen_g["models"]),
        "winner_dual": len(chosen_g["models"]) >= 2,
        "n_joins": n_joins, "join_counter": jc,
    }


def _fallback_record(entries: List[Dict[str, Any]], votes: Dict[int, int],
                     n_used: int, grouped: int, excluded: int) -> Dict[str, Any]:
    """NO_RESULTS（无组）→ 回退同池 arm_maj（与 _arm_vav 的 fallback 链一致）。"""
    fb = AP._arm_maj(POOL, entries, {POOL: votes})
    if fb["source"] == "no_pool":
        return {"source": "no_pool", "text": None, "votes": 0, "group_key": None,
                "group_size": 0, "instances_used": n_used,
                "vav_grouped": grouped, "vav_excluded": excluded,
                "winner_models": None, "winner_dual": False,
                "n_joins": None, "join_counter": None}
    return {"source": "fallback_maj", "text": fb["text"], "votes": fb["votes"],
            "group_key": None, "group_size": fb["votes"],
            "instances_used": n_used, "vav_grouped": grouped, "vav_excluded": excluded,
            "winner_models": None, "winner_dual": False,
            "n_joins": None, "join_counter": None}


def arm_baseline(entries: List[Dict[str, Any]], sigs_per_entry: List[List[str]],
                 votes: Dict[int, int], instances: List[str],
                 joins_cache: Dict[str, Tuple[int, str]]) -> Dict[str, Any]:
    """基线 arm_vav_multi_all（both 池）：与 adjudicate_pool._arm_vav 完全同语义
    （同一分组循环 + 同一 choose_group_vav + 同一组内代表选择），但用本文件的
    build_groups 以便携带组内模型集合（供判别力验证）。与 _arm_vav 产出的
    predicted_sql 逐题一致（冒烟阶段有对比验证）。"""
    n_used = len(instances)
    groups, grouped, excluded = build_groups(entries, sigs_per_entry, votes, n_used)
    if not groups:
        return _fallback_record(entries, votes, n_used, grouped, excluded)
    chosen = AP.choose_group_vav(groups)
    if chosen is None:
        return _fallback_record(entries, votes, n_used, grouped, excluded)
    return _base_record(entries, chosen, groups[chosen], "vav",
                        n_used, grouped, excluded, joins_cache)


def arm_soft_cross(w: int, entries: List[Dict[str, Any]],
                   sigs_per_entry: List[List[str]], votes: Dict[int, int],
                   instances: List[str],
                   joins_cache: Dict[str, Tuple[int, str]]) -> Dict[str, Any]:
    """组评分 = 组大小 + w × 组内模型数（1|2）；平票取向量字符串最大。"""
    n_used = len(instances)
    groups, grouped, excluded = build_groups(entries, sigs_per_entry, votes, n_used)
    ranked = rankable_groups(groups)
    if not ranked:
        return _fallback_record(entries, votes, n_used, grouped, excluded)
    chosen_key, chosen_g = max(
        ranked, key=lambda km: (km[1]["size"] + w * len(km[1]["models"]), str(km[0])))
    return _base_record(entries, chosen_key, chosen_g, "soft_cross",
                        n_used, grouped, excluded, joins_cache)


def arm_soft_ladder(entries: List[Dict[str, Any]], sigs_per_entry: List[List[str]],
                    votes: Dict[int, int], instances: List[str],
                    joins_cache: Dict[str, Tuple[int, str]]) -> Dict[str, Any]:
    """确定性阶梯：① 双模型组优先 ② 组大小 ③ 向量字符串最大。"""
    n_used = len(instances)
    groups, grouped, excluded = build_groups(entries, sigs_per_entry, votes, n_used)
    ranked = rankable_groups(groups)
    if not ranked:
        return _fallback_record(entries, votes, n_used, grouped, excluded)
    chosen_key, chosen_g = max(
        ranked, key=lambda km: (1 if len(km[1]["models"]) >= 2 else 0,
                                km[1]["size"], str(km[0])))
    return _base_record(entries, chosen_key, chosen_g, "soft_ladder",
                        n_used, grouped, excluded, joins_cache)


def arm_gated_structural(entries: List[Dict[str, Any]],
                         sigs_per_entry: List[List[str]], votes: Dict[int, int],
                         instances: List[str],
                         joins_cache: Dict[str, Tuple[int, str]]) -> Dict[str, Any]:
    """平票门控：top1/top2（最大组规则排序）组大小差 ≤1 或 top1 < 2 → 二次裁决
    （① 双模型组 ② 组大小 ③ JOIN 更少 ④ 向量字符串最大）；非触发沿用最大组规则
    （= 基线语义）。"""
    n_used = len(instances)
    groups, grouped, excluded = build_groups(entries, sigs_per_entry, votes, n_used)
    ranked = rankable_groups(groups)
    if not ranked:
        rec = _fallback_record(entries, votes, n_used, grouped, excluded)
        rec.update({"gated_triggered": False, "top1_size": None, "top2_size": None,
                    "gated_changed_winner": False})
        return rec
    ordered = sorted(ranked, key=lambda km: (km[1]["size"], str(km[0])), reverse=True)
    top1_key, top1_g = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else None
    size1 = top1_g["size"]
    size2 = top2[1]["size"] if top2 is not None else None
    triggered = (size1 < 2) or (top2 is not None and size1 - size2 <= 1)
    if not triggered:
        rec = _base_record(entries, top1_key, top1_g, "vav",
                           n_used, grouped, excluded, joins_cache)
        rec.update({"gated_triggered": False, "top1_size": size1, "top2_size": size2,
                    "gated_changed_winner": False})
        return rec

    def gated_key(km: Tuple[Tuple[str, ...], Dict[str, Any]]) -> Tuple:
        k, g = km
        rep = _group_rep(entries, g)
        n_joins, _jc = _joins_cached(rep, joins_cache)
        return (1 if len(g["models"]) >= 2 else 0, g["size"], -n_joins, str(k))

    chosen_key, chosen_g = max(ranked, key=gated_key)
    rec = _base_record(entries, chosen_key, chosen_g, "gated",
                       n_used, grouped, excluded, joins_cache)
    rec.update({"gated_triggered": True, "top1_size": size1, "top2_size": size2,
                "gated_changed_winner": str(chosen_key) != str(top1_key)})
    return rec


# ===================================================================
# 单题裁决
# ===================================================================


def adjudicate_question(item: Dict[str, Any], engine: AP.ExecutionEngine,
                        db_instances: List[str]) -> Dict[str, Any]:
    """对一题完成 去重 → 签名向量 → 6 臂（both 池）裁决。
    判定所需执行由 main 在 phase 2/3 并行补齐（本函数只读执行缓存）。
    返回含 "groups"（供 group_reps 阶段做组级正确率验证）。"""
    candidates = item.get("candidates") or []
    instances = db_instances

    entries = AP._dedupe(candidates)

    sigs_per_entry: List[List[str]] = []
    for e in entries:
        if not (e["sql_text"] or "").strip():
            sigs = [AP.ERROR_SIG] * len(instances)
        else:
            sigs = [AP.outcome_signature(engine.get(e["sql_text"], inst))
                    for inst in instances]
        sigs_per_entry.append(sigs)

    # both 池票数（与 adjudicate_pool 的 pool_entry_votes["both"] 一致）
    votes: Dict[int, int] = defaultdict(int)
    for c in candidates:
        ck = AP.normalize_for_dedup(c.get("sql"))
        for ei, e in enumerate(entries):
            if ck == e["key"]:
                votes[ei] += 1
                break

    joins_cache: Dict[str, Tuple[int, str]] = {}
    results: Dict[str, Dict[str, Any]] = {}
    results[BASELINE_ARM] = arm_baseline(entries, sigs_per_entry, votes, instances,
                                         joins_cache)
    for w in SOFT_CROSS_WS:
        results[f"arm_soft_cross_w{w}"] = arm_soft_cross(
            w, entries, sigs_per_entry, votes, instances, joins_cache)
    results["arm_soft_ladder"] = arm_soft_ladder(
        entries, sigs_per_entry, votes, instances, joins_cache)
    results["arm_gated_structural"] = arm_gated_structural(
        entries, sigs_per_entry, votes, instances, joins_cache)

    for rec in results.values():
        # empty_winner = 有胜者但胜者 SQL 文本为空（fallback_maj 选中空文本组）；
        # no_pool（池内无候选）不算 empty_winner（与 adjudicate_pool 同语义）
        rec["empty_winner"] = (rec["text"] == "")

    groups, _g, _e = build_groups(entries, sigs_per_entry, votes, len(instances))
    return {
        "item": item,
        "entries": entries,
        "groups": groups,
        "num_candidates": len(candidates),
        "num_unique_candidates": len(entries),
        "num_instances": len(instances),
        "results": results,
    }


# ===================================================================
# 主流程
# ===================================================================


def _agg_records(recs: List[Dict[str, Any]], key: str = "correct") -> Dict[str, Any]:
    n = len(recs)
    c = sum(1 for r in recs if r.get(key))
    return {"n": n, "correct": c, "accuracy": round(c / n, 4) if n else None}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="T1-1/T1-2 组级软排序 + 平票门控裁决（纯 CPU）")
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--spider-dir", type=Path, default=DEFAULT_SPIDER_DIR)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--query-timeout", type=float, default=30.0)
    ap.add_argument("--max-vm-steps", type=int, default=5_000_000)
    ap.add_argument("--row-cap", type=int, default=100_000)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--keep-distinct", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="只裁决前 N 题（冒烟）")
    ap.add_argument("--no-group-correctness", action="store_true",
                    help="跳过 group_reps 阶段（双模型组判别力验证需要该阶段，默认开启）")
    args = ap.parse_args(argv)

    AP.rng = random.Random(args.seed)

    items = AP._load_items(args.items)
    if args.limit:
        items = items[: args.limit]
    print(f"[adjudicate_soft] {len(items)} 题，实例上限 {args.max_instances or '全部'}，"
          f"线程 {args.threads}，查询超时 {args.query_timeout}s，"
          f"JOIN 计数方法 {'sqlglot' if _HAS_SQLGLOT else 'regex'}", file=sys.stderr)

    database_dir = args.spider_dir / "database"
    engine = AP.ExecutionEngine(args.threads, args.query_timeout, args.max_vm_steps,
                                args.row_cap)

    db_instances_cache: Dict[str, List[str]] = {}

    def instances_for(db_id: str) -> List[str]:
        if db_id not in db_instances_cache:
            db_instances_cache[db_id] = AP.list_instances(
                str(database_dir / db_id), db_id, args.max_instances)
        return db_instances_cache[db_id]

    # ---- Phase 1: 所有唯一候选 SQL × 实例 并行执行（跨候选缓存，与 AP 同模式）----
    phase1_tasks: List[Tuple[str, str]] = []
    for item in items:
        insts = instances_for(item.get("db_id", ""))
        for e in AP._dedupe(item.get("candidates") or []):
            text = (e["sql_text"] or "").strip()
            if not text:
                continue
            for inst in insts:
                phase1_tasks.append((text, inst))
    phase1_tasks = list(set(phase1_tasks))
    print(f"[adjudicate_soft] phase1: {len(phase1_tasks)} 个唯一 (sql, db_path) 任务",
          file=sys.stderr)
    t0 = time.perf_counter()
    engine.run(phase1_tasks, phase="grouping")
    print(f"[adjudicate_soft] phase1 完成: {engine._stats['grouping']}", file=sys.stderr)

    # ---- 每题裁决（胜者选择，不执行 SQL）----
    per_question: List[Dict[str, Any]] = []
    for qi, item in enumerate(items):
        per_question.append(adjudicate_question(
            item, engine, instances_for(item.get("db_id", ""))))
        if (qi + 1) % 50 == 0 or qi + 1 == len(items):
            print(f"[adjudicate_soft] 裁决 {qi + 1}/{len(items)} 题 "
                  f"({time.perf_counter() - t0:.1f}s)", file=sys.stderr)

    # ---- Phase 2: 判定所需（gold 变换后 + 各臂胜者变换后）SQL × 实例 ----
    phase2_tasks: List[Tuple[str, str]] = []
    for qc in per_question:
        item = qc["item"]
        insts = instances_for(item.get("db_id", ""))
        gold_t = AP.official_transform(item.get("gold_sql") or "", is_pred=False,
                                       keep_distinct=args.keep_distinct)
        for inst in insts:
            phase2_tasks.append((gold_t, inst))
        for arm, rec in qc["results"].items():
            if rec["text"] is None:
                continue
            pred_t = AP.official_transform(rec["text"], is_pred=True,
                                           keep_distinct=args.keep_distinct)
            for inst in insts:
                phase2_tasks.append((pred_t, inst))
    phase2_tasks = list(set(phase2_tasks))
    engine.run(phase2_tasks, phase="judgment")
    print(f"[adjudicate_soft] phase2 完成: {engine._stats['judgment']}", file=sys.stderr)

    # ---- Phase 3 (group_reps): 每个签名组代表 SQL 的官方判定（判别力验证）----
    group_reps: List[Dict[str, Any]] = []
    if not args.no_group_correctness:
        phase3_tasks: List[Tuple[str, str]] = []
        for qi, qc in enumerate(per_question):
            item = qc["item"]
            insts = instances_for(item.get("db_id", ""))
            for g in qc["groups"].values():
                rep = _group_rep(qc["entries"], g)
                rep_t = AP.official_transform(rep["sql_text"], is_pred=True,
                                              keep_distinct=args.keep_distinct)
                group_reps.append({
                    "qi": qi, "db_id": item.get("db_id", ""),
                    "dataset_index": item.get("dataset_index", item.get("di")),
                    "gold_raw": item.get("gold_sql") or "",
                    "size": g["size"], "dual": len(g["models"]) >= 2,
                    "models": sorted(str(m) for m in g["models"]),
                    "rep_text": rep["sql_text"],
                })
                for inst in insts:
                    phase3_tasks.append((rep_t, inst))
        phase3_tasks = list(set(phase3_tasks))
        print(f"[adjudicate_soft] phase3(group_reps): {len(group_reps)} 个组代表，"
              f"{len(phase3_tasks)} 个判定任务（缓存命中的不重跑）", file=sys.stderr)
        engine.run(phase3_tasks, phase="group_reps")
        print(f"[adjudicate_soft] phase3 完成: {engine._stats['group_reps']}",
              file=sys.stderr)

    # ---- 重新判定（phase 2/3 已补齐所有 outcome，纯内存计算）----
    for qc in per_question:
        item = qc["item"]
        gold_raw = item.get("gold_sql") or ""
        insts = instances_for(item.get("db_id", ""))
        for arm, rec in qc["results"].items():
            j = AP._judge_winner(rec["text"], gold_raw, insts, engine, args.keep_distinct)
            rec["is_correct"] = j["correct"]
            rec["gold_exec_error"] = j["gold_exec_error"]
            rec["order_matters"] = j["order_matters"]
    for gr in group_reps:
        insts = instances_for(gr["db_id"])
        j = AP._judge_winner(gr["rep_text"], gr["gold_raw"], insts, engine,
                             args.keep_distinct)
        gr["correct"] = j["correct"]

    # ---- 汇总 ----
    dataset_stats: Dict[str, Any] = {
        "total_questions": len(items),
        "questions_with_no_instances": 0,
        "questions_with_gold_exec_error": 0,
        "db_instance_count": {
            db: len(insts) for db, insts in db_instances_cache.items()},
    }
    total_cands = 0
    unique_cands = 0
    max_dup = 0
    for qc in per_question:
        total_cands += qc["num_candidates"]
        unique_cands += qc["num_unique_candidates"]
        for e in qc["entries"]:
            max_dup = max(max_dup, e["count"] - 1)
        if qc["num_instances"] == 0:
            dataset_stats["questions_with_no_instances"] += 1
        if any(r["gold_exec_error"] for r in qc["results"].values()):
            dataset_stats["questions_with_gold_exec_error"] += 1
    dedup_stats = {
        "total_candidates": total_cands,
        "unique_after_dedup": unique_cands,
        "merged_duplicates": total_cands - unique_cands,
        "max_dup_count": max_dup,
    }

    cells: Dict[str, Dict[str, Any]] = {}
    for arm in ALL_ARMS:
        cell: Dict[str, Any] = {
            "total": len(items), "correct": 0, "accuracy": 0.0,
            "winner_sources": Counter(), "empty_winner": 0, "gold_exec_error": 0,
            "candidates_available": 0,
            "gated_triggered": 0, "gated_triggered_correct": 0,
            "gated_changed_winner": 0,
        }
        for qc in per_question:
            rec = qc["results"][arm]
            if rec["source"] != "no_pool":
                cell["candidates_available"] += 1
            cell["winner_sources"][rec["source"]] += 1
            if rec.get("empty_winner"):
                cell["empty_winner"] += 1
            if rec.get("gold_exec_error"):
                cell["gold_exec_error"] += 1
            if rec.get("is_correct"):
                cell["correct"] += 1
            if rec.get("gated_triggered"):
                cell["gated_triggered"] += 1
                if rec.get("is_correct"):
                    cell["gated_triggered_correct"] += 1
            if rec.get("gated_changed_winner"):
                cell["gated_changed_winner"] += 1
        cell["accuracy"] = round(cell["correct"] / cell["total"], 4) if cell["total"] else 0.0
        cell["winner_sources"] = dict(cell["winner_sources"])
        cells[arm] = cell

    # ---- vs 基线 fixed/broken ----
    base_correct = [qc["results"][BASELINE_ARM]["is_correct"] for qc in per_question]
    vs_baseline: Dict[str, Dict[str, Any]] = {}
    for arm in NEW_ARMS:
        fixed = broken = same_r = same_w = 0
        f_idx: List[Any] = []
        b_idx: List[Any] = []
        for i, qc in enumerate(per_question):
            a = qc["results"][arm]["is_correct"]
            b = base_correct[i]
            idx = qc["item"].get("dataset_index", qc["item"].get("di"))
            if not b and a:
                fixed += 1
                f_idx.append(idx)
            elif b and not a:
                broken += 1
                b_idx.append(idx)
            elif b:
                same_r += 1
            else:
                same_w += 1
        vs_baseline[arm] = {
            "baseline_accuracy": cells[BASELINE_ARM]["accuracy"],
            "arm_accuracy": cells[arm]["accuracy"],
            "delta": round(cells[arm]["accuracy"] - cells[BASELINE_ARM]["accuracy"], 4),
            "fixed": fixed, "broken": broken, "net": fixed - broken,
            "same_right": same_r, "same_wrong": same_w,
            "fixed_indices": f_idx, "broken_indices": b_idx,
        }

    # ---- 门控触发统计（关键统计 ②）----
    gated_idx = [i for i, qc in enumerate(per_question)
                 if qc["results"]["arm_gated_structural"].get("gated_triggered")]

    def acc_on(arm: str, idxs: List[int]) -> Dict[str, Any]:
        n = len(idxs)
        c = sum(1 for i in idxs if per_question[i]["results"][arm]["is_correct"])
        return {"n": n, "correct": c, "accuracy": round(c / n, 4) if n else None}

    gated_analysis: Dict[str, Any] = {
        "triggered_questions": len(gated_idx),
        "baseline_on_triggered": acc_on(BASELINE_ARM, gated_idx),
        "per_arm_on_triggered": {arm: acc_on(arm, gated_idx) for arm in ALL_ARMS},
        "fixed_on_triggered": {}, "broken_on_triggered": {},
    }
    for arm in NEW_ARMS:
        gated_analysis["fixed_on_triggered"][arm] = sum(
            1 for i in gated_idx
            if not base_correct[i] and per_question[i]["results"][arm]["is_correct"])
        gated_analysis["broken_on_triggered"][arm] = sum(
            1 for i in gated_idx
            if base_correct[i] and not per_question[i]["results"][arm]["is_correct"])
    changed = [i for i in gated_idx
               if per_question[i]["results"]["arm_gated_structural"].get("gated_changed_winner")]
    gated_analysis["changed_winner"] = len(changed)
    gated_analysis["changed_improved"] = sum(
        1 for i in changed
        if not base_correct[i] and per_question[i]["results"]["arm_gated_structural"]["is_correct"])
    gated_analysis["changed_regressed"] = sum(
        1 for i in changed
        if base_correct[i] and not per_question[i]["results"]["arm_gated_structural"]["is_correct"])

    # ---- 跨模型信号判别力验证（关键统计 ①）----
    dual_grps = [g for g in group_reps if g["dual"]]
    single_grps = [g for g in group_reps if not g["dual"]]
    by_size: Dict[str, Dict[str, Any]] = {}
    for lo, hi, name in [(1, 1, "1"), (2, 4, "2-4"), (5, 9, "5-9"), (10, 999, ">=10")]:
        by_size[name] = {
            "dual": _agg_records([g for g in dual_grps if lo <= g["size"] <= hi]),
            "single": _agg_records([g for g in single_grps if lo <= g["size"] <= hi]),
        }
    winner_level: Dict[str, Dict[str, Any]] = {}
    for arm in ALL_ARMS:
        recs_d = [qc["results"][arm] for qc in per_question
                  if qc["results"][arm].get("group_key") is not None
                  and qc["results"][arm].get("winner_dual")]
        recs_s = [qc["results"][arm] for qc in per_question
                  if qc["results"][arm].get("group_key") is not None
                  and not qc["results"][arm].get("winner_dual")]
        winner_level[arm] = {"dual": _agg_records(recs_d, key="is_correct"),
                             "single": _agg_records(recs_s, key="is_correct")}
    cross_model_validation = {
        "group_level": {
            "n_groups": len(group_reps),
            "n_dual": len(dual_grps), "n_single": len(single_grps),
            "dual": _agg_records(dual_grps), "single": _agg_records(single_grps),
            "by_size": by_size,
        },
        "winner_level": winner_level,
        "note": ("组级正确率 = 组代表 SQL（min_sample_idx 最小）与 gold 的官方语义 "
                 "exec match；dual = 组内同时含 sft_phase1 与 sft_v2 两个模型的候选。"
                 "winner_level 仅统计来自真实签名组（非 fallback_maj/no_pool）的胜者。"),
    }

    total_wall = sum(v.get("wall_seconds", 0.0) for v in engine._stats.values())
    summary = {
        "meta": {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_items": str(args.items),
            "output_dir": str(args.out_dir),
            "spider_dir": str(args.spider_dir),
            "threads": args.threads,
            "query_timeout_seconds": args.query_timeout,
            "max_vm_steps": args.max_vm_steps,
            "row_cap": args.row_cap,
            "max_instances_cap": args.max_instances,
            "keep_distinct": args.keep_distinct,
            "seed": args.seed,
            "pool": POOL,
            "baseline_arm": BASELINE_ARM,
            "arms": ALL_ARMS,
            "join_counter_method": "sqlglot" if _HAS_SQLGLOT else "regex",
            "group_correctness_phase": not args.no_group_correctness,
            "semantics": (
                "grouping: bag semantics, row-sorted canonical, column order kept, "
                "no column permutation tolerance (same as adjudicate_pool); ranking "
                "keys: soft_cross = size + w*num_models then str(key) max; ladder = "
                "(dual-model first, size, str(key) max); gated trigger = |top1-top2|"
                "<=1 or top1<2, stage-2 key = (dual-model first, size, fewer JOINs, "
                "str(key) max), non-triggered = max-group rule; judgment: official "
                "eval_exec_match (postprocess + remove_distinct + replace_cur_year + "
                "result_eq with order_matters and column permutation), all instances "
                "must match; NO_RESULTS falls back to same-pool arm_maj"),
        },
        "dataset_stats": dataset_stats,
        "dedup_stats": dedup_stats,
        "execution_stats": {
            "grouping_phase": engine._stats.get("grouping", {}),
            "judgment_phase": engine._stats.get("judgment", {}),
            "group_reps_phase": engine._stats.get("group_reps", None),
            "total_wall_seconds": round(total_wall, 2),
        },
        "accuracy": cells,
        "vs_baseline": vs_baseline,
        "gated_analysis": gated_analysis,
        "cross_model_validation": cross_model_validation,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "group_level_correctness.json").write_text(
        json.dumps(group_reps, ensure_ascii=False, indent=1), encoding="utf-8")

    for arm in ALL_ARMS:
        out_items = []
        for qc in per_question:
            item = qc["item"]
            rec = qc["results"][arm]
            predicted = rec["text"]
            if not predicted:
                predicted = "SELECT 1"  # AGENTS.md 铁律 4：空预测不跳过
            out_items.append({
                "dataset_index": item.get("dataset_index", item.get("di")),
                "di": item.get("di", item.get("dataset_index")),
                "db_id": item.get("db_id", ""),
                "question": item.get("question", ""),
                "gold_sql": item.get("gold_sql") or "",
                "predicted_sql": predicted,
                "empty_winner": rec.get("empty_winner", False),
                "winner_source": rec["source"],
                "winner_votes": rec.get("votes", 0),
                "winner_group_size": rec.get("group_size", 0),
                "winner_group_key": rec.get("group_key"),
                "winner_models": rec.get("winner_models"),
                "winner_dual": rec.get("winner_dual"),
                "n_joins": rec.get("n_joins"),
                "join_counter": rec.get("join_counter"),
                "gated_triggered": rec.get("gated_triggered"),
                "gated_changed_winner": rec.get("gated_changed_winner"),
                "top1_size": rec.get("top1_size"),
                "top2_size": rec.get("top2_size"),
                "num_candidates": qc["num_candidates"],
                "num_unique_candidates": qc["num_unique_candidates"],
                "num_instances": qc["num_instances"],
                "instances_used": rec.get("instances_used", 0),
                "order_matters": rec.get("order_matters"),
                "is_correct": rec.get("is_correct", False),
                "gold_exec_error": rec.get("gold_exec_error", False),
                "vav_grouped_candidates": rec.get("vav_grouped", 0),
                "vav_excluded_candidates": rec.get("vav_excluded", 0),
            })
        (args.out_dir / f"items_{arm}.json").write_text(
            json.dumps(out_items, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 终端汇总 ----
    print("\n=== accuracy (correct / total) ===")
    for arm in ALL_ARMS:
        c = cells[arm]
        tag = "*" if arm == BASELINE_ARM else " "
        print(f"  {arm:22s} {c['correct']}/{c['total']} ({c['accuracy']:.4f}){tag}")
    print("\n=== vs baseline (fixed / broken / net) ===")
    for arm in NEW_ARMS:
        v = vs_baseline[arm]
        print(f"  {arm:22s} fixed={v['fixed']} broken={v['broken']} net={v['net']:+d} "
              f"delta={v['delta']:+.4f}")
    print(f"\n=== gated: triggered={gated_analysis['triggered_questions']} "
          f"changed_winner={gated_analysis['changed_winner']} "
          f"(improved={gated_analysis['changed_improved']} "
          f"regressed={gated_analysis['changed_regressed']}) ===")
    gl = cross_model_validation["group_level"]
    print(f"=== cross-model group-level: dual={gl['dual']} single={gl['single']} ===")
    print(f"\nsummary -> {args.out_dir / 'summary.json'}")
    print(f"items   -> {args.out_dir / 'items_<arm>.json'} ({len(ALL_ARMS)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
