#!/usr/bin/env python3
"""
src/adjudicate_order_aware.py — T1-4「顺序敏感分组」（order-aware grouping）裁决
升级（纯 CPU，无 GPU 依赖）。

动机（上一个实验发现的盲点）：MI-VAV 的分组签名是包语义（结果行排序后的规范
形，去除行序）。但官方 test-suite 判定是 order-aware：gold 含 'order by' 时行序
必须一致（order_matters）。后果：order-sensitive 的候选若行序错误，会被并进
「正确组」（行集合相同）→ 最大组胜出可能选错或污染组——「隐性彩票」。
官方语义：order_matters = gold_sql 含 'order by'（忽略大小写）
（tools/test_suite_eval/exec_eval.py 的判定）。

本脚本在 adjudicate_pool 的同一批 B1 候选池（1034 题 × 32 候选，sft_phase1/
sft_v2 双模型，全部 parse_success）上离线重算 2 臂，全部走官方语义判定
（exec match：postprocess → remove_distinct → replace_cur_year → result_eq，
全实例等价）：

  arm_vav_multi_all            基线 = adjudicate_pool 的包语义签名 + 最大组胜出
                               （both 池，74.3%），本 run 重算（同执行引擎/同
                               缓存语义），供 fixed/broken 与官方 EX 同口径对比
  arm_vav_multi_all_orderaware 新臂 = 顺序敏感签名 + 最大组胜出：
                               order_matters 题（gold 官方变换后文本含 'order
                               by'，忽略大小写）的 per-instance 签名保留行序
                               （不排序）；其余题维持包语义（行排序）。分组循环
                               /choose_group_vav/空零组过滤/NO_RESULTS→arm_maj
                               回退链全部与基线相同。

复用（import，不复制重写）adjudicate_pool 的执行引擎/去重/官方判定/choose_
group_vav/空零组过滤/arm_maj 回退；分组循环逐行镜像 adjudicate_pool._arm_vav
的入组语义，仅 per-instance 签名在 order_matters 题上改用保序签名
（rows_to_group_signature_ordered，与 AP 包语义签名同格式，parse_signature_
values / vector_is_empty / vector_is_all_zero 完全兼容）。

关键统计（summary.split_analysis / split_questions.json）：
  ① split_questions：候选组因顺序敏感签名而分裂的题数（包语义分组划分 vs
     顺序敏感分组划分不一致的题；必然 ⊆ order_matters 题），含每个分裂题的
     分裂明细（哪个包语义组裂成几个保序组、各片大小、胜者归属）；
  ② 分裂题上基线胜者 vs order-aware 胜者的对错迁移（fixed/broken/same_*），
     另给全量 vs_baseline（含 winner 是否变更）；
  ③ 官方 EX 对比：由 scripts/adjudicate_order_aware_cpu.slurm 用
     scripts/eval_official.sh 对两臂 items 复评并写
     official_ex_comparison.json（本脚本只产出两臂 items 文件）。

输出
  outputs/adjudicate_order_aware/summary.json
  outputs/adjudicate_order_aware/items_arm_orderaware.json（新臂 predicted_sql，
    与 scripts/eval_official.sh 兼容；空胜者写 "SELECT 1" 不跳过）
  outputs/adjudicate_order_aware/items_arm_vav_multi_all.json（基线，同 run 复算）
  outputs/adjudicate_order_aware/split_questions.json（分裂题明细）

用法
  # HPC 全量（默认 1034 题、全部实例）：
  python src/adjudicate_order_aware.py
  # 冒烟：--limit 30；实例数上限策略：--max-instances 8
"""

import argparse
import json
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
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "adjudicate_order_aware"
DEFAULT_SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"

POOL = "both"  # 两臂都在双模型全池上裁决（与基线 both 池同口径）
BASELINE_ARM = "arm_vav_multi_all"
ORDER_ARM = "arm_vav_multi_all_orderaware"
ITEM_FILE_BASELINE = "items_arm_vav_multi_all.json"
ITEM_FILE_ORDER = "items_arm_orderaware.json"
ARMS = [BASELINE_ARM, ORDER_ARM]


# ===================================================================
# 顺序敏感签名（与 AP 包语义签名同格式；只在 order_matters 题使用）
# ===================================================================


def rows_to_group_signature_ordered(rows: List[List[Any]],
                                    truncated: bool = False) -> str:
    """成功结果 → 'SUCCESS_VALUES:<sig>'，保序（不排序）行 canonical 字符串
    （每行 = _SEP_VAL 连接的带类型值 json；行间 _SEP_ROW 连接；重复行保留；
    行序 = 查询返回序）。与 AP.rows_to_group_signature 的唯一差别是不 sorted，
    格式/编码/TRUNC 标记完全一致，保证 AP.parse_signature_values /
    AP.vector_is_empty / AP.vector_is_all_zero 兼容。"""
    row_strings = [
        AP._SEP_VAL.join(
            json.dumps(AP._sig_value(v), ensure_ascii=False, separators=(",", ":"))
            for v in row
        )
        for row in rows
    ]
    sig = AP.SUCCESS_PREFIX + AP._SEP_ROW.join(row_strings)
    if truncated:
        sig += AP._SEP_VAL + AP._TRUNC_MARK
    return sig


def outcome_signature_ordered(outcome: Dict[str, Any]) -> str:
    """执行 outcome → 单实例保序分组签名（失败 → ERROR）。"""
    if not outcome.get("ok"):
        return AP.ERROR_SIG
    return rows_to_group_signature_ordered(
        outcome.get("rows") or [], outcome.get("truncated", False))


def gold_order_matters(gold_raw: Any, keep_distinct: bool) -> bool:
    """镜像官方判定口径（与 AP._judge_winner 完全一致）：gold 经官方变换后文本
    含 'order by'（忽略大小写）→ order_matters=True。"""
    g = AP.official_transform(gold_raw, is_pred=False, keep_distinct=keep_distinct)
    return "order by" in g.lower()


# ===================================================================
# 分组与裁决臂（分组循环逐行镜像 adjudicate_pool._arm_vav）
# ===================================================================


def build_groups(entries: List[Dict[str, Any]], sigs_per_entry: List[List[str]],
                 votes: Dict[int, int], n_instances: int) -> Tuple[Dict, int, int]:
    """镜像 adjudicate_pool._arm_vav 的入组循环：只 SUCCESS 候选入组（签名向量
    含任一 ERROR 分量的候选整条排除）；size = 池内票数加权（去重后按票数计）。
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
        g = groups.setdefault(key, {"size": 0, "members": []})
        g["size"] += cnt
        g["members"].append(ei)
    return groups, grouped, excluded


def partition_of(sigs_per_entry: List[List[str]], votes: Dict[int, int],
                 n_instances: int) -> Dict[int, Optional[Tuple[str, ...]]]:
    """候选划分（分裂检测用）：ei → 组 key（ERROR 排除 → None）。"""
    key_of: Dict[int, Optional[Tuple[str, ...]]] = {}
    for ei, _cnt in sorted(votes.items()):
        sigs = sigs_per_entry[ei][:n_instances]
        key_of[ei] = None if any(s == AP.ERROR_SIG for s in sigs) else tuple(sigs)
    return key_of


def _fallback_record(entries: List[Dict[str, Any]], votes: Dict[int, int],
                     n_used: int, grouped: int, excluded: int) -> Dict[str, Any]:
    """NO_RESULTS（无组）→ 回退同池 arm_maj（与 _arm_vav 的 fallback 链一致）。"""
    fb = AP._arm_maj(POOL, entries, {POOL: votes})
    if fb["source"] == "no_pool":
        return {"source": "no_pool", "text": None, "votes": 0, "group_key": None,
                "group_size": 0, "instances_used": n_used,
                "vav_grouped": grouped, "vav_excluded": excluded}
    return {"source": "fallback_maj", "text": fb["text"], "votes": fb["votes"],
            "group_key": None, "group_size": fb["votes"],
            "instances_used": n_used, "vav_grouped": grouped,
            "vav_excluded": excluded}


def arm_max_group(entries: List[Dict[str, Any]], sigs_per_entry: List[List[str]],
                  votes: Dict[int, int], instances: List[str],
                  source: str) -> Dict[str, Any]:
    """最大组胜出臂（对任意签名方案通用）：分组循环 + choose_group_vav 语义 +
    NO_RESULTS→arm_maj 回退 + 组内代表 min_sample_idx 最小者，全部与
    adjudicate_pool._arm_vav(k=-1) 一致；source 仅作标签区分两臂。"""
    n_used = len(instances)
    groups, grouped, excluded = build_groups(entries, sigs_per_entry, votes, n_used)
    if not groups:
        return _fallback_record(entries, votes, n_used, grouped, excluded)
    chosen = AP.choose_group_vav(groups)
    if chosen is None:
        return _fallback_record(entries, votes, n_used, grouped, excluded)
    members = groups[chosen]["members"]
    best_member = min(members, key=lambda ei: (entries[ei]["min_sample_idx"],
                                               entries[ei]["key"]))
    return {"source": source, "text": entries[best_member]["sql_text"],
            "votes": groups[chosen]["size"], "group_key": str(chosen),
            "group_size": groups[chosen]["size"], "instances_used": n_used,
            "vav_grouped": grouped, "vav_excluded": excluded}


# ===================================================================
# 单题裁决（含分裂检测）
# ===================================================================


def adjudicate_question(item: Dict[str, Any], engine: AP.ExecutionEngine,
                        db_instances: List[str],
                        keep_distinct: bool) -> Dict[str, Any]:
    """对一题完成 去重 → 双签名方案（包语义 / 顺序敏感）→ 两臂裁决 → 分裂检测。
    判定所需执行由 main 在 phase 2 并行补齐（本函数只读执行缓存）。"""
    candidates = item.get("candidates") or []
    instances = db_instances

    entries = AP._dedupe(candidates)

    order_matters = gold_order_matters(item.get("gold_sql") or "", keep_distinct)

    # 包语义签名向量（基线口径，与 adjudicate_pool 完全一致）
    bag_sigs_per_entry: List[List[str]] = []
    for e in entries:
        if not (e["sql_text"] or "").strip():
            sigs = [AP.ERROR_SIG] * len(instances)
        else:
            sigs = [AP.outcome_signature(engine.get(e["sql_text"], inst))
                    for inst in instances]
        bag_sigs_per_entry.append(sigs)

    # 顺序敏感签名向量：order_matters 题保序，其余题复用包语义向量
    if order_matters:
        order_sigs_per_entry: List[List[str]] = []
        for e, bag_sigs in zip(entries, bag_sigs_per_entry):
            if not (e["sql_text"] or "").strip():
                sigs = [AP.ERROR_SIG] * len(instances)
            else:
                sigs = [outcome_signature_ordered(engine.get(e["sql_text"], inst))
                        for inst in instances]
            order_sigs_per_entry.append(sigs)
    else:
        order_sigs_per_entry = bag_sigs_per_entry

    # both 池票数（与 adjudicate_pool 的 pool_entry_votes["both"] 一致）
    votes: Dict[int, int] = defaultdict(int)
    for c in candidates:
        ck = AP.normalize_for_dedup(c.get("sql"))
        for ei, e in enumerate(entries):
            if ck == e["key"]:
                votes[ei] += 1
                break

    results: Dict[str, Dict[str, Any]] = {
        BASELINE_ARM: arm_max_group(entries, bag_sigs_per_entry, votes, instances,
                                    source="vav"),
        ORDER_ARM: arm_max_group(entries, order_sigs_per_entry, votes, instances,
                                 source="vav_ordered"),
    }
    for rec in results.values():
        # empty_winner = 有胜者但胜者 SQL 文本为空（fallback_maj 选中空文本组）；
        # no_pool（池内无候选）不算 empty_winner（与 adjudicate_pool 同语义）
        rec["empty_winner"] = (rec["text"] == "")

    # ---- 分裂检测：包语义划分 vs 顺序敏感划分 ----
    n_used = len(instances)
    bag_partition = partition_of(bag_sigs_per_entry, votes, n_used)
    order_partition = partition_of(order_sigs_per_entry, votes, n_used)
    split = bag_partition != order_partition

    base_text = results[BASELINE_ARM]["text"]
    order_text = results[ORDER_ARM]["text"]
    winner_changed = (AP.normalize_for_dedup(base_text)
                      != AP.normalize_for_dedup(order_text))

    return {
        "item": item,
        "entries": entries,
        "num_candidates": len(candidates),
        "num_unique_candidates": len(entries),
        "num_instances": n_used,
        "order_matters": order_matters,
        "split": split,
        "winner_changed": winner_changed,
        "bag_groups": build_groups(entries, bag_sigs_per_entry, votes, n_used)[0],
        "order_groups": build_groups(entries, order_sigs_per_entry, votes, n_used)[0],
        "bag_partition": bag_partition,
        "order_partition": order_partition,
        "votes": dict(votes),
        "results": results,
    }


# ===================================================================
# 分裂题明细（关键统计 ① 的支撑数据）
# ===================================================================


def _sorted_group_entries(groups: Dict) -> List[Tuple[str, ...]]:
    """组 key 按 str(key) 排序（与 choose_group_vav 平票 tie-break 同口径）。"""
    return sorted(groups.keys(), key=str)


def split_question_detail(qc: Dict[str, Any], base_correct: bool,
                          order_correct: bool) -> Dict[str, Any]:
    """分裂题明细：包语义组 → 顺序敏感组的分裂映射、胜者归属、对错迁移。"""
    item = qc["item"]
    bag_groups = qc["bag_groups"]
    order_groups = qc["order_groups"]
    votes = qc["votes"]
    bag_partition = qc["bag_partition"]
    order_partition = qc["order_partition"]

    bag_keys = _sorted_group_entries(bag_groups)
    order_keys = _sorted_group_entries(order_groups)
    bag_id = {k: f"g{i}" for i, k in enumerate(bag_keys)}
    order_id = {k: f"o{i}" for i, k in enumerate(order_keys)}
    # 胜者记录里 group_key 是 str(组 key)，用 str 键映射直接匹配，无需 reparse
    bag_id_by_str = {str(k): v for k, v in bag_id.items()}
    order_id_by_str = {str(k): v for k, v in order_id.items()}
    base_key = qc["results"][BASELINE_ARM]["group_key"]
    order_key = qc["results"][ORDER_ARM]["group_key"]

    # 每个包语义组：其成员在顺序敏感方案下落入哪些组（>1 片 = 该组被分裂）
    splits: List[Dict[str, Any]] = []
    for bk in bag_keys:
        pieces: Dict[Tuple[str, ...], int] = defaultdict(int)
        for ei, bkey in bag_partition.items():
            if bkey != bk:
                continue
            ok = order_partition.get(ei)
            if ok is None:
                continue  # 双方案都排除的候选（ERROR）不入组，不参与分裂统计
            pieces[ok] += votes.get(ei, 0)
        if len(pieces) <= 1:
            continue
        splits.append({
            "bag_group_id": bag_id[bk],
            "bag_group_size": bag_groups[bk]["size"],
            "bag_group_is_baseline_winner": str(bk) == base_key,
            "pieces": [
                {"order_group_id": order_id[ok], "order_group_size": size,
                 "order_group_is_orderaware_winner": str(ok) == order_key}
                for ok, size in sorted(pieces.items(), key=lambda kv: str(kv[0]))
            ],
        })

    if not base_correct and order_correct:
        migration = "fixed"
    elif base_correct and not order_correct:
        migration = "broken"
    elif base_correct:
        migration = "same_right"
    else:
        migration = "same_wrong"

    return {
        "dataset_index": item.get("dataset_index", item.get("di")),
        "di": item.get("di", item.get("dataset_index")),
        "db_id": item.get("db_id", ""),
        "question": item.get("question", ""),
        "gold_sql": item.get("gold_sql") or "",
        "n_groups_bag": len(bag_keys),
        "n_groups_orderaware": len(order_keys),
        "winner_changed": qc["winner_changed"],
        "baseline_winner": {
            "sql": qc["results"][BASELINE_ARM]["text"],
            "source": qc["results"][BASELINE_ARM]["source"],
            "group_id": bag_id_by_str.get(base_key),
            "group_size": qc["results"][BASELINE_ARM]["group_size"],
            "is_correct": base_correct,
        },
        "orderaware_winner": {
            "sql": qc["results"][ORDER_ARM]["text"],
            "source": qc["results"][ORDER_ARM]["source"],
            "group_id": order_id_by_str.get(order_key),
            "group_size": qc["results"][ORDER_ARM]["group_size"],
            "is_correct": order_correct,
        },
        "migration": migration,
        "splits": splits,
        "n_candidates": qc["num_candidates"],
        "n_unique_candidates": qc["num_unique_candidates"],
    }


# ===================================================================
# 主流程
# ===================================================================


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="T1-4 顺序敏感分组裁决（纯 CPU）")
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
    args = ap.parse_args(argv)

    AP.rng = random.Random(args.seed)

    items = AP._load_items(args.items)
    if args.limit:
        items = items[: args.limit]
    print(f"[adjudicate_order_aware] {len(items)} 题，实例上限 "
          f"{args.max_instances or '全部'}，线程 {args.threads}，"
          f"查询超时 {args.query_timeout}s", file=sys.stderr)

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
    # 包语义与顺序敏感签名都从同一批缓存的 outcome.rows 计算（保序签名取返回序，
    # 包语义签名排序），零额外执行。
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
    print(f"[adjudicate_order_aware] phase1: {len(phase1_tasks)} 个唯一 (sql, db_path) "
          f"任务", file=sys.stderr)
    t0 = time.perf_counter()
    engine.run(phase1_tasks, phase="grouping")
    print(f"[adjudicate_order_aware] phase1 完成: {engine._stats['grouping']}",
          file=sys.stderr)

    # ---- 每题裁决（胜者选择 + 分裂检测，不执行 SQL）----
    per_question: List[Dict[str, Any]] = []
    for qi, item in enumerate(items):
        per_question.append(adjudicate_question(
            item, engine, instances_for(item.get("db_id", "")),
            args.keep_distinct))
        if (qi + 1) % 50 == 0 or qi + 1 == len(items):
            print(f"[adjudicate_order_aware] 裁决 {qi + 1}/{len(items)} 题 "
                  f"({time.perf_counter() - t0:.1f}s)", file=sys.stderr)

    # ---- Phase 2: 判定所需（gold 变换后 + 两臂胜者变换后）SQL × 实例 ----
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
    print(f"[adjudicate_order_aware] phase2 完成: {engine._stats['judgment']}",
          file=sys.stderr)

    # ---- 重新判定（phase 2 已补齐所有 outcome，纯内存计算）----
    for qc in per_question:
        item = qc["item"]
        gold_raw = item.get("gold_sql") or ""
        insts = instances_for(item.get("db_id", ""))
        for arm, rec in qc["results"].items():
            j = AP._judge_winner(rec["text"], gold_raw, insts, engine,
                                 args.keep_distinct)
            rec["is_correct"] = j["correct"]
            rec["gold_exec_error"] = j["gold_exec_error"]
            rec["order_matters"] = j["order_matters"]

    # ---- 汇总 ----
    base_correct = [qc["results"][BASELINE_ARM]["is_correct"] for qc in per_question]
    order_correct = [qc["results"][ORDER_ARM]["is_correct"] for qc in per_question]

    om_idx = [i for i, qc in enumerate(per_question) if qc["order_matters"]]
    split_idx = [i for i, qc in enumerate(per_question) if qc["split"]]
    changed_idx = [i for i, qc in enumerate(per_question) if qc["winner_changed"]]
    changed_split_idx = [i for i in split_idx if per_question[i]["winner_changed"]]

    # 一致性自检：非分裂题的签名划分相同 → 两臂胜者文本与对错必须逐题一致
    non_split_disagree = sum(
        1 for i in range(len(per_question))
        if not per_question[i]["split"] and (
            per_question[i]["winner_changed"]
            or base_correct[i] != order_correct[i]))

    def acc_on(corrects: List[bool], idxs: List[int]) -> Dict[str, Any]:
        n = len(idxs)
        c = sum(1 for i in idxs if corrects[i])
        return {"n": n, "correct": c, "accuracy": round(c / n, 4) if n else None}

    def migration_counts(idxs: List[int]) -> Dict[str, int]:
        fixed = broken = same_r = same_w = 0
        for i in idxs:
            b, o = base_correct[i], order_correct[i]
            if not b and o:
                fixed += 1
            elif b and not o:
                broken += 1
            elif b:
                same_r += 1
            else:
                same_w += 1
        return {"fixed": fixed, "broken": broken, "same_right": same_r,
                "same_wrong": same_w}

    dataset_stats: Dict[str, Any] = {
        "total_questions": len(items),
        "order_matters_questions": len(om_idx),
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
    for arm in ARMS:
        cell: Dict[str, Any] = {
            "total": len(items), "correct": 0, "accuracy": 0.0,
            "winner_sources": Counter(), "empty_winner": 0, "gold_exec_error": 0,
            "candidates_available": 0,
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
        cell["accuracy"] = round(cell["correct"] / cell["total"], 4) if cell["total"] else 0.0
        cell["winner_sources"] = dict(cell["winner_sources"])
        cells[arm] = cell

    # ---- vs 基线 fixed/broken（全量）----
    f_idx: List[Any] = []
    b_idx: List[Any] = []
    fixed = broken = same_r = same_w = 0
    for i, qc in enumerate(per_question):
        b, o = base_correct[i], order_correct[i]
        idx = qc["item"].get("dataset_index", qc["item"].get("di"))
        if not b and o:
            fixed += 1
            f_idx.append(idx)
        elif b and not o:
            broken += 1
            b_idx.append(idx)
        elif b:
            same_r += 1
        else:
            same_w += 1
    vs_baseline = {
        "baseline_accuracy": cells[BASELINE_ARM]["accuracy"],
        "arm_accuracy": cells[ORDER_ARM]["accuracy"],
        "delta": round(cells[ORDER_ARM]["accuracy"] - cells[BASELINE_ARM]["accuracy"], 4),
        "fixed": fixed, "broken": broken, "net": fixed - broken,
        "same_right": same_r, "same_wrong": same_w,
        "fixed_indices": f_idx, "broken_indices": b_idx,
    }

    # ---- 关键统计 ①②：分裂题数 + 分裂题上两臂对错迁移 ----
    split_analysis: Dict[str, Any] = {
        "order_matters_questions": len(om_idx),
        "split_questions": len(split_idx),
        "split_rate_over_order_matters": (
            round(len(split_idx) / len(om_idx), 4) if om_idx else None),
        "split_where_winner_changed": len(changed_split_idx),
        "winner_changed_total": len(changed_idx),
        "baseline_on_split": acc_on(base_correct, split_idx),
        "orderaware_on_split": acc_on(order_correct, split_idx),
        "migration_on_split": migration_counts(split_idx),
        "migration_on_split_winner_changed": migration_counts(changed_split_idx),
        "baseline_on_order_matters": acc_on(base_correct, om_idx),
        "orderaware_on_order_matters": acc_on(order_correct, om_idx),
        "migration_on_order_matters": migration_counts(om_idx),
        "non_split_questions": len(per_question) - len(split_idx),
        "non_split_arms_disagree": non_split_disagree,
    }

    # ---- 分裂题明细（summary 内嵌 + split_questions.json）----
    split_details: List[Dict[str, Any]] = []
    for i in split_idx:
        split_details.append(split_question_detail(
            per_question[i], base_correct[i], order_correct[i]))

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
            "orderaware_arm": ORDER_ARM,
            "semantics": (
                "order-aware grouping: for order_matters questions (gold official-"
                "transformed text contains 'order by', case-insensitive, same rule "
                "as tools/test_suite_eval/exec_eval.py) the per-instance group "
                "signature preserves row order (no row sorting, repeated rows kept, "
                "column order kept); other questions keep bag semantics (row-sorted "
                "canonical) identical to adjudicate_pool. Grouping loop / "
                "choose_group_vav (max-size, skip empty/all-zero, str(key) tie-"
                "break) / NO_RESULTS -> same-pool arm_maj fallback / winner rep "
                "(min_sample_idx) are identical to adjudicate_pool._arm_vav(k=-1). "
                "Judgment: official eval_exec_match mechanism (postprocess + "
                "remove_distinct + replace_cur_year + result_eq with order_matters "
                "and column permutation), all instances must match."),
        },
        "dataset_stats": dataset_stats,
        "dedup_stats": dedup_stats,
        "execution_stats": {
            "grouping_phase": engine._stats.get("grouping", {}),
            "judgment_phase": engine._stats.get("judgment", {}),
            "total_wall_seconds": round(total_wall, 2),
        },
        "accuracy": cells,
        "vs_baseline": vs_baseline,
        "split_analysis": split_analysis,
        "split_questions": split_details,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "split_questions.json").write_text(
        json.dumps(split_details, ensure_ascii=False, indent=2), encoding="utf-8")

    for arm, fname in [(BASELINE_ARM, ITEM_FILE_BASELINE),
                       (ORDER_ARM, ITEM_FILE_ORDER)]:
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
                "num_candidates": qc["num_candidates"],
                "num_unique_candidates": qc["num_unique_candidates"],
                "num_instances": qc["num_instances"],
                "instances_used": rec.get("instances_used", 0),
                "order_matters": qc["order_matters"],
                "split_question": qc["split"],
                "winner_changed_vs_baseline": (
                    qc["winner_changed"] if arm == ORDER_ARM else None),
                "is_correct": rec.get("is_correct", False),
                "gold_exec_error": rec.get("gold_exec_error", False),
                "vav_grouped_candidates": rec.get("vav_grouped", 0),
                "vav_excluded_candidates": rec.get("vav_excluded", 0),
            })
        (args.out_dir / fname).write_text(
            json.dumps(out_items, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 终端汇总 ----
    print("\n=== accuracy (correct / total) ===")
    for arm in ARMS:
        c = cells[arm]
        tag = "*" if arm == ORDER_ARM else " "
        print(f"  {arm:30s} {c['correct']}/{c['total']} ({c['accuracy']:.4f}){tag}")
    print("\n=== vs baseline ===")
    print(f"  fixed={vs_baseline['fixed']} broken={vs_baseline['broken']} "
          f"net={vs_baseline['net']:+d} delta={vs_baseline['delta']:+.4f}")
    print(f"\n=== split: order_matters={split_analysis['order_matters_questions']} "
          f"split={split_analysis['split_questions']} "
          f"winner_changed={split_analysis['split_where_winner_changed']} "
          f"migration_on_split={split_analysis['migration_on_split']} ===")
    print(f"=== 一致性自检：非分裂题两臂分歧 = {non_split_disagree}（应为 0）===")
    print(f"\nsummary -> {args.out_dir / 'summary.json'}")
    print(f"items   -> {args.out_dir / ITEM_FILE_BASELINE}, "
          f"{args.out_dir / ITEM_FILE_ORDER}")
    print(f"split   -> {args.out_dir / 'split_questions.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
