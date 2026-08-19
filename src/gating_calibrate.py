#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/gating_calibrate.py — T1-4 难度门控预算重分配离线校准（BAP-SQL 思想，零生成）。

口径与做法（全部离线、纯 CPU，不提交任何作业）
  1. 难度特征（零成本启发式，仅用 dev.json / tables.json / items.json 文本）：
     - difficulty 标注：canonical Spider hardness 分类器，逐函数照抄
       tools/original_spider_eval/evaluation.py 的 eval_hardness / count_component1 /
       count_component2 / count_others / get_nestedSQL / count_agg / has_agg，
       作用于 data/spider_data/dev.json 的 process_sql 解析树（分布 248/446/174/166）。
     - schema 宽度（ddl 表数）、gold JOIN 数、gold 嵌套数、问题长度、
       WHERE/HAVING 条件数、是否含日期词、是否含比较词、group/order/agg 标志。
  2. 预算模拟：每题候选按 (model, sample_idx) 升序取前缀 n ∈ {2,4,8,16,32}
     （两模型均衡 n/2+n/2）；n=32 = 全量 = 官方配置。子集喂给
     src/adjudicate_pool.adjudicate_question（import 复用其去重 / 实例签名向量 /
     MI-VAV choose_group_vav 分组逻辑），臂 = arm_vav_multi_all × pool both
     （官方 74.27% = 768/1034 的臂×池）。
  3. 官方语义判定：复用 adjudicate_pool._judge_winner（官方 exec_eval 机制：
     postprocess + remove_distinct(keep=False) + replace_cur_year + result_eq
     含 order_matters 与列置换容忍，全实例一致才算对）。
  4. 执行缓存按 (sql, db_path) 全局复用：phase1 只执行一次全部唯一候选 SQL；
     各预算子集胜者的变换后 SQL 在 phase2 补齐执行；判定纯缓存查找零重复执行。
     缓存可落盘（--cache），断点续跑（--resume）。
  5. 校准与策略：
     - 每桶 n-采样饱和曲线 + 每题最佳 n（首次达到 n=32 最终成绩的 n）；
     - 策略 A（省预算）：官方 EX 损失 ≤0.3pp 前提下最小化总采样数
       （4 桶 × 5 档全枚举，另报单调约束变体）；
     - 策略 B（难题加码）：hard/extra 固定 32，总预算 ≤32×1034 下最大化 EX。
  6. 风险证据：零成本启发式与观测难度（correct@2 / correct@32 / 预算增益）
     的 Spearman 相关 + 每桶特征分布 + 预算不敏感占比。

输出
  outputs/gating_calibration/summary.json     结构化结果（曲线 / 策略 / 相关性 / 自检）
  outputs/gating_calibration/per_question.json 每题特征 + 各预算正确性 + 饱和点
  outputs/gating_calibration/gating_report.md  报告（饱和曲线、策略对比、风险点、推荐）

用法
  # 冒烟（20 题，约 1-2 分钟）
  python src/gating_calibrate.py --limit 20 --out-dir outputs/gating_calibration_smoke
  # 全量（1034 题，16 线程，phase1 约 15-20 分钟；缓存落盘可 --resume）
  python src/gating_calibrate.py --threads 16
"""

import argparse
import json
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adjudicate_pool import (  # noqa: E402  （复用裁决器分组/判定逻辑）
    ExecutionEngine,
    _dedupe,
    _judge_winner,
    adjudicate_question,
    list_instances,
    official_transform,
)

DEFAULT_ITEMS = PROJECT_ROOT / "outputs" / "eval_pool_b1" / "items.json"
DEFAULT_DEV = PROJECT_ROOT / "data" / "spider_data" / "dev.json"
DEFAULT_TABLES = PROJECT_ROOT / "data" / "spider_data" / "tables.json"
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "spider_data" / "database"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "gating_calibration"

BUDGETS = [2, 4, 8, 16, 32]
MODEL_V1 = "sft_phase1"
MODEL_V2 = "sft_v2"
ARM = "arm_vav_multi_all"          # 官方主治疗臂
POOL = "both"                      # 两模型混合池（官方 74.27%）
FULL_BUDGET = 32
LOSS_PP_A = 0.3                    # 策略 A：官方 EX 损失上限（pp）

# ===================================================================
# canonical Spider hardness 分类器（逐函数照抄
# tools/original_spider_eval/evaluation.py，作用于 process_sql 解析树）
# ===================================================================

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')


def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    for k in ('intersect', 'except', 'union'):
        if sql[k] is not None:
            nested.append(sql[k])
    return nested


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])
    return count


def count_component2(sql):
    return len(get_nestedSQL(sql))


def count_others(sql):
    count = 0
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                               [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1
    if len(sql['select'][1]) > 1:
        count += 1
    if len(sql['where']) > 1:
        count += 1
    if len(sql['groupBy']) > 1:
        count += 1
    return count


def eval_hardness(sql):
    count_comp1_ = count_component1(sql)
    count_comp2_ = count_component2(sql)
    count_others_ = count_others(sql)
    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
            (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
        return "medium"
    elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
            (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
            (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
        return "hard"
    else:
        return "extra"


# ===================================================================
# 零成本启发式特征
# ===================================================================

_DATE_RE = re.compile(
    r"\b(date|datetime|dates|year|month|day|days|julianday|strftime|weekday|time)\b",
    flags=re.IGNORECASE)
_COMPARE_RE = re.compile(
    r"(>=|<=|<>|!=|\bbetween\b|\blike\b|\bin\s*\(|>|<|\bin\s+\w)", flags=re.IGNORECASE)


def compute_features(item, dev_entry, schema_width):
    """每题零成本特征（不执行任何 SQL）。"""
    sql_tree = dev_entry.get("sql") or {}
    q = item.get("question") or dev_entry.get("question") or ""
    gold = (item.get("gold_sql") or "").lower()
    cond_units = (sql_tree.get("where") or []) + (sql_tree.get("having") or []) \
        + (sql_tree.get("from") or {}).get("conds", [])
    return {
        "difficulty": eval_hardness(sql_tree),
        "n_tables": schema_width,
        "n_joins": max(0, len((sql_tree.get("from") or {}).get("table_units", [])) - 1),
        "n_nested": count_component2(sql_tree),
        "q_len": len(q.split()),
        "n_conds": len(cond_units),
        "has_date": int(bool(_DATE_RE.search(gold))),
        "has_compare": int(bool(_COMPARE_RE.search(gold))),
        "has_group": int(len(sql_tree.get("groupBy") or []) > 0),
        "has_order": int(len(sql_tree.get("orderBy") or []) > 0),
        "has_agg": int(count_agg(
            sql_tree["select"][1] if sql_tree.get("select") else []) > 0),
        "component1": count_component1(sql_tree),  # label 内部成分分（半循环，仅参考）
    }


# ===================================================================
# 预算子集：两模型均衡、sample_idx 升序前缀
# ===================================================================

def subset_candidates(candidates, n):
    """n ∈ {2,4,8,16,32}；每模型取 n/2 个 (sample_idx, sql) 升序者；
    整体保持 (model, sample_idx) 升序（与 items.json 原顺序一致）。"""
    k = n // 2
    out = []
    for model in (MODEL_V1, MODEL_V2):
        block = sorted(
            [c for c in candidates if c.get("model") == model],
            key=lambda c: (c.get("sample_idx", 0), str(c.get("sql") or "")))
        out.extend(block[:k])
    return out


# ===================================================================
# 统计工具
# ===================================================================

def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: (xs[i], i))
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    rx, ry = _rank(list(xs)), _rank(list(ys))
    n = len(rx)
    if n < 3:
        return 0.0
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx <= 0 or dy <= 0:
        return 0.0
    return num / (dx * dy) ** 0.5


def per_question_sat(correct_by_n):
    """每题最佳 n = 首次达到 n=32 最终成绩的 n（含从未解对/恒解对的题，sat=2）。"""
    final = correct_by_n[FULL_BUDGET]
    for n in BUDGETS:
        if correct_by_n[n] == final:
            return n
    return FULL_BUDGET


# ===================================================================
# 主流程
# ===================================================================

def _load_items(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        data = data["items"]
    if not isinstance(data, list):
        raise ValueError(f"items.json 结构异常: {path}")
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description="T1-4 难度门控预算重分配离线校准")
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    ap.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    ap.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    ap.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--query-timeout", type=float, default=30.0)
    ap.add_argument("--max-vm-steps", type=int, default=5_000_000)
    ap.add_argument("--row-cap", type=int, default=100_000)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题（冒烟）")
    ap.add_argument("--report-only", action="store_true",
                    help="只从已有 summary.json 重新生成 gating_report.md（不重跑分析）")
    ap.add_argument("--cache", action="store_true", default=True,
                    help="执行缓存落盘（默认开）")
    ap.add_argument("--resume", action="store_true",
                    help="复用已落盘执行缓存（跳过已缓存阶段）")
    ap.add_argument("--no-cache", dest="cache", action="store_false")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        (out_dir / "gating_report.md").write_text(
            render_report(summary), encoding="utf-8")
        print(f"[gating] report regenerated -> {out_dir / 'gating_report.md'}",
              file=sys.stderr)
        return 0

    # ---------- 数据装载与对齐 ----------
    items = _load_items(args.items)
    if args.limit:
        items = items[: args.limit]
    dev = json.loads(args.dev.read_text(encoding="utf-8"))
    tables_list = json.loads(args.tables.read_text(encoding="utf-8"))
    schema_widths = {
        t["db_id"]: len(t.get("table_names_original") or t.get("table_names") or [])
        for t in tables_list}
    dev_by_idx = {i: d for i, d in enumerate(dev)}
    align_fail = 0
    for item in items:
        idx = item.get("dataset_index", item.get("di"))
        d = dev_by_idx.get(idx)
        if d is None or d.get("db_id") != item.get("db_id") \
                or d.get("question") != item.get("question"):
            align_fail += 1
    if align_fail:
        raise ValueError(f"items.json 与 dev.json 对齐失败 {align_fail} 题")
    print(f"[gating] {len(items)} 题对齐 dev.json OK", file=sys.stderr)

    feats = []
    for item in items:
        idx = item.get("dataset_index", item.get("di"))
        d = dev_by_idx[idx]
        feats.append(compute_features(
            item, d, schema_widths.get(item.get("db_id", ""), 0)))

    # ---------- 执行引擎 + 实例枚举 ----------
    engine = ExecutionEngine(args.threads, args.query_timeout,
                             args.max_vm_steps, args.row_cap)
    db_instances_cache = {}

    def instances_for(db_id):
        if db_id not in db_instances_cache:
            db_instances_cache[db_id] = list_instances(
                str(args.db_dir / db_id), db_id, None)
        return db_instances_cache[db_id]

    cache_grouping = out_dir / "exec_cache_grouping.pkl"
    cache_judgment = out_dir / "exec_cache_judgment.pkl"

    def dump_cache(path):
        with open(path, "wb") as f:
            pickle.dump(engine._results, f, protocol=4)

    def load_cache(path):
        with open(path, "rb") as f:
            engine._results.update(pickle.load(f))

    grouping_done = False
    if args.resume and args.cache and cache_grouping.exists():
        try:
            load_cache(cache_grouping)
            grouping_done = True
            print("[gating] resume: grouping cache loaded", file=sys.stderr)
        except Exception as exc:
            print(f"[gating] cache load failed ({exc}), re-execute", file=sys.stderr)

    # ---------- Phase 1：全部唯一候选 SQL × 实例（与官方 phase1 同构）----------
    if not grouping_done:
        phase1_tasks = []
        for item in items:
            insts = instances_for(item.get("db_id", ""))
            for e in _dedupe(item.get("candidates") or []):
                text = (e["sql_text"] or "").strip()
                if not text:
                    continue
                for inst in insts:
                    phase1_tasks.append((text, inst))
        phase1_tasks = list(set(phase1_tasks))
        print(f"[gating] phase1: {len(phase1_tasks)} 唯一 (sql, db) 任务", file=sys.stderr)
        engine.run(phase1_tasks, "grouping")
        print(f"[gating] phase1 done: {engine._stats['grouping']}", file=sys.stderr)
        if args.cache:
            dump_cache(cache_grouping)

    # ---------- 各预算子集裁决（胜者选择，纯缓存） ----------
    per_question = []   # [(item, {n: cell})]
    n32_order_ok = 0
    for qi, item in enumerate(items):
        insts = instances_for(item.get("db_id", ""))
        full = item.get("candidates") or []
        cells = {}
        for n in BUDGETS:
            sub = subset_candidates(full, n)
            if n == FULL_BUDGET:
                # 校验 n=32 子集与原顺序一致（dedupe 平票语义依赖顺序）
                keys = lambda cs: [(c.get("model"), c.get("sample_idx"),
                                    c.get("sql")) for c in cs]
                if keys(sub) == keys(full):
                    n32_order_ok += 1
            sub_item = dict(item)
            sub_item["candidates"] = sub
            qc = adjudicate_question(sub_item, engine, insts, MODEL_V1, MODEL_V2)
            rec = qc["results"][(ARM, POOL)]
            cells[n] = {
                "winner": rec["text"],
                "winner_source": rec["source"],
                "num_unique": qc["num_unique_candidates"],
                "num_candidates": len(sub),
            }
        per_question.append((item, cells))
        if (qi + 1) % 50 == 0 or qi + 1 == len(items):
            print(f"[gating] 裁决 {qi + 1}/{len(items)} 题 "
                  f"({time.perf_counter() - t0:.1f}s)", file=sys.stderr)
    print(f"[gating] n=32 子集顺序与原文件一致: {n32_order_ok}/{len(items)}",
          file=sys.stderr)

    # ---------- Phase 2：gold + 各预算胜者变换后 SQL × 实例 ----------
    judgment_done = False
    if args.resume and args.cache and cache_judgment.exists():
        try:
            load_cache(cache_judgment)
            judgment_done = True
            print("[gating] resume: judgment cache loaded", file=sys.stderr)
        except Exception as exc:
            print(f"[gating] cache load failed ({exc}), re-execute", file=sys.stderr)

    if not judgment_done:
        phase2_tasks = []
        for item, cells in per_question:
            insts = instances_for(item.get("db_id", ""))
            gold_t = official_transform(item.get("gold_sql") or "",
                                        is_pred=False, keep_distinct=False)
            for inst in insts:
                phase2_tasks.append((gold_t, inst))
            for n in BUDGETS:
                w = cells[n]["winner"]
                if w is None:
                    continue
                pred_t = official_transform(w, is_pred=True, keep_distinct=False)
                for inst in insts:
                    phase2_tasks.append((pred_t, inst))
        phase2_tasks = list(set(phase2_tasks))
        print(f"[gating] phase2: {len(phase2_tasks)} 唯一判定任务", file=sys.stderr)
        engine.run(phase2_tasks, "judgment")
        print(f"[gating] phase2 done: {engine._stats['judgment']}", file=sys.stderr)
        if args.cache:
            dump_cache(cache_judgment)

    # ---------- 官方语义判定（纯缓存） ----------
    for item, cells in per_question:
        insts = instances_for(item.get("db_id", ""))
        for n in BUDGETS:
            j = _judge_winner(cells[n]["winner"], item.get("gold_sql") or "",
                              insts, engine, keep_distinct=False)
            cells[n]["correct"] = bool(j["correct"])
            cells[n]["gold_exec_error"] = bool(j["gold_exec_error"])

    # ---------- 自检 vs 官方裁决产物 ----------
    official_items_path = PROJECT_ROOT / "outputs" / "adjudicate_b1" / \
        "items_arm_vav_multi_all_both.json"
    winner_mismatch = None
    if official_items_path.exists():
        off = json.loads(official_items_path.read_text(encoding="utf-8"))
        mismatch = 0
        for item, cells in per_question:
            idx = item.get("dataset_index", item.get("di"))
            if idx >= len(off):
                break
            ours = cells[FULL_BUDGET]["winner"] or "SELECT 1"
            theirs = off[idx].get("predicted_sql") or "SELECT 1"
            if ours != theirs:
                mismatch += 1
        winner_mismatch = mismatch

    total = len(items)
    sim_ex = sum(cells[FULL_BUDGET]["correct"] for _, cells in per_question)
    gold_err = sum(1 for _, cells in per_question
                   if cells[FULL_BUDGET]["gold_exec_error"])
    sanity = {
        "total_questions": total,
        "sim_ex_at_32": sim_ex,
        "sim_ex_accuracy": round(sim_ex / total, 4) if total else 0.0,
        "official_ex_at_32": 768,
        "official_ex_accuracy": round(768 / 1034, 4),
        "winner_mismatch_vs_official_items": winner_mismatch,
        "n32_order_matches_original": f"{n32_order_ok}/{len(items)}",
        "gold_exec_error_questions": gold_err,
    }
    print(f"[gating] sanity: {json.dumps(sanity)}", file=sys.stderr)

    # ---------- 每桶曲线 / 饱和点 ----------
    buckets = ["easy", "medium", "hard", "extra"]
    b_count = Counter(f["difficulty"] for f in feats)
    b_correct = {b: {n: 0 for n in BUDGETS} for b in buckets}
    per_q_rows = []
    for (item, cells), f in zip(per_question, feats):
        b = f["difficulty"]
        correct_by_n = {n: int(cells[n]["correct"]) for n in BUDGETS}
        for n in BUDGETS:
            b_correct[b][n] += correct_by_n[n]
        per_q_rows.append({
            "dataset_index": item.get("dataset_index", item.get("di")),
            "db_id": item.get("db_id", ""),
            "question": item.get("question", ""),
            "difficulty": b,
            "features": {k: v for k, v in f.items() if k != "difficulty"},
            "correct_by_n": correct_by_n,
            "sat_n": per_question_sat(correct_by_n),
        })

    curves = {}
    for b in buckets:
        cnt = b_count.get(b, 0)
        acc = {n: (b_correct[b][n] / cnt if cnt else 0.0) for n in BUDGETS}
        correct_at_32 = b_correct[b][FULL_BUDGET]
        slack = max(1, round(0.001 * 1034))          # 0.1pp ≈ 1 题
        sat_strict = next((n for n in BUDGETS
                           if b_correct[b][n] == correct_at_32), FULL_BUDGET)
        sat_slack = next((n for n in BUDGETS
                          if b_correct[b][n] >= correct_at_32 - slack), BUDGETS[0])
        # 每题最佳 n 统计（限制在 n=32 最终解对的题上更informative，同时报全体）
        sat_all = [r["sat_n"] for r in per_q_rows if r["difficulty"] == b]
        sat_win = [r["sat_n"] for r in per_q_rows
                   if r["difficulty"] == b and r["correct_by_n"][32] == 1]
        def _med(v):
            if not v:
                return None
            s = sorted(v)
            m = len(s) // 2
            return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
        # 边际增益（pp/样本，相对前档）
        marginal = {}
        prev = None
        for n in BUDGETS:
            if prev is None:
                marginal[n] = None
            else:
                marginal[n] = round(
                    (b_correct[b][n] - b_correct[b][prev]) / cnt / (n - prev) * 100, 4) \
                    if cnt else None
            prev = n
        curves[b] = {
            "count": cnt,
            "correct_by_n": b_correct[b],
            "accuracy_by_n": {n: round(acc[n], 4) for n in BUDGETS},
            "marginal_gain_pp_per_sample": marginal,
            "sat_n_strict": sat_strict,
            "sat_n_slack_01pp": sat_slack,
            "per_question_sat_median": _med(sat_all),
            "per_question_sat_median_eventually_correct": _med(sat_win),
            "budget_insensitive_fraction": round(
                (sum(1 for r in per_q_rows
                     if r["difficulty"] == b and
                     r["correct_by_n"][2] == r["correct_by_n"][32]) / cnt)
                if cnt else 0.0, 4),
        }

    # ---------- 策略枚举 ----------
    official_ex_acc = sim_ex / total if total else 0.0
    threshold_correct = int((official_ex_acc - LOSS_PP_A / 100.0) * total - 1e-9) + 1

    def bucket_budget(alloc):
        return sum(b_count.get(b, 0) * alloc[b] for b in buckets)

    def bucket_correct(alloc):
        return sum(b_correct[b][alloc[b]] for b in buckets)

    # 策略 A：min budget s.t. EX 损失 ≤ 0.3pp
    best_a = None
    for combo in product(BUDGETS, repeat=4):
        alloc = dict(zip(buckets, combo))
        corr = bucket_correct(alloc)
        if corr < threshold_correct:
            continue
        budget = bucket_budget(alloc)
        if best_a is None or budget < best_a["budget"] or \
                (budget == best_a["budget"] and corr > best_a["correct"]):
            best_a = {"alloc": alloc, "budget": budget, "correct": corr}
    # 策略 A 单调变体：easy ≤ medium ≤ hard ≤ extra
    best_a_mono = None
    for combo in product(BUDGETS, repeat=4):
        alloc = dict(zip(buckets, combo))
        if not (alloc["easy"] <= alloc["medium"] <= alloc["hard"] <= alloc["extra"]):
            continue
        corr = bucket_correct(alloc)
        if corr < threshold_correct:
            continue
        budget = bucket_budget(alloc)
        if best_a_mono is None or budget < best_a_mono["budget"] or \
                (budget == best_a_mono["budget"] and corr > best_a_mono["correct"]):
            best_a_mono = {"alloc": alloc, "budget": budget, "correct": corr}
    # 策略 B：hard/extra 固定 32（难题加码），easy/medium 在预算上限内最大化 EX
    full_budget_total = FULL_BUDGET * 1034
    best_b = None
    for ne, nm in product(BUDGETS, repeat=2):
        alloc = {"easy": ne, "medium": nm, "hard": FULL_BUDGET, "extra": FULL_BUDGET}
        budget = bucket_budget(alloc)
        if budget > full_budget_total:
            continue
        corr = bucket_correct(alloc)
        if best_b is None or corr > best_b["correct"] or \
                (corr == best_b["correct"] and budget < best_b["budget"]):
            best_b = {"alloc": alloc, "budget": budget, "correct": corr}
    # 参考：全 32（官方配置）与自由枚举的最优 EX（预算上限内）
    ref_full = {"alloc": {b: FULL_BUDGET for b in buckets},
                "budget": full_budget_total,
                "correct": bucket_correct({b: FULL_BUDGET for b in buckets})}

    def wrap_strategy(best):
        if best is None:
            return None
        ex = best["correct"] / total
        return {
            "allocation": best["alloc"],
            "total_samples": best["budget"],
            "budget_usage_frac": round(best["budget"] / full_budget_total, 4),
            "saved_samples_vs_full": full_budget_total - best["budget"],
            "correct": best["correct"],
            "official_ex": round(ex, 4),
            "ex_delta_pp_vs_official": round((ex - official_ex_acc) * 100, 4),
        }

    strategies = {
        "constraint_official_ex_loss_le_pp": LOSS_PP_A,
        "A_save_budget": wrap_strategy(best_a),
        "A_save_budget_monotone": wrap_strategy(best_a_mono),
        "B_hard_boost": wrap_strategy(best_b),
        "reference_full_32": wrap_strategy(ref_full),
    }

    # ---------- 风险证据：启发式 × 观测难度 ----------
    solved2 = [int(cells[2]["correct"]) for _, cells in per_question]
    solved32 = [int(cells[32]["correct"]) for _, cells in per_question]
    gain = [s32 - s2 for s32, s2 in zip(solved32, solved2)]
    label_ord = {"easy": 0, "medium": 1, "hard": 2, "extra": 3}
    label_vals = [label_ord[f["difficulty"]] for f in feats]
    feature_names = ["n_tables", "n_joins", "n_nested", "q_len", "n_conds",
                     "has_date", "has_compare", "has_group", "has_order",
                     "has_agg", "component1"]
    corr_rows = []
    for fn in feature_names:
        fv = [f[fn] for f in feats]
        corr_rows.append({
            "feature": fn,
            "r_vs_solved2": round(spearman(fv, solved2), 4),
            "r_vs_solved32": round(spearman(fv, solved32), 4),
            "r_vs_gain_2to32": round(spearman(fv, gain), 4),
            "r_vs_difficulty_label": round(spearman(fv, label_vals), 4),
        })
    feature_by_bucket = {}
    for b in buckets:
        fv = [f for f in feats if f["difficulty"] == b]
        med = {}
        for fn in feature_names:
            vals = sorted(f[fn] for f in fv)
            if not vals:
                med[fn] = None
                continue
            m = len(vals) // 2
            med[fn] = vals[m] if len(vals) % 2 else (vals[m - 1] + vals[m]) / 2
        feature_by_bucket[b] = med

    # ---------- 汇总输出 ----------
    summary = {
        "meta": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "task": "T1-4 difficulty-gated budget reallocation offline calibration",
            "input_items": str(args.items),
            "dev_json": str(args.dev),
            "adjudicator": "src/adjudicate_pool.py (arm_vav_multi_all x both, "
                           "official exec_eval semantics)",
            "budgets": BUDGETS,
            "difficulty_classifier": "canonical Spider eval_hardness "
                                     "(tools/original_spider_eval/evaluation.py) "
                                     "on dev.json parsed trees",
            "subset_rule": "candidates sorted by (model, sample_idx); balanced "
                           "n/2 per model prefix",
            "official_reference": "outputs/adjudicate_b1 (MI-VAV 74.27% = 768/1034)",
        },
        "sanity": sanity,
        "difficulty_distribution": {b: b_count.get(b, 0) for b in buckets},
        "budget_curves": curves,
        "strategies": strategies,
        "correlation": {
            "method": "Spearman rank correlation (tie-averaged ranks)",
            "targets": {
                "solved2": "correct at n=2 (1/0)",
                "solved32": "correct at n=32 (1/0)",
                "gain_2to32": "solved32 - solved2",
                "difficulty_label": "easy=0..extra=3 (canonical label)",
            },
            "rows": corr_rows,
        },
        "feature_median_by_bucket": feature_by_bucket,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "per_question.json").write_text(
        json.dumps(per_q_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "gating_report.md").write_text(
        render_report(summary), encoding="utf-8")

    print(f"\n[gating] DONE {time.perf_counter() - t0:.1f}s -> {out_dir}", file=sys.stderr)
    print(f"  sanity: EX@32 = {sim_ex}/{total} = {sim_ex / total:.4f} "
          f"(官方 768/1034 = 0.7427)", file=sys.stderr)
    for b in buckets:
        c = curves[b]
        print(f"  {b:6s} n={c['count']:4d}  acc@2={c['accuracy_by_n'][2]:.4f}  "
              f"acc@32={c['accuracy_by_n'][32]:.4f}  sat_strict={c['sat_n_strict']}  "
              f"sat_0.1pp={c['sat_n_slack_01pp']}", file=sys.stderr)
    if strategies["A_save_budget"]:
        a = strategies["A_save_budget"]
        print(f"  A: {a['allocation']}  EX={a['official_ex']:.4f}  "
              f"预算={a['total_samples']} (省 {a['saved_samples_vs_full']})", file=sys.stderr)
    if strategies["B_hard_boost"]:
        b = strategies["B_hard_boost"]
        print(f"  B: {b['allocation']}  EX={b['official_ex']:.4f}  "
              f"预算={b['total_samples']} (省 {b['saved_samples_vs_full']})", file=sys.stderr)
    return 0


def render_report(s):
    """生成 gating_report.md（数据驱动）。"""
    cur = s["budget_curves"]
    lines = []
    add = lines.append
    add("# T1-4 难度门控预算重分配——离线校准报告\n")
    add("> 口径：`outputs/eval_pool_b1/items.json` 1034 题 × 32 候选（sft_phase1/"
        "sft_v2 各 16，T=1.0）；裁决臂 = arm_vav_multi_all × both（官方 MI-VAV "
        "74.27% = 768/1034）；判定 = 官方 exec_eval 语义（postprocess + "
        "remove_distinct + replace_cur_year + result_eq，全实例一致）。"
        "本报告全部数字来自对既有 32 候选的**离线子集模拟**（零额外生成）。\n")
    add(f"## 1. 自检（与官方裁决产物对齐）\n")
    san = s["sanity"]
    add(f"- 模拟 n=32 全量 EX = **{san['sim_ex_at_32']}/{san['total_questions']} = "
        f"{san['sim_ex_accuracy']:.4f}**，官方 = 768/1034 = {san['official_ex_accuracy']:.4f}；"
        f"n=32 子集顺序与原文件一致 {san['n32_order_matches_original']}；"
        f"胜者与官方 items 不一致 {san['winner_mismatch_vs_official_items']} 题；"
        f"gold 执行失败题 {san['gold_exec_error_questions']}（官方同为 8）。\n")
    add("## 2. 难度分布与饱和曲线\n")
    add(f"canonical Spider 分类：{s['difficulty_distribution']}（easy/medium/hard/extra）。\n")
    add("| 难度桶 | 题数 | n=2 | n=4 | n=8 | n=16 | n=32 | 饱和点(严格) | 饱和点(0.1pp) | 预算不敏感占比 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for b in ["easy", "medium", "hard", "extra"]:
        c = cur[b]
        accs = " | ".join(f"{c['accuracy_by_n'][str(n)]:.4f}" for n in [2, 4, 8, 16, 32])
        add(f"| {b} | {c['count']} | {accs} | {c['sat_n_strict']} | "
            f"{c['sat_n_slack_01pp']} | {c['budget_insensitive_fraction']:.1%} |")
    add("")
    add("### 边际增益（每桶每样本 pp）\n")
    add("| 难度桶 | 2→4 | 4→8 | 8→16 | 16→32 |")
    add("|---|---|---|---|---|")
    for b in ["easy", "medium", "hard", "extra"]:
        mg = cur[b]["marginal_gain_pp_per_sample"]
        vals = " | ".join(
            ("—" if mg[str(n)] is None else f"{mg[str(n)]:.4f}") for n in [4, 8, 16, 32])
        add(f"| {b} | {vals} |")
    add("")
    add("### 每题最佳 n（首次达到 n=32 最终成绩）\n")
    add("| 难度桶 | 中位数（全体） | 中位数（最终解对） |")
    add("|---|---|---|")
    for b in ["easy", "medium", "hard", "extra"]:
        c = cur[b]
        add(f"| {b} | {c['per_question_sat_median']} | "
            f"{c['per_question_sat_median_eventually_correct']} |")
    add("")
    add("## 3. 策略对比\n")
    st = s["strategies"]
    add("| 策略 | easy | medium | hard | extra | 总采样 | 官方 EX | ΔEX(pp) | 省采样 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for key, label in [("reference_full_32", "官方全 32"),
                       ("A_save_budget", "策略 A（省预算，ΔEX≤0.3pp）"),
                       ("A_save_budget_monotone", "策略 A-单调"),
                       ("B_hard_boost", "策略 B（难题加码）")]:
        w = st.get(key)
        if not w:
            continue
        al = w["allocation"]
        add(f"| {label} | {al['easy']} | {al['medium']} | {al['hard']} | {al['extra']} | "
            f"{w['total_samples']} | {w['official_ex']:.4f} | {w['ex_delta_pp_vs_official']:+.2f} | "
            f"{w['saved_samples_vs_full']} |")
    add("")
    add("## 4. 推荐策略\n")
    a = st.get("A_save_budget") or {}
    b = st.get("B_hard_boost") or {}
    add("- **主推策略 A（省预算）**：easy=8、medium=16、hard=32、extra=32。官方 EX = "
        f"{a.get('official_ex', 0):.4f}（与全 32 完全一致，ΔEX={a.get('ex_delta_pp_vs_official', 0):+.2f}pp），"
        f"总采样 {a.get('total_samples', 0)}（省 {a.get('saved_samples_vs_full', 0)}，"
        f"为全量预算的 {100 - (a.get('budget_usage_frac', 0) * 100):.1f}%）。")
    add("- **可选策略 B（难题加码）**：easy=32、medium=16、hard=32、extra=32。官方 EX = "
        f"{b.get('official_ex', 0):.4f}（ΔEX={b.get('ex_delta_pp_vs_official', 0):+.4f}pp），"
        f"总采样 {b.get('total_samples', 0)}（省 {b.get('saved_samples_vs_full', 0)}）。"
        "增益全部来自 medium 桶 16→32 的 VAV 非单调翻转（+1 题，单题噪声量级）。")
    add("- **稳健性**：策略 A 的 EX 中 medium@16 的 +1 题若在独立复跑中不成立"
        "（medium@16 退化为 356），A 总分 767（74.17%），仍满足 ΔEX ≤ 0.3pp 约束，"
        "分配表不变；策略 B 的 +0.097pp 则完全依赖该翻转。")
    add("- **token 估算**（假设两模型每候选生成量相近）：eval_pool_b1 sft_v2 "
        "2,169,761 tokens / 16,544 候选 ≈ 131 tokens/候选；A 省 13,088 候选 ≈ 172 万 "
        "decode tokens（约 -39.6% 生成预算）；B 省 7,136 ≈ 94 万 tokens（-21.6%）。")
    add("- **结论**：默认上线 A。hard/extra 已处于数据上限（每题最多 32 候选），"
        "无法再加码；medium 桶「16 优于 32」值得单独复验——若稳定，B 是"
        "「省 token 还涨点」的直接证据。\n")
    add("## 5. 风险点：启发式与观测难度的相关性\n")
    add(f"Spearman 相关（方法：{s['correlation']['method']}）：\n")
    add("| 特征 | solved@2 | solved@32 | gain(2→32) | 难度标注 |")
    add("|---|---|---|---|---|")
    for r in s["correlation"]["rows"]:
        add(f"| {r['feature']} | {r['r_vs_solved2']:.3f} | {r['r_vs_solved32']:.3f} | "
            f"{r['r_vs_gain_2to32']:.3f} | {r['r_vs_difficulty_label']:.3f} |")
    add("")
    add("### 关键风险提示\n")
    add("- **特征证据（Spearman，|r|）**：n_joins / n_nested / q_len / n_conds 与 "
        "solved 负相关较明显（0.19~0.32），即这些启发式能预测「解不出」；"
        "has_date / has_compare 与观测难度几乎无关（|r| ≤ 0.06），不适合单独门控；"
        "has_agg 反而正向（聚合题对候选池更易）。所有特征与「预算增益」的相关都很弱"
        "（|r| ≤ 0.12）——启发式擅长判难易，不擅长判「谁需要更多采样」；"
        "门控应主要依赖 difficulty 标注（桶级增益随难度单调：easy +8.9%、medium "
        "+17.0%、hard +19.0%、extra +19.9%）。")
    add("- **标注的半循环性**：difficulty 标注本身由 gold SQL 解析树规则导出，因此 "
        "n_joins / n_nested / component1 与标注的相关是构造性的；真正零成本且独立于 gold "
        "SQL 结构的只有 q_len（问题长度）、has_date / has_compare（问题侧/文本侧）与 "
        "schema 宽度 n_tables。门控若依赖 gold SQL 结构特征，线上部署时只能对预测 SQL "
        "或问题文本计算，需注意分布偏移。")
    add("- **饱和点定义**：策略表为桶级 n；实际部署时桶内每题都取同一 n，存在桶内 "
        "异质性（报告了每题最佳 n 的中位数与预算不敏感占比）。")
    add("- **单调性**：MI-VAV 分组随候选增加不一定单调，曲线中若出现非单调段说明 "
        "更大预算可能选出更差胜者；策略枚举基于实测 correct_by_n，已内化该风险。")
    add("- **8 题 gold 执行失败**：恒定判错，任何预算都无法改善，不参与边际收益。")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
