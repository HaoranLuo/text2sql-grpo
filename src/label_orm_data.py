#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/label_orm_data.py — T2 自训 ORM（GradeSQL 式 outcome reward model）训练数据打标。

纯 CPU、无 GPU 依赖（import reasoning_generator_agent 仅引入 torch 包，不加载模型）。

对 B1 候选池 outputs/eval_pool_b1/items.json（1034 题 × 32 候选）逐题：
  1. 按题去重候选 SQL（与 src/adjudicate_pool._dedupe 完全同口径：
     strip + lower + 空白折叠；同文本候选合并、票数加权信息保留）。
  2. 官方语义判定（复用 src/adjudicate_pool 全套官方 exec-match 机制，零复制）：
     eval_official.sh 清洗 → postprocess → remove_distinct(keep_distinct=False，官方默认)
     → replace_cur_year → result_eq（bag semantics + 列置换容忍 + gold 含 order by 时
     行序一致）；在本题【全部实例文件】上一致才算 correct。
  3. 输出 chat 格式样本：user = canonical 生成端 prompt（ReasoningGeneratorAgent.
     build_prompt + SpiderLoader.format_ddl，与 eval_pool_b1 生成候选时逐字一致）
     + Candidate SQL；assistant = Yes/No。附 label(1/0) 与元数据。
  4. 统计：正负比、按难度分布（canonical Spider hardness 分类器，逐函数照抄
     tools/original_spider_eval/evaluation.py 的 eval_hardness，作用于 dev.json
     的 process_sql 解析树）、每模型、每题正确候选数分布。
  5. 交叉核对（--adjudicate-items）：与 B1 裁决器 arm_maj 胜者的 is_correct 标签
     逐一比对（预期 100% 一致，双方走同一判定函数）。

整题剔除规则（标签不可靠的题，样本不入库，统计显式计数）：
  - gold SQL 在任一实例执行失败（official 语义下 gold 不可判）→ 整题剔除；
  - 无实例文件的题（官方语义「空实例集合恒真」→ 全对，无判别信息）→ 整题剔除。

数据卫生：样本不含 gold_sql（防 ORM 记忆泄漏）；gold 只存在于 items.json 原始源。

输出（全部新文件，不覆盖任何既有产物）
  data/orm_train.json        训练样本（chat messages + label + 元数据）
  data/orm_questions.json    每题元数据（难度/候选数/正确数/arm_maj 胜者正确性，
                             供 question-level dev 划分与「败局题」ranking 评估）
  data/orm_label_stats.json  整体统计 + 交叉核对结果

用法
  # 冒烟（登录节点 CPU，100 题 ≈ 1.5-2 分钟；自动与裁决器交叉核对）
  envs/reasoning3b/bin/python src/label_orm_data.py --limit 100 --threads 16 \
      --out /tmp/orm_train_smoke.json --questions-out /tmp/orm_questions_smoke.json \
      --stats-out /tmp/orm_label_stats_smoke.json
  # 全量（20596 唯一候选 × ~34.75 实例 ≈ 75 万次 sqlite 只读执行，16 线程 ~20 分钟）
  envs/reasoning3b/bin/python src/label_orm_data.py --threads 16
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(PROJECT_ROOT / "src"),
           str(PROJECT_ROOT / "tools"),
           str(PROJECT_ROOT / "tools" / "original_spider_eval")):  # process_sql 依赖
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adjudicate_pool import (  # noqa: E402  官方 exec-match 判定全套机制
    ExecutionEngine,
    _dedupe,
    _judge_winner,
    _load_items,
    list_instances,
    normalize_for_dedup,
    official_transform,
)
# canonical 生成端 prompt（与 eval_pool_b1 候选生成逐字一致；模块级 import 仅引入
# torch/transformers，不加载模型，CPU 可用）
from reasoning_generator_agent import ReasoningGeneratorAgent  # noqa: E402
from spider_utils import SpiderLoader  # noqa: E402
# canonical Spider hardness 分类器：组件计数函数来自原版官方 evaluation.py；
# eval_hardness 组装逻辑逐函数照抄 src/gating_calibrate.py（其自身亦为官方照抄）
from original_spider_eval.evaluation import (  # noqa: E402
    count_component1,
    count_component2,
    count_others,
)

DEFAULT_ITEMS = PROJECT_ROOT / "outputs" / "eval_pool_b1" / "items.json"
DEFAULT_SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"
DEFAULT_OUT = PROJECT_ROOT / "data" / "orm_train.json"
DEFAULT_QUESTIONS_OUT = PROJECT_ROOT / "data" / "orm_questions.json"
DEFAULT_STATS_OUT = PROJECT_ROOT / "data" / "orm_label_stats.json"
DEFAULT_ADJUDICATE = PROJECT_ROOT / "outputs" / "adjudicate_b1" / "items_arm_maj_both.json"

YES_STR, NO_STR = "Yes", "No"


def eval_hardness(sql: Any) -> str:
    """canonical Spider hardness 分类器（easy/medium/hard/extra），逐函数照抄
    src/gating_calibrate.py（= 官方 evaluation.py 组件函数 + 官方判定树），
    作用于 dev.json 的 process_sql 解析树（官方分布 248/446/174/166）。"""
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


def build_orm_prompt(question: str, ddl_schema: str, candidate_sql: str) -> str:
    """ORM user 侧输入：schema/question 表示与生成端 canonical prompt 逐字一致
    （VavSampler prompt_style='default' = ReasoningGeneratorAgent.build_prompt(
    question, ddl_schema, schema_links=None, evidence=None, dialect='sqlite')），
    再追加候选 SQL 与 Yes/No 判定指令。
    一致性理由：ORM 判定对象正是该 prompt 分布下生成的候选；schema/question 表示
    与生成端相同可减小 train/infer 分布漂移（报告风险点 #2）。"""
    base = ReasoningGeneratorAgent.build_prompt(
        question=question, ddl_schema=ddl_schema,
        schema_links=None, evidence=None, dialect="sqlite")
    cand = (candidate_sql or "").strip().replace("```", "`")
    return (
        f"{base}\n\nCandidate SQL Query:\n```sql\n{cand}\n```\n\n"
        "Task: Judge whether the candidate SQL query above correctly answers the "
        "question (execution-equivalent to the gold query). "
        "Answer with only Yes or No."
    )


def difficulty_of(dev_sql_tree: Any) -> str:
    """canonical Spider hardness：easy/medium/hard/extra（作用于 dev.json 的
    process_sql 解析树；官方分布 248/446/174/166）。"""
    try:
        return eval_hardness(dev_sql_tree or {})
    except Exception:
        return "unknown"


def load_adjudicator_map(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    """B1 裁决器 items（如 arm_maj_both）：{dataset_index: {predicted_sql, is_correct,
    winner_source, winner_votes, empty_winner}}。用于交叉核对 + 每题败局标记。"""
    out: Dict[int, Dict[str, Any]] = {}
    if not path or not Path(path).exists():
        return out
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            data = data["items"]
    except Exception as exc:
        print(f"[label] 裁决器文件不可读（跳过交叉核对）: {exc}", file=sys.stderr)
        return out
    for rec in data:
        qid = rec.get("dataset_index", rec.get("di"))
        if qid is None:
            continue
        out[int(qid)] = {
            "predicted_sql": rec.get("predicted_sql"),
            "is_correct": bool(rec.get("is_correct", False)),
            "winner_source": rec.get("winner_source"),
            "winner_votes": rec.get("winner_votes"),
            "empty_winner": rec.get("empty_winner", False),
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="T2 ORM 训练数据打标（纯 CPU）")
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    ap.add_argument("--spider-dir", type=Path, default=DEFAULT_SPIDER_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="训练样本输出")
    ap.add_argument("--questions-out", type=Path, default=DEFAULT_QUESTIONS_OUT)
    ap.add_argument("--stats-out", type=Path, default=DEFAULT_STATS_OUT)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--query-timeout", type=float, default=30.0)
    ap.add_argument("--max-vm-steps", type=int, default=5_000_000)
    ap.add_argument("--row-cap", type=int, default=100_000)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--keep-distinct", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="只打标前 N 题（冒烟）")
    ap.add_argument("--adjudicate-items", type=Path, action="append", default=None,
                    help="B1 裁决器 items 文件（交叉核对，可多次；默认自动探测 "
                         "outputs/adjudicate_b1/items_arm_maj_both.json）")
    ap.add_argument("--no-crosscheck", action="store_true",
                    help="禁用与裁决器的交叉核对")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    items = _load_items(args.items)
    if args.limit:
        items = items[: args.limit]
    print(f"[label] {len(items)} 题 | 线程 {args.threads} | 实例上限 "
          f"{args.max_instances or '全部'} | keep_distinct={args.keep_distinct}",
          file=sys.stderr)

    # ---- 难度来源：dev.json process_sql 解析树（按 dataset_index 顺序对齐）----
    dev_entries: List[Dict[str, Any]] = []
    dev_path = args.spider_dir / "dev.json"
    if dev_path.exists():
        try:
            dev_entries = json.loads(dev_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[label] dev.json 不可读，难度全部 unknown: {exc}", file=sys.stderr)

    # ---- 交叉核对来源：B1 裁决器 arm_maj（同一判定函数，预期 100% 一致）----
    adjudicator_files: List[Path] = []
    if not args.no_crosscheck:
        if args.adjudicate_items:
            adjudicator_files = list(args.adjudicate_items)
        elif DEFAULT_ADJUDICATE.exists():
            adjudicator_files = [DEFAULT_ADJUDICATE]
    adj_maps: Dict[str, Dict[int, Dict[str, Any]]] = {
        str(p): load_adjudicator_map(p) for p in adjudicator_files}

    database_dir = args.spider_dir / "database"
    loader = SpiderLoader(str(args.spider_dir))
    engine = ExecutionEngine(args.threads, args.query_timeout,
                             args.max_vm_steps, args.row_cap)
    db_instances_cache: Dict[str, List[str]] = {}
    ddl_cache: Dict[str, str] = {}

    def instances_for(db_id: str) -> List[str]:
        if db_id not in db_instances_cache:
            db_instances_cache[db_id] = list_instances(
                str(database_dir / db_id), db_id, args.max_instances)
        return db_instances_cache[db_id]

    def ddl_for(db_id: str) -> str:
        if db_id not in ddl_cache:
            ddl_cache[db_id] = loader.format_ddl(db_id)
        return ddl_cache[db_id]

    # ---- Phase 1：所有 (变换后 SQL × 实例) 唯一任务一次执行（全局缓存）----
    # 判定只读 official_transform 后的文本执行结果；与 adjudicate_pool phase1+2
    # 等价，但这里对【每个唯一候选】而非仅胜者做官方变换（同题候选去重 + 跨题
    # (sql, db) 全局缓存控制总执行量）。
    tasks: set = set()
    qc_list: List[Tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]] = []
    for item in items:
        insts = instances_for(item.get("db_id", ""))
        entries = _dedupe(item.get("candidates") or [])
        qc_list.append((item, insts, entries))
        gold_t = official_transform(item.get("gold_sql") or "", is_pred=False,
                                    keep_distinct=args.keep_distinct)
        for inst in insts:
            tasks.add((gold_t, inst))
        for e in entries:
            pred_t = official_transform(e["sql_text"], is_pred=True,
                                        keep_distinct=args.keep_distinct)
            if not pred_t.strip():
                continue  # 空 SQL：engine.get 直接合成失败，无需执行
            for inst in insts:
                tasks.add((pred_t, inst))
    print(f"[label] phase1: {len(tasks)} 个唯一 (sql, db_path) 执行任务",
          file=sys.stderr)
    engine.run(sorted(tasks), phase="orm_label")
    print(f"[label] phase1 完成: {engine._stats['orm_label']}", file=sys.stderr)

    # ---- Phase 2：逐题逐候选官方判定（纯缓存查找，零重复执行）----
    samples: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []
    excluded_gold_err = 0
    excluded_no_inst = 0
    prompt_chars: List[int] = []

    for qi, (item, insts, entries) in enumerate(qc_list):
        qid = item.get("dataset_index", item.get("di"))
        db_id = item.get("db_id", "")
        question = item.get("question", "")
        gold_raw = item.get("gold_sql") or ""
        diff = difficulty_of(dev_entries[qid].get("sql")) if 0 <= int(qid) < len(dev_entries) \
            else "unknown"

        judged = [(e, _judge_winner(e["sql_text"], gold_raw, insts, engine,
                                    args.keep_distinct)) for e in entries]

        if not insts:
            excluded_no_inst += 1
            continue
        if any(j["gold_exec_error"] for _, j in judged):
            excluded_gold_err += 1
            continue

        q_correct = 0
        for e, j in judged:
            label = 1 if j["correct"] else 0
            q_correct += label
            prompt = build_orm_prompt(question, ddl_for(db_id), e["sql_text"])
            prompt_chars.append(len(prompt))
            samples.append({
                "question_id": int(qid),
                "db_id": db_id,
                "difficulty": diff,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant",
                     "content": YES_STR if label == 1 else NO_STR},
                ],
                "label": label,
                "candidate_sql": e["sql_text"],
                "dup_count": int(e["count"]),
                "models": sorted(e["models"]),
            })

        # 每题败局标记优先取 arm_maj 裁决结果（文件名含 'maj'），否则取第一个文件
        maj_source = next((m for name, m in adj_maps.items() if "maj" in name),
                          next(iter(adj_maps.values()), {}) if adj_maps else {})
        maj = maj_source.get(int(qid))
        questions.append({
            "question_id": int(qid),
            "db_id": db_id,
            "difficulty": diff,
            "num_instances": len(insts),
            "num_unique_candidates": len(entries),
            "num_total_votes": len(item.get("candidates") or []),
            "num_correct": q_correct,
            "num_incorrect": len(entries) - q_correct,
            "order_matters": judged[0][1]["order_matters"] if judged else None,
            "maj_correct": maj["is_correct"] if maj else None,
            "maj_winner_source": maj.get("winner_source") if maj else None,
            "maj_votes": maj.get("winner_votes") if maj else None,
        })

        if (qi + 1) % 50 == 0 or qi + 1 == len(qc_list):
            print(f"[label] 判定 {qi + 1}/{len(qc_list)} 题 "
                  f"({time.perf_counter() - t0:.1f}s, 样本 {len(samples)})",
                  file=sys.stderr)

    # ---- 交叉核对：裁决器 winner 的 is_correct vs 本脚本同 SQL 标签 ----
    crosscheck: Dict[str, Any] = {}
    labeled_qids = {int(q["question_id"]) for q in questions}
    for fname, adj in adj_maps.items():
        matched = consistent = mismatched = missing = 0
        mismatch_examples: List[Dict[str, Any]] = []
        label_by_norm: Dict[Tuple[int, str], int] = {}
        for s in samples:
            label_by_norm[(s["question_id"],
                           normalize_for_dedup(s["candidate_sql"]))] = s["label"]
        for qid, rec in adj.items():
            if qid not in labeled_qids:
                continue  # 不在本次打标子集（--limit 冒烟时的其余题目）
            w_norm = normalize_for_dedup(rec.get("predicted_sql") or "")
            if rec.get("empty_winner") and w_norm == normalize_for_dedup("SELECT 1"):
                continue  # 裁决器对空胜者的占位回写，无对应真实候选
            mine = label_by_norm.get((qid, w_norm))
            if mine is None:
                missing += 1
                continue
            matched += 1
            if bool(mine) == bool(rec["is_correct"]):
                consistent += 1
            else:
                mismatched += 1
                mismatch_examples.append({
                    "question_id": qid,
                    "adjudicator_is_correct": rec["is_correct"],
                    "orm_label": mine,
                    "winner_sql": rec.get("predicted_sql"),
                })
        crosscheck[fname] = {
            "questions_checked": matched, "consistent": consistent,
            "mismatched": mismatched, "missing": missing,
            "mismatch_examples": mismatch_examples[:10],
        }
        print(f"[label] crosscheck {fname}: 命中 {matched} | 一致 {consistent} | "
              f"不一致 {mismatched} | 缺失 {missing}", file=sys.stderr)

    # ---- 统计 ----
    n = len(samples)
    pos = sum(1 for s in samples if s["label"] == 1)
    neg = n - pos
    per_diff: Dict[str, Dict[str, int]] = {}
    for s in samples:
        d = per_diff.setdefault(s["difficulty"], {"samples": 0, "positive": 0})
        d["samples"] += 1
        d["positive"] += s["label"]
    per_model: Dict[str, Dict[str, int]] = {}
    for s in samples:
        for m in s["models"]:
            d = per_model.setdefault(m, {"samples": 0, "positive": 0})
            d["samples"] += 1
            d["positive"] += s["label"]
    all_correct = sum(1 for q in questions if q["num_correct"] == q["num_unique_candidates"])
    all_wrong = sum(1 for q in questions if q["num_correct"] == 0)
    correct_per_q = [q["num_correct"] for q in questions]
    unique_per_q = [q["num_unique_candidates"] for q in questions]
    q_diff = Counter(q["difficulty"] for q in questions)
    prompt_chars.sort()

    stats = {
        "meta": {
            "created_by": "src/label_orm_data.py",
            "input_items": str(args.items),
            "spider_dir": str(args.spider_dir),
            "keep_distinct": args.keep_distinct,
            "max_instances_cap": args.max_instances,
            "semantics": (
                "official eval_exec_match per unique candidate: postprocess + "
                "remove_distinct + replace_cur_year + result_eq (bag semantics, "
                "column permutation tolerance, order_matters per gold), all "
                "instances must match; questions with gold_exec_error or no "
                "instances excluded entirely"),
        },
        "dataset": {
            "total_questions": len(items),
            "questions_labeled": len(questions),
            "questions_excluded_gold_exec_error": excluded_gold_err,
            "questions_excluded_no_instances": excluded_no_inst,
            "total_unique_candidates": sum(unique_per_q),
            "samples": n,
            "positive": pos,
            "negative": neg,
            "pos_ratio": round(pos / n, 4) if n else None,
            "questions_all_correct": all_correct,
            "questions_all_wrong": all_wrong,
            "questions_mixed": len(questions) - all_correct - all_wrong,
            "correct_per_question": {
                "mean": round(sum(correct_per_q) / len(correct_per_q), 3) if correct_per_q else None,
                "min": min(correct_per_q) if correct_per_q else None,
                "max": max(correct_per_q) if correct_per_q else None,
                "hist": dict(sorted(Counter(correct_per_q).items())),
            },
            "unique_per_question_mean": round(sum(unique_per_q) / len(unique_per_q), 2)
            if unique_per_q else None,
            "questions_by_difficulty": dict(sorted(q_diff.items())),
        },
        "per_difficulty": {
            d: {"samples": v["samples"], "positive": v["positive"],
                "pos_ratio": round(v["positive"] / v["samples"], 4) if v["samples"] else None}
            for d, v in sorted(per_diff.items())
        },
        "per_model": {
            m: {"samples": v["samples"], "positive": v["positive"],
                "pos_ratio": round(v["positive"] / v["samples"], 4) if v["samples"] else None}
            for m, v in sorted(per_model.items())
        },
        "prompt_chars": {
            "mean": round(sum(prompt_chars) / len(prompt_chars), 1) if prompt_chars else None,
            "p90": prompt_chars[int(len(prompt_chars) * 0.9)] if prompt_chars else None,
            "max": prompt_chars[-1] if prompt_chars else None,
        },
        "crosscheck": crosscheck,
        "execution_stats": {k: v for k, v in engine._stats.items()},
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.questions_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(samples, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    args.questions_out.write_text(
        json.dumps(questions, ensure_ascii=False, indent=1), encoding="utf-8")
    args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print("\n=== ORM 打标统计 ===")
    print(f"题目: {len(items)}（剔除 gold_exec_error {excluded_gold_err} / "
          f"无实例 {excluded_no_inst}）")
    print(f"样本: {n} | 正 {pos} / 负 {neg} | 正负比 {pos / neg:.3f}" if neg else f"样本: {n}")
    print(f"每题正确候选数: mean {stats['dataset']['correct_per_question']['mean']} "
          f"min {stats['dataset']['correct_per_question']['min']} "
          f"max {stats['dataset']['correct_per_question']['max']}")
    print(f"全对题 {all_correct} / 全错题 {all_wrong} / 混合题 "
          f"{len(questions) - all_correct - all_wrong}")
    for d, v in sorted(per_diff.items()):
        print(f"  难度 {d:8s}: {v['samples']:6d} 样本 正率 {v['positive'] / v['samples']:.3f}"
              if v["samples"] else f"  难度 {d:8s}: 0")
    print(f"prompt 字符数: mean {stats['prompt_chars']['mean']} "
          f"p90 {stats['prompt_chars']['p90']} max {stats['prompt_chars']['max']}")
    print(f"执行: {stats['execution_stats']}")
    print(f"输出: {args.out} / {args.questions_out} / {args.stats_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
