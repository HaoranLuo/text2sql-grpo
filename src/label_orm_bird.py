#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/label_orm_bird.py — BIRD 判卷老师适配：候选池自动打标（纯 CPU，无 GPU 依赖）。

【实验文档】docs/BIRD_JUDGE_ADAPTATION_PLAN.md（预注册：dev 74 题 / bal2 / 1 epoch）。

对 BIRD 候选池 outputs/eval_pool_bird/items.json（1534 题 × 4 模型 × 16 采样，
dataset_index = question_id 0..1533，与 dev.json 顺序一致）逐题打 Yes/No 标签：

  1. 按题去重候选 SQL（与 adjudicate_pool._dedupe 完全同口径：strip + lower +
     空白折叠；同文本候选合并、count 票数加权、models 保留）。
  2. 官方语义判定——【逐字复刻】BIRD 官方评估器
     tmp_idea_research/finer-sql/evaluation/official_bird_evaluation/
     evaluation_bird_ex.py 的 execute_sql()：
       predicted_res = cursor.execute(pred).fetchall()
       ground_truth_res = cursor.execute(gold).fetchall()
       res = 1 if set(predicted_res) == set(ground_truth_res) else 0
     即：行序无关、重复行折叠（set 语义）、列序敏感（tuple 内顺序即 SELECT
     列序）、无 Spider 的 postprocess/remove_distinct/replace_cur_year/列置换。
     候选执行失败/超时 → label 0（官方 res=0）；gold 执行失败/超时 → 整题剔除。
     注意与 Spider 口径的差异：BIRD 官方是【结果集相等】，不是包语义+列置换。
  3. ORM 判卷 prompt = orm_selection.build_orm_prompt(question, ddl, sql)
     （与 src/bird_select.py --phase prep 打分时逐字一致；evidence=None——
     推理侧不带 evidence，训练侧必须同口径，防 train/infer 漂移）。运行时若
     环境可 import label_orm_data 则做逐字一致性自检。
  4. 输出 chat 格式样本（user = prompt；assistant = Yes/No；附 label(1/0)、
     candidate_sql、dup_count、models 等元数据），与 data/orm_train.json 字段
     完全同构 → train_orm.py 零改动直接吃。
  5. 分层截断变体（--max-per-question K + --cap-out）：每题保留全部正样本，
     不足 K 时用负样本随机补齐（seed 固定）——便宜适配的 cap12 配方入口；
     正样本超 K 的题（全对/近全对）随机取 K 条正样本。变体题目集合与全量一致
     → train_orm.py --seed 42 的 dev 划分在全量/cap/bal 变体上完全一致。
  6. 交叉核对数据准备（不在本脚本内执行，见方案 V1/V2）：
     每题元数据记录 maj_correct = arm_vav 胜者的官方 is_correct
     （outputs/bird_select/items_arm_vav.json，供 dev 侧「败局题救回率」
     rank_acc_maj_wrong 评估）。

整题剔除规则（与 src/label_orm_data.py 同思路）：
  - gold SQL 执行失败/超时（官方语义下 gold 不可判）→ 整题剔除；
  - gold 结果被行上限截断 → 整题剔除（无法可靠判等）。
  BIRD dev 每库单实例 sqlite（dev_databases/<db_id>/<db_id>.sqlite，11 库），
  无 test-suite 变体 → 每题恰好 1 个实例。

数据卫生：样本不含 gold_sql；gold 只存在于 items.json 原始源。

输出（全部新文件，不覆盖任何既有产物）
  data/orm_train_bird.json       全量训练样本（chat + label + 元数据）
  data/orm_train_bird_cap12.json 分层截断变体（--max-per-question 12 时）
  data/orm_questions_bird.json   每题元数据（difficulty/候选数/正确数/
                                 maj_correct=arm_vav 胜者正确性）
  data/orm_label_stats_bird.json 整体统计 + 剔除清单

用法
  # 冒烟（登录节点 CPU，--limit 30 ≈ 1-2 分钟）
  envs/reasoning3b/bin/python src/label_orm_bird.py --limit 30 \
      --out /tmp/orm_train_bird_smoke.json \
      --questions-out /tmp/orm_questions_bird_smoke.json \
      --stats-out /tmp/orm_label_stats_bird_smoke.json
  # 全量（~80k 唯一 (sql, db) 执行任务，16 线程 ≈ 12-20 分钟，见方案）
  envs/reasoning3b/bin/python src/label_orm_bird.py \
      --threads 16 --max-per-question 12 \
      --cap-out data/orm_train_bird_cap12.json
"""

import argparse
import json
import random
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(PROJECT_ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adjudicate_pool as AP  # noqa: E402  _dedupe / ExecutionEngine（纯 CPU）
# ORM 判卷 prompt（与 bird_select 打分逐字一致；VllmScorer 延迟 import，纯 CPU 可用）
from orm_selection import build_orm_prompt  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "bird" / "bird_dev" / "dev_20240627"
DEFAULT_ITEMS = PROJECT_ROOT / "outputs" / "eval_pool_bird" / "items.json"
DEFAULT_DB_ROOT = DATA_DIR / "dev_databases"
DEFAULT_OUT = PROJECT_ROOT / "data" / "orm_train_bird.json"
DEFAULT_QUESTIONS_OUT = PROJECT_ROOT / "data" / "orm_questions_bird.json"
DEFAULT_STATS_OUT = PROJECT_ROOT / "data" / "orm_label_stats_bird.json"
DEFAULT_VAV_ITEMS = PROJECT_ROOT / "outputs" / "bird_select" / "items_arm_vav.json"

YES_STR, NO_STR = "Yes", "No"


# ---------------------------------------------------------------------------
# BIRD 官方语义判定
# ---------------------------------------------------------------------------

def bird_result_set_eq(pred_rows: List[List[Any]], gold_rows: List[List[Any]]) -> bool:
    """【逐字复刻】evaluation_bird_ex.py: set(predicted_res) == set(ground_truth_res)。

    - 行序无关（set 语义）；
    - 重复行折叠（多行相同只算一次——官方如此，必须保留，不可自作主张改 multiset）；
    - 列序敏感（tuple 内顺序 = SELECT 列序；无列置换容忍）；
    - 类型敏感（同一库同存储类型，pred/gold 取回类型天然一致）。
    sqlite fetchall 的所有列类型（None/int/float/str/bytes）均可哈希；str 兜底
    分支仅为防御性代码，正常不可能触发。"""
    try:
        return set(map(tuple, pred_rows)) == set(map(tuple, gold_rows))
    except TypeError:  # 理论上不会发生（sqlite 行值均可哈希）
        return sorted(map(str, pred_rows)) == sorted(map(str, gold_rows))


def read_ddl(db_root: Path, db_id: str) -> str:
    """BIRD schema DDL：sqlite_master 的 CREATE TABLE 语句按表名排序拼接。
    （与 src/bird_select.read_ddl、src/gen_bird_pool.py 同款实现——ORM 判卷
    prompt 的 DDL 表示必须与打分时逐字一致。）"""
    db_path = Path(db_root) / db_id / f"{db_id}.sqlite"
    if not db_path.is_file():
        return ""
    con = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY name").fetchall()
    finally:
        con.close()
    return "\n".join(r[0] for r in rows)


def load_vav_correct(path: Optional[Path]) -> Dict[int, bool]:
    """arm_vav 胜者官方正确性（items_arm_vav.json 的 is_correct，按 dataset_index）。"""
    out: Dict[int, bool] = {}
    if not path or not Path(path).exists():
        return out
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            data = data["items"]
    except Exception as exc:
        print(f"[label_bird] arm_vav items 不可读（maj_correct 置 None）: {exc}",
              file=sys.stderr)
        return out
    for rec in data:
        qid = rec.get("dataset_index", rec.get("di"))
        if qid is None:
            continue
        out[int(qid)] = bool(rec.get("is_correct", False))
    return out


def stratified_cap(entries: List[Dict[str, Any]], labels: List[int], k: int,
                   rng: random.Random) -> List[Dict[str, Any]]:
    """每题分层截断：正样本全保留；不足 K 用负样本随机补齐；正样本超 K（全对/
    近全对题）随机取 K 条正样本。返回保留条目（保持原 entries 相对顺序）。"""
    if k <= 0 or len(entries) <= k:
        return list(entries)
    pos = [e for e, l in zip(entries, labels) if l == 1]
    neg = [e for e, l in zip(entries, labels) if l == 0]
    if len(pos) >= k:
        chosen = set(id(e) for e in rng.sample(pos, k))
    else:
        chosen = set(id(e) for e in pos)
        chosen |= set(id(e) for e in rng.sample(neg, min(len(neg), k - len(pos))))
    return [e for e in entries if id(e) in chosen]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="BIRD ORM 训练数据打标（官方 set-of-tuples 语义，纯 CPU）")
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    ap.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="全量训练样本输出")
    ap.add_argument("--questions-out", type=Path, default=DEFAULT_QUESTIONS_OUT)
    ap.add_argument("--stats-out", type=Path, default=DEFAULT_STATS_OUT)
    ap.add_argument("--cap-out", type=Path, default=None,
                    help="分层截断变体输出（需配合 --max-per-question）")
    ap.add_argument("--max-per-question", type=int, default=0,
                    help="每题最多保留的唯一样本数（0=不截断；便宜适配 cap12 配方）")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--query-timeout", type=float, default=30.0,
                    help="单查询墙钟超时（官方 meta_time_out=30.0）")
    ap.add_argument("--max-vm-steps", type=int, default=5_000_000,
                    help="SQLite VM 步数上限（防节点 CPU 被病态查询拖垮）")
    ap.add_argument("--row-cap", type=int, default=500_000,
                    help="行数上限（官方无上限；提高以贴近官方，超限保守处理）")
    ap.add_argument("--limit", type=int, default=None, help="只打标前 N 题（冒烟）")
    ap.add_argument("--vav-items", type=Path, default=None,
                    help="arm_vav items（maj_correct 来源；默认自动探测）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    items = AP._load_items(args.items)
    if args.limit:
        items = items[: args.limit]
    print(f"[label_bird] {len(items)} 题 | 线程 {args.threads} | timeout "
          f"{args.query_timeout}s | cap={args.max_per_question or 'off'}",
          file=sys.stderr)

    db_root = Path(args.db_root)
    engine = AP.ExecutionEngine(args.threads, args.query_timeout,
                                args.max_vm_steps, args.row_cap)
    rng = random.Random(args.seed)

    # ---- prompt 一致性自检（可选）：与 Spider label_orm_data 的同名函数逐字比对 ----
    prompt_check = "passed"
    try:
        from label_orm_data import build_orm_prompt as _canonical_prompt  # noqa: E402
        if _canonical_prompt("__q__", "__ddl__", "SELECT 1;") != \
                build_orm_prompt("__q__", "__ddl__", "SELECT 1;"):
            raise AssertionError("build_orm_prompt 与 label_orm_data 不一致")
        print("[label_bird] build_orm_prompt 与 label_orm_data 逐字一致性自检通过")
    except ImportError:
        prompt_check = "skipped-no-nltk"
        print("[label_bird] WARN 本环境无法 import label_orm_data（缺 nltk），"
              "跳过自检（bird_select 打分用同一 orm_selection.build_orm_prompt）")
    except Exception as exc:
        prompt_check = f"FAILED: {exc}"

    # ---- arm_vav 胜者正确性（maj_correct 来源）----
    vav_path = args.vav_items if args.vav_items else (
        DEFAULT_VAV_ITEMS if DEFAULT_VAV_ITEMS.exists() else None)
    vav_correct = load_vav_correct(vav_path)

    # ---- 实例与 DDL ----
    def db_path_for(db_id: str) -> str:
        return str(Path(db_root) / db_id / f"{db_id}.sqlite")

    def instances_for(db_id: str) -> List[str]:
        p = Path(db_root) / db_id / f"{db_id}.sqlite"
        return [str(p)] if p.is_file() else []

    ddl_cache: Dict[str, str] = {}

    def ddl_for(db_id: str) -> str:
        if db_id not in ddl_cache:
            ddl_cache[db_id] = read_ddl(db_root, db_id)
        return ddl_cache[db_id]

    # ---- Phase 1：所有 (唯一候选 SQL ∪ gold SQL) × 单实例 一次执行（全局缓存）----
    qc_list: List[Tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]] = []
    tasks: set = set()
    for item in items:
        db_id = item.get("db_id", "")
        insts = instances_for(db_id)
        entries = AP._dedupe(item.get("candidates") or [])
        qc_list.append((item, insts, entries))
        for inst in insts:
            tasks.add((item.get("gold_sql") or "", inst))
            for e in entries:
                text = (e["sql_text"] or "").strip()
                if text:
                    tasks.add((text, inst))
    print(f"[label_bird] phase1: {len(tasks)} 个唯一 (sql, db_path) 执行任务",
          file=sys.stderr)
    engine.run(sorted(tasks), phase="orm_label_bird")
    print(f"[label_bird] phase1 完成: {engine._stats['orm_label_bird']}",
          file=sys.stderr)

    # ---- Phase 2：逐题逐候选官方判定（纯缓存查找，零重复执行）----
    samples: List[Dict[str, Any]] = []
    cap_samples: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []
    excluded_gold_fail: List[int] = []
    excluded_gold_trunc: List[int] = []
    excluded_no_inst: List[int] = []
    prompt_chars: List[int] = []
    cap_stats: Dict[str, Any] = {}

    for qi, (item, insts, entries) in enumerate(qc_list):
        qid = item.get("dataset_index", item.get("di"))
        db_id = item.get("db_id", "")
        question = item.get("question", "")
        gold_raw = item.get("gold_sql") or ""
        difficulty = item.get("difficulty", "")
        n_used = len(insts)

        if not insts:
            excluded_no_inst.append(int(qid))
            continue

        # gold 官方语义可判性检查（任一实例失败/截断 → 整题剔除）
        gold_ok = True
        gold_rows: Optional[List[tuple]] = None
        gold_fail_reason = None
        if n_used == 1:
            g_out = engine.get(gold_raw, insts[0])
            if g_out is None or not g_out["ok"]:
                gold_ok, gold_fail_reason = False, g_out.get("error_type") if g_out else "none"
            elif g_out.get("truncated"):
                gold_ok, gold_fail_reason = False, "truncated"
            else:
                gold_rows = [tuple(r) for r in g_out["rows"]]
        else:  # BIRD dev 恒为单实例；防御性多实例全一致口径（同 Spider 思路）
            per_inst = [engine.get(gold_raw, inst) for inst in insts]
            if any(o is None or not o["ok"] for o in per_inst):
                gold_ok = False
                gold_fail_reason = "some_instance_failed"
            elif any(o.get("truncated") for o in per_inst):
                gold_ok = False
                gold_fail_reason = "truncated"
            else:
                gold_rows = [tuple(r) for r in per_inst[0]["rows"]]
        if not gold_ok:
            (excluded_gold_trunc if gold_fail_reason == "truncated"
             else excluded_gold_fail).append(int(qid))
            continue

        # 逐候选判定（BIRD dev 每库单实例 → 只用 insts[0]）
        labels: List[int] = []
        for e in entries:
            p_out = engine.get(e["sql_text"], insts[0])
            if p_out is None or not p_out["ok"] or p_out.get("truncated"):
                labels.append(0)  # 执行失败/超时/截断 → No（官方 res=0）
                continue
            p_rows = [tuple(r) for r in p_out["rows"]]
            labels.append(1 if bird_result_set_eq(p_rows, gold_rows) else 0)

        q_correct = sum(labels)
        ddl = ddl_for(db_id)

        def make_sample(e: Dict[str, Any], label: int) -> Dict[str, Any]:
            prompt = build_orm_prompt(question, ddl, e["sql_text"])
            prompt_chars.append(len(prompt))
            return {
                "question_id": int(qid),
                "db_id": db_id,
                "difficulty": difficulty,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant",
                     "content": YES_STR if label == 1 else NO_STR},
                ],
                "label": label,
                "candidate_sql": e["sql_text"],
                "dup_count": int(e["count"]),
                "models": sorted(e["models"]),
            }

        for e, label in zip(entries, labels):
            samples.append(make_sample(e, label))

        # 分层截断变体
        if args.max_per_question and args.cap_out:
            kept = stratified_cap(entries, labels, args.max_per_question, rng)
            kept_ids = {id(e) for e in kept}
            for e, label in zip(entries, labels):
                if id(e) in kept_ids:
                    cap_samples.append(make_sample(e, label))

        maj = vav_correct.get(int(qid))
        questions.append({
            "question_id": int(qid),
            "db_id": db_id,
            "difficulty": difficulty,
            "num_instances": n_used,
            "num_unique_candidates": len(entries),
            "num_total_votes": len(item.get("candidates") or []),
            "num_correct": q_correct,
            "num_incorrect": len(entries) - q_correct,
            "maj_correct": maj,  # = arm_vav 胜者官方正确性（败局题救回率评估）
            "maj_winner_source": "arm_vav" if maj is not None else None,
        })

        if (qi + 1) % 50 == 0 or qi + 1 == len(qc_list):
            print(f"[label_bird] 判定 {qi + 1}/{len(qc_list)} 题 "
                  f"({time.perf_counter() - t0:.1f}s, 样本 {len(samples)})",
                  file=sys.stderr)

    if args.cap_out:
        cap_stats = {
            "enabled": bool(args.cap_out),
            "max_per_question": args.max_per_question,
            "seed": args.seed,
            "note": ("分层截断：正样本全保留，不足 K 负样本随机补齐；正样本超 K "
                     "随机取 K。题目集合与全量一致 → train_orm.py --seed 42 dev "
                     "划分在全量/cap/bal 变体上完全一致"),
        }

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
    prompt_chars.sort()

    stats = {
        "meta": {
            "created_by": "src/label_orm_bird.py",
            "input_items": str(args.items),
            "db_root": str(args.db_root),
            "semantics": ("BIRD 官方 evaluation_bird_ex.py execute_sql 逐字复刻："
                          "set(predicted_rows) == set(gold_rows)（行序无关、重复行"
                          "折叠、列序敏感、无 remove_distinct/列置换/replace_cur_year"
                          "等 Spider 清洗）；候选执行失败/超时/截断 → No；gold 失败/"
                          "截断 → 整题剔除；每库单实例 sqlite"),
            "prompt_note": ("ORM prompt = orm_selection.build_orm_prompt(question, "
                            "read_ddl, sql)（evidence=None，与 bird_select 打分逐字"
                            "一致）；chat template + add_generation_prompt，左截断 "
                            "2048 在训练 tokenize 时发生（train_orm.py 同口径）"),
            "prompt_consistency_check": prompt_check,
            "query_timeout": args.query_timeout,
            "max_vm_steps": args.max_vm_steps,
            "row_cap": args.row_cap,
            "seed": args.seed,
            "cap": cap_stats,
        },
        "dataset": {
            "total_questions": len(items),
            "questions_labeled": len(questions),
            "questions_excluded_gold_fail": len(excluded_gold_fail),
            "questions_excluded_gold_fail_ids": excluded_gold_fail,
            "questions_excluded_gold_truncated": len(excluded_gold_trunc),
            "questions_excluded_no_instances": len(excluded_no_inst),
            "total_unique_candidates": sum(q["num_unique_candidates"] for q in questions),
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
            "questions_by_difficulty": dict(sorted(Counter(
                q["difficulty"] for q in questions).items())),
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
        "cap_variant": {
            "file": str(args.cap_out),
            "samples": len(cap_samples),
            "positive": sum(1 for s in cap_samples if s["label"] == 1),
            "questions": len({s["question_id"] for s in cap_samples}),
        } if args.cap_out else None,
        "maj_correct_source": str(vav_path),
        "maj_correct_coverage": sum(1 for q in questions if q["maj_correct"] is not None),
        "execution_stats": {k: v for k, v in engine._stats.items()},
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }

    for p in (args.out, args.questions_out, args.stats_out):
        p.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(samples, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    args.questions_out.write_text(
        json.dumps(questions, ensure_ascii=False, indent=1), encoding="utf-8")
    args.stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    if args.cap_out and cap_samples:
        args.cap_out.parent.mkdir(parents=True, exist_ok=True)
        args.cap_out.write_text(json.dumps(cap_samples, ensure_ascii=False, indent=1),
                                encoding="utf-8")

    print("\n=== BIRD ORM 打标统计 ===")
    print(f"题目: {len(items)}（剔除 gold_fail {len(excluded_gold_fail)} / "
          f"gold_truncated {len(excluded_gold_trunc)} / 无实例 {len(excluded_no_inst)}）")
    print(f"样本: {n} | 正 {pos} / 负 {neg} | 正率 {pos / n:.3f}" if n else f"样本: {n}")
    print(f"全对题 {all_correct} / 全错题 {all_wrong} / 混合题 "
          f"{len(questions) - all_correct - all_wrong}")
    for d, v in sorted(per_diff.items()):
        print(f"  难度 {d:14s}: {v['samples']:6d} 样本 正率 "
              f"{v['positive'] / v['samples']:.3f}" if v["samples"] else
              f"  难度 {d:14s}: 0")
    print(f"prompt 字符数: mean {stats['prompt_chars']['mean']} "
          f"p90 {stats['prompt_chars']['p90']} max {stats['prompt_chars']['max']}")
    print(f"执行: {stats['execution_stats']}")
    print(f"输出: {args.out} / {args.questions_out} / {args.stats_out}")
    if args.cap_out:
        print(f"cap 变体: {args.cap_out}（{len(cap_samples)} 样本）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
