#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BIRD dev 裁决 + ORM 组头选择 + 官方 EX 评估（三层方法移植，三阶段执行）。

把 Spider 管线的三层方法移植到 BIRD dev（dev_20240627，1534 题 11 库、
每库单实例 sqlite——BIRD 无 test-suite 变体）：

  --phase prep  （CPU, reasoning3b）
      去重（AP._dedupe）→ 单实例执行（AP.ExecutionEngine，每库一个
      <db_id>.sqlite）→ 签名 = AP.outcome_signature(执行结果)，分组 =
      AS.build_groups（MI-VAV 同口径：只 SUCCESS 候选入组、size = 池内票数
      加权、rankable = AS.rankable_groups 跳过空组/全零组）→ arm_vav 胜者
      （AP.choose_group_vav）→ 对各 rankable 组代表生成 ORM 判卷 prompt
      （orm_selection.build_orm_prompt 同款）→ 落 work/prep.json +
      work/orm_payloads.json。
  --phase score （GPU, vllmenv）
      orm_selection.VllmScorer（checkpoints/orm_b1，logprobs=20，
      max_length=2048 左截断）只打 rankable 组代表 → work/orm_scores.json。
  --phase final （CPU, reasoning3b）
      arm_vav（MI-VAV 基线）/ arm_orm_grouphead（size × P(Yes) 取最大，
      平票 → 组大小 → str(key)，与 orm_selection 同口径）胜者 →
      官方格式 predict_dev.json（[[question, "sql\\t----- bird -----\\tdb_id"], ...]，
      与 dev.json 顺序对齐；空胜者写 SELECT 1 不跳过）→ 调 FINER 仓库自带
      官方评估器 evaluation_bird_ex.py（唯一有效数字：官方执行准确率，
      含 simple/moderate/challenging/total 分列）→ outputs/bird_select/
      summary.json。

官方评估器调用（与 FINER scripts/eval_bird.sh 最后一段同款）：
  evaluation_bird_ex.py --db_root_path <dev_databases 以 / 结尾> \
    --predicted_sql_json_path <arm>/predict_dev.json --data_mode dev \
    --ground_truth_sql_path dev.sql --num_cpus 12 --mode_predict gpt \
    --diff_json_path dev.json --meta_time_out 30.0

用法：
  envs/reasoning3b/bin/python src/bird_select.py --phase prep \
      --items outputs/eval_pool_bird/items.json --out-dir outputs/bird_select
  envs/vllmenv/bin/python src/bird_select.py --phase score \
      --out-dir outputs/bird_select
  envs/reasoning3b/bin/python src/bird_select.py --phase final \
      --out-dir outputs/bird_select --num-cpus 12
  冒烟：各阶段加 --limit 10（final 阶段自动用前 N 行 dev.sql / 前 N 条
  dev.json 做临时金标/难度文件，完整走通官方评估链路）。
"""
import argparse
import json
import random
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(PROJECT_ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adjudicate_pool as AP  # noqa: E402  去重/执行引擎/签名/choose_group_vav（纯 CPU）
import adjudicate_soft as AS  # noqa: E402  build_groups/rankable_groups/_group_rep/_base_record/_fallback_record
from orm_selection import VllmScorer, build_orm_prompt  # noqa: E402  VllmScorer 延迟 import torch/vLLM

DATA_DIR = PROJECT_ROOT / "data" / "bird" / "bird_dev" / "dev_20240627"
DEFAULT_ITEMS = PROJECT_ROOT / "outputs" / "eval_pool_bird" / "items.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "bird_select"
DEFAULT_DB_ROOT = DATA_DIR / "dev_databases"
DEFAULT_DATA_JSON = DATA_DIR / "dev.json"
DEFAULT_GROUND_TRUTH = DATA_DIR / "dev.sql"
DEFAULT_EVALUATOR = (PROJECT_ROOT / "tmp_idea_research" / "finer-sql" / "evaluation"
                     / "official_bird_evaluation" / "evaluation_bird_ex.py")
DEFAULT_ORM_CKPT = PROJECT_ROOT / "checkpoints" / "orm_b1"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"
DEFAULT_MERGE_PYTHON = PROJECT_ROOT / "envs" / "reasoning3b" / "bin" / "python"

ARMS = ["arm_vav", "arm_orm_grouphead"]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="BIRD dev 三层管线：prep（分组）/ score（ORM）/ final（裁决+官方 EX）")
    ap.add_argument("--phase", choices=["prep", "score", "final"], required=True)
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS,
                    help="生成阶段输出的 items.json（prep 阶段读取）")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="中间产物目录（默认 <out-dir>/work）")
    ap.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    ap.add_argument("--data-json", type=Path, default=DEFAULT_DATA_JSON)
    ap.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    ap.add_argument("--evaluator-py", type=Path, default=DEFAULT_EVALUATOR)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--query-timeout", type=float, default=30.0)
    ap.add_argument("--max-vm-steps", type=int, default=5_000_000)
    ap.add_argument("--row-cap", type=int, default=100_000)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 题（冒烟）")
    ap.add_argument("--seed", type=int, default=0)
    # score 阶段（与 orm_selection.VllmScorer 一致）
    ap.add_argument("--orm-checkpoint", type=Path, default=DEFAULT_ORM_CKPT)
    ap.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    ap.add_argument("--merge-python", default=str(DEFAULT_MERGE_PYTHON))
    ap.add_argument("--max-length", type=int, default=2048,
                    help="ORM prompt 左截断长度（与 orm_selection 同口径）")
    ap.add_argument("--logprobs-topk", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--max-num-seqs", type=int, default=None)
    # final 阶段（官方评估器参数）
    ap.add_argument("--num-cpus", type=int, default=12)
    ap.add_argument("--meta-time-out", type=float, default=30.0)
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def read_ddl(db_root: Path, db_id: str) -> str:
    """BIRD schema DDL：sqlite_master 的 CREATE TABLE 语句按表名排序拼接。
    （与 src/gen_bird_pool.py 同款实现；此处独立复制，避免 prep 阶段拖入
    torch/vLLM import 链。）"""
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


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Phase 1：prep（CPU）——去重 → 单实例执行 → 分组 → arm_vav + ORM payloads
# ---------------------------------------------------------------------------

def phase_prep(args: argparse.Namespace) -> None:
    AP.rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    work = Path(args.work_dir) if args.work_dir else out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    items = AP._load_items(Path(args.items))
    if args.limit:
        items = items[: args.limit]
    print(f"[bird_prep] {len(items)} 题 | out={out_dir} | work={work}",
          file=sys.stderr)

    db_root = Path(args.db_root)
    engine = AP.ExecutionEngine(args.threads, args.query_timeout,
                                args.max_vm_steps, args.row_cap)

    def instances_for(db_id: str) -> List[str]:
        p = Path(db_root) / db_id / f"{db_id}.sqlite"
        return [str(p)] if p.is_file() else []

    # ---- Phase 1：所有唯一候选 SQL × 单实例 并行执行（跨候选缓存）----
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
    print(f"[bird_prep] phase1: {len(phase1_tasks)} 个唯一 (sql, db_path) 任务",
          file=sys.stderr)
    t0 = time.perf_counter()
    engine.run(phase1_tasks, phase="grouping")
    print(f"[bird_prep] phase1 完成: {engine._stats['grouping']}",
          file=sys.stderr)

    # ---- 每题：签名分组 + arm_vav + ORM payloads ----
    ddl_cache: Dict[str, str] = {}
    prep: List[Dict[str, Any]] = []
    orm_payloads: List[Dict[str, Any]] = []
    joins_cache: Dict[str, Tuple[int, str]] = {}
    n_ddl_missing = 0

    for qi, item in enumerate(items):
        entries = AP._dedupe(item.get("candidates") or [])
        insts = instances_for(item.get("db_id", ""))
        n_used = len(insts)

        sigs_per_entry: List[List[str]] = []
        for e in entries:
            if not (e["sql_text"] or "").strip():
                sigs = [AP.ERROR_SIG] * n_used
            else:
                sigs = [AP.outcome_signature(engine.get(e["sql_text"], inst))
                        for inst in insts]
            sigs_per_entry.append(sigs)

        votes: Dict[int, int] = defaultdict(int)
        for c in item.get("candidates") or []:
            ck = AP.normalize_for_dedup(c.get("sql"))
            for ei, e in enumerate(entries):
                if ck == e["key"]:
                    votes[ei] += 1
                    break

        groups, grouped, excluded = AS.build_groups(entries, sigs_per_entry,
                                                    votes, n_used)
        ranked = AS.rankable_groups(groups)

        # ---- arm_vav（MI-VAV 基线，both 池，与 Spider 同口径）----
        if not groups:
            rec_vav = AS._fallback_record(entries, votes, n_used, grouped, excluded)
        else:
            chosen = AP.choose_group_vav(groups)
            if chosen is None:
                rec_vav = AS._fallback_record(entries, votes, n_used, grouped, excluded)
            else:
                rec_vav = AS._base_record(entries, chosen, groups[chosen], "vav",
                                          n_used, grouped, excluded, joins_cache)
        rec_vav["empty_winner"] = (rec_vav["text"] == "")

        # ---- ORM 判卷 prompt（只打 rankable 组代表）----
        db_id = item.get("db_id", "")
        if db_id not in ddl_cache:
            ddl_cache[db_id] = read_ddl(db_root, db_id)
        ddl = ddl_cache[db_id]
        if not ddl:
            n_ddl_missing += 1
        ranked_info: List[Dict[str, Any]] = []
        for key, g in ranked:
            rep = AS._group_rep(entries, g)
            ei = entries.index(rep)
            ranked_info.append({
                "key": str(key), "size": g["size"], "rep_ei": ei,
                "models": sorted(str(m) for m in g["models"]),
                "rep_text": rep["sql_text"],
            })
            prompt = build_orm_prompt(item.get("question", ""), ddl,
                                      rep["sql_text"])
            orm_payloads.append({
                "qi": qi, "ei": ei, "question": item.get("question", ""),
                "db_id": db_id, "prompt": prompt,
            })

        prep.append({
            "qi": qi,
            "dataset_index": item.get("dataset_index", item.get("di")),
            "db_id": db_id,
            "question": item.get("question", ""),
            "gold_sql": item.get("gold_sql") or "",
            "difficulty": item.get("difficulty", ""),
            "num_candidates": len(item.get("candidates") or []),
            "num_unique_candidates": len(entries),
            "num_instances": n_used,
            "entries": [{"key": e["key"], "sql_text": e["sql_text"],
                         "count": e["count"], "min_sample_idx": e["min_sample_idx"],
                         "models": sorted(str(m) for m in e["models"])}
                        for e in entries],
            "sigs_per_entry": sigs_per_entry,
            "votes": {str(ei): v for ei, v in votes.items()},
            "groups_meta": {
                "n_groups": len(groups),
                "grouped": grouped, "excluded": excluded,
                "ranked": ranked_info,
            },
            "arm_vav": rec_vav,
        })
        if (qi + 1) % 50 == 0 or qi + 1 == len(items):
            print(f"[bird_prep] 裁决 {qi + 1}/{len(items)} 题 "
                  f"({time.perf_counter() - t0:.1f}s)", file=sys.stderr)

    _write_json(work / "prep.json",
                {"meta": {"items_file": str(args.items), "limit": args.limit,
                          "n_questions": len(items), "db_root": str(db_root),
                          "n_ddl_missing": n_ddl_missing,
                          "created_at": datetime.now(timezone.utc).isoformat()},
                 "items": prep})
    _write_json(work / "orm_payloads.json", orm_payloads)
    _write_json(work / "prep_exec_stats.json", {
        "grouping_phase": engine._stats.get("grouping", {}),
        "total_wall_seconds": round(time.perf_counter() - t0, 2),
        "n_orm_payloads": len(orm_payloads),
        "n_ddl_missing": n_ddl_missing,
    })
    print(f"[bird_prep] DONE: {len(orm_payloads)} 个 rankable 组代表待 ORM 打分")
    print(f"[bird_prep] work -> {work}")


# ---------------------------------------------------------------------------
# Phase 2：score（GPU, vllmenv）——VllmScorer 只打 rankable 组代表
# ---------------------------------------------------------------------------

def phase_score(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    work = Path(args.work_dir) if args.work_dir else out_dir / "work"
    payloads_file = work / "orm_payloads.json"
    if not payloads_file.exists():
        raise RuntimeError(f"缺少 {payloads_file}——先跑 --phase prep")
    payloads = json.loads(payloads_file.read_text(encoding="utf-8"))
    if not payloads:
        raise RuntimeError("orm_payloads.json 为空（无 rankable 组代表可打分）")
    print(f"[bird_score] {len(payloads)} 个组代表待打分", file=sys.stderr)

    from argparse import Namespace  # noqa: E402
    ns = Namespace(
        base_model=args.base_model,
        orm_checkpoint=Path(args.orm_checkpoint),
        merge_python=args.merge_python,
        max_length=args.max_length,
        logprobs_topk=args.logprobs_topk,
        chunk_size=args.chunk_size,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
    )
    scorer = VllmScorer(ns)
    t0 = time.perf_counter()
    scores = scorer.score([(p["qi"], p["ei"], p["prompt"]) for p in payloads])
    print(f"[bird_score] 打分完成: {len(scores)} 候选 "
          f"({time.perf_counter() - t0:.1f}s, mode={scorer.stats['mode']})",
          file=sys.stderr)

    n_missing = sum(1 for s in scores if s is None)
    if n_missing:
        raise RuntimeError(f"[bird_score] {n_missing} 个组代表缺分（内部错误）")
    _write_json(work / "orm_scores.json", {
        "entries": [{"qi": p["qi"], "ei": p["ei"], "score": float(s)}
                    for p, s in zip(payloads, scores)],
        "stats": scorer.stats,
    })
    print(f"[bird_score] DONE -> {work / 'orm_scores.json'}")


# ---------------------------------------------------------------------------
# Phase 3：final（CPU）——裁决 + 官方格式预测 + FINER 官方评估器
# ---------------------------------------------------------------------------

def _arm_rec_to_jsonable(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v if not isinstance(v, (tuple, set)) else str(v))
            for k, v in rec.items()}


def run_official_evaluator(predict_path: Path, args: argparse.Namespace,
                           n_questions: int, work: Path,
                           arm: str) -> Dict[str, Any]:
    """调用 FINER 官方评估器并解析 accuracy 表（simple/moderate/challenging/total）。"""
    db_root = str(Path(args.db_root).resolve()).rstrip("/") + "/"  # 官方拼接要求以 / 结尾
    gt = args.ground_truth
    diff = args.data_json
    temp_files = []
    if n_questions < 1534:  # 冒烟：临时金标/难度文件（前 N 行/条）
        gt_lines = gt.read_text(encoding="utf-8").splitlines()
        diff_data = json.loads(diff.read_text(encoding="utf-8"))
        gt = work / "smoke_dev.sql"
        diff = work / "smoke_dev.json"
        gt.write_text("\n".join(gt_lines[:n_questions]), encoding="utf-8")
        _write_json(diff, diff_data[:n_questions])
        temp_files += [gt, diff]
    cmd = [
        sys.executable, str(args.evaluator_py),
        "--db_root_path", db_root,
        "--predicted_sql_json_path", str(predict_path),
        "--data_mode", "dev",
        "--ground_truth_sql_path", str(gt),
        "--num_cpus", str(args.num_cpus),
        "--mode_predict", "gpt",
        "--diff_json_path", str(diff),
        "--meta_time_out", str(args.meta_time_out),
    ]
    print(f"[bird_final] official eval [{arm}]: {' '.join(cmd)}", file=sys.stderr)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT,
                          timeout=7200)
    log_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path = work / f"official_{arm}.log"
    log_path.write_text(log_text, encoding="utf-8")
    for f in temp_files:
        try:
            f.unlink()
        except OSError:
            pass
    if proc.returncode != 0:
        raise RuntimeError(
            f"官方评估器退出码 {proc.returncode} [{arm}]，日志: {log_path}")

    m_acc = re.search(r"accuracy\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                      log_text)
    m_cnt = re.search(r"count\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", log_text)
    if not m_acc:
        raise RuntimeError(f"官方评估器输出中未找到 accuracy 表 [{arm}]，日志: {log_path}")
    out = {
        "arm": arm,
        "returncode": proc.returncode,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "simple": float(m_acc.group(1)), "moderate": float(m_acc.group(2)),
        "challenging": float(m_acc.group(3)), "total": float(m_acc.group(4)),
        "counts": {"simple": int(m_cnt.group(1)), "moderate": int(m_cnt.group(2)),
                   "challenging": int(m_cnt.group(3)), "total": int(m_cnt.group(4))}
        if m_cnt else None,
        "log": str(log_path),
        "eval_result_json": str(predict_path.parent / "eval_result_dev.json"),
    }
    print(f"[bird_final] official [{arm}]: "
          f"simple={out['simple']} moderate={out['moderate']} "
          f"challenging={out['challenging']} total={out['total']}", file=sys.stderr)
    return out


def phase_final(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    work = Path(args.work_dir) if args.work_dir else out_dir / "work"
    prep_file = work / "prep.json"
    scores_file = work / "orm_scores.json"
    if not prep_file.exists():
        raise RuntimeError(f"缺少 {prep_file}——先跑 --phase prep")
    if not scores_file.exists():
        raise RuntimeError(f"缺少 {scores_file}——先跑 --phase score")

    prep = json.loads(prep_file.read_text(encoding="utf-8"))
    qcs = prep["items"]
    n_questions = len(qcs)
    score_entries = json.loads(scores_file.read_text(encoding="utf-8"))["entries"]
    score_map: Dict[Tuple[int, int], float] = {}
    for e in score_entries:
        score_map[(int(e["qi"]), int(e["ei"]))] = float(e["score"])

    # 预测文件顺序必须与 dev.json / dev.sql 顺序一致（官方评估器按 enumerate 对齐）
    qids = [qc["dataset_index"] for qc in qcs]
    if qids != sorted(qids) or len(set(qids)) != len(qids):
        raise RuntimeError("prep 条目顺序/ID 异常（必须与 dev.json 顺序一致）")

    per_arm: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in ARMS}
    n_missing_scores = 0
    for qc in qcs:
        qi = qc["qi"]
        entries = qc["entries"]
        ranked = qc["groups_meta"]["ranked"]
        rec_vav = qc["arm_vav"]

        # ---- arm_orm_grouphead：size × P(Yes) 取最大（平票 → size → str(key)）----
        if not ranked:
            rec_orm = dict(rec_vav)
            rec_orm["source"] = rec_vav.get("source", "fallback")
            rec_orm["orm_score"] = None
            rec_orm["orm_fallback"] = True
        else:
            def ghead_key(kg: Dict[str, Any], s: float) -> Tuple[float, int, str]:
                return (float(kg["size"]) * s, int(kg["size"]), kg["key"])

            best = None
            best_key = None
            for kg in ranked:
                s = score_map.get((qi, int(kg["rep_ei"])))
                if s is None:
                    n_missing_scores += 1
                    s = 0.0  # 缺分回填为 0（保守：size 项退化为组大小裁决）
                k = ghead_key(kg, s)
                if best_key is None or k > best_key:
                    best, best_key = kg, k
            rep = entries[int(best["rep_ei"])]
            rec_orm = {
                "source": "orm_grouphead", "text": rep["sql_text"],
                "votes": best["size"], "group_key": best["key"],
                "group_size": best["size"],
                "instances_used": qc["num_instances"],
                "vav_grouped": qc["groups_meta"]["grouped"],
                "vav_excluded": qc["groups_meta"]["excluded"],
                "orm_score": score_map.get((qi, int(best["rep_ei"]))),
                "orm_fallback": False,
            }

        for arm, rec in (("arm_vav", rec_vav), ("arm_orm_grouphead", rec_orm)):
            predicted = rec["text"]
            if not predicted:
                predicted = "SELECT 1"  # 铁律：空预测不跳过
            per_arm[arm].append({
                "dataset_index": qc["dataset_index"],
                "db_id": qc["db_id"],
                "question": qc["question"],
                "gold_sql": qc["gold_sql"],
                "difficulty": qc["difficulty"],
                "predicted_sql": predicted,
                "empty_winner": bool(rec.get("empty_winner")),
                "winner_source": rec.get("source"),
                "winner_votes": rec.get("votes", 0),
                "winner_group_size": rec.get("group_size", 0),
                "orm_score": rec.get("orm_score"),
                "num_candidates": qc["num_candidates"],
                "num_unique_candidates": qc["num_unique_candidates"],
                "num_instances": qc["num_instances"],
            })

    if n_missing_scores:
        print(f"[bird_final] WARN {n_missing_scores} 个组代表缺分（按 0 分处理）",
              file=sys.stderr)

    # ---- 写 items + 官方格式 predict_dev.json ----
    official_results: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        rows = per_arm[arm]
        _write_json(out_dir / f"items_{arm}.json", rows)
        pred_list = [
            [r["question"], f"{r['predicted_sql']}\t----- bird -----\t{r['db_id']}"]
            for r in rows
        ]
        _write_json(arm_dir / "predict_dev.json", pred_list)
        official_results[arm] = run_official_evaluator(
            arm_dir / "predict_dev.json", args, n_questions, work, arm)

    # ---- 汇总 ----
    winner_sources: Dict[str, Dict[str, int]] = {}
    for arm in ARMS:
        winner_sources[arm] = dict(
            Counter(r["winner_source"] for r in per_arm[arm]))
    gen_summary = None
    gen_sum_path = Path(args.items).parent / "summary.json"
    if gen_sum_path.exists():
        try:
            gen_summary = json.loads(gen_sum_path.read_text(encoding="utf-8"))
        except Exception:
            gen_summary = None
    difficulty_dist = dict(Counter(qc["difficulty"] for qc in qcs))
    prep_stats = json.loads((work / "prep_exec_stats.json").read_text(encoding="utf-8"))
    scoring_stats = json.loads(scores_file.read_text(encoding="utf-8")).get("stats")

    summary = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": "src/bird_select.py",
            "items_file": str(args.items),
            "out_dir": str(out_dir),
            "work_dir": str(work),
            "db_root": str(args.db_root),
            "ground_truth": str(args.ground_truth),
            "diff_json": str(args.data_json),
            "evaluator": str(args.evaluator_py),
            "n_questions": n_questions,
            "limit": args.limit,
            "official_evaluator_cmd": (
                "evaluation_bird_ex.py --db_root_path <dev_databases/> "
                "--predicted_sql_json_path <arm>/predict_dev.json --data_mode dev "
                f"--ground_truth_sql_path {args.ground_truth} --num_cpus {args.num_cpus} "
                "--mode_predict gpt "
                f"--diff_json_path {args.data_json} --meta_time_out {args.meta_time_out}"),
            "method": (
                "三层方法移植（Spider B1/ORM 管线 → BIRD dev 单实例）：候选池 = 4 源 "
                "checkpoint × 16 采样（gen_bird_pool.py）；执行分组 = 单实例签名 "
                "（AP.outcome_signature）+ AS.build_groups（MI-VAV 同口径，只 SUCCESS "
                "候选入组、size=池内票数加权）；arm_vav = AP.choose_group_vav 最大组；"
                "arm_orm_grouphead = size × P(Yes)（orm_b1, logprobs=20, "
                "build_orm_prompt 同款，只打 rankable 组代表），平票 → 组大小 → str(key)；"
                "NO_RESULTS 回退同池 arm_maj；空胜者写 SELECT 1。唯一有效数字 = FINER "
                "官方评估器 evaluation_bird_ex.py 的执行准确率。"),
            "orm_prompt_note": (
                "ORM 判卷 prompt = orm_selection.build_orm_prompt 原函数（逐字同款）："
                "canonical 生成端 prompt（不含 evidence，与 Spider ORM 训练/推理同口径）"
                "+ Candidate SQL Query 块 + Yes/No 指令；chat template 左截断 2048。"
                "BIRD DDL 较长（最长 ~2386 token），左截断会保留 question 尾部 + 候选 "
                "SQL + 指令，截掉大部分 DDL 头部——与 Spider 训练口径一致的刻意选择。"),
            "gen_prompt_note": (
                "生成端 prompt = sqlite_master CREATE TABLE DDL + question + evidence"
                "（有值才加，ReasoningGeneratorAgent.build_prompt canonical 模板，"
                "dialect=sqlite），截断 3072（实测 1534 题最长 2386，全量不截断）。"),
        },
        "difficulty_distribution": difficulty_dist,
        "prep_exec_stats": prep_stats,
        "scoring_stats": scoring_stats,
        "n_missing_scores": n_missing_scores,
        "winner_sources": winner_sources,
        "official_exec_accuracy": official_results,
        "generation_summary": gen_summary,
    }
    _write_json(out_dir / "summary.json", summary)

    print("\n=== BIRD 官方执行准确率（FINER evaluation_bird_ex.py）===")
    for arm in ARMS:
        r = official_results[arm]
        print(f"  {arm:22s} simple={r['simple']:.2f} moderate={r['moderate']:.2f} "
              f"challenging={r['challenging']:.2f} total={r['total']:.2f} "
              f"({r['counts']['total']} 题)" if r.get("counts") else
              f"  {arm:22s} total={r['total']:.2f}")
    print(f"\nsummary -> {out_dir / 'summary.json'}")
    for arm in ARMS:
        print(f"items    -> {out_dir / f'items_{arm}.json'}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.phase == "prep":
        phase_prep(args)
    elif args.phase == "score":
        phase_score(args)
    else:
        phase_final(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
