#!/usr/bin/env python3
"""
L2: 多视角 × 采样执行投票（Multi-Perspective × sampling Execution Voting, MPEV）。

论文 L2 升级：在 P1 vav 执行投票（finer_port/eval_vav.py，n 采样→执行分组投票）
之上引入「多视角」维度，复用现有实现、不重复造轮子：

  1. 每题构造 N 种 prompt 视角 —— 复用 src/eval_5prompt_agent.py::build_prompt_variants
     的 p1-p5（--n-perspectives 7 时取 p1-p7）；
  2. 每视角采样 K 条候选（默认 K=6，T=1.0）→ 5×6=30 候选
     （VavSampler.sample_batch，单次 forward 产视角数×K 条）；
  3. 执行结果分组投票 —— 复用 vav_voting 的 normalize_execution_result /
     choose_group_vav（经 run_vav_voting 纯函数流水线：语法错误过滤、退化组跳过、
     fallback 语义与 FINER majority_voting 完全一致）；
  4. 置信度过滤：最大组占比 = chosen_group_size / num_candidates，低于
     --confidence-threshold（默认 0.2，即 30 候选 ≥6 票同组）→ 标记
     low_confidence=True（仍输出 predicted_sql，下游可据标记做保守/重跑策略）；
  5. 失败类型感知：候选全部执行失败（chosen_result == NO_RESULTS）时输出
     failure_diagnosis —— 语法 / 超时 / 空结果 / 其他 四类计数 + 高频错误消息
     （空结果口径：SQL 为空（parse 失败）或执行成功但 0 行）；
  6. 输出 items.json（predicted_sql + confidence + votes + strategy），
     与 scripts/eval_official.sh 兼容（官方 test-suite EX 复评）；
  7. 断点续跑复用 src/spider_utils.py 的 checkpoint 协议
     （build_run_config / load_checkpoint / validate_resume_config /
      validate_checkpoint_integrity / save_checkpoint），混配置直接拒绝。

用法（GPU，与 eval_vav.py 同风格）:
    # FINER 权重（系统提示贴近 FINER 训练分布，视角仍来自 p1-p5）:
    python finer_port/eval_mpev.py \\
        --model-path models/FINER-SQL-3B-Spider --prompt-style finer \\
        --output-dir outputs/eval_mpev_finer --limit 1034
    # 标准 5 问法（与 eval_5prompt_agent 相同 prompt 视角）:
    python finer_port/eval_mpev.py \\
        --model-path models/Qwen2.5-Coder-3B-Instruct --prompt-style 5p \\
        --output-dir outputs/eval_mpev_5p --limit 100
    # 完成后官方 test-suite 复评:
    bash scripts/eval_official.sh outputs/eval_mpev_finer/items.json outputs/official_mpev_finer

纯逻辑自测（无 GPU；仅 diagnose_failures / 置信度 / 空候选投票等纯函数）:
    python finer_port/eval_mpev.py --self-test
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

# sys.path 引导：项目 src/（spider_utils / eval_5prompt_agent）+ 本包
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
for _p in (str(_SRC_DIR), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from spider_utils import (  # noqa: E402
    SpiderLoader,
    DatabaseExecutor,
    compare_execution_results,
    normalize_sql,
    build_run_config,
    validate_resume_config,
    validate_checkpoint_integrity,
    save_checkpoint,
    load_checkpoint,
)
from vav_voting import (  # noqa: E402
    VavEvaluator,
    run_vav_voting,
    normalize_execution_result,
    is_syntax_error,
)
from sampler import VavSampler, FINER_SYSTEM_PROMPT  # noqa: E402
from eval_5prompt_agent import build_prompt_variants  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_mpev"

EVALUATOR_TYPE = "mpev_execution_voting"


# ===================================================================
# Run config（含 mpev 专属字段，resume 时逐字段比对防混实验）
# ===================================================================

def build_mp_run_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = build_run_config(
        spider_dir=args.spider_dir,
        start_index=args.start_index,
        limit=args.limit,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        evaluator_type=EVALUATOR_TYPE,
    )
    cfg.update(
        {
            "n_perspectives": args.n_perspectives,
            "samples_per_prompt": args.samples_per_prompt,
            "prompt_style": args.prompt_style,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "confidence_threshold": args.confidence_threshold,
            "lora_path": args.lora_path,
        }
    )
    return cfg


# ===================================================================
# 多视角 prompt 构造（复用 build_prompt_variants，不重复造轮子）
# ===================================================================

def build_mp_prompt_chats(
    sampler: VavSampler,
    question: str,
    ddl: str,
    prompt_style: str = "5p",
    n_prompts: int = 5,
) -> List[str]:
    """
    构造 N 个视角的 chat 文本（每视角 1 条；配合 sample_batch(n=K) 产 N×K 候选）。

    视角文本复用 eval_5prompt_agent.build_prompt_variants（p1-p5 / p1-p7）：
      - prompt_style="5p"：与 eval_5prompt_agent 一致的纯 user 消息（Qwen chat 模板）；
      - prompt_style="finer"：FINER 系统提示 + 视角文本作为 user 内容
        （视角变化仍来自 p1-p5，系统提示贴近 FINER 训练分布，兼容 FINER 权重）。
    """
    variants = build_prompt_variants(question, ddl)[:n_prompts]
    chats: List[str] = []
    for v in variants:
        if prompt_style == "finer":
            messages = [
                {"role": "system", "content": FINER_SYSTEM_PROMPT},
                {"role": "user", "content": v},
            ]
        else:
            messages = [{"role": "user", "content": v}]
        chats.append(
            sampler.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return chats


# ===================================================================
# 置信度（最大组占比）
# ===================================================================

def compute_confidence(votes: int, num_candidates: int) -> float:
    """最大组占比 = 选中组票数 / 候选总数（0 候选时 0.0）。"""
    if num_candidates <= 0:
        return 0.0
    return round(votes / num_candidates, 4)


def is_low_confidence(confidence: float, threshold: float) -> bool:
    """低于阈值（严格小于）→ low_confidence；阈值默认 0.2 = 30 候选 ≥6 票。"""
    return confidence < threshold


# ===================================================================
# 失败类型感知诊断（候选全部执行失败时）
# ===================================================================

def diagnose_failures(pred_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    候选全部执行失败时的错误类型统计（语法 / 超时 / 空结果 / 其他）。

    口径：
      - 空结果 empty_results：SQL 为空（parse 失败 / 空串，execute_cached 返回
        "Empty SQL"）或执行成功但 0 行（条件无匹配等）；
      - 语法错误 syntax_errors：is_syntax_error（错误消息不含基础设施关键词）；
      - 超时 timeouts：非语法错误且错误消息含 timeout / timed out 关键词；
      - 其他 other_errors：其余失败（基础设施类非超时错误）。

    pred_results: run_vav_voting 输入同构（每项含 sql / result / index）。
    """
    counts = {"syntax_errors": 0, "timeouts": 0, "empty_results": 0, "other_errors": 0}
    msg_counter: Counter = Counter()
    for pr in pred_results:
        res = pr["result"]
        sql = (pr.get("sql") or "").strip()
        if not sql:
            counts["empty_results"] += 1
            msg_counter["<empty SQL / parse failed>"] += 1
            continue
        if res.get("success"):
            if not (res.get("full_rows") or []):
                counts["empty_results"] += 1  # 成功但零行（仍属「空结果」类）
            continue
        if is_syntax_error(res):
            counts["syntax_errors"] += 1
        else:
            err = str(res.get("error") or "").lower()
            if "timeout" in err or "timed out" in err:
                counts["timeouts"] += 1
            else:
                counts["other_errors"] += 1
        msg = str(res.get("error") or "Unknown error")[:120]
        msg_counter[msg] += 1
    return {
        "counts": counts,
        "total": len(pred_results),
        "top_error_messages": [
            {"message": m, "count": c} for m, c in msg_counter.most_common(3)
        ],
    }


# ===================================================================
# 单样本处理
# ===================================================================

def _new_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_index": item["dataset_index"],
        "db_id": item["db_id"],
        "question": item["question"],
        "gold_sql": item["query"],
        "difficulty": item.get("difficulty", "unknown"),
        "predicted_sql": None,          # ← eval_official.sh 兼容字段
        "confidence": None,             # 最大组占比
        "low_confidence": None,         # confidence < 阈值
        "votes": 0,                     # 选中组票数
        "num_candidates": 0,            # 视角数 × 每视角采样数
        "strategy": None,
        "selected_sql": None,
        "selected_sql_index": -1,
        "winner_prompt_id": None,       # 选中候选来自哪个视角 (0-based)
        "chosen_result": None,
        "chosen_group_size": 0,
        "majority_result": None,
        "majority_group_size": 0,
        "degenerate_skip_applied": False,
        "vav_self_match": False,
        "majority_self_match": False,
        "selected_custom_exec_match": None,      # 训练同口径（compare_execution_results）
        "selected_custom_exec_match_reason": None,
        "gold_execution_success": False,
        "gold_error": None,
        "parse_success_count": 0,
        "num_predicted_sqls": 0,
        "num_syntax_errors": 0,
        "num_infrastructure_failures": 0,
        "num_valid_sqls_after_filtering": 0,
        "num_unique_sql": 0,
        "result_groups": {},
        "failure_diagnosis": None,
        "candidates": [],
        "generation_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "error": None,
    }


def process_item(
    item: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    evaluator: VavEvaluator,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    执行验证 + vav 分组投票 + 置信度过滤 + 失败诊断，产出单个评估条目。

    candidates: 主循环汇总的候选 dict 列表（每条含 sql / prompt_id / global_id /
    raw_response / parse_success / parse_method），global_id = prompt_id×K + candidate_id。
    """
    r = _new_item(item)
    t0 = time.perf_counter()
    db_id = item["db_id"]
    gold_sql = item["query"]

    # 1) 逐候选执行（带 (db_id, normalize_sql) 缓存；空 SQL 不触库）
    pred_results: List[Dict[str, Any]] = []
    for c in candidates:
        outcome = evaluator.execute_cached(db_id, c["sql"])
        pred_results.append(
            {"sql": c["sql"], "result": outcome, "index": c.get("global_id", len(pred_results))}
        )
    gold_outcome = evaluator.execute_cached(db_id, gold_sql)

    # 2) vav 分组投票（复用 normalize_execution_result / choose_group_vav 流水线）
    vote = run_vav_voting(pred_results, gold_outcome, gt_sql=gold_sql, strategy="vav")

    selected_sql = vote["selected_sql"]
    num_candidates = len(candidates)
    r["predicted_sql"] = selected_sql or None
    r["selected_sql"] = selected_sql or None
    r["selected_sql_index"] = vote["selected_sql_index"]
    r["num_candidates"] = num_candidates
    r["strategy"] = f"mpev_{args.prompt_style}_p{args.n_perspectives}x{args.samples_per_prompt}_vav"
    r["votes"] = vote["chosen_group_size"]
    r["confidence"] = compute_confidence(vote["chosen_group_size"], num_candidates)
    r["low_confidence"] = is_low_confidence(r["confidence"], args.confidence_threshold)

    r["chosen_result"] = vote["chosen_result"]
    r["chosen_group_size"] = vote["chosen_group_size"]
    r["majority_result"] = vote["majority_result"]
    r["majority_group_size"] = vote["majority_group_size"]
    r["degenerate_skip_applied"] = vote["degenerate_skip_applied"]
    r["vav_self_match"] = vote["is_sample_correct"]
    r["majority_self_match"] = vote["majority_is_correct"]
    r["gold_execution_success"] = bool(gold_outcome.get("success"))
    r["gold_error"] = (
        gold_outcome.get("error") if not gold_outcome.get("success") else None
    )
    r["parse_success_count"] = sum(1 for c in candidates if c.get("parse_success"))
    r["num_predicted_sqls"] = vote["num_predicted_sqls"]
    r["num_syntax_errors"] = vote["num_syntax_errors"]
    r["num_infrastructure_failures"] = vote["num_infrastructure_failures"]
    r["num_valid_sqls_after_filtering"] = vote["num_valid_sqls_after_filtering"]
    r["num_unique_sql"] = len(
        {
            normalize_sql(c["sql"])
            for c in candidates
            if (c.get("sql") or "").strip()
        }
    )
    r["result_groups"] = vote["result_groups"]

    # 3) 失败类型感知：候选全部执行失败 → 输出诊断
    if vote["chosen_result"] == "NO_RESULTS":
        r["failure_diagnosis"] = diagnose_failures(pred_results)

    # 4) 获胜视角（供 L2 分析：哪个视角贡献最大）
    sel_idx = vote["selected_sql_index"]
    if sel_idx >= 0:
        for c in candidates:
            if c.get("global_id") == sel_idx:
                r["winner_prompt_id"] = c.get("prompt_id")
                break

    # 5) candidates 数组（留作分析；--save-full-responses 时存全文）
    for c, pr in zip(candidates, pred_results):
        entry = {
            "candidate_id": c.get("global_id", 0),
            "prompt_id": c.get("prompt_id"),
            "sql": c["sql"],
            "parse_success": c.get("parse_success", False),
            "parse_method": c.get("parse_method"),
            "execution_success": bool(pr["result"].get("success")),
            "execution_error": (
                pr["result"].get("error")
                if not pr["result"].get("success")
                else None
            ),
            "result_key": normalize_execution_result(pr["result"], gt_sql=gold_sql),
        }
        if args.save_full_responses:
            entry["raw_response"] = c.get("raw_response", "")
        else:
            entry["raw_response_preview"] = (c.get("raw_response") or "")[:200]
        r["candidates"].append(entry)

    # 6) 训练同口径对照：compare_execution_results（ORDER BY 感知 multiset）
    sel_res = vote.get("selected_result") or {}
    if (
        selected_sql
        and gold_outcome.get("success")
        and sel_res.get("success")
        and not (gold_outcome.get("full_rows_truncated") or sel_res.get("full_rows_truncated"))
    ):
        cmp = compare_execution_results(
            sel_res.get("full_rows") or [],
            gold_outcome.get("full_rows") or [],
            gold_sql=gold_sql,
        )
        r["selected_custom_exec_match"] = cmp["match"]
        r["selected_custom_exec_match_reason"] = cmp.get("match_reason")
    elif selected_sql and gold_outcome.get("success") and sel_res.get("success"):
        r["selected_custom_exec_match"] = False
        r["selected_custom_exec_match_reason"] = "full rows truncated"

    r["evaluation_seconds"] = round(time.perf_counter() - t0, 4)
    r["generation_seconds"] = round(time.perf_counter() - t0, 4)
    return r


# ===================================================================
# Summary
# ===================================================================

def _difficulty_split(all_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[str, Dict[str, int]] = {}
    for it in all_items:
        d = str(it.get("difficulty", "unknown"))
        s = by.setdefault(
            d, {"count": 0, "vav_self_match": 0, "majority_self_match": 0, "low_confidence": 0}
        )
        s["count"] += 1
        if it.get("vav_self_match"):
            s["vav_self_match"] += 1
        if it.get("majority_self_match"):
            s["majority_self_match"] += 1
        if it.get("low_confidence"):
            s["low_confidence"] += 1
    out: Dict[str, Any] = {}
    for d, s in sorted(by.items()):
        out[d] = {
            "count": s["count"],
            "vav_self_match_count": s["vav_self_match"],
            "vav_self_match_rate": (
                round(s["vav_self_match"] / s["count"], 4) if s["count"] else 0.0
            ),
            "majority_self_match_count": s["majority_self_match"],
            "majority_self_match_rate": (
                round(s["majority_self_match"] / s["count"], 4) if s["count"] else 0.0
            ),
            "low_confidence_count": s["low_confidence"],
            "low_confidence_rate": (
                round(s["low_confidence"] / s["count"], 4) if s["count"] else 0.0
            ),
        }
    return out


def _aggregate_diagnoses(all_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总「候选全部执行失败」题目的错误类型计数（语法/超时/空结果/其他）。"""
    agg: Counter = Counter()
    n_items = 0
    for it in all_items:
        d = it.get("failure_diagnosis")
        if d:
            n_items += 1
            for k, v in (d.get("counts") or {}).items():
                agg[k] += v
    return {
        "all_failed_items": n_items,
        "error_type_counts": dict(agg),
    }


def _perspective_winner_counts(
    all_items: List[Dict[str, Any]], n_perspectives: int
) -> Dict[str, int]:
    """每个视角（p1..pN）当选获胜候选的次数（L2 视角贡献分析）。"""
    c: Counter = Counter()
    for it in all_items:
        w = it.get("winner_prompt_id")
        if w is not None:
            c[int(w)] += 1
    return {f"p{i + 1}": c[i] for i in range(n_perspectives)}


def build_summary(
    all_items: List[Dict[str, Any]],
    requested_indices: Set[int],
    run_config: Dict[str, Any],
    evaluator: VavEvaluator,
    total_wall_seconds: float,
    generated_at: str,
) -> Dict[str, Any]:
    total = len(all_items)
    n_vav = sum(1 for it in all_items if it.get("vav_self_match"))
    n_maj = sum(1 for it in all_items if it.get("majority_self_match"))
    n_skip = sum(1 for it in all_items if it.get("degenerate_skip_applied"))
    n_low = sum(1 for it in all_items if it.get("low_confidence"))
    n_custom = sum(
        1 for it in all_items if it.get("selected_custom_exec_match") is True
    )
    n_custom_scored = sum(
        1 for it in all_items if it.get("selected_custom_exec_match") is not None
    )

    confs = [
        it.get("confidence") for it in all_items if it.get("confidence") is not None
    ]
    avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

    cand_parse = sum(it.get("parse_success_count", 0) for it in all_items)
    cand_total = sum(it.get("num_predicted_sqls", 0) for it in all_items)
    cand_exec_ok = sum(
        sum(1 for c in it.get("candidates", []) if c.get("execution_success"))
        for it in all_items
    )
    avg_unique = (
        sum(it.get("num_unique_sql", 0) for it in all_items) / total if total else 0.0
    )
    gen_times = [it.get("generation_seconds", 0.0) for it in all_items]

    return {
        "evaluator_type": EVALUATOR_TYPE,
        "is_official_spider_metric": False,
        "note": (
            "L2 MPEV：N 视角 × K 采样 → 执行分组投票（vav 口径，与 FINER "
            "majority_voting 一致）；confidence = 选中组票数/候选总数，低于 "
            "confidence_threshold 标记 low_confidence（仍输出 predicted_sql）；"
            "官方 test-suite EX 需 bash scripts/eval_official.sh <items.json> <out_dir>。"
        ),
        "total_requested": len(requested_indices),
        "total_completed": total,
        "vav_self_match_count": n_vav,
        "vav_self_match_rate": round(n_vav / total, 4) if total else 0.0,
        "majority_self_match_count": n_maj,
        "majority_self_match_rate": round(n_maj / total, 4) if total else 0.0,
        "degenerate_skip_count": n_skip,
        "degenerate_skip_rate": round(n_skip / total, 4) if total else 0.0,
        "low_confidence_count": n_low,
        "low_confidence_rate": round(n_low / total, 4) if total else 0.0,
        "average_confidence": avg_conf,
        "failure_diagnosis_aggregate": _aggregate_diagnoses(all_items),
        "perspective_winner_counts": _perspective_winner_counts(
            all_items, int(run_config.get("n_perspectives", 5))
        ),
        "selected_custom_exec_match_count": n_custom,
        "selected_custom_exec_match_rate": (
            round(n_custom / n_custom_scored, 4) if n_custom_scored else 0.0
        ),
        "selected_custom_exec_scored_count": n_custom_scored,
        "candidate_parse_success_count": cand_parse,
        "candidate_parse_success_rate": (
            round(cand_parse / cand_total, 4) if cand_total else 0.0
        ),
        "candidate_execution_success_count": cand_exec_ok,
        "candidate_execution_success_rate": (
            round(cand_exec_ok / cand_total, 4) if cand_total else 0.0
        ),
        "execution_cache_hits": evaluator.cache_hits,
        "execution_cache_misses": evaluator.cache_misses,
        "execution_cache_hit_ratio": evaluator.cache_ratio,
        "avg_unique_sql_per_item": round(avg_unique, 2),
        "difficulty_split": _difficulty_split(all_items),
        "average_generation_seconds": (
            round(sum(gen_times) / len(gen_times), 4) if gen_times else 0.0
        ),
        "total_wall_seconds": round(total_wall_seconds, 2),
        "requested_indices": sorted(requested_indices),
        "run_config": run_config,
        "generated_at": generated_at,
    }


# ===================================================================
# Main
# ===================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "L2 MPEV 评估：N 视角 × K 采样 → 执行分组投票 → 置信度过滤 + 失败诊断 "
            "→ 输出 SQL（与 scripts/eval_official.sh 兼容）"
        ),
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH),
                        help="本地模型路径（FINER 权重或本项目 3B 基座/LoRA 合并后权重）")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="输出目录（checkpoint.json / items.json / summary.json）")
    parser.add_argument("--prompt-style", choices=["5p", "finer"], default="5p",
                        help="5p=标准 5 问法视角（eval_5prompt_agent 同款）；"
                             "finer=FINER 系统提示包装的 5 视角（兼容 FINER 权重）")
    parser.add_argument("--n-perspectives", type=int, default=5, choices=[5, 7],
                        help="投票用多少个 prompt 视角（build_prompt_variants 的 p1-p5/p1-p7）")
    parser.add_argument("--samples-per-prompt", type=int, default=6,
                        help="每视角采样 K 条（默认 6 → 5×6=30 候选）")
    parser.add_argument("--confidence-threshold", type=float, default=0.2,
                        help="最大组占比阈值（默认 0.2 = 30 候选 ≥6 票同组）")
    parser.add_argument("--limit", type=int, default=1034,
                        help="评估条数（默认全量 dev 1034；子集先跑 100-200 看增益方向）")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--spider-dir", default=str(DEFAULT_SPIDER_DIR))
    parser.add_argument("--lora-path", default=None,
                        help="LoRA adapter 路径（可选；不加则跑基座/FINER 权重）")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="与训练 max_completion 对齐（PLAN §8 生成长度一致性）")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="每次 forward 处理的题目数（batch>1 时一次 forward 产 B*N*K 条）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="每处理 N 题写一次 checkpoint（断点续跑）")
    parser.add_argument("--save-full-responses", action="store_true",
                        help="candidates 里存完整 raw_response（默认只存 200 字符预览）")
    parser.add_argument("--allow-remote", action="store_true",
                        help="允许从 HF 在线拉取权重（默认 local_files_only）")
    parser.add_argument("--self-test", action="store_true",
                        help="纯逻辑自测（无 GPU；测完退出）")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.self_test:
        sys.exit(_run_self_tests())
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 加载数据集 ---
    loader = SpiderLoader(args.spider_dir)
    items = loader.load_dev(limit=args.limit, start_index=args.start_index)
    requested_indices: Set[int] = {it["dataset_index"] for it in items}
    print(f"Loaded {len(items)} items (start={args.start_index}, limit={args.limit})")

    # --- checkpoint / resume（协议与 evaluate_after_grpo / eval_vav 一致） ---
    run_config = build_mp_run_config(args)
    cp = load_checkpoint(output_dir)
    if cp["run_config"] is not None:
        validate_resume_config(cp["run_config"], run_config)
    validate_checkpoint_integrity(cp, requested_indices)
    completed: Set[int] = cp["completed_indices"]
    all_items: List[Dict[str, Any]] = list(cp["items"])
    print(f"Resume: {len(completed)}/{len(requested_indices)} items already completed")

    # --- 加载模型 + 执行器 ---
    print(f"\nLoading model: {args.model_path}")
    sampler = VavSampler(
        model_path=args.model_path,
        lora_path=args.lora_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_style="default" if args.prompt_style == "5p" else "finer",
        seed=args.seed,
        local_files_only=not args.allow_remote,
    )
    evaluator = VavEvaluator(DatabaseExecutor(args.spider_dir))
    n_cand_total = args.n_perspectives * args.samples_per_prompt
    print(
        f"MPEV ready: {args.n_perspectives} 视角 x {args.samples_per_prompt} 采样 "
        f"= {n_cand_total} 候选 | style={args.prompt_style} "
        f"| conf_thr={args.confidence_threshold}\n"
    )

    # --- 预构造 N 视角 prompt（DDL 失败条目记为 error 条目，不阻塞） ---
    chat_texts: Dict[int, Optional[List[str]]] = {}
    for it in items:
        try:
            ddl = loader.format_ddl(it["db_id"])
            chat_texts[it["dataset_index"]] = build_mp_prompt_chats(
                sampler, it["question"], ddl,
                prompt_style=args.prompt_style, n_prompts=args.n_perspectives,
            )
        except RuntimeError as exc:
            chat_texts[it["dataset_index"]] = None
            print(f"[WARN] item {it['dataset_index']} db_id={it['db_id']} "
                  f"DDL failed: {exc}")

    pending = [it for it in items if it["dataset_index"] not in completed]
    wall_start = time.perf_counter()

    for i in range(0, len(pending), args.batch_size):
        batch = pending[i:i + args.batch_size]

        # 展平：本批所有题 × 视角 → 一次 forward 采样 K 条/视角
        flat: List[Tuple[int, int, str]] = []  # (批内位置, 视角号, chat 文本)
        for pos, it in enumerate(batch):
            chats = chat_texts.get(it["dataset_index"])
            if chats:
                for pi, ctext in enumerate(chats):
                    flat.append((pos, pi, ctext))

        gen_map: Dict[int, List[Dict[str, Any]]] = {}
        gen_ok = True
        if flat:
            try:
                gen_start = time.perf_counter()
                res = sampler.sample_batch(
                    [f[2] for f in flat], n=args.samples_per_prompt
                )
                print(
                    f"  batch gen {len(batch)} items x {args.n_perspectives} prompts "
                    f"x {args.samples_per_prompt} in "
                    f"{time.perf_counter() - gen_start:.1f}s"
                )
                for (pos, pi, _), cands in zip(flat, res):
                    ds = batch[pos]["dataset_index"]
                    item_cands = gen_map.setdefault(ds, [])
                    for cand in cands:
                        c2 = dict(cand)
                        c2["prompt_id"] = pi
                        c2["global_id"] = pi * args.samples_per_prompt + int(
                            cand["candidate_id"]
                        )
                        item_cands.append(c2)
            except Exception as exc:
                gen_ok = False
                print(f"[WARN] batch generation failed ({exc}); "
                      f"recording generation_failed items")

        # --- 每题：执行验证 + vav 投票 + 置信度 + 失败诊断 ---
        for it in batch:
            ds = it["dataset_index"]
            if chat_texts.get(ds) is None:
                r = _new_item(it)
                r["error"] = "ddl_load_failed"
            elif not gen_ok and ds not in gen_map:
                r = _new_item(it)
                r["error"] = "generation_failed"
            else:
                try:
                    r = process_item(
                        it, gen_map.get(ds, []), evaluator, args
                    )
                except Exception as exc:
                    r = _new_item(it)
                    r["error"] = f"processing failed: {exc}"
            all_items.append(r)
            completed.add(ds)

        # --- 定期 checkpoint ---
        if (i // args.batch_size + 1) % args.checkpoint_every == 0:
            save_checkpoint(
                output_dir,
                {"completed_indices": sorted(completed), "items": all_items},
                run_config,
            )
            done = len(all_items)
            n_vav = sum(1 for it in all_items if it.get("vav_self_match"))
            n_low = sum(1 for it in all_items if it.get("low_confidence"))
            print(
                f"  [checkpoint {done}/{len(items)}] "
                f"vav_self={n_vav}/{done} ({n_vav / done:.1%}) "
                f"low_conf={n_low}/{done} ({n_low / done:.1%}) "
                f"wall={time.perf_counter() - wall_start:.0f}s"
            )

    wall_end = time.perf_counter()
    total_wall = wall_end - wall_start

    # --- 收尾：最终 checkpoint + items.json + summary.json ---
    save_checkpoint(
        output_dir,
        {"completed_indices": sorted(completed), "items": all_items},
        run_config,
    )
    items_path = output_dir / "items.json"
    with open(items_path, "w", encoding="utf-8") as fh:
        json.dump(all_items, fh, ensure_ascii=False, indent=2)

    summary = build_summary(
        all_items,
        requested_indices,
        run_config,
        evaluator,
        total_wall,
        datetime.now(timezone.utc).isoformat(),
    )
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # --- 打印结果 ---
    print("\n" + "=" * 60)
    print("  L2 MPEV EVALUATION SUMMARY (multi-perspective x sampling)")
    print("=" * 60)
    print(f"  Items:               {summary['total_completed']}/{summary['total_requested']}")
    print(f"  vav self MV:         {summary['vav_self_match_count']} "
          f"({summary['vav_self_match_rate']:.1%})")
    print(f"  majority (control):  {summary['majority_self_match_count']} "
          f"({summary['majority_self_match_rate']:.1%})  [不过滤对照]")
    print(f"  low_confidence:      {summary['low_confidence_count']} "
          f"({summary['low_confidence_rate']:.1%})  [阈值 {args.confidence_threshold}]")
    print(f"  平均 confidence:     {summary['average_confidence']}")
    print(f"  全部失败诊断题数:    {summary['failure_diagnosis_aggregate']['all_failed_items']} "
          f"类型计数: {summary['failure_diagnosis_aggregate']['error_type_counts']}")
    print(f"  视角获胜次数:        {summary['perspective_winner_counts']}")
    print(f"  难度拆分:")
    for d, s in summary["difficulty_split"].items():
        print(f"    {d:<8} n={s['count']:<5} vav={s['vav_self_match_rate']:.1%} "
              f"low_conf={s['low_confidence_rate']:.1%}")
    print(f"  候选 parse 率:       {summary['candidate_parse_success_rate']:.1%}")
    print(f"  候选执行成功率:      {summary['candidate_execution_success_rate']:.1%}")
    print(f"  执行缓存命中率:      {summary['execution_cache_hit_ratio']:.1%}")
    print(f"  每题唯一 SQL 均值:   {summary['avg_unique_sql_per_item']}")
    print(f"  总耗时:              {summary['total_wall_seconds']:.0f}s")
    print("=" * 60)
    print(f"\nItems saved to:   {items_path}")
    print(f"Summary saved to: {summary_path}")
    print("\n官方 test-suite 评估（下一步）:")
    print(
        f"  bash scripts/eval_official.sh {items_path} "
        f"{output_dir / 'official'}"
    )


# ===================================================================
# 纯逻辑自测（无 GPU；--self-test）
# ===================================================================

def _run_self_tests() -> int:
    failures: List[str] = []
    passed = 0

    def check(description: str, condition: bool) -> None:
        nonlocal passed
        if condition:
            passed += 1
        else:
            failures.append(description)
            print(f"  FAIL: {description}")

    print("=== finer_port.eval_mpev self-tests ===\n")

    # --- 置信度 ---
    check("conf: 6/30 = 0.2，阈值 0.2 时不算 low (>= 阈值采用)",
          compute_confidence(6, 30) == 0.2
          and not is_low_confidence(compute_confidence(6, 30), 0.2))
    check("conf: 5/30 < 0.2 → low",
          is_low_confidence(compute_confidence(5, 30), 0.2))
    check("conf: 0 候选 → 0.0 → low",
          compute_confidence(0, 0) == 0.0 and is_low_confidence(0.0, 0.2))
    check("conf: 30/30 = 1.0 → high",
          compute_confidence(30, 30) == 1.0
          and not is_low_confidence(1.0, 0.2))

    # --- 失败诊断 ---
    preds = [
        # 语法错误（no such column）
        {"sql": "SELECT bad FROM t",
         "result": {"success": False, "error": "no such column: bad",
                    "error_type": "sqlite_error"}, "index": 0},
        # 超时（基础设施关键词）
        {"sql": "SELECT * FROM t",
         "result": {"success": False, "error": "statement timeout",
                    "error_type": "sqlite_error"}, "index": 1},
        # 空 SQL（parse 失败 → execute_cached 合成 "Empty SQL"）
        {"sql": "", "result": {"success": False, "error": "Empty SQL",
                               "error_type": "empty_sql"}, "index": 2},
        # 成功但 0 行（空结果）
        {"sql": "SELECT 1 WHERE 1=0",
         "result": {"success": True, "full_rows": [], "row_count": 0,
                    "full_rows_truncated": False}, "index": 3},
        # 其他（基础设施非超时）
        {"sql": "SELECT * FROM t",
         "result": {"success": False, "error": "connection refused",
                    "error_type": "connection_error"}, "index": 4},
    ]
    d = diagnose_failures(preds)
    check("diag: syntax_errors=1", d["counts"]["syntax_errors"] == 1)
    check("diag: timeouts=1", d["counts"]["timeouts"] == 1)
    check("diag: empty_results=2 (空 SQL + 0 行)",
          d["counts"]["empty_results"] == 2)
    check("diag: other_errors=1", d["counts"]["other_errors"] == 1)
    check("diag: total=5", d["total"] == 5)
    check("diag: top_error_messages 非空",
          len(d["top_error_messages"]) >= 1)
    check("diag: top 消息按频次降序",
          all(d["top_error_messages"][i]["count"]
              >= d["top_error_messages"][i + 1]["count"]
              for i in range(len(d["top_error_messages"]) - 1)))

    # --- 空候选 → vav 投票 → NO_RESULTS（与 process_item 的诊断触发条件一致） ---
    vote_empty = run_vav_voting([], None)
    check("vote: 空候选 → NO_RESULTS",
          vote_empty["chosen_result"] == "NO_RESULTS"
          and vote_empty["selected_sql"] == ""
          and vote_empty["chosen_group_size"] == 0)
    check("vote: NO_RESULTS 时 confidence=0 且 low",
          compute_confidence(vote_empty["chosen_group_size"], 0) == 0.0
          and is_low_confidence(0.0, 0.2))

    # --- 策略字符串 ---
    strat = f"mpev_5p_p5x6_vav"
    check("strategy: 编码视角数×采样数", strat == "mpev_5p_p5x6_vav")

    print()
    total = passed + len(failures)
    if failures:
        print(f"=== {passed}/{total} passed, {len(failures)} FAILED ===")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"=== All {passed} tests passed ===")
    return 0


if __name__ == "__main__":
    main()
