#!/usr/bin/env python3
"""
文献弹药 8.3: 投票失败诊断曲线（maj@K curve diagnostic）评估脚本。

背景: SFT 检查点（checkpoints/sft_phase1, LoRA）在 Spider dev 前 100 条上做
**同 prompt 采样投票诊断**——用 maj@K 曲线 + Pass@K/Average@K + 执行分组数 +
Self-BLEU 近似，回答「采样投票为什么失败/在哪里失败」（候选多样性不足？
组碎片化？多数达不到绝对多数？）。

设计（与 finer_port/eval_vav.py P1 同口径，复用不重写）:
  1. 每题用 canonical prompt（VavSampler prompt_style="default" =
     ReasoningGeneratorAgent.build_prompt，与训练/既有 vav 评估一致）
     一次性采样 N = max(K_list) 条候选（单次 forward, num_return_sequences=N,
     do_sample T=1.0 top_p=1.0, seed 固定）——**一次采样存全部, 再按前 K 条
     前缀逐档计算**, 避免对每个 K 重复推理；
  2. 执行分组投票判定: 复用 finer_port/vav_voting.py 的 run_vav_voting
     （执行结果相同归组, vav 口径: 仅 SUCCESS_VALUES: 组、跳过空/全零退化组、
     取最大组, fallback 语义与 FINER majority_voting 一致）。注意空结果组
     SUCCESS_VALUES 签名（空签名组硬跳过但 fallback 可选中）——直接复用
     vav_voting 的 normalize_execution_result / choose_group_vav, 不重写；
  3. 候选正确性（Pass@K / Average@K / maj 判定统一口径）=
     执行成功 + 与 gold 行集合相等（vav_voting.results_equal,
     header-agnostic, 与 FINER 自评口径一致）；
  4. Self-BLEU 近似: 对候选 SQL 正则分词后做 Zhu et al. 2018 口径的
     clipped n-gram precision（n=1..4, +1 Laplace 平滑, 无 nltk 依赖）,
     反映候选多样性（低 Self-BLEU = 组碎片化/投票无共识）；
  5. 输出 items.json（逐题: 各 K 的 maj 判定/组数/Pass/Average/绝对多数/self-bleu;
     predicted_sql = 最大 K 档的投票结果, 与 scripts/eval_official.sh 兼容）
     + summary.json（各 K 的 maj@K 准确率、Pass@K/Average@K、平均执行分组数、
     绝对多数率/绝对多数命中率、平均 Self-BLEU）；
  6. 断点续跑复用 src/spider_utils.py 的 checkpoint 协议
     （build_run_config / load_checkpoint / validate_resume_config /
     validate_checkpoint_integrity / save_checkpoint），混配置直接拒绝。

口径说明（写入 summary.note）:
  - maj@K 准确率 = 前 K 条候选执行分组投票选中组与 gold 行集合相等的题占比
    （分母 = 全部完成题数, 采样不足 K 条按实际条数判, 保守计入分母）；
  - 绝对多数 = chosen_group_size > K/2（严格过半）；绝对多数命中率 =
    绝对多数存在且命中 / 绝对多数存在题数（条件命中率）。

用法（GPU, 与 eval_vav.py 同风格）:
    python src/eval_maj_diag.py \
        --lora-path checkpoints/sft_phase1 \
        --model-path models/Qwen2.5-Coder-3B-Instruct \
        --spider-dir data/spider_data \
        --limit 100 --k-list 4,8,16,32 --temperature 1.0 \
        --output-dir outputs/eval_maj_diag_100
    # 完成后官方 test-suite 复评（对 K=最大档的 items）:
    bash scripts/eval_official.sh outputs/eval_maj_diag_100/items.json \
        outputs/official_maj_diag_100
"""

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

# sys.path 引导: 项目 src/（spider_utils）+ finer_port/（sampler / vav_voting）
_SRC_DIR = Path(__file__).resolve().parent
_FINER_DIR = Path(__file__).resolve().parent.parent / "finer_port"
for _p in (str(_SRC_DIR), str(_FINER_DIR)):
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
    results_equal,
)
from sampler import VavSampler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"
DEFAULT_LORA_PATH = PROJECT_ROOT / "checkpoints" / "sft_phase1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_maj_diag_100"

EVALUATOR_TYPE = "maj_diag_execution_voting"


# ===================================================================
# 参数
# ===================================================================

def parse_k_list(s: str) -> List[int]:
    """解析 --k-list "4,8,16,32" → 去重、升序的 int 列表（每档都 >= 1）。"""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("empty --k-list")
    ks: List[int] = []
    for p in parts:
        try:
            k = int(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid K value: {p!r}")
        if k < 1:
            raise argparse.ArgumentTypeError(f"K must be >= 1, got {k}")
        ks.append(k)
    return sorted(set(ks))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "文献弹药 8.3 投票失败诊断: 同 prompt 采样 max(K) 条候选, "
            "按前 K 条前缀逐档计算 maj@K 曲线 + Pass@K/Average@K + "
            "执行分组数 + Self-BLEU 近似（vav 口径, 与 eval_vav P1 一致）"
        ),
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH),
                        help="本地基座模型路径")
    parser.add_argument("--lora-path", default=str(DEFAULT_LORA_PATH),
                        help="LoRA adapter 路径（默认 checkpoints/sft_phase1）")
    parser.add_argument("--spider-dir", default=str(DEFAULT_SPIDER_DIR),
                        help="Spider 数据集根目录")
    parser.add_argument("--limit", type=int, default=100,
                        help="评估条数（默认 dev 前 100 条）")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--k-list", type=parse_k_list, default="4,8,16,32",
                        help="maj@K 各档 K 值, 逗号分隔（默认 4,8,16,32; "
                             "实际采样数 = max(K_list)）")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度（默认 1.0, 与 eval_vav 一致）")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="与训练 max_completion 对齐（PLAN §8 生成长度一致性）")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="每次 forward 处理的题目数（一次 forward 产 B*max(K) 条）")
    parser.add_argument("--seed", type=int, default=0,
                        help="采样种子（前缀曲线依赖 num_return_sequences 的固定顺序）")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="每处理 N 题写一次 checkpoint（断点续跑）")
    parser.add_argument("--save-full-responses", action="store_true",
                        help="candidates 里存完整 raw_response（默认只存 200 字符预览）")
    parser.add_argument("--allow-remote", action="store_true",
                        help="允许从 HF 在线拉取权重（默认 local_files_only）")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="输出目录（checkpoint.json / items.json / summary.json）")
    return parser.parse_args(argv)


# ===================================================================
# Run config（resume 时逐字段比对防混实验）
# ===================================================================

def build_majdiag_run_config(args: argparse.Namespace) -> Dict[str, Any]:
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
            "k_list": args.k_list,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "prompt_style": "default",  # 同 prompt 诊断: canonical prompt 固定
            "lora_path": args.lora_path,
            "seed": args.seed,
        }
    )
    return cfg


# ===================================================================
# Self-BLEU 近似（SQL token 级, 无 nltk 依赖）
# ===================================================================

# SQL 正则分词: 标识符 / 数字 / 多字符运算符 / 单个标点
_SQL_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|<=|>=|<>|!=|->|[^\sA-Za-z0-9_]"
)


def _tokenize_sql(sql: str) -> List[str]:
    """SQL 正则分词（多字符运算符优先于单字符标点）。"""
    return _SQL_TOKEN_RE.findall(sql or "")


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def self_bleu_approx(sqls: List[str], max_n: int = 4) -> Optional[float]:
    """
    Self-BLEU 近似（Zhu et al. 2018 口径, 本地无 nltk 实现）。

    对每条候选 SQL 分词后, 计算其相对其余候选的 clipped n-gram precision
    （n=1..max_n, +1 Laplace 平滑避免 log(0)）, 几何平均后对候选取平均。
    高 Self-BLEU = 候选高度重复（投票无意义）; 低 Self-BLEU = 组碎片化。
    有效（非空 token）候选 < 2 条时返回 None。
    """
    tok_sents = [_tokenize_sql(s) for s in sqls]
    tok_sents = [s for s in tok_sents if s]
    if len(tok_sents) < 2:
        return None

    scores: List[float] = []
    for i, cand in enumerate(tok_sents):
        others = [s for j, s in enumerate(tok_sents) if j != i]
        ref_counts: Dict[int, Counter] = {}
        for n in range(1, max_n + 1):
            ref_counts[n] = Counter()
            for s in others:
                ref_counts[n].update(_ngrams(s, n))

        precisions: List[float] = []
        for n in range(1, max_n + 1):
            cand_ngrams = _ngrams(cand, n)
            if not cand_ngrams:
                continue  # 候选短于 n → 跳过该档
            cand_counter = Counter(cand_ngrams)
            matched = sum(
                min(cnt, ref_counts[n][g]) for g, cnt in cand_counter.items()
            )
            # +1 Laplace 平滑: 全零匹配时精度 > 0, 几何平均可计算
            precisions.append((matched + 1.0) / (len(cand_ngrams) + 1.0))

        if precisions:
            scores.append(
                math.exp(sum(math.log(p) for p in precisions) / len(precisions))
            )

    return round(sum(scores) / len(scores), 4) if scores else None


# ===================================================================
# 单样本处理
# ===================================================================

def _candidate_correct(
    outcome: Dict[str, Any],
    gold_outcome: Dict[str, Any],
) -> bool:
    """候选正确性 = 执行成功 + 与 gold 行集合相等（vav 口径, header-agnostic）。"""
    if not outcome.get("success") or not gold_outcome.get("success"):
        return False
    return results_equal(
        gold_outcome.get("full_rows") or [],
        outcome.get("full_rows") or [],
    )


def _empty_k_entry() -> Dict[str, Any]:
    """K 前缀诊断条目的零值模板（error 条目也保持结构完整, 供 summary 聚合）。"""
    return {
        "prefix_len": 0,
        "maj_correct": False,
        "chosen_result": "NO_RESULTS",
        "chosen_group_size": 0,
        "majority_result": "NO_RESULTS",
        "majority_group_size": 0,
        "degenerate_skip_applied": False,
        "num_exec_groups": 0,
        "num_syntax_errors": 0,
        "num_valid_sqls_after_filtering": 0,
        "pass_at_k": False,
        "average_at_k": 0.0,
        "has_abs_majority": False,
        "abs_majority_correct": False,
        "self_bleu_sql": None,
    }


def _new_item(item: Dict[str, Any], k_list: List[int]) -> Dict[str, Any]:
    return {
        "dataset_index": item["dataset_index"],
        "db_id": item["db_id"],
        "question": item["question"],
        "gold_sql": item["query"],
        "difficulty": item.get("difficulty", "unknown"),
        "predicted_sql": None,          # ← 最大 K 档投票结果, eval_official.sh 兼容
        "selected_sql_index": -1,
        "vav_self_match": False,        # 最大 K 档 vav 自评（header-agnostic）
        "majority_self_match": False,   # 最大 K 档 majority 不过滤对照
        "selected_custom_exec_match": None,  # 最大 K 档训练同口径（ORDER BY 感知）
        "selected_custom_exec_match_reason": None,
        "gold_execution_success": False,
        "gold_error": None,
        "num_sampled": 0,
        "parse_success_count": 0,
        "num_unique_sql": 0,
        "candidates": [],
        "maj_at_k": {str(k): _empty_k_entry() for k in k_list},
        "generation_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "error": None,
    }


def process_item(
    item: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    evaluator: VavEvaluator,
    k_list: List[int],
    save_full_responses: bool,
) -> Dict[str, Any]:
    """
    执行验证（全部候选一次执行, 带缓存）+ 各 K 前缀投票诊断, 产出单个评估条目。

    candidates: VavSampler 输出的候选 dict 列表（raw_response/sql/parse_success/
    parse_method/candidate_id），顺序 = num_return_sequences 固定顺序（seed 固定）。
    前缀曲线不重新执行 SQL：执行结果在 max(K) 上算一次, 各 K 只重跑
    run_vav_voting 纯函数（分组/投票/正确性判定, 零额外执行开销）。
    """
    r = _new_item(item, k_list)
    t0 = time.perf_counter()
    db_id = item["db_id"]
    gold_sql = item["query"]

    # 1) 逐候选执行（带 (db_id, normalize_sql) 缓存; 空 SQL 不触库）+ gold
    pred_results: List[Dict[str, Any]] = []
    for c in candidates:
        outcome = evaluator.execute_cached(db_id, c["sql"])
        pred_results.append(
            {"sql": c["sql"], "result": outcome, "index": c["candidate_id"]}
        )
    gold_outcome = evaluator.execute_cached(db_id, gold_sql)

    r["gold_execution_success"] = bool(gold_outcome.get("success"))
    r["gold_error"] = (
        gold_outcome.get("error") if not gold_outcome.get("success") else None
    )
    r["num_sampled"] = len(pred_results)
    r["parse_success_count"] = sum(1 for c in candidates if c.get("parse_success"))
    r["num_unique_sql"] = len(
        {
            normalize_sql(c["sql"])
            for c in candidates
            if (c.get("sql") or "").strip()
        }
    )

    # 候选正确性 flags（Pass@K / Average@K 用前缀复用）
    correct_flags = [
        _candidate_correct(pr["result"], gold_outcome) for pr in pred_results
    ]

    # 2) candidates 数组（留作分析; 默认只存 raw_response 前 200 字符预览）
    for idx, (c, pr) in enumerate(zip(candidates, pred_results)):
        entry = {
            "candidate_id": c["candidate_id"],
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
            "correct_vav": correct_flags[idx] if idx < len(correct_flags) else False,
        }
        if save_full_responses:
            entry["raw_response"] = c.get("raw_response", "")
        else:
            entry["raw_response_preview"] = (c.get("raw_response") or "")[:200]
        r["candidates"].append(entry)

    # 3) 各 K 前缀投票诊断（一次采样, 前缀复用; k_list 升序）
    last_vote: Optional[Dict[str, Any]] = None
    n_total = len(pred_results)
    for k in k_list:
        entry = _empty_k_entry()
        k_eff = min(k, n_total)
        entry["prefix_len"] = k_eff
        if k_eff > 0:
            vote = run_vav_voting(
                pred_results[:k_eff], gold_outcome, gt_sql=gold_sql, strategy="vav"
            )
            entry["maj_correct"] = bool(vote["is_sample_correct"])
            entry["chosen_result"] = vote["chosen_result"]
            entry["chosen_group_size"] = vote["chosen_group_size"]
            entry["majority_result"] = vote["majority_result"]
            entry["majority_group_size"] = vote["majority_group_size"]
            entry["degenerate_skip_applied"] = vote["degenerate_skip_applied"]
            entry["num_exec_groups"] = len(vote["result_groups"])
            entry["num_syntax_errors"] = vote["num_syntax_errors"]
            entry["num_valid_sqls_after_filtering"] = (
                vote["num_valid_sqls_after_filtering"]
            )
            prefix_flags = correct_flags[:k_eff]
            entry["pass_at_k"] = bool(any(prefix_flags))
            entry["average_at_k"] = round(sum(prefix_flags) / k_eff, 4)
            # 绝对多数 = 选中组票数严格过半（> K/2, 相对 K 条前缀）
            entry["has_abs_majority"] = entry["chosen_group_size"] > k_eff / 2
            entry["abs_majority_correct"] = (
                entry["has_abs_majority"] and entry["maj_correct"]
            )
            entry["self_bleu_sql"] = self_bleu_approx(
                [pr["sql"] for pr in pred_results[:k_eff]]
            )
            last_vote = vote
        r["maj_at_k"][str(k)] = entry

    # 4) 最大 K 档 → predicted_sql（官方口径 + 训练同口径对照; 与 eval_vav 一致）
    if last_vote is not None and last_vote.get("selected_sql"):
        r["predicted_sql"] = last_vote["selected_sql"]
        r["selected_sql_index"] = last_vote["selected_sql_index"]
        r["vav_self_match"] = bool(last_vote["is_sample_correct"])
        r["majority_self_match"] = bool(last_vote["majority_is_correct"])
        sel_res = last_vote.get("selected_result") or {}
        if (
            gold_outcome.get("success")
            and sel_res.get("success")
            and not (
                gold_outcome.get("full_rows_truncated")
                or sel_res.get("full_rows_truncated")
            )
        ):
            cmp = compare_execution_results(
                sel_res.get("full_rows") or [],
                gold_outcome.get("full_rows") or [],
                gold_sql=gold_sql,
            )
            r["selected_custom_exec_match"] = cmp["match"]
            r["selected_custom_exec_match_reason"] = cmp.get("match_reason")
        elif gold_outcome.get("success") and sel_res.get("success"):
            r["selected_custom_exec_match"] = False
            r["selected_custom_exec_match_reason"] = "full rows truncated"

    r["evaluation_seconds"] = round(time.perf_counter() - t0, 4)
    r["generation_seconds"] = round(time.perf_counter() - t0, 4)
    return r


# ===================================================================
# Summary
# ===================================================================

def _k_entry(it: Dict[str, Any], key: str) -> Dict[str, Any]:
    """安全取 K 前缀条目（缺字段时回退零值模板, 兼容异常/旧 checkpoint 条目）。"""
    maj = it.get("maj_at_k")
    if isinstance(maj, dict):
        ent = maj.get(key)
        if isinstance(ent, dict):
            return ent
    return _empty_k_entry()


def _per_k_summary(
    all_items: List[Dict[str, Any]],
    k: int,
    total: int,
) -> Dict[str, Any]:
    """单档 K 的汇总（maj@K / Pass@K / Average@K / 分组数 / 绝对多数 / Self-BLEU）。"""
    key = str(k)
    ent: Dict[str, Any] = {"K": k, "n_items": total}

    ent["n_items_with_full_prefix"] = sum(
        1 for it in all_items if _k_entry(it, key)["prefix_len"] >= k
    )

    n_maj = sum(1 for it in all_items if _k_entry(it, key)["maj_correct"])
    ent["maj_at_k_correct_count"] = n_maj
    ent["maj_at_k_accuracy"] = round(n_maj / total, 4) if total else 0.0

    n_pass = sum(1 for it in all_items if _k_entry(it, key)["pass_at_k"])
    ent["pass_at_k_count"] = n_pass
    ent["pass_at_k_rate"] = round(n_pass / total, 4) if total else 0.0

    avg_list = [
        _k_entry(it, key)["average_at_k"]
        for it in all_items if _k_entry(it, key)["prefix_len"] > 0
    ]
    ent["average_at_k"] = round(sum(avg_list) / len(avg_list), 4) if avg_list else 0.0

    groups_list = [
        _k_entry(it, key)["num_exec_groups"]
        for it in all_items if _k_entry(it, key)["prefix_len"] > 0
    ]
    ent["avg_exec_groups"] = (
        round(sum(groups_list) / len(groups_list), 4) if groups_list else 0.0
    )

    chosen_list = [
        _k_entry(it, key)["chosen_group_size"]
        for it in all_items if _k_entry(it, key)["prefix_len"] > 0
    ]
    ent["avg_chosen_group_size"] = (
        round(sum(chosen_list) / len(chosen_list), 4) if chosen_list else 0.0
    )

    n_abs = sum(1 for it in all_items if _k_entry(it, key)["has_abs_majority"])
    n_abs_correct = sum(
        1 for it in all_items if _k_entry(it, key)["abs_majority_correct"]
    )
    ent["abs_majority_count"] = n_abs
    ent["abs_majority_rate"] = round(n_abs / total, 4) if total else 0.0
    # 绝对多数命中率 = 绝对多数存在且命中 / 绝对多数存在（条件命中率）
    ent["abs_majority_hit_rate"] = (
        round(n_abs_correct / n_abs, 4) if n_abs else 0.0
    )

    sb_list = [
        _k_entry(it, key).get("self_bleu_sql")
        for it in all_items
        if _k_entry(it, key).get("self_bleu_sql") is not None
    ]
    ent["avg_self_bleu_sql"] = (
        round(sum(sb_list) / len(sb_list), 4) if sb_list else None
    )
    return ent


def build_summary(
    all_items: List[Dict[str, Any]],
    requested_indices: Set[int],
    run_config: Dict[str, Any],
    k_list: List[int],
    evaluator: VavEvaluator,
    total_wall_seconds: float,
    generated_at: str,
) -> Dict[str, Any]:
    total = len(all_items)
    per_k = [_per_k_summary(all_items, k, total) for k in k_list]

    cand_parse = sum(it.get("parse_success_count", 0) for it in all_items)
    cand_total = sum(it.get("num_sampled", 0) for it in all_items)
    cand_exec_ok = sum(
        sum(1 for c in it.get("candidates", []) if c.get("execution_success"))
        for it in all_items
    )
    avg_unique = (
        sum(it.get("num_unique_sql", 0) for it in all_items) / total
        if total else 0.0
    )
    gen_times = [it.get("generation_seconds", 0.0) for it in all_items]
    n_error = sum(1 for it in all_items if it.get("error"))

    max_k = k_list[-1] if k_list else 0
    n_custom = sum(
        1 for it in all_items if it.get("selected_custom_exec_match") is True
    )
    n_custom_scored = sum(
        1 for it in all_items if it.get("selected_custom_exec_match") is not None
    )

    return {
        "evaluator_type": EVALUATOR_TYPE,
        "is_official_spider_metric": False,
        "note": (
            "文献弹药 8.3 投票失败诊断: 同 prompt 一次采样 max(K) 条, 按前 K 条前缀"
            "逐档计算。maj@K 判定 = vav 执行分组投票（SUCCESS_VALUES 组, 跳过空/全零"
            "退化组, FINER majority_voting 同口径, 直接复用 finer_port/vav_voting）;"
            "候选正确性（maj/Pass@K/Average@K 统一）= header-agnostic 行集合相等"
            "（results_equal）; has_abs_majority = chosen_group_size > K/2（严格过半）;"
            "abs_majority_hit_rate = 绝对多数存在且命中 / 绝对多数存在题数（条件命中率）;"
            "avg_self_bleu_sql = SQL token 级 Self-BLEU 近似（clipped n-gram precision, "
            "n=1..4, +1 Laplace 平滑, 无 nltk）; 官方 test-suite EX 需 "
            "bash scripts/eval_official.sh <items.json> <out_dir>（用最大 K 档 "
            "predicted_sql）。"
        ),
        "total_requested": len(requested_indices),
        "total_completed": total,
        "error_items": n_error,
        "max_k": max_k,
        "per_k": per_k,
        "vav_self_match_count": sum(
            1 for it in all_items if it.get("vav_self_match")
        ),
        "vav_self_match_rate": (
            round(sum(1 for it in all_items if it.get("vav_self_match")) / total, 4)
            if total else 0.0
        ),
        "majority_self_match_count": sum(
            1 for it in all_items if it.get("majority_self_match")
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

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    max_k = args.k_list[-1]  # k_list 升序 → 采样数 = 最大档

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 加载数据集 ---
    loader = SpiderLoader(args.spider_dir)
    items = loader.load_dev(limit=args.limit, start_index=args.start_index)
    requested_indices: Set[int] = {it["dataset_index"] for it in items}
    print(f"Loaded {len(items)} items (start={args.start_index}, limit={args.limit})")
    print(f"K list: {args.k_list} (sample once with max K = {max_k})")

    # --- checkpoint / resume（协议与 eval_vav / eval_mpev 一致） ---
    run_config = build_majdiag_run_config(args)
    cp = load_checkpoint(output_dir)
    if cp["run_config"] is not None:
        validate_resume_config(cp["run_config"], run_config)
    validate_checkpoint_integrity(cp, requested_indices)
    completed: Set[int] = cp["completed_indices"]
    all_items: List[Dict[str, Any]] = list(cp["items"])
    print(f"Resume: {len(completed)}/{len(requested_indices)} items already completed")

    # --- 加载模型 + 执行器（复用 VavSampler / VavEvaluator, 不重写） ---
    print(f"\nLoading model: {args.model_path}")
    print(f"LoRA adapter: {args.lora_path}")
    sampler = VavSampler(
        model_path=args.model_path,
        lora_path=args.lora_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_style="default",  # 同 prompt: canonical prompt（与训练/既有评估一致）
        seed=args.seed,
        local_files_only=not args.allow_remote,
    )
    evaluator = VavEvaluator(DatabaseExecutor(args.spider_dir))
    print(f"Sampler ready. n-samples={max_k}, T={args.temperature}, seed={args.seed}\n")

    # --- 预构造 prompt（DDL 失败条目记为 error 条目, 不阻塞） ---
    chat_texts: Dict[int, Optional[str]] = {}
    for it in items:
        try:
            ddl = loader.format_ddl(it["db_id"])
            chat_texts[it["dataset_index"]] = sampler.build_chat_text(
                it["question"], ddl
            )
        except RuntimeError as exc:
            chat_texts[it["dataset_index"]] = None
            print(f"[WARN] item {it['dataset_index']} db_id={it['db_id']} "
                  f"DDL failed: {exc}")

    pending = [it for it in items if it["dataset_index"] not in completed]
    wall_start = time.perf_counter()

    for i in range(0, len(pending), args.batch_size):
        batch = pending[i:i + args.batch_size]
        batch_prompts: List[Tuple[int, str]] = [
            (it["dataset_index"], chat_texts[it["dataset_index"]])
            for it in batch
            if chat_texts.get(it["dataset_index"]) is not None
        ]
        gen_map: Dict[int, List[Dict[str, Any]]] = {}
        if batch_prompts:
            try:
                gen_start = time.perf_counter()
                gen_results = sampler.sample_batch(
                    [p for _, p in batch_prompts], n=max_k
                )
                print(
                    f"  batch gen {len(batch_prompts)} items x {max_k} "
                    f"in {time.perf_counter() - gen_start:.1f}s"
                )
                for (ds, _), cands in zip(batch_prompts, gen_results):
                    gen_map[ds] = cands
            except Exception as exc:
                print(f"[WARN] batch generation failed ({exc}); "
                      f"recording error items for this batch")
                for ds, _ in batch_prompts:
                    gen_map[ds] = []

        # --- 每题: 执行 + 各 K 前缀投票诊断 ---
        for it in batch:
            ds = it["dataset_index"]
            if chat_texts.get(ds) is None:
                r = _new_item(it, args.k_list)
                r["error"] = "ddl_load_failed"
            else:
                try:
                    r = process_item(
                        it, gen_map.get(ds, []), evaluator,
                        args.k_list, args.save_full_responses,
                    )
                except Exception as exc:
                    r = _new_item(it, args.k_list)
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
            if done:
                last_key = str(args.k_list[-1])
                n_maj = sum(
                    1 for it in all_items
                    if it["maj_at_k"][last_key]["maj_correct"]
                )
                print(
                    f"  [checkpoint {done}/{len(items)}] "
                    f"maj@{args.k_list[-1]}={n_maj}/{done} ({n_maj / done:.1%}) "
                    f"wall={time.perf_counter() - wall_start:.0f}s"
                )

    wall_end = time.perf_counter()
    total_wall = wall_end - wall_start

    # --- 收尾: 最终 checkpoint + items.json + summary.json ---
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
        args.k_list,
        evaluator,
        total_wall,
        datetime.now(timezone.utc).isoformat(),
    )
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # --- 打印结果 ---
    print("\n" + "=" * 66)
    print("  VOTING FAILURE DIAGNOSIS SUMMARY (maj@K curve)")
    print("=" * 66)
    print(f"  Items: {summary['total_completed']}/{summary['total_requested']} "
          f"(error={summary['error_items']})")
    print(f"  {'K':>4} | {'maj@K':>7} | {'pass@K':>7} | {'avg@K':>6} | "
          f"{'groups':>6} | {'absMaj':>6} | {'absHit':>6} | {'selfBL':>6}")
    for ent in summary["per_k"]:
        sb = f"{ent['avg_self_bleu_sql']:.3f}" if ent["avg_self_bleu_sql"] is not None else "-"
        print(
            f"  {ent['K']:>4} | {ent['maj_at_k_accuracy']:>7.1%} | "
            f"{ent['pass_at_k_rate']:>7.1%} | {ent['average_at_k']:>6.3f} | "
            f"{ent['avg_exec_groups']:>6.2f} | {ent['abs_majority_rate']:>6.1%} | "
            f"{ent['abs_majority_hit_rate']:>6.1%} | {sb:>6}"
        )
    print(f"  最大 K 档 vav 自评:     {summary['vav_self_match_rate']:.1%}")
    print(f"  最大 K 档训练同口径:   {summary['selected_custom_exec_match_rate']:.1%}")
    print(f"  候选 parse 率:         {summary['candidate_parse_success_rate']:.1%}")
    print(f"  候选执行成功率:        {summary['candidate_execution_success_rate']:.1%}")
    print(f"  执行缓存命中率:        {summary['execution_cache_hit_ratio']:.1%}")
    print(f"  每题唯一 SQL 均值:     {summary['avg_unique_sql_per_item']}")
    print(f"  总耗时:                {summary['total_wall_seconds']:.0f}s")
    print("=" * 66)
    print(f"\nItems saved to:   {items_path}")
    print(f"Summary saved to: {summary_path}")
    print("\n官方 test-suite 评估（下一步, K=最大档 items）:")
    print(
        f"  bash scripts/eval_official.sh {items_path} "
        f"{output_dir / 'official'}"
    )


if __name__ == "__main__":
    main()
