#!/usr/bin/env python3
"""投票诊断曲线温度网格实验（maj@K × temperature grid）评估脚本。

背景: 在 eval_maj_diag（文献弹药 8.3 投票失败诊断曲线, 单温度 T=1.0,
K∈{4,8,16,32}, sft_phase1）的基础上, 对 **sft_v2 检查点 (LoRA)** 做温度网格:
    maj@{8,16,32,64} × 温度{0.7,1.0}
回答「采样温度对投票失败诊断曲线的影响」——低温(0.7)候选更集中(Self-BLEU 高 /
执行分组少)、高温(1.0)候选更多样, 各自 maj@K 曲线如何变化, 绝对多数命中率如何。

设计（前缀复用采样直接照抄 eval_maj_diag 的实现方式）:
  1. 每温度档用 canonical prompt（VavSampler prompt_style="default"）**一次采样
     N = max(K_list) = 64 条候选**（单次 forward, num_return_sequences=N,
     do_sample T=temp top_p=1.0），按前 K∈{8,16,32,64} 条前缀逐档计算，避免对
     每个 K 重复推理——两个温度档共 2 次前向采样/题；
  2. 两个温度档复用同一个 VavSampler（同一份模型权重），逐档只改
     sampler.temperature 属性并 torch.manual_seed(seed) 重置 RNG——每档从同一
     RNG 状态出发, 保证「同题同前缀序列」下温度是唯一变量（受控温度对比）;
  3. 执行分组投票判定: 复用 finer_port/vav_voting.py 的 run_vav_voting
     （执行结果相同归组, vav 口径: 仅 SUCCESS_VALUES 组、跳过空/全零退化组、
     取最大组, fallback 语义与 FINER majority_voting 一致）——与 eval_maj_diag
     完全同引擎, maj@K 曲线可直接对比; 执行结果在 max(K) 上算一次, 各 K 只重跑
     run_vav_voting 纯函数（零额外执行开销）;
  4. 正确性双口径:
     - vav 口径（与 eval_maj_diag 同）: header-agnostic 行集合相等
       （vav_voting.results_equal, FINER 自评口径）;
     - 项目自定义执行匹配（本实验 headline, 任务指定）: src/spider_utils.py
       的 compare_execution_results（"match" 字段, ORDER BY 感知 multiset）。
     官方 test-suite 复评由主控另跑（每温度档 max-K 的 predicted_sql 已存于
     items.json）;
  5. Self-BLEU 近似: 复用 eval_maj_diag.self_bleu_approx（SQL token 级 clipped
     n-gram precision, n=1..4, +1 Laplace 平滑, 无 nltk）——与 maj_diag 同口径
     （任务允许的「token 集合 Jaccard」为备选近似, 此处选 maj_diag 同款以保证
     曲线可比; 口径已写入 summary.note）;
  6. 输出 items.json（逐题: 每温度档 candidates + 各 K 的 maj 判定/组数/Pass/
     绝对多数/self-bleu; grid[<temp>].predicted_sql = 该档最大 K 的投票结果,
     官方复评按温度档提取）+ summary.json（(K, T) 网格汇总表 cells）;
  7. 断点续跑复用 src/spider_utils.py 的 checkpoint 协议
     （build_run_config / load_checkpoint / validate_resume_config /
     validate_checkpoint_integrity / save_checkpoint），混配置直接拒绝;
     断点粒度 = 整题（两温度档都完成才记 completed; 中断最多重跑 1 题 ×
     已开始的温度档, 与 eval_maj_diag 的整题粒度一致）。

口径说明（写入 summary.note）:
  - maj@K 准确率 = 前 K 条候选执行分组投票选中组与 gold 行集合相等的题占比
    （vav 口径, 分母 = 全部完成题数）; maj_custom 口径 = 选中 SQL 执行结果与
    gold 经 compare_execution_results 判定 match（分母 = 可判定题数:
    双方执行成功且均未截断）;
  - Pass@K = K 个候选里存在与 gold 执行等价的（vav 口径 results_equal;
    custom 口径 compare_execution_results）;
  - 绝对多数 = chosen_group_size > K/2（严格过半, 相对 K 条前缀）; 绝对多数
    命中率 = 绝对多数存在且命中 / 绝对多数存在题数（条件命中率）;
  - Self-BLEU 越低 = 候选越多样（组碎片化/投票无共识）; 越高 = 候选重复。

用法（GPU, 与 eval_maj_diag.py 同风格）:
    python src/eval_maj_temp_grid.py \
        --lora-path checkpoints/sft_v2 \
        --base-model models/Qwen2.5-Coder-3B-Instruct \
        --spider-dir data/spider_data \
        --n 100 --grid "8,16,32,64:0.7,1.0" \
        --output-dir outputs/eval_maj_temp_grid
    # 官方 test-suite 复评由主控另跑（对各温度档 max-K 的 predicted_sql,
    # 见 items.json 的 grid[<temp>].predicted_sql）:
    # bash scripts/eval_official.sh <per-temp items.json> <out_dir>
"""

import argparse
import json
import sys
import time
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
# Self-BLEU 近似口径与 eval_maj_diag 完全一致（复用不重写, 保证曲线可比）
from eval_maj_diag import self_bleu_approx  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"
DEFAULT_LORA_PATH = PROJECT_ROOT / "checkpoints" / "sft_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_maj_temp_grid"

EVALUATOR_TYPE = "maj_temp_grid_execution_voting"


# ===================================================================
# 参数
# ===================================================================

def parse_grid(s: str) -> Tuple[List[int], List[float]]:
    """解析 --grid "8,16,32,64:0.7,1.0" → (k_list 升序, temps 升序)。"""
    if s.count(":") != 1:
        raise argparse.ArgumentTypeError(
            f"invalid --grid {s!r}: expected '<k1,k2,...>:<t1,t2,...>'"
        )
    ks_raw, ts_raw = s.split(":")
    ks_raw = [p.strip() for p in ks_raw.split(",") if p.strip()]
    ts_raw = [p.strip() for p in ts_raw.split(",") if p.strip()]
    if not ks_raw or not ts_raw:
        raise argparse.ArgumentTypeError(
            f"invalid --grid {s!r}: both sides of ':' must be non-empty"
        )
    k_list: List[int] = []
    for p in ks_raw:
        try:
            k = int(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid K value in --grid: {p!r}")
        if k < 1:
            raise argparse.ArgumentTypeError(f"K must be >= 1, got {k}")
        k_list.append(k)
    temps: List[float] = []
    for p in ts_raw:
        try:
            t = float(p)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"invalid temperature value in --grid: {p!r}"
            )
        if not 0.0 < t <= 2.0:
            raise argparse.ArgumentTypeError(
                f"temperature must be in (0, 2], got {t}"
            )
        temps.append(t)
    return sorted(set(k_list)), sorted(set(temps))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "投票诊断曲线温度网格: sft_v2 检查点, maj@K×温度网格。"
            "每温度档同 prompt 一次采样 max(K) 条候选, 按前 K 条前缀逐档计算 "
            "maj@K(双口径) + Pass@K + 执行分组数 + 绝对多数命中率 + Self-BLEU 近似"
        ),
    )
    parser.add_argument("--lora-path", default=str(DEFAULT_LORA_PATH),
                        help="LoRA adapter 路径（默认 checkpoints/sft_v2）")
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL),
                        help="本地基座模型路径（默认 models/Qwen2.5-Coder-3B-Instruct）")
    parser.add_argument("--spider-dir", default=str(DEFAULT_SPIDER_DIR),
                        help="Spider 数据集根目录")
    parser.add_argument("--n", type=int, default=100,
                        help="评估条数（默认 Spider dev 前 100 条, 按 dataset_index(di) 字段切片）")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--grid", type=parse_grid, default="8,16,32,64:0.7,1.0",
                        help="maj@K × 温度网格 '<k1,k2,...>:<t1,t2,...>'"
                             "（默认 8,16,32,64:0.7,1.0; 实际采样数 = max(K)）")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="与训练 max_completion 对齐（PLAN §8 生成长度一致性）")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="每次 forward 处理的题目数（一次 forward 产 B*max(K) 条）")
    parser.add_argument("--seed", type=int, default=0,
                        help="采样种子（每个温度档从同一 RNG 状态出发, 前缀序列可比）")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="每处理 N 题写一次 checkpoint（断点续跑, 粒度=整题）")
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

def build_grid_run_config(
    args: argparse.Namespace,
    k_list: List[int],
    temps: List[float],
) -> Dict[str, Any]:
    cfg = build_run_config(
        spider_dir=args.spider_dir,
        start_index=args.start_index,
        limit=args.n,
        model_path=args.base_model,
        max_new_tokens=args.max_new_tokens,
        evaluator_type=EVALUATOR_TYPE,
    )
    cfg.update(
        {
            "k_list": k_list,
            "temperatures": temps,
            "top_p": args.top_p,
            "prompt_style": "default",  # 同 prompt 诊断: canonical prompt 固定
            "lora_path": args.lora_path,
            "seed": args.seed,
        }
    )
    return cfg


# ===================================================================
# 正确性判定（双口径）
# ===================================================================

def _temp_key(temp: float) -> str:
    """温度档 JSON 键（"0.7" / "1.0", round 去浮点噪声）。"""
    return str(round(float(temp), 4))


def _candidate_correct_vav(
    outcome: Dict[str, Any],
    gold_outcome: Dict[str, Any],
) -> bool:
    """vav 口径候选正确性 = 执行成功 + header-agnostic 行集合相等（maj_diag 同款）。"""
    if not outcome.get("success") or not gold_outcome.get("success"):
        return False
    return results_equal(
        gold_outcome.get("full_rows") or [],
        outcome.get("full_rows") or [],
    )


def _candidate_correct_custom(
    outcome: Dict[str, Any],
    gold_outcome: Dict[str, Any],
    gold_sql: str,
) -> bool:
    """项目自定义执行匹配口径（compare_execution_results, ORDER BY 感知 multiset）。

    截断/失败一律判 False（保守, 与 summary 口径一致）。
    """
    if not outcome.get("success") or not gold_outcome.get("success"):
        return False
    if outcome.get("full_rows_truncated") or gold_outcome.get("full_rows_truncated"):
        return False
    return compare_execution_results(
        gold_outcome.get("full_rows") or [],
        outcome.get("full_rows") or [],
        gold_sql=gold_sql,
    )["match"]


def _custom_match_of(
    pred_res: Dict[str, Any],
    gold_outcome: Dict[str, Any],
    gold_sql: str,
) -> Tuple[Optional[bool], Optional[str]]:
    """选中 SQL 执行结果 vs gold 的项目自定义口径判定 → (match|None, reason|None)。

    None = 不可判定（任一方执行失败）; 双方成功但截断 → False + "full rows truncated"。
    """
    if not pred_res.get("success") or not gold_outcome.get("success"):
        return None, "gold or selected execution failed"
    if pred_res.get("full_rows_truncated") or gold_outcome.get("full_rows_truncated"):
        return False, "full rows truncated"
    cmp = compare_execution_results(
        pred_res.get("full_rows") or [],
        gold_outcome.get("full_rows") or [],
        gold_sql=gold_sql,
    )
    return cmp["match"], cmp.get("match_reason")


# ===================================================================
# 条目模板
# ===================================================================

def _empty_cell() -> Dict[str, Any]:
    """(K, temp) 前缀诊断条目的零值模板（error 条目也保持结构完整, 供 summary 聚合）。"""
    return {
        "prefix_len": 0,
        "maj_correct": False,               # vav 口径（与 eval_maj_diag 同）
        "maj_custom_exec_match": None,      # 项目自定义口径（compare_execution_results）
        "maj_custom_exec_match_reason": None,
        "chosen_result": "NO_RESULTS",
        "chosen_group_size": 0,
        "majority_result": "NO_RESULTS",
        "majority_group_size": 0,
        "degenerate_skip_applied": False,
        "num_exec_groups": 0,
        "num_syntax_errors": 0,
        "num_valid_sqls_after_filtering": 0,
        "pass_at_k": False,                 # vav 口径
        "pass_at_k_custom": False,          # 项目自定义口径
        "average_at_k": 0.0,
        "average_at_k_custom": 0.0,
        "has_abs_majority": False,
        "abs_majority_correct": False,
        "abs_majority_correct_custom": False,
        "self_bleu_sql": None,
    }


def _empty_temp_entry(
    temp: Optional[float] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """单温度档条目零值模板（error 条目也保持结构完整）。"""
    entry: Dict[str, Any] = {
        "temperature": temp,
        "num_sampled": 0,
        "parse_success_count": 0,
        "num_unique_sql": 0,
        "candidates": [],
        "maj_at_k": {},
        "predicted_sql": None,          # ← 该温度档最大 K 的投票结果
        "selected_sql_index": -1,
        "vav_self_match": False,        # 最大 K 档 vav 自评（header-agnostic）
        "selected_custom_exec_match": None,  # 最大 K 档项目自定义口径
        "selected_custom_exec_match_reason": None,
        "generation_seconds": 0.0,
        "evaluation_seconds": 0.0,
    }
    if error is not None:
        entry["error"] = error
    return entry


# ===================================================================
# 单温度档处理
# ===================================================================

def process_temp_entry(
    item: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    evaluator: VavEvaluator,
    k_list: List[int],
    gold_outcome: Dict[str, Any],
    save_full_responses: bool,
) -> Dict[str, Any]:
    """
    执行验证（全部候选一次执行, 带缓存）+ 各 K 前缀投票诊断, 产出单温度档条目。

    candidates: VavSampler 输出的候选 dict 列表（raw_response/sql/parse_success/
    parse_method/candidate_id），顺序 = num_return_sequences 固定顺序。
    前缀曲线不重新执行 SQL：执行结果在 max(K) 上算一次, 各 K 只重跑
    run_vav_voting 纯函数（分组/投票/正确性判定, 零额外执行开销）。
    """
    entry = _empty_temp_entry()  # temperature 字段由 finalize_item 按档回填
    t0 = time.perf_counter()
    db_id = item["db_id"]
    gold_sql = item["query"]

    # 1) 逐候选执行（带 (db_id, normalize_sql) 缓存, 跨温度档共享; 空 SQL 不触库）
    pred_results: List[Dict[str, Any]] = []
    for c in candidates:
        outcome = evaluator.execute_cached(db_id, c["sql"])
        pred_results.append(
            {"sql": c["sql"], "result": outcome, "index": c["candidate_id"]}
        )

    entry["num_sampled"] = len(pred_results)
    entry["parse_success_count"] = sum(1 for c in candidates if c.get("parse_success"))
    entry["num_unique_sql"] = len(
        {
            normalize_sql(c["sql"])
            for c in candidates
            if (c.get("sql") or "").strip()
        }
    )

    # 候选正确性 flags（Pass@K 前缀复用, 双口径）
    flags_vav = [
        _candidate_correct_vav(pr["result"], gold_outcome) for pr in pred_results
    ]
    flags_custom = [
        _candidate_correct_custom(pr["result"], gold_outcome, gold_sql)
        for pr in pred_results
    ]

    # 2) candidates 数组（留作分析; 默认只存 raw_response 前 200 字符预览）
    for idx, (c, pr) in enumerate(zip(candidates, pred_results)):
        cand_entry = {
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
            "correct_vav": flags_vav[idx] if idx < len(flags_vav) else False,
            "correct_custom": (
                flags_custom[idx] if idx < len(flags_custom) else False
            ),
        }
        if save_full_responses:
            cand_entry["raw_response"] = c.get("raw_response", "")
        else:
            cand_entry["raw_response_preview"] = (c.get("raw_response") or "")[:200]
        entry["candidates"].append(cand_entry)

    # 3) 各 K 前缀投票诊断（一次采样, 前缀复用; k_list 升序）
    last_vote: Optional[Dict[str, Any]] = None
    n_total = len(pred_results)
    for k in k_list:
        cell = _empty_cell()
        k_eff = min(k, n_total)
        cell["prefix_len"] = k_eff
        if k_eff > 0:
            vote = run_vav_voting(
                pred_results[:k_eff], gold_outcome, gt_sql=gold_sql, strategy="vav"
            )
            cell["maj_correct"] = bool(vote["is_sample_correct"])
            cell["chosen_result"] = vote["chosen_result"]
            cell["chosen_group_size"] = vote["chosen_group_size"]
            cell["majority_result"] = vote["majority_result"]
            cell["majority_group_size"] = vote["majority_group_size"]
            cell["degenerate_skip_applied"] = vote["degenerate_skip_applied"]
            cell["num_exec_groups"] = len(vote["result_groups"])
            cell["num_syntax_errors"] = vote["num_syntax_errors"]
            cell["num_valid_sqls_after_filtering"] = (
                vote["num_valid_sqls_after_filtering"]
            )
            sel_res = vote.get("selected_result") or {}
            cmp, reason = _custom_match_of(sel_res, gold_outcome, gold_sql)
            cell["maj_custom_exec_match"] = cmp
            cell["maj_custom_exec_match_reason"] = reason
            prefix_vav = flags_vav[:k_eff]
            prefix_custom = flags_custom[:k_eff]
            cell["pass_at_k"] = bool(any(prefix_vav))
            cell["pass_at_k_custom"] = bool(any(prefix_custom))
            cell["average_at_k"] = round(sum(prefix_vav) / k_eff, 4)
            cell["average_at_k_custom"] = round(sum(prefix_custom) / k_eff, 4)
            # 绝对多数 = 选中组票数严格过半（> K/2, 相对 K 条前缀）
            cell["has_abs_majority"] = cell["chosen_group_size"] > k_eff / 2
            cell["abs_majority_correct"] = (
                cell["has_abs_majority"] and cell["maj_correct"]
            )
            cell["abs_majority_correct_custom"] = (
                cell["has_abs_majority"] and cell["maj_custom_exec_match"] is True
            )
            cell["self_bleu_sql"] = self_bleu_approx(
                [pr["sql"] for pr in pred_results[:k_eff]]
            )
            last_vote = vote
        entry["maj_at_k"][str(k)] = cell

    # 4) 最大 K 档 → predicted_sql（官方口径按温度档提取, 与 eval_maj_diag 一致）
    if last_vote is not None and last_vote.get("selected_sql"):
        entry["predicted_sql"] = last_vote["selected_sql"]
        entry["selected_sql_index"] = last_vote["selected_sql_index"]
        entry["vav_self_match"] = bool(last_vote["is_sample_correct"])
        sel_res = last_vote.get("selected_result") or {}
        cmp, reason = _custom_match_of(sel_res, gold_outcome, gold_sql)
        entry["selected_custom_exec_match"] = cmp
        entry["selected_custom_exec_match_reason"] = reason

    entry["evaluation_seconds"] = round(time.perf_counter() - t0, 4)
    entry["generation_seconds"] = round(time.perf_counter() - t0, 4)
    return entry


# ===================================================================
# 条目收尾（两温度档都完成才产出最终条目）
# ===================================================================

def finalize_item(
    item: Dict[str, Any],
    acc_entry: Dict[str, Any],
    k_list: List[int],
    temps: List[float],
    gold_outcome: Dict[str, Any],
) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        "dataset_index": item["dataset_index"],
        "db_id": item["db_id"],
        "question": item["question"],
        "gold_sql": item["query"],
        "difficulty": item.get("difficulty", "unknown"),
        "grid": {},
        "gold_execution_success": bool(gold_outcome.get("success")),
        "gold_error": (
            gold_outcome.get("error") if not gold_outcome.get("success") else None
        ),
        "error": acc_entry.get("error"),
    }
    for temp in temps:
        tkey = _temp_key(temp)
        tent = acc_entry.get("entries", {}).get(tkey)
        if isinstance(tent, dict):
            tent["temperature"] = temp
        else:
            tent = _empty_temp_entry(
                temp, error=acc_entry.get("error") or "processing failed"
            )
            tent["maj_at_k"] = {str(k): _empty_cell() for k in k_list}
        r["grid"][tkey] = tent
    return r


# ===================================================================
# Summary
# ===================================================================

def _cell_entry(it: Dict[str, Any], tkey: str, k: int) -> Dict[str, Any]:
    """安全取 (temp, K) 前缀条目（缺字段时回退零值模板, 兼容异常/旧 checkpoint 条目）。"""
    grid = it.get("grid")
    if isinstance(grid, dict):
        tent = grid.get(tkey)
        if isinstance(tent, dict):
            maj = tent.get("maj_at_k")
            if isinstance(maj, dict):
                ent = maj.get(str(k))
                if isinstance(ent, dict):
                    return ent
    return _empty_cell()


def _per_cell_summary(
    all_items: List[Dict[str, Any]],
    k: int,
    temp: float,
    total: int,
) -> Dict[str, Any]:
    """单格 (K, temp) 的汇总（maj@K 双口径 / Pass@K 双口径 / 分组数 / 绝对多数 / Self-BLEU）。"""
    tkey = _temp_key(temp)
    cell: Dict[str, Any] = {"K": k, "temperature": temp, "n_items": total}

    cell["n_items_with_full_prefix"] = sum(
        1 for it in all_items if _cell_entry(it, tkey, k)["prefix_len"] >= k
    )

    n_maj = sum(1 for it in all_items if _cell_entry(it, tkey, k)["maj_correct"])
    cell["maj_at_k_correct_count"] = n_maj
    cell["maj_at_k_accuracy"] = round(n_maj / total, 4) if total else 0.0

    n_maj_custom = sum(
        1 for it in all_items
        if _cell_entry(it, tkey, k)["maj_custom_exec_match"] is True
    )
    n_maj_custom_scored = sum(
        1 for it in all_items
        if _cell_entry(it, tkey, k)["maj_custom_exec_match"] is not None
    )
    cell["maj_custom_match_count"] = n_maj_custom
    cell["maj_custom_match_scored_count"] = n_maj_custom_scored
    cell["maj_custom_match_rate"] = (
        round(n_maj_custom / n_maj_custom_scored, 4) if n_maj_custom_scored else 0.0
    )

    n_pass = sum(1 for it in all_items if _cell_entry(it, tkey, k)["pass_at_k"])
    cell["pass_at_k_count"] = n_pass
    cell["pass_at_k_rate"] = round(n_pass / total, 4) if total else 0.0

    n_pass_custom = sum(
        1 for it in all_items if _cell_entry(it, tkey, k)["pass_at_k_custom"]
    )
    cell["pass_at_k_custom_count"] = n_pass_custom
    cell["pass_at_k_custom_rate"] = round(n_pass_custom / total, 4) if total else 0.0

    avg_vav = [
        _cell_entry(it, tkey, k)["average_at_k"]
        for it in all_items if _cell_entry(it, tkey, k)["prefix_len"] > 0
    ]
    cell["average_at_k"] = round(sum(avg_vav) / len(avg_vav), 4) if avg_vav else 0.0
    avg_custom = [
        _cell_entry(it, tkey, k)["average_at_k_custom"]
        for it in all_items if _cell_entry(it, tkey, k)["prefix_len"] > 0
    ]
    cell["average_at_k_custom"] = (
        round(sum(avg_custom) / len(avg_custom), 4) if avg_custom else 0.0
    )

    groups_list = [
        _cell_entry(it, tkey, k)["num_exec_groups"]
        for it in all_items if _cell_entry(it, tkey, k)["prefix_len"] > 0
    ]
    cell["avg_exec_groups"] = (
        round(sum(groups_list) / len(groups_list), 4) if groups_list else 0.0
    )

    chosen_list = [
        _cell_entry(it, tkey, k)["chosen_group_size"]
        for it in all_items if _cell_entry(it, tkey, k)["prefix_len"] > 0
    ]
    cell["avg_chosen_group_size"] = (
        round(sum(chosen_list) / len(chosen_list), 4) if chosen_list else 0.0
    )

    n_abs = sum(1 for it in all_items if _cell_entry(it, tkey, k)["has_abs_majority"])
    n_abs_correct = sum(
        1 for it in all_items if _cell_entry(it, tkey, k)["abs_majority_correct"]
    )
    n_abs_correct_custom = sum(
        1 for it in all_items
        if _cell_entry(it, tkey, k)["abs_majority_correct_custom"]
    )
    cell["abs_majority_count"] = n_abs
    cell["abs_majority_rate"] = round(n_abs / total, 4) if total else 0.0
    # 绝对多数命中率 = 绝对多数存在且命中 / 绝对多数存在（条件命中率, 双口径）
    cell["abs_majority_hit_rate"] = (
        round(n_abs_correct / n_abs, 4) if n_abs else 0.0
    )
    cell["abs_majority_hit_rate_custom"] = (
        round(n_abs_correct_custom / n_abs, 4) if n_abs else 0.0
    )

    sb_list = [
        _cell_entry(it, tkey, k).get("self_bleu_sql")
        for it in all_items
        if _cell_entry(it, tkey, k).get("self_bleu_sql") is not None
    ]
    cell["avg_self_bleu_sql"] = (
        round(sum(sb_list) / len(sb_list), 4) if sb_list else None
    )
    return cell


def build_summary(
    all_items: List[Dict[str, Any]],
    requested_indices: Set[int],
    run_config: Dict[str, Any],
    k_list: List[int],
    temps: List[float],
    evaluator: VavEvaluator,
    total_wall_seconds: float,
    generated_at: str,
) -> Dict[str, Any]:
    total = len(all_items)
    cells = [
        _per_cell_summary(all_items, k, temp, total)
        for temp in temps
        for k in k_list
    ]

    cand_parse = sum(
        sum(t.get("parse_success_count", 0) for t in it.get("grid", {}).values())
        for it in all_items
    )
    cand_total = sum(
        sum(t.get("num_sampled", 0) for t in it.get("grid", {}).values())
        for it in all_items
    )
    cand_exec_ok = sum(
        sum(1 for c in t.get("candidates", []) if c.get("execution_success"))
        for it in all_items
        for t in it.get("grid", {}).values()
    )
    unique_list = [
        t.get("num_unique_sql", 0)
        for it in all_items
        for t in it.get("grid", {}).values()
        if t.get("num_sampled", 0) > 0
    ]
    avg_unique = round(sum(unique_list) / len(unique_list), 2) if unique_list else 0.0
    n_error = sum(1 for it in all_items if it.get("error"))

    return {
        "evaluator_type": EVALUATOR_TYPE,
        "is_official_spider_metric": False,
        "note": (
            "投票诊断曲线温度网格: 每温度档同 prompt 一次采样 max(K)=64 条, 按前 K 条"
            "前缀逐档计算（两个温度档共用同一 VavSampler, 每档重置同一 seed → 前缀序列"
            "同 RNG 流, 温度是唯一变量）。maj@K 判定 = vav 执行分组投票（SUCCESS_VALUES "
            "组, 跳过空/全零退化组, FINER majority_voting 同口径, 与 eval_maj_diag "
            "同引擎, 曲线可直接对比）; 正确性双口径: maj_at_k_accuracy / pass_at_k_rate "
            "为 header-agnostic 行集合相等（results_equal, vav 口径）; "
            "maj_custom_match_rate / pass_at_k_custom_rate 为项目自定义执行匹配 "
            "（compare_execution_results, ORDER BY 感知 multiset, 本实验 headline, "
            "非官方 Spider EX/EM）; has_abs_majority = chosen_group_size > K/2 "
            "（严格过半, 相对 K 条前缀）; abs_majority_hit_rate = 绝对多数存在且命中 / "
            "绝对多数存在题数（条件命中率）; avg_self_bleu_sql = SQL token 级 Self-BLEU "
            "近似（复用 eval_maj_diag.self_bleu_approx: clipped n-gram precision "
            "n=1..4, +1 Laplace 平滑, 无 nltk; 数值越低候选越多样）; 官方 test-suite "
            "EX 复评由主控另跑（每温度档 max-K 的 predicted_sql 存于 items.json 的 "
            "grid[<temp>].predicted_sql, 按温度档提取）。"
        ),
        "total_requested": len(requested_indices),
        "total_completed": total,
        "error_items": n_error,
        "max_k": k_list[-1] if k_list else 0,
        "k_list": k_list,
        "temperatures": temps,
        "cells": cells,
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
        "avg_unique_sql_per_temp_entry": avg_unique,
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
    k_list, temps = args.grid
    max_k = k_list[-1]  # k_list 升序 → 每温度档采样数 = 最大档

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 加载数据集 ---
    loader = SpiderLoader(args.spider_dir)
    items = loader.load_dev(limit=args.n, start_index=args.start_index)
    requested_indices: Set[int] = {it["dataset_index"] for it in items}
    print(f"Loaded {len(items)} items (start={args.start_index}, limit={args.n})")
    print(f"Grid: K={k_list} x T={temps} ({len(k_list) * len(temps)} cells, "
          f"sample once per temperature with max K = {max_k})")

    # --- checkpoint / resume（协议与 eval_maj_diag / eval_mpev 一致） ---
    run_config = build_grid_run_config(args, k_list, temps)
    cp = load_checkpoint(output_dir)
    if cp["run_config"] is not None:
        validate_resume_config(cp["run_config"], run_config)
    validate_checkpoint_integrity(cp, requested_indices)
    completed: Set[int] = cp["completed_indices"]
    all_items: List[Dict[str, Any]] = list(cp["items"])
    print(f"Resume: {len(completed)}/{len(requested_indices)} items already completed")

    # --- 加载模型 + 执行器（复用 VavSampler / VavEvaluator, 不重写） ---
    print(f"\nLoading model: {args.base_model}")
    print(f"LoRA adapter: {args.lora_path}")
    sampler = VavSampler(
        model_path=args.base_model,
        lora_path=args.lora_path,
        max_new_tokens=args.max_new_tokens,
        temperature=temps[-1],  # 每档运行时再逐档改 sampler.temperature
        top_p=args.top_p,
        prompt_style="default",  # 同 prompt: canonical prompt（与训练/既有评估一致）
        seed=args.seed,
        local_files_only=not args.allow_remote,
    )
    evaluator = VavEvaluator(DatabaseExecutor(args.spider_dir))
    print(f"Sampler ready. n-samples={max_k}, T={temps}, seed={args.seed}\n")

    # --- 预构造 prompt + gold 执行（跨温度档共用, 各算一次） ---
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
    gold_map: Dict[int, Dict[str, Any]] = {
        it["dataset_index"]: evaluator.execute_cached(it["db_id"], it["query"])
        for it in items
    }

    pending = [it for it in items if it["dataset_index"] not in completed]
    # 在途条目累积器: {ds: {"item": item, "entries": {temp_key: temp_entry|None},
    #                       "error": str|None}}
    acc: Dict[int, Dict[str, Any]] = {}
    for it in pending:
        ds = it["dataset_index"]
        acc[ds] = {"item": it, "entries": {}, "error": None}
        if chat_texts.get(ds) is None:
            acc[ds]["error"] = "ddl_load_failed"
            for temp in temps:
                acc[ds]["entries"][_temp_key(temp)] = None

    wall_start = time.perf_counter()
    batch_counter = 0

    # --- 逐温度档: 一次采样 64, 前缀 K∈{8,16,32,64} 逐档计算 ---
    for t_idx, temp in enumerate(temps):
        tkey = _temp_key(temp)
        # 每档从同一 RNG 状态出发 → 同题同前缀序列下温度是唯一变量
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        sampler.temperature = temp
        print(f"\n=== temperature={temp} ({t_idx + 1}/{len(temps)}), "
              f"n={max_k} per item ===")

        for i in range(0, len(pending), args.batch_size):
            batch = pending[i:i + args.batch_size]
            batch_counter += 1
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
                          f"recording empty candidates for this batch")
                    for ds, _ in batch_prompts:
                        gen_map[ds] = []

            # --- 每题: 执行 + 各 K 前缀投票诊断（单温度档） ---
            for it in batch:
                ds = it["dataset_index"]
                if chat_texts.get(ds) is None and acc[ds]["error"] is None:
                    acc[ds]["error"] = "ddl_load_failed"
                if acc[ds]["error"] is not None:
                    acc[ds]["entries"][tkey] = None
                else:
                    try:
                        acc[ds]["entries"][tkey] = process_temp_entry(
                            it, gen_map.get(ds, []), evaluator,
                            k_list, gold_map.get(ds, {}), args.save_full_responses,
                        )
                    except Exception as exc:
                        acc[ds]["entries"][tkey] = None
                        acc[ds]["error"] = f"temperature {tkey} processing failed: {exc}"

            # --- 两温度档都完成的题 → 产出最终条目 ---
            for it in batch:
                ds = it["dataset_index"]
                if ds in completed:
                    continue
                if len(acc[ds]["entries"]) == len(temps):
                    all_items.append(finalize_item(
                        it, acc[ds], k_list, temps, gold_map.get(ds, {})
                    ))
                    completed.add(ds)

            # --- 定期 checkpoint（只存整题完成的条目, 断点粒度=整题） ---
            if batch_counter % args.checkpoint_every == 0:
                save_checkpoint(
                    output_dir,
                    {"completed_indices": sorted(completed), "items": all_items},
                    run_config,
                )
                done = len(all_items)
                if done:
                    tkey_last = _temp_key(temps[-1])
                    last_k = str(k_list[-1])
                    n_maj = sum(
                        1 for it in all_items
                        if _cell_entry(it, tkey_last, k_list[-1])["maj_correct"]
                    )
                    print(
                        f"  [checkpoint {done}/{len(items)}] "
                        f"maj@{k_list[-1]}(T={temps[-1]})={n_maj}/{done} "
                        f"({n_maj / done:.1%}) "
                        f"wall={time.perf_counter() - wall_start:.0f}s"
                    )

    # --- 兜底: 收尾任何未完成条目（异常路径） ---
    for it in pending:
        ds = it["dataset_index"]
        if ds not in completed:
            all_items.append(finalize_item(
                it, acc.get(ds, {"entries": {}, "error": "missing accumulator"}),
                k_list, temps, gold_map.get(ds, {})
            ))
            completed.add(ds)

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
        k_list,
        temps,
        evaluator,
        total_wall,
        datetime.now(timezone.utc).isoformat(),
    )
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # --- 打印网格表 ---
    print("\n" + "=" * 86)
    print("  VOTING DIAGNOSIS CURVE x TEMPERATURE GRID SUMMARY")
    print("=" * 86)
    print(f"  Items: {summary['total_completed']}/{summary['total_requested']} "
          f"(error={summary['error_items']})")
    print(
        f"  {'K':>4} | {'T':>5} | {'majVav':>7} | {'majCust':>8} | "
        f"{'passVav':>7} | {'passCust':>8} | {'groups':>6} | "
        f"{'absMaj':>6} | {'absHit':>6} | {'selfBL':>6}"
    )
    for cell in summary["cells"]:
        sb = (f"{cell['avg_self_bleu_sql']:.3f}"
              if cell["avg_self_bleu_sql"] is not None else "-")
        print(
            f"  {cell['K']:>4} | {cell['temperature']:>5g} | "
            f"{cell['maj_at_k_accuracy']:>7.1%} | {cell['maj_custom_match_rate']:>8.1%} | "
            f"{cell['pass_at_k_rate']:>7.1%} | {cell['pass_at_k_custom_rate']:>8.1%} | "
            f"{cell['avg_exec_groups']:>6.2f} | {cell['abs_majority_rate']:>6.1%} | "
            f"{cell['abs_majority_hit_rate_custom']:>6.1%} | {sb:>6}"
        )
    print(f"  候选 parse 率:         {summary['candidate_parse_success_rate']:.1%}")
    print(f"  候选执行成功率:        {summary['candidate_execution_success_rate']:.1%}")
    print(f"  执行缓存命中率:        {summary['execution_cache_hit_ratio']:.1%}")
    print(f"  温度档条目唯一 SQL 均值: {summary['avg_unique_sql_per_temp_entry']}")
    print(f"  总耗时:                {summary['total_wall_seconds']:.0f}s")
    print("=" * 86)
    print(f"\nItems saved to:   {items_path}")
    print(f"Summary saved to: {summary_path}")
    print("\n官方 test-suite 评估（下一步, 由主控另跑, 按温度档提取 predicted_sql）:")
    print(f"  # bash scripts/eval_official.sh <per-temp items.json> "
          f"{output_dir / 'official_<temp>'}")


if __name__ == "__main__":
    main()
