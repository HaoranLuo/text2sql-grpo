#!/usr/bin/env python3
"""
文献弹药: memory 质心过滤投票（centroid-filtered voting）评估脚本。

背景: L7 机制论文「选择缺口」修复尝试。离线分析已出数: 5p 投票中 votes=3
（3:2 争议）题的多数派正确率仅 34.7%, 且按数据库聚集（world_1 16% / car_1 0%）。
本脚本用「memory 质心过滤」对低一致争议题做一致性干预: 同 prompt 采样出 n 个
候选后, 计算每个候选与「群体」（其余候选）的平均相似度（质心分数）, 剔除质心
相似度最低的 20% / 30% 离群候选后再做执行分组投票, 对照无过滤基线 maj@n。

设计（复用 eval_maj_diag.py 的采样与统计函数, 不重写）:
  1. 采样: 同 prompt（VavSampler prompt_style="default"）一次采样 n=16 条候选
     （单次 forward, num_return_sequences=n, do_sample T=1.0, seed 固定）;
  2. 执行: 全部候选 + gold 各执行一次（VavEvaluator, (db_id, normalize_sql) 缓存）,
     解析失败/执行失败的候选尝试全部留痕;
  3. 相似度口径（写入 summary.note, 执行结果分组优先——语义等价更可靠）:
     - 投票资格池 = 非语法错误候选（is_syntax_error, 与 run_vav_voting 同口径）;
     - 两两相似度: 双方执行成功 → 归一化执行签名（normalize_execution_result）
       相同 ? 1.0 : 0.0; 任一方执行失败 → 回退 SQL token 集合 Jaccard
       （sqlglot.tokenize, sqlglot 缺失时回退正则分词, 双方均空 token 记 1.0）;
     - **两种口径均落盘**: 每候选 avg_similarity（混合）/ avg_similarity_exec
       （纯执行签名对均值）/ avg_similarity_text（纯文本 Jaccard 对均值）+
       逐题 similarity_pair_counts 两种对的计数;
     - 质心分数 = 与其余资格候选相似度的均值（混合口径）;
  4. 票数分层: votes = 基线执行分组的多数组票数（majority_group_size, 与 5p 口径
     的「票数」同构——n=16 同 prompt 采样下 votes<=3 为碎片化低一致题）。summary
     按 votes ∈ {0,1,2,3,4,5,6+} 分层输出 baseline vs 过滤后正确率, 重点看低一致
     争议题（3:2 同构档）是否被过滤救回; items.json 逐题含 votes, 可按 db_id 做
     数据库聚集分析;
  5. 过滤: 按 (质心分数升序, candidate_id 升序) 剔除最底 round(K*r) 个
     （K=资格候选数, 至少 1 个、至多 K-1 个）, r 为 --drop-ratio 各档;
     对保留候选复用 run_vav_voting(strategy="vav") 执行分组投票;
     --min-votes N: 仅对基线票数 <= N 的低一致题启用过滤（0=全部启用; 建议 3,
     高一致题不动防误杀——未启用过滤的题 filtered 条目 = 基线结果, 逐题标记
     filter_applied=False）;
  6. 基线对照: 全 n 候选同一投票口径（vav strategy）, predicted_sql = 基线
     投票结果（scripts/eval_official.sh 兼容——官方口径仅对基线臂有意义）;
  7. 输出 items.json + summary.json（基线 maj@n vs 各档过滤后 maj@n + fixed/broken
     迁移统计 + votes 分层）; 断点续跑复用 src/spider_utils.py 的 checkpoint 协议。

用法（GPU, 与 eval_maj_diag 同风格）:
    python src/eval_mem_filter.py \
        --lora-path checkpoints/sft_v2 \
        --model-path models/Qwen2.5-Coder-3B-Instruct \
        --spider-dir data/spider_data \
        --limit 100 --n 16 --drop-ratio 0.2,0.3 --min-votes 3 \
        --output-dir outputs/eval_mem_filter
"""

import argparse
import json
import re
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
from sampler import VavSampler  # noqa: E402

# sqlglot 为软依赖（文本相似度回退口径优先用其 tokenize; 缺失时回退正则分词）
try:
    import sqlglot  # type: ignore
    _SQLGLOT_AVAILABLE = bool(hasattr(sqlglot, "tokenize"))
except ImportError:  # pragma: no cover
    sqlglot = None
    _SQLGLOT_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"
DEFAULT_LORA_PATH = PROJECT_ROOT / "checkpoints" / "sft_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_mem_filter"

EVALUATOR_TYPE = "mem_filter_centroid_voting"

# votes 分层桶顺序（votes = 基线执行分组多数组票数; >5 归入 "6+"）
VOTES_BUCKET_ORDER = ["0", "1", "2", "3", "4", "5", "6+"]


# ===================================================================
# 参数
# ===================================================================

def parse_drop_ratios(s: str) -> List[float]:
    """解析 --drop-ratio "0.2,0.3" → 去重、升序的 float 列表（每档都在 (0, 1)）。"""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("empty --drop-ratio")
    ratios: List[float] = []
    for p in parts:
        try:
            r = float(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid drop ratio: {p!r}")
        if not (0.0 < r < 1.0):
            raise argparse.ArgumentTypeError(f"drop ratio must be in (0, 1), got {r}")
        ratios.append(r)
    return sorted(set(ratios))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "文献弹药 memory 质心过滤投票: 同 prompt 采样 n 条候选, 按质心相似度"
            "剔除最底 20%/30% 离群候选后执行分组投票, 对照无过滤基线 maj@n"
            "（vav 口径, 与 eval_maj_diag 一致; votes 分层 + --min-votes 低一致题门控）"
        ),
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH),
                        help="本地基座模型路径")
    parser.add_argument("--lora-path", default=str(DEFAULT_LORA_PATH),
                        help="LoRA adapter 路径（默认 checkpoints/sft_v2）")
    parser.add_argument("--spider-dir", default=str(DEFAULT_SPIDER_DIR),
                        help="Spider 数据集根目录")
    parser.add_argument("--limit", type=int, default=100,
                        help="评估条数（默认 dev 前 100 条, 内部诊断口径）")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--n", type=int, default=16,
                        help="每题采样候选数（默认 16, 对应 maj@16 诊断档）")
    parser.add_argument("--drop-ratio", type=parse_drop_ratios, default="0.2,0.3",
                        dest="drop_ratios",
                        help="质心过滤剔除比例档位, 逗号分隔, 每档在 (0,1)"
                             "（默认 0.2,0.3）")
    parser.add_argument("--min-votes", type=int, default=0,
                        help="仅对基线票数 <= 该值的低一致题启用质心过滤"
                             "（默认 0=全部启用; 建议 3=只滤低一致争议题, "
                             "高一致题不动防误杀）")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度（默认 1.0, 与 eval_maj_diag 一致）")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="与训练 max_completion 对齐（PLAN §8 生成长度一致性）")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="每次 forward 处理的题目数（一次 forward 产 B*n 条）")
    parser.add_argument("--seed", type=int, default=0,
                        help="采样种子（候选顺序依赖 num_return_sequences 的固定顺序）")
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

def build_memfilter_run_config(args: argparse.Namespace) -> Dict[str, Any]:
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
            "n": args.n,
            "drop_ratios": args.drop_ratios,
            "min_votes": args.min_votes,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "prompt_style": "default",  # 同 prompt: canonical prompt 固定
            "lora_path": args.lora_path,
            "seed": args.seed,
        }
    )
    return cfg


# ===================================================================
# 文本相似度回退（SQL token 集合 Jaccard）
# ===================================================================

# SQL 正则分词（sqlglot 缺失时的回退）: 标识符 / 数字 / 多字符运算符 / 单个标点
_SQL_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|<=|>=|<>|!=|->|[^\sA-Za-z0-9_]"
)


def _tokenize_sql_regex(sql: str) -> List[str]:
    return _SQL_TOKEN_RE.findall(sql or "")


def _sql_text_tokens(sql: str) -> Set[str]:
    """
    SQL token 集合: sqlglot.tokenize 优先（含 token 类型, 文本大写归一）;
    sqlglot 缺失或 tokenize 抛错时回退正则分词（文本大写归一）。
    """
    if _SQLGLOT_AVAILABLE:
        try:
            toks = sqlglot.tokenize(sql or "")
            out: Set[str] = set()
            for t in toks:
                tt = str(getattr(t, "token_type", "")).split(".")[-1]
                if tt in ("", "WHITESPACE", "BREAK"):
                    continue
                out.add(f"{tt}:{(t.text or '').upper()}")
            if out:
                return out
        except Exception:
            pass  # sqlglot tokenize 失败 → 回退正则分词
    return {t.upper() for t in _tokenize_sql_regex(sql or "")}


def _token_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if not union:
        return 1.0
    return len(a & b) / union


# ===================================================================
# 质心相似度计算与过滤（纯函数, 零执行开销）
# ===================================================================

def compute_centroid_scores(
    pred_results: List[Dict[str, Any]],
    gold_sql: str,
) -> Tuple[List[int], Dict[int, Optional[float]], Dict[int, Optional[float]],
           Dict[int, Optional[float]], Optional[float], int, int]:
    """
    在投票资格池（非语法错误候选, 与 run_vav_voting 同口径）上计算两两相似度
    与质心分数（混合口径 + 两种单一口径均返回）。

    相似度口径（执行结果分组优先）:
      - 双方执行成功 → 归一化执行签名（normalize_execution_result）相同 ? 1.0 : 0.0;
      - 任一方执行失败 → 回退 SQL token 集合 Jaccard（_sql_text_tokens）。

    返回: (eligible_ids, avg_sim, avg_sim_exec, avg_sim_text,
           avg_pairwise_sim, exec_pair_count, text_pair_count)
      eligible_ids: pred_results 下标列表（保持采样顺序）;
      avg_sim:  {下标: 混合口径质心分数}（与其余资格候选相似度均值, 无其余候选时 None）;
      avg_sim_exec:  {下标: 纯执行签名对均值}（该候选没有任何执行签名对时为 None）;
      avg_sim_text:  {下标: 纯文本 Jaccard 对均值}（该候选没有任何文本对时为 None）;
      avg_pairwise_sim: 全部资格候选对的平均相似度（不足 2 个资格候选时为 None）。
    """
    n = len(pred_results)
    eligible = [
        i for i in range(n) if not is_syntax_error(pred_results[i]["result"])
    ]
    sigs: Dict[int, Optional[str]] = {}
    tokens: Dict[int, Set[str]] = {}
    for i in eligible:
        pr = pred_results[i]
        if pr["result"].get("success", False):
            sigs[i] = normalize_execution_result(pr["result"], gt_sql=gold_sql)
        else:
            sigs[i] = None
        tokens[i] = _sql_text_tokens(pr["sql"])

    avg_sim: Dict[int, Optional[float]] = {}
    avg_sim_exec: Dict[int, Optional[float]] = {}
    avg_sim_text: Dict[int, Optional[float]] = {}
    exec_pairs = 0
    text_pairs = 0
    pair_sims: List[float] = []
    for i in eligible:
        acc = 0.0
        cnt = 0
        exec_acc = 0.0
        exec_cnt = 0
        text_acc = 0.0
        text_cnt = 0
        for j in eligible:
            if i == j:
                continue
            if sigs[i] is not None and sigs[j] is not None:
                s = 1.0 if sigs[i] == sigs[j] else 0.0
                exec_pairs += 1
                exec_acc += s
                exec_cnt += 1
            else:
                s = _token_jaccard(tokens[i], tokens[j])
                text_pairs += 1
                text_acc += s
                text_cnt += 1
            acc += s
            cnt += 1
            pair_sims.append(s)
        avg_sim[i] = (acc / cnt) if cnt else None
        avg_sim_exec[i] = (exec_acc / exec_cnt) if exec_cnt else None
        avg_sim_text[i] = (text_acc / text_cnt) if text_cnt else None

    avg_pairwise = sum(pair_sims) / len(pair_sims) if pair_sims else None
    return (
        eligible, avg_sim, avg_sim_exec, avg_sim_text,
        avg_pairwise, exec_pairs, text_pairs,
    )


def centroid_filter(
    eligible: List[int],
    avg_sim: Dict[int, Optional[float]],
    drop_ratio: float,
) -> Tuple[List[int], List[int], Optional[float]]:
    """
    剔除质心相似度最低的 round(K * drop_ratio) 个资格候选。

    K = len(eligible); K < 2 或 ratio <= 0 时不过滤（返回全保留）。
    剔除数 clamp 到 [1, K-1]（至少保留 1 个候选）。
    同分（质心分数相同）按 candidate_id 升序先剔——确定性 tie-break。

    返回: (kept_ids, dropped_ids, threshold)
      kept_ids / dropped_ids: 保持原始采样顺序;
      threshold: 最后被剔除候选的质心分数（切割线, 供分析; K<2 时为 None）。
    """
    k = len(eligible)
    if k < 2 or drop_ratio <= 0:
        return list(eligible), [], None
    n_drop = max(1, min(int(round(k * drop_ratio)), k - 1))
    order = sorted(eligible, key=lambda i: (avg_sim.get(i) or 0.0, i))
    dropped = order[:n_drop]
    dropped_set = set(dropped)
    kept = [i for i in eligible if i not in dropped_set]
    threshold = avg_sim.get(dropped[-1])
    return kept, dropped, threshold


# ===================================================================
# 单样本处理
# ===================================================================

def _empty_vote_entry() -> Dict[str, Any]:
    """投票条目零值模板（error 条目也保持结构完整, 供 summary 聚合）。"""
    return {
        "num_candidates": 0,
        "maj_correct": False,
        "chosen_result": "NO_RESULTS",
        "chosen_group_size": 0,
        "majority_result": "NO_RESULTS",
        "majority_group_size": 0,
        "majority_is_correct": False,
        "degenerate_skip_applied": False,
        "num_exec_groups": 0,
        "num_syntax_errors": 0,
        "num_valid_sqls_after_filtering": 0,
        "selected_sql_index": -1,
    }


def _empty_filtered_vote_entry() -> Dict[str, Any]:
    entry = _empty_vote_entry()
    entry.update(
        {
            "drop_ratio": None,
            "filter_applied": False,
            "dropped_count": 0,
            "kept_count": 0,
            "dropped_indices": [],
            "kept_indices": [],
            "avg_sim_drop_threshold": None,
        }
    )
    return entry


def _vote_from_results(vote: Dict[str, Any]) -> Dict[str, Any]:
    """把 run_vav_voting 的返回精简为可落盘的标量字段（不落 full_rows）。"""
    return {
        "num_candidates": vote["num_predicted_sqls"],
        "maj_correct": bool(vote["is_sample_correct"]),
        "chosen_result": vote["chosen_result"],
        "chosen_group_size": vote["chosen_group_size"],
        "majority_result": vote["majority_result"],
        "majority_group_size": vote["majority_group_size"],
        "majority_is_correct": bool(vote["majority_is_correct"]),
        "degenerate_skip_applied": bool(vote["degenerate_skip_applied"]),
        "num_exec_groups": len(vote["result_groups"]),
        "num_syntax_errors": vote["num_syntax_errors"],
        "num_valid_sqls_after_filtering": vote["num_valid_sqls_after_filtering"],
        "selected_sql_index": vote["selected_sql_index"],
    }


def _new_item(item: Dict[str, Any], drop_ratios: List[float]) -> Dict[str, Any]:
    return {
        "dataset_index": item["dataset_index"],
        "db_id": item["db_id"],
        "question": item["question"],
        "gold_sql": item["query"],
        "difficulty": item.get("difficulty", "unknown"),
        "predicted_sql": None,          # ← 基线投票结果, eval_official.sh 兼容
        "selected_sql_index": -1,
        "gold_execution_success": False,
        "gold_error": None,
        "num_sampled": 0,
        "parse_success_count": 0,
        "num_unique_sql": 0,
        "num_eligible": 0,
        "num_syntax_errors": 0,
        "votes": 0,                     # 基线执行分组多数组票数（5p「票数」同构）
        "similarity_mode": "execution_result_grouping_with_text_jaccard_fallback",
        "avg_pairwise_similarity": None,
        "similarity_pair_counts": {
            "execution_result_pairs": 0,
            "text_jaccard_pairs": 0,
        },
        "candidates": [],
        "baseline_vote": _empty_vote_entry(),
        "filtered_votes": {str(r): _empty_filtered_vote_entry() for r in drop_ratios},
        "filter_applied": {str(r): False for r in drop_ratios},
        "generation_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "error": None,
    }


def process_item(
    item: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    evaluator: VavEvaluator,
    drop_ratios: List[float],
    min_votes: int,
    save_full_responses: bool,
) -> Dict[str, Any]:
    """
    执行验证（全部候选一次执行, 带缓存）→ 质心分数 → 基线投票 + 各 drop 档
    过滤后投票（--min-votes 门控）, 产出单个评估条目。

    candidates: VavSampler 输出的候选 dict 列表（raw_response/sql/parse_success/
    parse_method/candidate_id），顺序 = num_return_sequences 固定顺序（seed 固定）。
    过滤后投票零额外执行开销: run_vav_voting 是纯函数（只重新分组/投票）。

    votes 分层: votes = 基线投票的多数组票数（majority_group_size, 执行分组口径）。
    --min-votes N: 仅 votes <= N 的题启用过滤; 其余题 filtered 条目 = 基线结果
    （filter_applied=False, 防误杀高一致题）。
    """
    r = _new_item(item, drop_ratios)
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
    r["num_syntax_errors"] = sum(
        1 for pr in pred_results if is_syntax_error(pr["result"])
    )

    # 2) 质心分数（投票资格池 = 非语法错误候选, 与 run_vav_voting 同口径;
    #    混合 + 执行签名 + 文本 Jaccard 三种口径均落盘）
    (eligible, avg_sim, avg_sim_exec, avg_sim_text,
     avg_pairwise, exec_pairs, text_pairs) = compute_centroid_scores(
        pred_results, gold_sql
    )
    eligible_set = set(eligible)
    r["num_eligible"] = len(eligible)
    r["avg_pairwise_similarity"] = (
        round(avg_pairwise, 4) if avg_pairwise is not None else None
    )
    r["similarity_pair_counts"] = {
        "execution_result_pairs": exec_pairs,
        "text_jaccard_pairs": text_pairs,
    }

    # 3) 基线投票（全 n 候选, vav 口径）→ votes 分层
    baseline_vote = run_vav_voting(
        pred_results, gold_outcome, gt_sql=gold_sql, strategy="vav"
    )
    r["baseline_vote"] = _vote_from_results(baseline_vote)
    r["votes"] = baseline_vote["majority_group_size"]
    if baseline_vote.get("selected_sql"):
        r["predicted_sql"] = baseline_vote["selected_sql"]
        r["selected_sql_index"] = baseline_vote["selected_sql_index"]

    # 4) 各 drop 档: 质心过滤（--min-votes 门控）→ 保留候选重新投票
    kept_per_ratio: Dict[str, Set[int]] = {}
    dropped_per_ratio: Dict[str, Set[int]] = {}
    for ratio in drop_ratios:
        key = str(ratio)
        filter_enabled = (min_votes <= 0) or (r["votes"] <= min_votes)
        if filter_enabled:
            kept, dropped, threshold = centroid_filter(eligible, avg_sim, ratio)
            kept_per_ratio[key] = set(kept)
            dropped_per_ratio[key] = set(dropped)
            vote = run_vav_voting(
                [pred_results[i] for i in kept],
                gold_outcome,
                gt_sql=gold_sql,
                strategy="vav",
            )
            entry = _vote_from_results(vote)
            entry.update(
                {
                    "drop_ratio": ratio,
                    "filter_applied": True,
                    "dropped_count": len(dropped),
                    "kept_count": len(kept),
                    "dropped_indices": sorted(dropped),
                    "kept_indices": sorted(kept),
                    "avg_sim_drop_threshold": (
                        round(threshold, 4) if threshold is not None else None
                    ),
                }
            )
        else:
            # 高一致题不动: filtered 条目 = 基线结果, 防误杀
            kept_per_ratio[key] = set(range(len(pred_results)))
            dropped_per_ratio[key] = set()
            entry = _vote_from_results(baseline_vote)
            entry.update(
                {
                    "drop_ratio": ratio,
                    "filter_applied": False,
                    "dropped_count": 0,
                    "kept_count": len(pred_results),
                    "dropped_indices": [],
                    "kept_indices": sorted(range(len(pred_results))),
                    "avg_sim_drop_threshold": None,
                }
            )
        r["filtered_votes"][key] = entry
        r["filter_applied"][key] = entry["filter_applied"]

    # 5) candidates 数组（留作分析; 默认只存 raw_response 前 200 字符预览）
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
            "eligible": idx in eligible_set,
            "avg_similarity": (
                round(avg_sim[idx], 4) if avg_sim.get(idx) is not None else None
            ),
            "avg_similarity_exec": (
                round(avg_sim_exec[idx], 4)
                if avg_sim_exec.get(idx) is not None else None
            ),
            "avg_similarity_text": (
                round(avg_sim_text[idx], 4)
                if avg_sim_text.get(idx) is not None else None
            ),
            "dropped": {
                key: idx in dropped_per_ratio[key] for key in dropped_per_ratio
            },
        }
        if save_full_responses:
            entry["raw_response"] = c.get("raw_response", "")
        else:
            entry["raw_response_preview"] = (c.get("raw_response") or "")[:200]
        r["candidates"].append(entry)

    r["evaluation_seconds"] = round(time.perf_counter() - t0, 4)
    r["generation_seconds"] = round(time.perf_counter() - t0, 4)
    return r


# ===================================================================
# Summary
# ===================================================================

def _votes_bucket(votes: int) -> str:
    if votes <= 0:
        return "0"
    return str(votes) if votes <= 5 else "6+"


def build_summary(
    all_items: List[Dict[str, Any]],
    requested_indices: Set[int],
    run_config: Dict[str, Any],
    drop_ratios: List[float],
    evaluator: VavEvaluator,
    total_wall_seconds: float,
    generated_at: str,
) -> Dict[str, Any]:
    total = len(all_items)
    n_error = sum(1 for it in all_items if it.get("error"))

    n_base = sum(
        1 for it in all_items if (it.get("baseline_vote") or {}).get("maj_correct")
    )

    filtered: Dict[str, Dict[str, Any]] = {}
    for ratio in drop_ratios:
        key = str(ratio)
        n_correct = 0
        n_fixed = 0   # 基线错 → 过滤后对
        n_broken = 0  # 基线对 → 过滤后错
        n_applied = 0
        dropped_counts: List[int] = []
        kept_counts: List[int] = []
        for it in all_items:
            b = bool((it.get("baseline_vote") or {}).get("maj_correct"))
            fv = (it.get("filtered_votes") or {}).get(key) or {}
            f = bool(fv.get("maj_correct"))
            n_correct += int(f)
            if fv.get("filter_applied"):
                n_applied += 1
            if not b and f:
                n_fixed += 1
            if b and not f:
                n_broken += 1
            dropped_counts.append(int(fv.get("dropped_count", 0) or 0))
            kept_counts.append(int(fv.get("kept_count", 0) or 0))
        filtered[key] = {
            "drop_ratio": ratio,
            "maj_correct_count": n_correct,
            "maj_accuracy": round(n_correct / total, 4) if total else 0.0,
            "fixed_count": n_fixed,
            "broken_count": n_broken,
            "filter_applied_count": n_applied,
            "avg_dropped_count": (
                round(sum(dropped_counts) / len(dropped_counts), 2)
                if dropped_counts else 0.0
            ),
            "avg_kept_count": (
                round(sum(kept_counts) / len(kept_counts), 2)
                if kept_counts else 0.0
            ),
        }

    # --- votes 分层（重点: 低一致争议题是否被过滤救回） ---
    strata: Dict[str, Dict[str, Any]] = {}
    for it in all_items:
        v = int(it.get("votes", 0) or 0)
        bucket = _votes_bucket(v)
        s = strata.setdefault(
            bucket,
            {
                "votes_bucket": bucket,
                "count": 0,
                "baseline_correct": 0,
                "filtered": {
                    str(r): {"correct": 0, "n_filter_applied": 0}
                    for r in drop_ratios
                },
            },
        )
        s["count"] += 1
        if (it.get("baseline_vote") or {}).get("maj_correct"):
            s["baseline_correct"] += 1
        for ratio in drop_ratios:
            key = str(ratio)
            fv = (it.get("filtered_votes") or {}).get(key) or {}
            if fv.get("maj_correct"):
                s["filtered"][key]["correct"] += 1
            if fv.get("filter_applied"):
                s["filtered"][key]["n_filter_applied"] += 1

    votes_strata: Dict[str, Dict[str, Any]] = {}
    for bucket in VOTES_BUCKET_ORDER:
        s = strata.get(bucket)
        if s is None:
            continue
        s["baseline_accuracy"] = (
            round(s["baseline_correct"] / s["count"], 4) if s["count"] else 0.0
        )
        for key, f in s["filtered"].items():
            f["accuracy"] = round(f["correct"] / s["count"], 4) if s["count"] else 0.0
        votes_strata[bucket] = s

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
    avg_pair_sims = [
        it["avg_pairwise_similarity"]
        for it in all_items
        if it.get("avg_pairwise_similarity") is not None
    ]
    exec_pairs = sum(
        (it.get("similarity_pair_counts") or {}).get("execution_result_pairs", 0)
        for it in all_items
    )
    text_pairs = sum(
        (it.get("similarity_pair_counts") or {}).get("text_jaccard_pairs", 0)
        for it in all_items
    )
    n_text_fallback = sum(
        1 for it in all_items
        if (it.get("similarity_pair_counts") or {}).get("text_jaccard_pairs", 0) > 0
    )
    gen_times = [it.get("generation_seconds", 0.0) for it in all_items]

    return {
        "evaluator_type": EVALUATOR_TYPE,
        "is_official_spider_metric": False,
        "note": (
            "memory 质心过滤投票（L7 选择缺口修复尝试）。相似度口径: 执行结果分组优先"
            "——双方执行成功时相似度 = 归一化执行签名（vav_voting."
            "normalize_execution_result）相同 ? 1.0 : 0.0（语义等价更可靠）; 任一方"
            "执行失败（含基础设施失败）时回退文本相似度 = SQL token 集合 Jaccard"
            "（sqlglot.tokenize, sqlglot 缺失时回退正则分词, 双方均空 token 记 1.0）。"
            "两种口径均落盘: 每候选 avg_similarity（混合）/avg_similarity_exec/"
            "avg_similarity_text + 逐题 similarity_pair_counts 对计数。投票资格池 = "
            "非语法错误候选（is_syntax_error, 与 run_vav_voting 同口径）; 质心分数 = "
            "与其余资格候选相似度均值; 剔除最底 round(K*r) 个（K=资格候选数, 至少 1 "
            "个、至多 K-1 个, 同分按 candidate_id 升序先剔）; 过滤后对保留候选复用 "
            "run_vav_voting(strategy=vav) 执行分组投票; maj 判定 = 选中组与 gold 行"
            "集合相等（results_equal, header-agnostic）。votes = 基线多数组票数"
            "（majority_group_size, 执行分组口径, 与 5p「票数」同构）; --min-votes N "
            "门控: 仅 votes<=N 的题启用过滤, 高一致题不动（filtered 条目=基线结果, "
            "filter_applied=False, 防误杀）。fixed = 基线错→过滤后对, broken = 基线对"
            "→过滤后错。predicted_sql = 基线臂结果（scripts/eval_official.sh 仅对"
            "基线臂有意义）。"
        ),
        "total_requested": len(requested_indices),
        "total_completed": total,
        "error_items": n_error,
        "n": run_config.get("n"),
        "min_votes": run_config.get("min_votes"),
        "drop_ratios": drop_ratios,
        "baseline": {
            "maj_correct_count": n_base,
            "maj_accuracy": round(n_base / total, 4) if total else 0.0,
        },
        "filtered": filtered,
        "votes_strata": votes_strata,
        "avg_pairwise_similarity": (
            round(sum(avg_pair_sims) / len(avg_pair_sims), 4)
            if avg_pair_sims else None
        ),
        "similarity_pair_counts": {
            "execution_result_pairs": exec_pairs,
            "text_jaccard_pairs": text_pairs,
        },
        "items_with_text_fallback": n_text_fallback,
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
    if args.n < 1:
        raise SystemExit("--n must be >= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 加载数据集 ---
    loader = SpiderLoader(args.spider_dir)
    items = loader.load_dev(limit=args.limit, start_index=args.start_index)
    requested_indices: Set[int] = {it["dataset_index"] for it in items}
    print(f"Loaded {len(items)} items (start={args.start_index}, limit={args.limit})")
    print(f"n={args.n}, drop-ratios={args.drop_ratios}, min-votes={args.min_votes}, "
          f"text-sim fallback: sqlglot={'yes' if _SQLGLOT_AVAILABLE else 'no (regex)'}")

    # --- checkpoint / resume（协议与 eval_vav / eval_maj_diag 一致） ---
    run_config = build_memfilter_run_config(args)
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
    print(f"Sampler ready. n-samples={args.n}, T={args.temperature}, seed={args.seed}\n")

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
                    [p for _, p in batch_prompts], n=args.n
                )
                print(
                    f"  batch gen {len(batch_prompts)} items x {args.n} "
                    f"in {time.perf_counter() - gen_start:.1f}s"
                )
                for (ds, _), cands in zip(batch_prompts, gen_results):
                    gen_map[ds] = cands
            except Exception as exc:
                print(f"[WARN] batch generation failed ({exc}); "
                      f"recording error items for this batch")
                for ds, _ in batch_prompts:
                    gen_map[ds] = []

        # --- 每题: 执行 + 质心过滤 + 基线/过滤后投票 ---
        for it in batch:
            ds = it["dataset_index"]
            if chat_texts.get(ds) is None:
                r = _new_item(it, args.drop_ratios)
                r["error"] = "ddl_load_failed"
            else:
                try:
                    r = process_item(
                        it, gen_map.get(ds, []), evaluator,
                        args.drop_ratios, args.min_votes, args.save_full_responses,
                    )
                except Exception as exc:
                    r = _new_item(it, args.drop_ratios)
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
                n_base = sum(
                    1 for it in all_items
                    if (it.get("baseline_vote") or {}).get("maj_correct")
                )
                print(
                    f"  [checkpoint {done}/{len(items)}] "
                    f"baseline maj={n_base}/{done} ({n_base / done:.1%}) "
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
        args.drop_ratios,
        evaluator,
        total_wall,
        datetime.now(timezone.utc).isoformat(),
    )
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # --- 打印结果 ---
    print("\n" + "=" * 66)
    print("  MEMORY CENTROID FILTER VOTING SUMMARY")
    print("=" * 66)
    print(f"  Items: {summary['total_completed']}/{summary['total_requested']} "
          f"(error={summary['error_items']})")
    print(f"  baseline maj@{summary['n']}:  "
          f"{summary['baseline']['maj_correct_count']} "
          f"({summary['baseline']['maj_accuracy']:.1%})")
    for key, ent in summary["filtered"].items():
        print(
            f"  drop {ent['drop_ratio']:.1%}: maj={ent['maj_correct_count']} "
            f"({ent['maj_accuracy']:.1%})  fixed={ent['fixed_count']} "
            f"broken={ent['broken_count']}  applied={ent['filter_applied_count']}/{summary['total_completed']}  "
            f"avg_dropped={ent['avg_dropped_count']} "
            f"avg_kept={ent['avg_kept_count']}"
        )
    print("  votes 分层 (votes = 基线多数组票数):")
    for bucket, s in summary["votes_strata"].items():
        f0 = s["filtered"][str(args.drop_ratios[0])]
        print(
            f"    votes={bucket:>3}  n={s['count']:<4}  "
            f"baseline={s['baseline_accuracy']:.1%}  "
            f"filtered({args.drop_ratios[0]})={f0['accuracy']:.1%} "
            f"(applied {f0['n_filter_applied']}/{s['count']})"
        )
    if summary["avg_pairwise_similarity"] is not None:
        print(f"  平均两两相似度:       {summary['avg_pairwise_similarity']}")
    print(f"  相似度对计数: 执行签名={summary['similarity_pair_counts']['execution_result_pairs']}, "
          f"文本Jaccard={summary['similarity_pair_counts']['text_jaccard_pairs']} "
          f"({summary['items_with_text_fallback']} 题使用文本回退)")
    print(f"  候选 parse 率:         {summary['candidate_parse_success_rate']:.1%}")
    print(f"  候选执行成功率:        {summary['candidate_execution_success_rate']:.1%}")
    print(f"  执行缓存命中率:        {summary['execution_cache_hit_ratio']:.1%}")
    print(f"  每题唯一 SQL 均值:     {summary['avg_unique_sql_per_item']}")
    print(f"  总耗时:                {summary['total_wall_seconds']:.0f}s")
    print("=" * 66)
    print(f"\nItems saved to:   {items_path}")
    print(f"Summary saved to: {summary_path}")
    print("\n官方 test-suite 评估（仅基线臂, predicted_sql）:")
    print(
        f"  bash scripts/eval_official.sh {items_path} "
        f"{output_dir / 'official_baseline'}"
    )


if __name__ == "__main__":
    main()
