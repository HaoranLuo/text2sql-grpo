#!/usr/bin/env python3
"""
P1 评估入口：n 候选采样 → 执行验证 → vav 分组投票 → 输出 SQL（FINER-SQL 同口径）。

对应 PLAN §3.1（src/eval_vav_1034.py 的移植版，放在 finer_port/ 下不动 src/）：
  1. 加载 Spider dev（默认全量 1034，--limit/--start-index 可切子集），
     prompt 用单一 canonical prompt（默认 ReasoningGeneratorAgent.build_prompt，
     --prompt-style finer 为 FINER 模板对照臂）。
  2. 采样：HF transformers 路径，单次 forward 产 n=30 条（do_sample T=1.0 top_p=1.0，
     max_new_tokens=2048）；逐条独立 extract_sql（FINER 权重走 </think> 后内容）。
  3. 执行分组：本地 DatabaseExecutor + (db_id, normalize_sql) 缓存；
     vav 投票（跳过退化组）+ majority 不过滤对照臂。
  4. 输出：每题 selected_sql 写回 items.json 的 predicted_sql（保留 dataset_index
     结构，与 scripts/eval_official.sh 兼容），candidates 数组留作分析；
     checkpoint.json 断点续跑（1034×30 次生成必须可续）。
  5. summary.json：vav 自评 MV accuracy / majority 对照 / 难度拆分 easy/medium/hard/extra
     （对照 FINER 94.8/90.1/78.2/64.5）/ 训练同口径 compare_execution_results 对照。

用法（HPC 上建议用集群 python）：
    python finer_port/eval_vav.py \\
        --model-path models/FINER-SQL-3B-Spider \\
        --output-dir outputs/eval_vav_finer \\
        --n-samples 30 --limit 1034 --prompt-style finer
    # 完成后跑官方 test-suite：
    bash scripts/eval_official.sh outputs/eval_vav_finer/items.json outputs/official_vav_finer
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

# sys.path 引导：项目 src/（spider_utils）+ 本包
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
)
from sampler import VavSampler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval_vav"


# ===================================================================
# Run config（含 vav 专属字段，resume 时逐字段比对防混实验）
# ===================================================================

def build_vav_run_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = build_run_config(
        spider_dir=args.spider_dir,
        start_index=args.start_index,
        limit=args.limit,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
    )
    cfg.update(
        {
            "n_samples": args.n_samples,
            "prompt_style": args.prompt_style,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "lora_path": args.lora_path,
        }
    )
    return cfg


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
        "selected_sql": None,
        "selected_sql_index": -1,
        "chosen_strategy": None,
        "vav_result": None,
        "vav_group_size": 0,
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
    执行验证 + vav 投票，产出单个评估条目。

    candidates: VavSampler 输出的候选 dict 列表（每条含 raw_response/sql/
    parse_success/parse_method）。
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
            {"sql": c["sql"], "result": outcome, "index": c["candidate_id"]}
        )
    gold_outcome = evaluator.execute_cached(db_id, gold_sql)

    # 2) vav 投票（含 majority 不过滤对照）
    vote = run_vav_voting(
        pred_results, gold_outcome, gt_sql=gold_sql, strategy="vav"
    )

    selected_sql = vote["selected_sql"]
    r["predicted_sql"] = selected_sql or None
    r["selected_sql"] = selected_sql or None
    r["selected_sql_index"] = vote["selected_sql_index"]
    r["chosen_strategy"] = vote["chosen_strategy"]
    r["vav_result"] = vote["chosen_result"]
    r["vav_group_size"] = vote["chosen_group_size"]
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
    # 组元数据（含组内 SQL 与正确性标注，供分析）
    r["result_groups"] = vote["result_groups"]

    # 3) candidates 数组（留作分析；默认只存 raw_response 前 200 字符预览，
    #    --save-full-responses 时存全文）
    for c, pr in zip(candidates, pred_results):
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
        }
        if args.save_full_responses:
            entry["raw_response"] = c.get("raw_response", "")
        else:
            entry["raw_response_preview"] = (c.get("raw_response") or "")[:200]
        r["candidates"].append(entry)

    # 4) 训练同口径对照：compare_execution_results（ORDER BY 感知 multiset，
    #    PLAN §8「奖励/评估口径漂移」要求三方口径都记录）
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
    # P2 修复：generation_seconds 落盘（采样+执行+投票总耗时，供 summary 统计）
    r["generation_seconds"] = round(time.perf_counter() - t0, 4)
    return r


# ===================================================================
# Summary
# ===================================================================

def _difficulty_split(all_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[str, Dict[str, int]] = {}
    for it in all_items:
        d = str(it.get("difficulty", "unknown"))
        s = by.setdefault(d, {"count": 0, "vav_self_match": 0, "majority_self_match": 0})
        s["count"] += 1
        if it.get("vav_self_match"):
            s["vav_self_match"] += 1
        if it.get("majority_self_match"):
            s["majority_self_match"] += 1
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
        }
    return out


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
    n_custom = sum(
        1 for it in all_items if it.get("selected_custom_exec_match") is True
    )
    n_custom_scored = sum(
        1 for it in all_items if it.get("selected_custom_exec_match") is not None
    )

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
        "evaluator_type": "vav_execution_voting",
        "is_official_spider_metric": False,
        "note": (
            "vav 自评口径 = header-agnostic 行集合相等（FINER majority_voting 同款）；"
            "官方 test-suite EX 需 bash scripts/eval_official.sh <items.json> <out_dir>；"
            "自评与官方差 ~0.9pp 属正常（官方多实例执行 + ORDER BY 更严格）。"
        ),
        "total_requested": len(requested_indices),
        "total_completed": total,
        "vav_self_match_count": n_vav,
        "vav_self_match_rate": round(n_vav / total, 4) if total else 0.0,
        "majority_self_match_count": n_maj,
        "majority_self_match_rate": round(n_maj / total, 4) if total else 0.0,
        "degenerate_skip_count": n_skip,
        "degenerate_skip_rate": round(n_skip / total, 4) if total else 0.0,
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
            "P1 vav 投票评估：n 候选采样 → 执行验证 → vav 分组投票 → 输出 SQL "
            "（FINER-SQL 同口径，与 scripts/eval_official.sh 兼容）"
        ),
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH),
                        help="本地模型路径（FINER 权重或本项目 3B 基座/LoRA 合并后权重）")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="输出目录（checkpoint.json / items.json / summary.json）")
    parser.add_argument("--n-samples", type=int, default=30,
                        help="每题采样候选数（默认 30，FINER 官方口径）")
    parser.add_argument("--limit", type=int, default=1034,
                        help="评估条数（默认全量 dev 1034；子集先跑 100-200 看增益方向）")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--spider-dir", default=str(DEFAULT_SPIDER_DIR))
    parser.add_argument("--lora-path", default=None,
                        help="LoRA adapter 路径（可选；不加则跑基座/FINER 权重）")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="与训练 max_completion 对齐（PLAN §8 生成长度一致性）")
    parser.add_argument("--prompt-style", choices=["default", "finer"], default="default",
                        help="default=项目 canonical prompt；finer=FINER 系统提示模板对照臂")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="每次 forward 处理的题目数（batch>1 时一次 forward 产 B*n 条，更省显存但更慢）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="每处理 N 题写一次 checkpoint（断点续跑）")
    parser.add_argument("--save-full-responses", action="store_true",
                        help="candidates 里存完整 raw_response（默认只存 200 字符预览）")
    parser.add_argument("--allow-remote", action="store_true",
                        help="允许从 HF 在线拉取权重（默认 local_files_only）")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 加载数据集 ---
    loader = SpiderLoader(args.spider_dir)
    items = loader.load_dev(limit=args.limit, start_index=args.start_index)
    requested_indices: Set[int] = {it["dataset_index"] for it in items}
    print(f"Loaded {len(items)} items (start={args.start_index}, limit={args.limit})")

    # --- checkpoint / resume（协议与 evaluate_after_grpo 一致） ---
    run_config = build_vav_run_config(args)
    cp = load_checkpoint(output_dir)
    if cp["run_config"] is not None:
        validate_resume_config(cp["run_config"], run_config)
    validate_checkpoint_integrity(cp, requested_indices)
    completed: Set[int] = cp["completed_indices"]
    all_items: List[Dict[str, Any]] = list(cp["items"])
    print(
        f"Resume: {len(completed)}/{len(requested_indices)} items already completed"
    )

    # --- 加载模型 + 执行器 ---
    print(f"\nLoading model: {args.model_path}")
    sampler = VavSampler(
        model_path=args.model_path,
        lora_path=args.lora_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_style=args.prompt_style,
        seed=args.seed,
        local_files_only=not args.allow_remote,
    )
    evaluator = VavEvaluator(DatabaseExecutor(args.spider_dir))
    print(f"Sampler ready. n-samples={args.n_samples}, style={args.prompt_style}\n")

    # --- 预构造 prompt（DDL 失败条目记为 error 条目，不阻塞） ---
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
                    [p for _, p in batch_prompts], n=args.n_samples
                )
                print(
                    f"  batch gen {len(batch_prompts)} items x {args.n_samples} "
                    f"in {time.perf_counter() - gen_start:.1f}s"
                )
                for (ds, _), cands in zip(batch_prompts, gen_results):
                    gen_map[ds] = cands
            except Exception as exc:
                print(f"[WARN] batch generation failed ({exc}); "
                      f"recording error items for this batch")
                for ds, _ in batch_prompts:
                    gen_map[ds] = []

        # --- 每题：执行验证 + vav 投票 ---
        for it in batch:
            ds = it["dataset_index"]
            if chat_texts.get(ds) is None:
                r = _new_item(it)
                r["error"] = "ddl_load_failed"
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
            print(
                f"  [checkpoint {done}/{len(items)}] "
                f"vav_self={n_vav}/{done} ({n_vav / done:.1%}) "
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
    print("  P1 VAV VOTING EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Items:              {summary['total_completed']}/{summary['total_requested']}")
    print(f"  vav self MV:        {summary['vav_self_match_count']} "
          f"({summary['vav_self_match_rate']:.1%})")
    print(f"  majority (control): {summary['majority_self_match_count']} "
          f"({summary['majority_self_match_rate']:.1%})  [不过滤对照]")
    print(f"  degenerate-skip 影响题数: {summary['degenerate_skip_count']} "
          f"({summary['degenerate_skip_rate']:.1%})")
    print(f"  custom exec match:  {summary['selected_custom_exec_match_count']} "
          f"({summary['selected_custom_exec_match_rate']:.1%})  [训练同口径]")
    print(f"  候选 parse 率:      {summary['candidate_parse_success_rate']:.1%}")
    print(f"  候选执行成功率:     {summary['candidate_execution_success_rate']:.1%}")
    print(f"  执行缓存命中率:     {summary['execution_cache_hit_ratio']:.1%}")
    print(f"  每题唯一 SQL 均值:  {summary['avg_unique_sql_per_item']}")
    print(f"  难度拆分:")
    for d, s in summary["difficulty_split"].items():
        print(f"    {d:<8} n={s['count']:<5} vav={s['vav_self_match_rate']:.1%} "
              f"majority={s['majority_self_match_rate']:.1%}")
    print(f"  总耗时:             {summary['total_wall_seconds']:.0f}s")
    print("=" * 60)
    print(f"\nItems saved to:   {items_path}")
    print(f"Summary saved to: {summary_path}")
    print("\n官方 test-suite 评估（下一步）:")
    print(
        f"  bash scripts/eval_official.sh {items_path} "
        f"{output_dir / 'official'}"
    )


if __name__ == "__main__":
    main()
