#!/usr/bin/env python3
"""
Post-GRPO evaluation: run the trained model on the same 100 Spider dev items
that were used for the pre-training baseline.

Produces:
    outputs/grpo_post_train_100/
        checkpoint.json
        items.json
        summary.json

Compare with:
    outputs/baseline_pretrain_100/summary.json

Usage:
    python src/evaluate_after_grpo.py \
        --lora-path checkpoints/grpo_lora \
        --limit 100
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import torch

# Optional: prettier progress bars
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "grpo_post_train_100"

# ---------------------------------------------------------------------------
# Reuse everything from spider_utils and the agent
# ---------------------------------------------------------------------------
from spider_utils import (  # noqa: E402
    SpiderLoader,
    DatabaseExecutor,
    compute_normalized_sql_string_match,
    compare_execution_results,
    compute_summary,
    EVALUATOR_TYPE,
    build_run_config,
    validate_resume_config,
    check_duplicate_indices,
    validate_checkpoint_integrity,
    validate_agent_candidate,
    save_checkpoint,
    load_checkpoint,
)
from reasoning_generator_agent import ReasoningGeneratorAgent  # noqa: E402


# ===================================================================
# Per-item processing ( mirrors run_spider_baseline.process_one_item )
# ===================================================================

def _evaluate_one(
    item: Dict[str, Any],
    gen_result: Dict[str, Any],
    db_executor: DatabaseExecutor,
    loader: SpiderLoader,
) -> Dict[str, Any]:
    """Run SQL execution + comparison for one item, using pre-computed gen_result."""
    dataset_index: int = item["dataset_index"]
    db_id: str = item["db_id"]
    question: str = item["question"]
    gold_sql: str = item["query"]

    result: Dict[str, Any] = {
        "dataset_index": dataset_index, "db_id": db_id,
        "question": question, "gold_sql": gold_sql,
        "raw_model_response": None, "predicted_sql": None,
        "parse_success": False, "parse_method": None,
        "prediction_execution_success": False, "gold_execution_success": False,
        "prediction_error": None, "gold_error": None,
        "custom_execution_match": False, "normalized_sql_string_match": False,
        "match_reason": None, "generation_seconds": 0.0, "evaluation_seconds": 0.0,
        "ddl_source": None,
        "gold_rows_saved": [], "predicted_rows_saved": [],
        "gold_row_count": 0, "predicted_row_count": 0,
        "full_rows_truncated": False,
    }

    eval_start = time.perf_counter()

    # DDL source
    try:
        _ddl, ddl_source = loader.get_ddl_with_source(db_id)
        result["ddl_source"] = ddl_source
    except RuntimeError:
        result["prediction_error"] = f"Failed to get DDL for db_id={db_id}"
        result["match_reason"] = "ddl_load_failed"
        result["evaluation_seconds"] = round(time.perf_counter() - eval_start, 4)
        return result

    # Execute gold SQL
    gold_outcome = db_executor.execute(db_id, gold_sql)
    if gold_outcome["success"]:
        result["gold_execution_success"] = True
        result["gold_rows_saved"] = gold_outcome["saved_rows"]
        result["gold_row_count"] = gold_outcome["row_count"]
        result["_gold_full_rows"] = gold_outcome["full_rows"]
        result["_gold_full_rows_truncated"] = gold_outcome["full_rows_truncated"]
    else:
        result["gold_execution_success"] = False
        result["gold_error"] = gold_outcome["error"]

    # Validate agent output
    vc = validate_agent_candidate(gen_result)
    result["raw_model_response"] = vc["raw_response"]
    result["predicted_sql"] = vc["sql"]
    # Take only the first statement (model sometimes outputs multiple SQL).
    # Quote-aware: ';' inside string literals must survive.
    if result["predicted_sql"]:
        from reasoning_generator_agent import ReasoningGeneratorAgent
        split_at = ReasoningGeneratorAgent._find_top_level_semicolon(result["predicted_sql"])
        if split_at is not None:
            result["predicted_sql"] = result["predicted_sql"][:split_at].strip() + ";"
        elif not result["predicted_sql"].endswith(";"):
            result["predicted_sql"] = result["predicted_sql"].strip() + ";"
    result["parse_success"] = vc["parse_success"]
    result["parse_method"] = vc["parse_method"]
    result["generation_seconds"] = round(vc["generation_seconds"], 4)

    if not vc["valid"] or not vc["parse_success"]:
        result["prediction_error"] = "SQL extraction failed" if vc["valid"] else f"Invalid: {vc['error']}"
        result["match_reason"] = "sql_parse_failed"
        result["evaluation_seconds"] = round(time.perf_counter() - eval_start, 4)
        _cleanup(result)
        return result

    # Execute predicted SQL
    pred_outcome = db_executor.execute(db_id, result["predicted_sql"])
    if pred_outcome["success"]:
        result["prediction_execution_success"] = True
        result["predicted_rows_saved"] = pred_outcome["saved_rows"]
        result["predicted_row_count"] = pred_outcome["row_count"]
        result["_pred_full_rows"] = pred_outcome["full_rows"]
        result["_pred_full_rows_truncated"] = pred_outcome["full_rows_truncated"]
    else:
        result["prediction_execution_success"] = False
        result["prediction_error"] = pred_outcome["error"]

    # Custom execution match
    both_ok = result["gold_execution_success"] and result["prediction_execution_success"]
    any_truncated = (
        result.get("_gold_full_rows_truncated", False)
        or result.get("_pred_full_rows_truncated", False)
    )

    if both_ok:
        if any_truncated:
            result["custom_execution_match"] = False
            result["full_rows_truncated"] = True
            result["match_reason"] = "Cannot compare: full rows truncated."
        else:
            exec_cmp = compare_execution_results(
                result["_pred_full_rows"], result["_gold_full_rows"], gold_sql=gold_sql,
            )
            result["custom_execution_match"] = exec_cmp["match"]
            result["match_reason"] = exec_cmp["match_reason"]
    elif result["gold_execution_success"] and not result["prediction_execution_success"]:
        result["custom_execution_match"] = False
        result["match_reason"] = f"Prediction execution failed: {result.get('prediction_error')}"
    elif not result["gold_execution_success"] and result["prediction_execution_success"]:
        result["custom_execution_match"] = False
        result["match_reason"] = f"Gold execution failed: {result.get('gold_error')}"
    else:
        result["custom_execution_match"] = False
        result["match_reason"] = "Both prediction and gold execution failed"

    # Normalized SQL string match
    if result["parse_success"] and result["predicted_sql"]:
        str_cmp = compute_normalized_sql_string_match(
            result["predicted_sql"], gold_sql,
        )
        result["normalized_sql_string_match"] = str_cmp["match"]

    result["evaluation_seconds"] = round(time.perf_counter() - eval_start, 4)
    _cleanup(result)
    return result


def _cleanup(result: Dict[str, Any]) -> None:
    for key in (
        "_gold_full_rows", "_pred_full_rows",
        "_gold_full_rows_truncated", "_pred_full_rows_truncated",
    ):
        result.pop(key, None)


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GRPO-trained model on Spider dev (same slice as baseline)",
    )
    parser.add_argument(
        "--lora-path", default=None,
        help="Path to the saved LoRA adapter directory (omit for baseline)",
    )
    parser.add_argument(
        "--spider-dir", default=str(DEFAULT_SPIDER_DIR),
        help="Root of the Spider dataset",
    )
    parser.add_argument(
        "--model-path", default=str(DEFAULT_MODEL_PATH),
        help="Path to the base model",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output files",
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Number of dev.json items to evaluate (default: 100)",
    )
    parser.add_argument(
        "--start-index", type=int, default=0,
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Number of items to process in one GPU forward pass (default: 8). "
             "Higher = faster but more VRAM. A40 48GB can handle 8-16.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    loader = SpiderLoader(args.spider_dir)
    items = loader.load_dev(limit=args.limit, start_index=args.start_index)
    requested_indices: Set[int] = {it["dataset_index"] for it in items}
    print(f"Evaluating {len(items)} items (start={args.start_index}, limit={args.limit})")
    print(f"Batch size: {args.batch_size}")

    # --- Load agent with LoRA ---
    lora_path = args.lora_path if args.lora_path else None
    print(f"\nLoading base model: {args.model_path}")
    if lora_path:
        print(f"Loading LoRA adapter: {lora_path}")
    else:
        print("No LoRA adapter — running baseline inference")
    agent = ReasoningGeneratorAgent(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        lora_path=lora_path,
    )
    print(f"Agent ready. GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB\n")

    db_executor = DatabaseExecutor(args.spider_dir)

    # --- Pre-build all request dicts (DDL loading) ---
    print("Loading DDL for all items...")
    gen_requests: List[Dict[str, Any]] = []
    items_with_ddl: List[Dict[str, Any]] = []

    for item in items:
        try:
            ddl_schema, ddl_source = loader.get_ddl_with_source(item["db_id"])
        except RuntimeError:
            # Skip items with missing DDL — rare
            continue
        gen_requests.append({
            "question": item["question"],
            "ddl_schema": ddl_schema,
            "schema_links": None,
            "evidence": None,
            "dialect": "sqlite",
        })
        items_with_ddl.append(item)

    n_items = len(gen_requests)
    print(f"Ready to evaluate {n_items} items (DDL loaded)\n")

    # --- Process in batches ---
    wall_start = time.perf_counter()
    all_items: List[Dict[str, Any]] = []
    num_batches = (n_items + args.batch_size - 1) // args.batch_size

    # Use tqdm if available, otherwise simple counter
    batch_iter = range(0, n_items, args.batch_size)
    if _HAS_TQDM:
        batch_iter = tqdm(
            batch_iter,
            desc="Evaluating",
            unit="batch",
            total=num_batches,
            ncols=100,
        )
    else:
        print(f"Processing {n_items} items in {num_batches} batches...")

    match_count = 0
    parse_count = 0
    exec_count = 0

    for batch_start in batch_iter:
        batch_end = min(batch_start + args.batch_size, n_items)
        batch_requests = gen_requests[batch_start:batch_end]
        batch_items = items_with_ddl[batch_start:batch_end]

        # ── Batched GPU generation (fast!) ──
        try:
            gen_results = agent.generate_batch(batch_requests)
        except Exception as exc:
            # Fallback: individual processing on batch failure
            if _HAS_TQDM:
                batch_iter.write(f"  ⚠ Batch failed ({exc}), falling back to single...")
            gen_results = []
            for req in batch_requests:
                try:
                    gen_results.append(agent.generate(**req))
                except Exception as exc2:
                    gen_results.append({
                        "candidates": [{
                            "candidate_id": 0,
                            "raw_response": "",
                            "sql": "",
                            "parse_success": False,
                            "parse_method": None,
                        }],
                        "metadata": {"generation_seconds": 0.0},
                    })

        # ── Per-item evaluation (CPU/SQLite, fast) ──
        for item, gen_result in zip(batch_items, gen_results):
            result = _evaluate_one(item, gen_result, db_executor, loader)
            all_items.append(result)

            if result["parse_success"]:
                parse_count += 1
            if result["prediction_execution_success"]:
                exec_count += 1
            if result["custom_execution_match"]:
                match_count += 1

        # Update progress description
        if _HAS_TQDM:
            n_done = len(all_items)
            batch_iter.set_postfix(
                parse=f"{parse_count}/{n_done}",
                exec=f"{exec_count}/{n_done}",
                match=f"{match_count}/{n_done}",
                rate=f"{match_count/n_done:.0%}" if n_done else "-",
            )

        # Periodic checkpoint
        if (batch_start // args.batch_size + 1) % 5 == 0:
            cp = {"completed_indices": sorted([r["dataset_index"] for r in all_items]), "items": all_items}
            cp_path = output_dir / "checkpoint.json"
            tmp_path = output_dir / "checkpoint.json.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(cp, fh, ensure_ascii=False, indent=2)
            os.replace(str(tmp_path), str(cp_path))

    wall_end = time.perf_counter()
    total_wall = round(wall_end - wall_start, 2)
    rate = n_items / total_wall if total_wall > 0 else 0
    print(f"\nAll {len(all_items)} items processed in {total_wall:.2f}s ({rate:.1f} items/s)")

    # --- Summary ---
    summary = compute_summary(all_items, requested_indices=requested_indices)
    summary["total_wall_seconds"] = total_wall
    summary["evaluator_type"] = EVALUATOR_TYPE
    summary["is_official_spider_metric"] = False
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["requested_indices"] = sorted(requested_indices)
    summary["start_index"] = args.start_index
    summary["limit"] = args.limit
    summary["lora_path"] = args.lora_path

    # --- Save ---
    items_path = output_dir / "items.json"
    summary_path = output_dir / "summary.json"
    with open(items_path, "w", encoding="utf-8") as fh:
        json.dump(all_items, fh, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\nItems saved to: {items_path}")
    print(f"Summary saved to: {summary_path}")

    # --- Print summary ---
    print("\n" + "=" * 50)
    print("  POST-TRAINING EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Parse success:         {summary['parse_success_count']}/{summary['total_completed']} ({summary['parse_success_rate']:.1%})")
    print(f"  Execution success:     {summary['prediction_execution_success_count']}/{summary['total_completed']} ({summary['prediction_execution_success_rate']:.1%})")
    print(f"  Execution match:       {summary['custom_execution_match_count']}/{summary['total_completed']} ({summary['custom_execution_match_rate']:.1%})")
    print(f"  SQL string match:      {summary['normalized_sql_string_match_count']}/{summary['total_completed']} ({summary['normalized_sql_string_match_rate']:.1%})")
    print(f"  Avg generation time:   {summary['average_generation_seconds']:.2f}s")
    print(f"  Total wall time:       {summary['total_wall_seconds']:.2f}s")
    print("=" * 50)

    # --- Comparison hint ---
    baseline_path = PROJECT_ROOT / "outputs" / "baseline_pretrain_100" / "summary.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as fh:
            baseline = json.load(fh)
        print("\n📊 COMPARISON (before vs after GRPO):")
        print(f"  Baseline (pre-train):  {baseline['custom_execution_match_rate']:.1%}")
        print(f"  After GRPO:            {summary['custom_execution_match_rate']:.1%}")
        diff = summary['custom_execution_match_rate'] - baseline['custom_execution_match_rate']
        direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        print(f"  Change:                {diff:+.1%} {direction}")
    else:
        print(f"\n⚠  Baseline not found at {baseline_path} — cannot compute delta.")


if __name__ == "__main__":
    main()
