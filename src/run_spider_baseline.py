#!/usr/bin/env python3
"""
Run the Reasoning Generator Agent on a fixed slice of Spider dev.json
(pre-training baseline, no fine-tuning).

Produces:
    {output_dir}/
        checkpoint.json   — atomic per-item progress (includes run_config)
        items.json        — per-item detailed results
        summary.json      — aggregate metrics

Metrics are CUSTOM only — NOT official Spider EX or EM.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

# ---------------------------------------------------------------------------
# All checkpoint / config / validation helpers live in spider_utils so they
# can be tested without importing the agent (which requires torch / GPU).
# ---------------------------------------------------------------------------
from spider_utils import (  # noqa: E402
    SpiderLoader,
    DatabaseExecutor,
    compute_normalized_sql_string_match,
    compare_execution_results,
    compute_summary,
    # --- checkpoint & config ---
    EVALUATOR_TYPE,
    build_run_config,
    validate_resume_config,
    check_duplicate_indices,
    validate_checkpoint_integrity,
    validate_agent_candidate,
    save_checkpoint,
    load_checkpoint,
)


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spider dev first-N baseline (pre-training, custom metrics only)",
    )
    parser.add_argument(
        "--spider-dir", required=True,
        help="Root of the Spider dataset (contains dev.json, tables.json, database/).",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for checkpoint.json, items.json, summary.json.",
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Max number of dev.json items to process (default: 100).",
    )
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="0-based start index into dev.json (default: 0).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing checkpoint.json in --output-dir.",
    )
    parser.add_argument(
        "--model-path",
        default=(
            "/gpfs/work/aac/jiahuiwang24/"
            "reasoning_generator_3b/models/Qwen2.5-Coder-3B-Instruct"
        ),
        help="Path to the local Qwen2.5-Coder-3B-Instruct model.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Max tokens to generate per item (default: 512).",
    )
    return parser.parse_args()


# ===================================================================
# Official Spider evaluator check
# ===================================================================

def check_official_evaluator(spider_dir: Path) -> Dict[str, Any]:
    candidate_paths = [
        spider_dir / "evaluation.py",
        spider_dir / "evaluate.py",
        spider_dir / "test_suite_eval" / "evaluation.py",
        spider_dir.parent / "test_suite_eval" / "evaluation.py",
    ]
    found = None
    for p in candidate_paths:
        if p.exists():
            found = str(p)
            break
    return {
        "official_evaluator_found": found is not None,
        "official_evaluator_path": found,
        "recommendation": (
            "When ready for official metrics, use the Spider evaluator at "
            "https://github.com/taoyds/test-suite-sql-eval. "
            "Feed it predicted SQL files in the format it expects. "
            "Our custom_execution_match is for rapid pre-training baselines only."
        ),
    }


# ===================================================================
# Per-item failure result (used when an unexpected exception escapes)
# ===================================================================

def _make_failure_result(item: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
    return {
        "dataset_index": item["dataset_index"],
        "db_id": item.get("db_id", "unknown"),
        "question": item.get("question", ""),
        "gold_sql": item.get("query", ""),
        "raw_model_response": None,
        "predicted_sql": None,
        "parse_success": False,
        "parse_method": None,
        "prediction_execution_success": False,
        "gold_execution_success": False,
        "prediction_error": error_msg,
        "gold_error": None,
        "custom_execution_match": False,
        "normalized_sql_string_match": False,
        "match_reason": "unexpected_exception_in_process_one_item",
        "generation_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "ddl_source": None,
        "gold_rows_saved": [],
        "predicted_rows_saved": [],
        "gold_row_count": 0,
        "predicted_row_count": 0,
        "full_rows_truncated": False,
    }


# ===================================================================
# Per-item processing
# ===================================================================

def process_one_item(
    item: Dict[str, Any],
    agent: Any,
    db_executor: DatabaseExecutor,
    loader: SpiderLoader,
) -> Dict[str, Any]:
    """Run the full pipeline for one Spider dev item.  Each SQL executed once."""

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

    # -- Step 1: DDL --
    try:
        ddl_schema, ddl_source = loader.get_ddl_with_source(db_id)
        result["ddl_source"] = ddl_source
    except RuntimeError:
        result["prediction_error"] = f"Failed to get DDL for db_id={db_id}"
        result["match_reason"] = "ddl_load_failed"
        result["evaluation_seconds"] = round(time.perf_counter() - eval_start, 4)
        return result

    # -- Step 2: Execute gold SQL (once) --
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
        result["_gold_full_rows"] = []
        result["_gold_full_rows_truncated"] = False

    # -- Step 3: Agent generation --
    try:
        gen_result = agent.generate(
            question=question, ddl_schema=ddl_schema,
            schema_links=None, evidence=None, dialect="sqlite", candidate_count=1,
        )
    except Exception as exc:
        result["prediction_error"] = f"Agent generation failed: {exc}"
        result["match_reason"] = "agent_generation_exception"
        result["evaluation_seconds"] = round(time.perf_counter() - eval_start, 4)
        _cleanup_temp_keys(result)
        return result

    # -- Step 3b: Validate agent output structure --
    vc = validate_agent_candidate(gen_result)
    result["raw_model_response"] = vc["raw_response"]
    result["predicted_sql"] = vc["sql"]
    result["parse_success"] = vc["parse_success"]
    result["parse_method"] = vc["parse_method"]
    result["generation_seconds"] = round(vc["generation_seconds"], 4)

    if not vc["valid"]:
        result["prediction_error"] = f"Agent output validation failed: {vc['error']}"
        result["match_reason"] = "agent_output_invalid"
        result["evaluation_seconds"] = round(time.perf_counter() - eval_start, 4)
        _cleanup_temp_keys(result)
        return result

    if not vc["parse_success"]:
        result["prediction_error"] = result.get("prediction_error") or (
            "SQL extraction failed from model output"
        )
        result["match_reason"] = "sql_parse_failed"
        result["evaluation_seconds"] = round(time.perf_counter() - eval_start, 4)
        _cleanup_temp_keys(result)
        return result

    # -- Step 4: Execute predicted SQL (once) --
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
        result["_pred_full_rows"] = []
        result["_pred_full_rows_truncated"] = False

    # -- Step 5: Custom execution match --
    both_ok = result["gold_execution_success"] and result["prediction_execution_success"]
    any_truncated = (
        result.get("_gold_full_rows_truncated", False)
        or result.get("_pred_full_rows_truncated", False)
    )

    if both_ok:
        if any_truncated:
            result["custom_execution_match"] = False
            result["full_rows_truncated"] = True
            result["match_reason"] = (
                "Cannot compare: full rows truncated. "
                f"gold_truncated={result['_gold_full_rows_truncated']}, "
                f"pred_truncated={result['_pred_full_rows_truncated']}"
            )
        else:
            exec_cmp = compare_execution_results(
                result["_pred_full_rows"], result["_gold_full_rows"], gold_sql=gold_sql,
            )
            result["custom_execution_match"] = exec_cmp["match"]
            result["match_reason"] = exec_cmp["match_reason"]
    elif result["gold_execution_success"] and not result["prediction_execution_success"]:
        result["custom_execution_match"] = False
        result["match_reason"] = (
            f"Prediction execution failed: {result.get('prediction_error', 'unknown')}"
        )
    elif not result["gold_execution_success"] and result["prediction_execution_success"]:
        result["custom_execution_match"] = False
        result["match_reason"] = (
            f"Gold execution failed: {result.get('gold_error', 'unknown')}"
        )
    else:
        result["custom_execution_match"] = False
        result["match_reason"] = "Both prediction and gold execution failed"

    # -- Step 6: Normalized SQL string match --
    if result["parse_success"] and result["predicted_sql"]:
        str_cmp = compute_normalized_sql_string_match(
            result["predicted_sql"], gold_sql,
        )
        result["normalized_sql_string_match"] = str_cmp["match"]

    result["evaluation_seconds"] = round(time.perf_counter() - eval_start, 4)
    _cleanup_temp_keys(result)
    return result


def _cleanup_temp_keys(result: Dict[str, Any]) -> None:
    for key in (
        "_gold_full_rows", "_pred_full_rows",
        "_gold_full_rows_truncated", "_pred_full_rows_truncated",
    ):
        result.pop(key, None)


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    args = parse_args()
    spider_dir = Path(args.spider_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_config = build_run_config(
        spider_dir=args.spider_dir,
        start_index=args.start_index,
        limit=args.limit,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
    )

    # --- Official evaluator check ---
    eval_info = check_official_evaluator(spider_dir)
    print("=== Official Spider Evaluator Check ===")
    print(json.dumps(eval_info, indent=2))
    print()

    # --- Load dataset ---
    print(f"Loading Spider dev.json from: {spider_dir / 'dev.json'}")
    loader = SpiderLoader(str(spider_dir))
    items = loader.load_dev(limit=args.limit, start_index=args.start_index)
    requested_count = len(items)
    print(f"Requested {requested_count} items (start={args.start_index}, limit={args.limit})")
    if requested_count == 0:
        print("No items to process. Exiting.")
        sys.exit(0)

    requested_indices: Set[int] = {it["dataset_index"] for it in items}

    # --- Checkpoint / resume ---
    if args.resume:
        checkpoint = load_checkpoint(output_dir)
        stored_config = checkpoint.get("run_config")
        if stored_config is None:
            print(
                "ERROR: --resume requested but checkpoint has no run_config. "
                "This checkpoint was created by an older version of this script. "
                "Please start a fresh run without --resume."
            )
            sys.exit(1)

        validate_resume_config(stored_config, current_config)
        print("Run-config validation passed — resuming from checkpoint.")

        # Full integrity check
        validate_checkpoint_integrity(checkpoint, requested_indices)
        check_duplicate_indices(checkpoint["items"])
        print("Checkpoint integrity check passed.")
    else:
        checkpoint = {"completed_indices": set(), "items": [], "run_config": None}

    already_done: Set[int] = checkpoint.get("completed_indices", set())  # type: ignore[assignment]
    todo = [it for it in items if it["dataset_index"] not in already_done]
    print(f"Already completed: {len(already_done)}")
    print(f"Remaining: {len(todo)}")

    if not todo:
        print("All requested items already completed. Regenerating summary.")
        all_items: List[Dict[str, Any]] = checkpoint.get("items", [])  # type: ignore[assignment]
        summary = compute_summary(all_items, requested_indices=requested_indices)
        summary["total_wall_seconds"] = 0.0
        summary["evaluator_type"] = EVALUATOR_TYPE
        summary["is_official_spider_metric"] = False
        summary["official_evaluator"] = eval_info
        summary["generated_at"] = datetime.now(timezone.utc).isoformat()
        summary["requested_indices"] = sorted(requested_indices)
        summary["start_index"] = args.start_index
        summary["limit"] = args.limit
        _write_final(output_dir, all_items, summary)
        return

    # --- Initialize Agent (model loads once) ---
    print("\nInitializing Reasoning Generator Agent...")
    from reasoning_generator_agent import ReasoningGeneratorAgent  # noqa: E402
    agent = ReasoningGeneratorAgent(
        model_path=args.model_path, max_new_tokens=args.max_new_tokens,
    )
    print("Agent ready.\n")

    db_executor = DatabaseExecutor(str(spider_dir))

    # --- Process ---
    wall_start = time.perf_counter()
    all_items: List[Dict[str, Any]] = list(checkpoint.get("items", []))  # type: ignore[arg-type]

    for idx, item in enumerate(todo, start=1):
        di = item["dataset_index"]
        print(
            f"[{idx}/{len(todo)}] dataset_index={di}  "
            f"db_id={item['db_id']}  question={item['question'][:80]}..."
        )

        # ---- per-item safety net ----
        try:
            result = process_one_item(item, agent, db_executor, loader)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt — saving checkpoint and exiting.")
            save_checkpoint(output_dir, checkpoint, current_config)  # type: ignore[arg-type]
            raise
        except SystemExit:
            raise
        except Exception as exc:
            result = _make_failure_result(
                item,
                f"Unhandled exception in process_one_item: {type(exc).__name__}: {exc}",
            )
            print(f"  !! UNHANDLED EXCEPTION: {type(exc).__name__}: {exc}")

        # Summary line
        status_parts = []
        status_parts.append("parse✓" if result["parse_success"] else "parse✗")
        status_parts.append("pred_ex✓" if result["prediction_execution_success"] else "pred_ex✗")
        status_parts.append("match✓" if result["custom_execution_match"] else "match✗")
        print(
            f"  -> {' | '.join(status_parts)}  "
            f"gen={result['generation_seconds']:.1f}s  "
            f"eval={result['evaluation_seconds']:.3f}s  "
            f"ddl_src={result['ddl_source']}"
        )

        all_items.append(result)
        checkpoint["completed_indices"].add(di)  # type: ignore[union-attr]
        checkpoint["items"] = all_items  # type: ignore[index]
        save_checkpoint(output_dir, checkpoint, current_config)  # type: ignore[arg-type]

    wall_end = time.perf_counter()
    total_wall = round(wall_end - wall_start, 2)
    print(f"\nAll {len(todo)} items processed in {total_wall:.2f} s")

    # --- Summary ---
    summary = compute_summary(all_items, requested_indices=requested_indices)
    summary["total_wall_seconds"] = total_wall
    summary["evaluator_type"] = EVALUATOR_TYPE
    summary["is_official_spider_metric"] = False
    summary["official_evaluator"] = eval_info
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["requested_indices"] = sorted(requested_indices)
    summary["start_index"] = args.start_index
    summary["limit"] = args.limit

    _write_final(output_dir, all_items, summary)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _write_final(
    output_dir: Path,
    items: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    items_path = output_dir / "items.json"
    summary_path = output_dir / "summary.json"
    with open(items_path, "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"Items saved to: {items_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
