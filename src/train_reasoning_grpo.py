#!/usr/bin/env python3
"""
GRPO training script for the Reasoning Generator Agent.

Trains Qwen2.5-Coder-3B-Instruct on Spider Text-to-SQL data
using Group Relative Policy Optimization (GRPO) with LoRA.

Usage:
    python train_reasoning_grpo.py \
        --num-train 100 \
        --num-generations 4 \
        --max-steps 50 \
        --output-dir /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/checkpoints/grpo_lora
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# Our own utilities — reuse spider_utils, do NOT rewrite.
# ---------------------------------------------------------------------------
from spider_utils import (
    SpiderLoader, DatabaseExecutor, compare_execution_results,
)
# Row-overlap helpers from spider_utils (tested there): ORDER BY detection is
# string-literal aware, and value normalization keeps float rounding consistent
# with the executor's own comparison logic.
from spider_utils import _has_order_by, _normalize_value_for_comparison, _rows_to_counter

# Atomic reward (FINER-SQL): dense structural credit via SQL parse Jaccard.
# Lazy-imported so sqlglot is NOT a hard dependency for non-atomic rewards.
_atomic_reward = None

def _get_atomic_reward():
    global _atomic_reward
    if _atomic_reward is None:
        from atomic_reward import AtomicOpsReward
        _atomic_reward = AtomicOpsReward(dialect="sqlite")
    return _atomic_reward

# Import agent's build_prompt and extract_sql for guaranteed consistency.
# Both are @staticmethod — no GPU needed.
from reasoning_generator_agent import ReasoningGeneratorAgent


# ===================================================================
# Paths (relative to this file: src/ → project root)
# ===================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"
MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "grpo_lora"


# ===================================================================
# SQL extraction (same regex logic as ReasoningGeneratorAgent)
# ===================================================================

def extract_sql(text: str) -> str:
    """Extract SQL from model completion using the agent's own parser.

    TRL passes completions as plain strings for string-prompts, but as
    list-of-dicts (messages) for chat-format prompts. Normalize first.
    """
    if isinstance(text, list):
        # Chat format: join the message contents
        parts = []
        for msg in text:
            if isinstance(msg, dict):
                c = msg.get("content", "")
                parts.append(c if isinstance(c, str) else str(c))
            else:
                parts.append(str(msg))
        text = "\n".join(parts)
    elif not isinstance(text, str):
        text = str(text)

    parsed = ReasoningGeneratorAgent.extract_sql(text)
    sql = parsed["sql"] if parsed["parse_success"] else ""
    # Take only the first statement (Model sometimes generates multiple).
    # Use the quote-aware splitter so ';' inside string literals survives.
    if sql:
        split_at = ReasoningGeneratorAgent._find_top_level_semicolon(sql)
        if split_at is not None:
            sql = sql[:split_at].strip() + ";"
        elif not sql.endswith(";"):
            sql = sql.strip() + ";"
    return sql


# ===================================================================
# Dataset builder
# ===================================================================

def build_dataset(spider_dir: str, limit: int = 100, filter_gold: bool = False) -> Dataset:
    """Build a HuggingFace Dataset from Spider train data.

    Each row contains:
        prompt   — the full instruction text (matching agent inference format)
        question — original NL question
        query    — gold SQL (for reward computation, NEVER shown to model)
        db_id    — database identifier (for SQLite execution)
        ddl      — DDL schema (informational, not used directly by reward func)

    filter_gold=True 剔除 gold SQL 无法在库上执行成功的样本
    （减少奖励噪声：多义/坏标注的题目会让模型学到错误模式）
    """
    loader = SpiderLoader(spider_dir)

    train_path = Path(spider_dir) / "train_spider.json"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    with open(train_path, "r", encoding="utf-8") as fh:
        train_data = json.load(fh)

    if limit > 0:
        train_data = train_data[:limit]

    records: List[Dict[str, str]] = []
    skipped = 0

    # 执行过滤用（懒加载，只有 filter_gold=True 才建 executor）
    executor = None
    if filter_gold:
        from spider_utils import DatabaseExecutor
        executor = DatabaseExecutor(spider_dir)

    for item in train_data:
        question: str = item["question"]
        query: str = item["query"]
        db_id: str = item["db_id"]

        # 可选：gold SQL 执行过滤（剔除执行失败的坏标注样本）
        if executor is not None:
            gold_out = executor.execute(db_id, query)
            if not gold_out["success"]:
                skipped += 1
                continue

        # Obtain DDL (prefer sqlite_master, fallback to tables.json)
        try:
            ddl, _source = loader.get_ddl_with_source(db_id)
        except RuntimeError:
            skipped += 1
            continue

        # Use agent's own build_prompt for 100 % consistency with inference
        prompt = ReasoningGeneratorAgent.build_prompt(
            question=question,
            ddl_schema=ddl,
            schema_links=None,
            evidence=None,
            dialect="sqlite",
        )

        # CRITICAL: store prompt as a CHAT MESSAGES LIST (not plain string)!
        # TRL's maybe_apply_chat_template() only applies the Qwen chat template
        # (with <|im_start|> markers) when the prompt is a list of message dicts.
        # A plain string is passed to the model verbatim, so the model never
        # sees the chat format → generates only ~32 tokens then stops (its
        # "chatty" behavior is gated on the chat markers). This was the root
        # cause of all short completions / failed GRPO runs.
        records.append({
            "prompt": [{"role": "user", "content": prompt}],
            "question": question,
            "query": query,
            "db_id": db_id,
            "ddl": ddl,
        })

    if skipped:
        print(f"⚠  Skipped {skipped} example(s) with missing DDL")
    print(f"Built dataset: {len(records)} records")

    return Dataset.from_list(records)


# ===================================================================
# Reward function factory
# ===================================================================

def _partial_row_overlap(
    pred_rows: List[List[Any]],
    gold_rows: List[List[Any]],
    gold_sql: str,
) -> float:
    """Fraction of gold rows matched by prediction (0.0–1.0).

    - If gold SQL has ORDER BY: rows must match positionally (order matters).
    - Otherwise: order-insensitive, duplicate-aware overlap via Counter
      (set operations over hashed row tuples — O(n), no nested loops).
    """
    gold_total = len(gold_rows)
    if gold_total == 0:
        return 0.0

    if _has_order_by(gold_sql):
        # Order matters — compare row-by-row in position
        overlap = 0
        for pr, gr in zip(pred_rows, gold_rows):
            pn = tuple(_normalize_value_for_comparison(v) for v in pr)
            gn = tuple(_normalize_value_for_comparison(v) for v in gr)
            if pn == gn:
                overlap += 1
        return overlap / gold_total

    # No ORDER BY — order-insensitive overlap, duplicate counts preserved
    gold_counter = _rows_to_counter(gold_rows)
    overlap = 0
    for pr in pred_rows:
        key = tuple(_normalize_value_for_comparison(v) for v in pr)
        if gold_counter.get(key, 0) > 0:
            gold_counter[key] -= 1
            overlap += 1
    return overlap / gold_total


def _references_schema_table(
    sql: str,
    gold_sql: str,
    db_id: str,
    executor: DatabaseExecutor,
) -> bool:
    """Anti-reward-hacking check: predicted SQL must reference a real table.

    A stub like ``SELECT 1`` or ``SELECT * FROM sqlite_master`` executes fine
    but answers nothing. We check that the SQL mentions at least one table
    name that also appears in the gold SQL (they share the same database),
    and that it is not just metadata-system-table access.
    """
    import re as _re

    # Extract table names from gold SQL (they are known-correct for this db)
    gold_tables = set(_re.findall(r"\bFROM\s+([a-zA-Z_]\w*)", gold_sql, _re.IGNORECASE))
    gold_tables |= set(_re.findall(r"\bJOIN\s+([a-zA-Z_]\w*)", gold_sql, _re.IGNORECASE))
    if not gold_tables:
        # Fallback: check against actual DB schema
        ddl_tables = set()
        try:
            from spider_utils import SpiderLoader
            loader = SpiderLoader(executor._database_dir.parent)
            ddl, _ = loader.get_ddl_with_source(db_id)
            ddl_tables = set(_re.findall(r"CREATE\s+TABLE\s+[`\"']?([a-zA-Z_]\w*)", ddl, _re.IGNORECASE))
        except Exception:
            pass
        gold_tables = ddl_tables

    # CTE gold (WITH x AS ...) — regex FROM only sees the CTE alias 'x',
    # so the intersection check would wrongly reject. In that case, fall
    # back to the DB schema (already computed above when gold_tables empty)
    # and only block obvious stubs.
    if _re.search(r"\bWITH\b", gold_sql, _re.IGNORECASE):
        pred_tables = set(_re.findall(r"\bFROM\s+([a-zA-Z_]\w*)", sql, _re.IGNORECASE))
        pred_tables |= set(_re.findall(r"\bJOIN\s+([a-zA-Z_]\w*)", sql, _re.IGNORECASE))
        # Reject only if the prediction references nothing at all (stub)
        return len(pred_tables) > 0

    pred_tables = set(_re.findall(r"\bFROM\s+([a-zA-Z_]\w*)", sql, _re.IGNORECASE))
    pred_tables |= set(_re.findall(r"\bJOIN\s+([a-zA-Z_]\w*)", sql, _re.IGNORECASE))

    # Must reference at least one gold table
    if not (pred_tables & gold_tables):
        return False
    return True


def create_reward_function(spider_dir: str, reward_type: str = "three_level"):
    """Return a reward function suitable for TRL's GRPOTrainer.

    reward_type:
        "binary"      — 1.0 exact match, 0.0 otherwise (original)
        "three_level" — 1.0 exact match, 0.1 executable but wrong, 0.0 invalid
        "partial"     — 1.0 exact match; executable but wrong gets fractional
                        credit: overlap/gold_rows when column counts match,
                        else a flat 0.2 executability bonus; 0.0 not executable
    """
    executor = DatabaseExecutor(spider_dir)  # read-only, safety-checked

    def reward_func(
        completions: List[str],
        query: List[str],
        db_id: List[str],
        **kwargs: Any,
    ) -> List[float]:
        rewards: List[float] = []

        for completion, gold_sql, db in zip(completions, query, db_id):
            sql = extract_sql(completion)
            if not sql:
                rewards.append(0.0)
                continue

            # ANTI-REWARD-HACKING: SQL must reference at least one table from
            # the gold SQL's database schema. Stub queries like "SELECT 1" or
            # "SELECT * FROM sqlite_master" that execute fine but answer nothing
            # get 0.0 even though they are executable (otherwise the model
            # learns to emit short stub SQL for the 0.1 executability bonus).
            if not _references_schema_table(sql, gold_sql, db, executor):
                rewards.append(0.0)
                continue

            # Execute both predicted and gold SQL
            pred_outcome = executor.execute(db, sql)
            gold_outcome = executor.execute(db, gold_sql)

            if pred_outcome["success"] and gold_outcome["success"]:
                # Use full_rows (NOT saved_rows) to match the evaluation
                # protocol exactly — saved_rows caps at 1000 and would
                # silently diverge from eval for >1000-row results.
                # Guard truncation like _evaluate_one does.
                if pred_outcome["full_rows_truncated"] or gold_outcome["full_rows_truncated"]:
                    rewards.append(0.0)
                    continue
                pred_rows = pred_outcome["full_rows"]
                gold_rows = gold_outcome["full_rows"]

                # CRITICAL: use the SAME comparison logic as evaluation
                # (compare_execution_results is ORDER BY-aware and
                # order-insensitive for unordered results — matches eval metric).
                exec_cmp = compare_execution_results(
                    pred_rows, gold_rows, gold_sql=gold_sql,
                )
                is_match = exec_cmp["match"]

                if is_match:
                    rewards.append(1.0)           # exact match (same as eval)
                elif reward_type == "three_level":
                    rewards.append(0.1)           # executable but wrong
                elif reward_type == "partial":
                    # Fractional credit from execution-result overlap
                    pred_cols = len(pred_rows[0]) if pred_rows else None
                    gold_cols = len(gold_rows[0]) if gold_rows else None
                    if pred_cols is not None and pred_cols == gold_cols:
                        rewards.append(
                            _partial_row_overlap(pred_rows, gold_rows, gold_sql)
                        )
                    else:
                        # Structurally different (or empty-gold) — small bonus
                        # for being executable
                        rewards.append(0.2)
                elif reward_type == "atomic":
                    # FINER-SQL atomic reward: Jaccard of parsed SQL structure
                    # (dense credit for "partially correct" SQL)
                    atomic_score = _get_atomic_reward().score_against_list(
                        sql, [gold_sql],
                    )
                    # Combine: 0.3 executability + 0.7 atomic (dense signal)
                    rewards.append(0.3 + 0.7 * atomic_score)
                else:
                    rewards.append(0.0)           # binary: wrong = 0

            elif pred_outcome["success"] and not gold_outcome["success"]:
                # Gold failed (rare) but pred works — give executability bonus
                if reward_type == "three_level":
                    rewards.append(0.1)
                elif reward_type == "partial":
                    rewards.append(0.2)
                else:
                    rewards.append(0.0)
            else:
                # Prediction failed, or both failed
                rewards.append(0.0)

        return rewards

    return reward_func


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    # ── GPU performance: TF32 + autotune ──
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        description="GRPO training for Reasoning Generator",
    )
    parser.add_argument(
        "--num-train", type=int, default=100,
        help="Number of Spider training examples (default: 100)",
    )
    parser.add_argument(
        "--num-generations", type=int, default=4,
        help="Number of completions per prompt for GRPO group (default: 4)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Maximum training steps (default: 50)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-6,
        help="Learning rate (default: 5e-6)",
    )
    parser.add_argument(
        "--beta", type=float, default=0.04,
        help="KL penalty coefficient (default: 0.04)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for generation (default: 0.7)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved LoRA adapter",
    )
    parser.add_argument(
        "--spider-dir", type=str, default=str(SPIDER_DIR),
        help="Root of Spider dataset",
    )
    parser.add_argument(
        "--model-path", type=str, default=str(MODEL_PATH),
        help="Path to local model",
    )
    parser.add_argument(
        "--reward-type", type=str, default="three_level",
        choices=["binary", "three_level", "partial", "atomic"],
        help="Reward function: binary (0/1), three_level (1.0/0.1/0.0), "
             "or partial (1.0 / row-overlap fraction / 0.2 executable / 0.0)",
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=None,
        help="per_device_train_batch_size 覆盖值（默认 num_generations*4；"
             "24GB 3090 训练 OOM 时调小到 8/4）",
    )
    parser.add_argument(
        "--lora-init", type=str, default=None,
        help="在已有 LoRA（如 SFT 冷启动）基础上继续 GRPO（真正的两阶段）",
    )
    parser.add_argument(
        "--filter-gold", action="store_true",
        help="剔除 gold SQL 执行失败的训练样本（减少奖励噪声）",
    )
    args = parser.parse_args()

    spider_dir = args.spider_dir
    model_path = args.model_path
    output_dir = args.output_dir

    # ------------------------------------------------------------------
    # 1. Build dataset
    # ------------------------------------------------------------------
    print(f"Building dataset from: {spider_dir}")
    dataset = build_dataset(spider_dir, limit=args.num_train,
                            filter_gold=args.filter_gold)
    print(f"Dataset: {len(dataset)} examples")
    print(f"First prompt length: {len(dataset[0]['prompt'])} chars")
    print(f"First db_id: {dataset[0]['db_id']}")

    # ------------------------------------------------------------------
    # 2. Load model & tokenizer
    # ------------------------------------------------------------------
    print(f"\nLoading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=True,
    )
    if args.lora_init:
        print(f"Loading existing LoRA (SFT cold-start): {args.lora_init}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_init, is_trainable=True)
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    # CRITICAL: Qwen has a default pad_token (<|endoftext|> id=151643) DIFFERENT
    # from eos_token (<|im_end|> id=151645), AND model.config.pad_token_id keeps
    # the OLD pad (151643) which is ALSO in eos_token_id=[151645, 151643].
    # Result: padding positions are treated as EOS → generation stops at ~29
    # tokens. Must unify BOTH tokenizer AND model config.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded. GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB")

    # ------------------------------------------------------------------
    # 3. LoRA configuration
    # ------------------------------------------------------------------
    # 已有 LoRA（lora-init）时不再新建 peft_config（TRL 不重复包装）
    lora_config = None
    if not args.lora_init:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    print(f"\nLoRA: {lora_config if lora_config else '(继承自 --lora-init 的适配器)'}")
    if lora_config:
        print(f"r={lora_config.r}, alpha={lora_config.lora_alpha}")
        print(f"Target modules: {lora_config.target_modules}")

    # ------------------------------------------------------------------
    # 4. GRPO configuration
    # ------------------------------------------------------------------
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        # 4x num_generations so each step processes 4 distinct prompts;
        # batch_size=num_generations would process only 1 prompt/step and
        # leave most of the dataset unseen within max_steps.
        # --train-batch-size overrides (24GB GPUs OOM with 4x on 500+ samples).
        per_device_train_batch_size=(
            args.train_batch_size if args.train_batch_size
            else args.num_generations * 4
        ),
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_steps=25,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_prompt_length=1536,
        max_completion_length=512,  # MUST match inference max_new_tokens (512)
        temperature=args.temperature,
        beta=args.beta,
        remove_unused_columns=False,   # CRITICAL: keep query, db_id for reward
        bf16=True,
        dataloader_num_workers=2,   # parallelize reward computation
        report_to="none",              # no wandb / tensorboard on HPC
    )

    print(f"\nGRPO: num_generations={args.num_generations}, "
          f"max_steps={args.max_steps}, lr={args.learning_rate}, "
          f"beta={args.beta}, temperature={args.temperature}")

    # ------------------------------------------------------------------
    # 5. Build reward function
    # ------------------------------------------------------------------
    reward_func = create_reward_function(spider_dir, reward_type=args.reward_type)
    if args.reward_type == "three_level":
        rtype_label = "three_level (1.0/0.1/0.0)"
    elif args.reward_type == "partial":
        rtype_label = "partial (1.0 / row-overlap / 0.2 executable / 0.0)"
    else:
        rtype_label = "binary (1.0/0.0)"
    print(f"\nReward function: {rtype_label}")

    # ------------------------------------------------------------------
    # 6. Trainer
    # ------------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,  # None → 使用 --lora-init 传入的适配器继续训练
    )

    print("\n" + "=" * 60)
    print("Starting GRPO training...")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
    except Exception as exc:
        print(f"\nTraining error: {type(exc).__name__}: {exc}")
        raise

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    print(f"\nSaving LoRA adapter to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Also save a small metadata file for traceability
    meta = {
        "base_model": model_path,
        "spider_dir": spider_dir,
        "num_train_examples": args.num_train,
        "num_generations": args.num_generations,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": list(lora_config.target_modules),
    }
    meta_path = Path(output_dir) / "training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"Metadata saved to: {meta_path}")
    print("\nDone! ✅")
    print(f"\nNext step: run evaluation with:")
    print(f"  python src/evaluate_after_grpo.py --lora-path {output_dir}")


if __name__ == "__main__":
    main()
