#!/usr/bin/env python3
"""
FINER Step-1 style reasoning distillation generator for Spider (DeepSeek API).

Reads the first N Spider train examples, calls DeepSeek (--model deepseek-chat
or deepseek-reasoner) with a "think first, then answer" prompt, and appends one
JSON line per example to a JSONL file, for later SFT / GRPO cold-start use.

The API response is stored raw; it is expected to look like:

    <think>...reasoning...</think>
    ```sql
    SELECT ...
    ```

Each output line contains at minimum:
    index / question / db_id / gold_sql / response / success
plus model, error, the exact messages sent, ddl, token usage and timestamp.

Features
--------
- Asynchronous concurrency (--concurrency, asyncio + AsyncOpenAI) with an
  optional per-call delay (--sleep) and exponential-backoff retries
  (--max-retries) for rate limiting / transient failures.
- Resume: on re-run, indices already present in the output JSONL are skipped.
  --retry-failed re-runs only the previously failed indices.
- Cost estimate printed before the run; actual cost report (from API usage)
  printed after.
- --dry-run builds the full plan (DDL + prompts + token/cost estimate) without
  calling the API - useful on machines without an API key.

Usage
-----
    export DEEPSEEK_API_KEY=sk-xxx
    python src/gen_reasoning_data.py --model deepseek-reasoner --num-train 1000
    python src/gen_reasoning_data.py --dry-run              # no API calls

Notes
-----
- Default input paths point to the HPC cluster; override with --train-file /
  --spider-dir when running elsewhere (local Spider dirs also work).
- deepseek-reasoner does not support `temperature`; it is omitted automatically
  and the model's hidden chain-of-thought is recovered via `reasoning_content`
  and wrapped as <think>...</think> (same trick as FINER's
  model_pool_generate.py). deepseek-chat is instructed to emit the <think> and
  ```sql blocks itself.
- DDL comes from the Spider database files (sqlite_master) with a tables.json
  fallback, via spider_utils.SpiderLoader.
- Output lines are written as tasks complete, so line order may not be
  sequential (each line carries its own `index`). Do NOT run two instances
  with the same --output at the same time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Make `spider_utils` importable no matter where this script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spider_utils import SpiderLoader

try:
    import tiktoken  # optional, only used for the pre-run token estimate
except Exception:
    tiktoken = None

try:
    from openai import AsyncOpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    AsyncOpenAI = None
    _OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants / config
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_FILE = (
    "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/"
    "data/spider_data/train_spider.json"
)
DEFAULT_SPIDER_DIR = (
    "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data"
)
DEFAULT_BASE_URL = "https://api.deepseek.com"
MODEL_CHOICES = ("deepseek-chat", "deepseek-reasoner")

# DeepSeek pricing, USD per 1M tokens (standard price, cache-miss default).
# Prices change - verify at https://api-docs.deepseek.com/quick_start/pricing
PRICING_USD_PER_M: Dict[str, Dict[str, float]] = {
    "deepseek-chat": {"input_miss": 0.27, "input_hit": 0.07, "output": 1.10},
    "deepseek-reasoner": {"input_miss": 0.55, "input_hit": 0.14, "output": 2.19},
}
ASSUMED_AVG_OUTPUT_TOKENS = 600  # used by the pre-run cost estimate only

# ---------------------------------------------------------------------------
# Prompts (mimic FINER model_pool_generate.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_CHAT = """You are a meticulous SQL expert. Given a database schema and a natural-language question, think step by step about which tables, columns, joins, filters and aggregations are needed, then produce exactly one valid SQLite SQL query.

Follow this exact response format:
<think>
[your step-by-step reasoning]
</think>
```sql
[the SQL query]
```

Rules:
- Output exactly one SQL statement.
- The SQL must be executable on SQLite.
- Use only tables and columns that appear in the schema.
- Do not invent database objects.
- Do not include any explanatory text outside the <think> section."""

SYSTEM_PROMPT_REASONER = """You are a meticulous SQL expert. Given a database schema and a natural-language question, generate a single, correct SQL query.

Rules:
- Output exactly one SQL statement.
- The SQL must be executable on SQLite.
- Use only tables and columns that appear in the schema.
- Do not invent database objects.
- Put the final SQL inside a ```sql code block and output nothing else."""


def build_user_prompt(ddl: str, question: str) -> str:
    return (
        "Database Schema:\n"
        f"{ddl}\n\n"
        f"Question: {question}\n\n"
        "Think step by step, then output <think>...</think> followed by the SQL "
        "in a ```sql code block."
    )


def build_messages(model: str, ddl: str, question: str) -> List[Dict[str, str]]:
    system = SYSTEM_PROMPT_REASONER if model == "deepseek-reasoner" else SYSTEM_PROMPT_CHAT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_prompt(ddl, question)},
    ]


# ---------------------------------------------------------------------------
# Token estimation (pre-run only; real counts come from API usage)
# ---------------------------------------------------------------------------

_ENC: Any = None


def _get_encoding():
    global _ENC
    if _ENC is None and tiktoken is not None:
        try:
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENC = False
    return _ENC if _ENC else None


def estimate_tokens(text: str) -> int:
    enc = _get_encoding()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)  # fallback heuristic


# ---------------------------------------------------------------------------
# DeepSeek call (async, retry with exponential backoff)
# ---------------------------------------------------------------------------


async def call_deepseek(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[str], Any, Optional[Exception]]:
    """One chat completion. Returns (text, reasoning_content, usage, error).

    For deepseek-reasoner the composed response is
    '<think>\\n<reasoning>\\n</think>\\n<answer>' (FINER convention).
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
    }
    if model != "deepseek-reasoner":
        kwargs["temperature"] = args.temperature  # reasoner does not support it

    last_error: Optional[Exception] = None
    for attempt in range(args.max_retries + 1):
        try:
            resp = await client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                text = f"<think>\n{reasoning}\n</think>\n{content}"
            else:
                text = content
            return text, reasoning, resp.usage, None
        except Exception as exc:  # network / 429 / 5xx / anything
            last_error = exc
            if attempt < args.max_retries:
                wait = min(30.0, 2.0 ** attempt) * (1 + random.random() * 0.25)
                await asyncio.sleep(wait)
    return None, None, None, last_error


def _usage_int(usage: Any, field: str) -> Optional[int]:
    if usage is None:
        return None
    val = getattr(usage, field, None)
    return val if isinstance(val, int) else None


# ---------------------------------------------------------------------------
# Records / resume
# ---------------------------------------------------------------------------


def make_base_record(idx: int, item: Dict[str, Any], ddl: str) -> Dict[str, Any]:
    return {
        "index": idx,
        "question": item.get("question", ""),
        "db_id": item.get("db_id", ""),
        "gold_sql": item.get("query", item.get("gold_sql", "")),
        "success": False,
        "model": "",
        "error": None,
        "response": None,
        "reasoning_content": None,
        "messages": None,
        "ddl": ddl,
        "prompt_tokens": None,
        "completion_tokens": None,
        "prompt_cache_hit_tokens": None,
        "prompt_cache_miss_tokens": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


async def process_one(
    idx: int,
    item: Dict[str, Any],
    ddl: str,
    client: Any,
    sem: asyncio.Semaphore,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    record = make_base_record(idx, item, ddl)
    record["model"] = args.model
    question = record["question"]
    messages = build_messages(args.model, ddl, question)
    record["messages"] = messages

    async with sem:
        if args.sleep > 0:
            await asyncio.sleep(args.sleep)  # stagger starts to avoid bursts
        text, reasoning, usage, err = await call_deepseek(client, args.model, messages, args)

    if err is not None:
        record["error"] = f"{type(err).__name__}: {err}"
        return record

    record["success"] = True
    record["response"] = text
    record["reasoning_content"] = reasoning
    record["prompt_tokens"] = _usage_int(usage, "prompt_tokens")
    record["completion_tokens"] = _usage_int(usage, "completion_tokens")
    record["prompt_cache_hit_tokens"] = _usage_int(usage, "prompt_cache_hit_tokens")
    record["prompt_cache_miss_tokens"] = _usage_int(usage, "prompt_cache_miss_tokens")
    return record


def load_done(output_path: Path) -> Tuple[Set[int], Set[int]]:
    """Return (success_indices, failed_indices) already recorded in the JSONL."""
    done_success: Set[int] = set()
    done_failed: Set[int] = set()
    if not output_path.exists():
        return done_success, done_failed
    with open(output_path, "r", encoding="utf-8-sig") as fh:  # tolerate BOM
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = rec.get("index")
            if not isinstance(idx, int):
                continue
            (done_success if rec.get("success") else done_failed).add(idx)
    return done_success, done_failed


def write_records(output_path: Path, records: List[Dict[str, Any]]) -> None:
    """Append records (JSONL). flush() after every line so a crash loses at
    most the in-flight item."""
    with open(output_path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()


async def generate_async(
    pending: List[Tuple[int, Dict[str, Any]]],
    ddl_map: Dict[int, str],
    client: Any,
    args: argparse.Namespace,
    output_path: Path,
) -> List[Dict[str, Any]]:
    """Run all pending items concurrently; append each result as it completes."""
    sem = asyncio.Semaphore(args.concurrency)
    tasks = {
        asyncio.create_task(process_one(idx, item, ddl_map[idx], client, sem, args)): idx
        for idx, item in pending
    }
    done_count = 0
    success_count = 0
    records: List[Dict[str, Any]] = []
    with open(output_path, "a", encoding="utf-8") as fh:
        for coro in asyncio.as_completed(tasks):
            rec = await coro
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            records.append(rec)
            done_count += 1
            success_count += 1 if rec["success"] else 0
            if done_count % 25 == 0 or done_count == len(tasks):
                print(
                    f"  [{done_count}/{len(tasks)}] success={success_count} "
                    f"fail={done_count - success_count}",
                    flush=True,
                )
    return records


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------


def estimate_input_tokens(items: List[Tuple[int, Dict[str, Any]]], ddl_map: Dict[int, str], model: str) -> int:
    total = 0
    for idx, item in items:
        messages = build_messages(model, ddl_map[idx], item.get("question", ""))
        total += sum(estimate_tokens(m["content"]) for m in messages)
    return total


def estimate_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING_USD_PER_M.get(model, PRICING_USD_PER_M["deepseek-chat"])
    return (input_tokens / 1e6) * p["input_miss"] + (output_tokens / 1e6) * p["output"]


def compute_actual_cost(records: List[Dict[str, Any]], model: str) -> Tuple[float, int, int, int]:
    p = PRICING_USD_PER_M.get(model, PRICING_USD_PER_M["deepseek-chat"])
    in_hit = in_miss = out = 0
    for r in records:
        if not r.get("success"):
            continue
        hit = r.get("prompt_cache_hit_tokens")
        miss = r.get("prompt_cache_miss_tokens")
        if isinstance(hit, int) and isinstance(miss, int):
            in_hit += hit
            in_miss += miss
        else:
            in_miss += r.get("prompt_tokens") or 0
        out += r.get("completion_tokens") or 0
    usd = (
        (in_miss / 1e6) * p["input_miss"]
        + (in_hit / 1e6) * p["input_hit"]
        + (out / 1e6) * p["output"]
    )
    return usd, in_hit, in_miss, out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="FINER Step-1 style reasoning distillation generator "
                    "(Spider train -> DeepSeek <think> + SQL -> JSONL)."
    )
    ap.add_argument("--train-file", default=DEFAULT_TRAIN_FILE,
                    help=f"Spider train_spider.json (default: {DEFAULT_TRAIN_FILE})")
    ap.add_argument("--spider-dir", default=DEFAULT_SPIDER_DIR,
                    help="Spider data dir containing tables.json / database/ "
                         "for DDL extraction")
    ap.add_argument("--num-train", type=int, default=1000,
                    help="use the first N Spider train examples (0 = all)")
    ap.add_argument("--output", default="",
                    help="output JSONL path (default: "
                         "<spider-dir>/../reasoning_data/<model>_spider_train_think.jsonl)")
    ap.add_argument("--model", choices=MODEL_CHOICES, default="deepseek-chat")
    ap.add_argument("--api-key", default=None,
                    help="DeepSeek API key (default: env DEEPSEEK_API_KEY)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="sampling temperature (ignored for deepseek-reasoner)")
    ap.add_argument("--max-tokens", type=int, default=1500,
                    help="max output tokens per call (reasoner needs >= 1024)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="number of concurrent API calls")
    ap.add_argument("--sleep", type=float, default=0.05,
                    help="seconds each task sleeps before its API call "
                         "(stagger starts; effective peak rate ~ concurrency/sleep)")
    ap.add_argument("--max-retries", type=int, default=3,
                    help="retries per call with exponential backoff")
    ap.add_argument("--timeout", type=int, default=120,
                    help="per-request timeout in seconds")
    ap.add_argument("--dry-run", action="store_true",
                    help="load data, build prompts, print token/cost estimate, "
                         "do NOT call the API")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-run indices previously recorded as failed")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not _OPENAI_AVAILABLE and not args.dry_run:
        print("ERROR: the 'openai' package is required (pip install openai).")
        return 1

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: no DeepSeek API key. Set DEEPSEEK_API_KEY or pass --api-key.")
        return 1

    train_path = Path(args.train_file)
    if not train_path.exists():
        print(f"ERROR: train file not found: {train_path}")
        return 1
    with open(train_path, "r", encoding="utf-8") as fh:
        train_data = json.load(fh)
    slice_items = train_data if args.num_train <= 0 else train_data[: args.num_train]

    output_path = Path(args.output) if args.output else (
        Path(args.spider_dir).parent / "reasoning_data"
        / f"{args.model}_spider_train_think.jsonl"
    )

    # --- DDL for every requested db (fail fast before any API spend) -------
    loader = SpiderLoader(args.spider_dir)
    ddl_cache: Dict[str, str] = {}
    missing_ddl: Set[str] = set()
    for item in slice_items:
        db_id = item.get("db_id", "")
        if db_id in ddl_cache or db_id in missing_ddl:
            continue
        try:
            ddl_cache[db_id] = loader.get_ddl_with_source(db_id)[0]
        except Exception as exc:
            missing_ddl.add(db_id)
            print(f"WARNING: no DDL for db_id={db_id!r} ({type(exc).__name__}); "
                  f"its examples will be recorded as failed (no API call).")

    # --- resume bookkeeping -------------------------------------------------
    done_success, done_failed = load_done(output_path)
    skip = done_success | done_failed
    if args.retry_failed:
        skip = done_success  # re-run previously failed ones

    pending: List[Tuple[int, Dict[str, Any]]] = []
    fail_records: List[Dict[str, Any]] = []
    ddl_map: Dict[int, str] = {}
    for idx, item in enumerate(slice_items):
        if idx in skip:
            continue
        ddl = ddl_cache.get(item.get("db_id", ""))
        if ddl is None:
            rec = make_base_record(idx, item, None)
            rec["model"] = args.model
            rec["error"] = "ddl_not_found: no DDL available for this db_id"
            fail_records.append(rec)
            continue
        pending.append((idx, item))
        ddl_map[idx] = ddl

    print("=" * 70)
    print(f"model            : {args.model}")
    print(f"base url         : {args.base_url}")
    print(f"train file       : {train_path}")
    print(f"spider dir       : {Path(args.spider_dir)}")
    print(f"examples loaded  : {len(slice_items)} (requested {args.num_train})")
    print(f"already done     : {len(done_success)} success / {len(done_failed)} failed "
          f"(skipping {len(skip)})")
    print(f"to process       : {len(pending)} API calls + {len(fail_records)} ddl-missing "
          f"records (no API)")
    print(f"output           : {output_path}")
    print(f"concurrency      : {args.concurrency} | sleep {args.sleep}s | "
          f"max_retries {args.max_retries} | timeout {args.timeout}s")
    print("=" * 70)

    # --- pre-run cost estimate ---------------------------------------------
    est_input = estimate_input_tokens(pending, ddl_map, args.model)
    est_output = ASSUMED_AVG_OUTPUT_TOKENS * len(pending)
    est_cost = estimate_usd(args.model, est_input, est_output)
    p = PRICING_USD_PER_M.get(args.model, PRICING_USD_PER_M["deepseek-chat"])
    print(f"COST ESTIMATE (pre-run, worst-case cache miss):")
    print(f"  est input tokens : {est_input:,}  (assumed avg output {ASSUMED_AVG_OUTPUT_TOKENS}/item)")
    print(f"  est output tokens: {est_output:,}")
    print(f"  est cost         : ~${est_cost:.2f} USD "
          f"({args.model}: in ${p['input_miss']}/M, out ${p['output']}/M; "
          f"verify prices at api-docs.deepseek.com/quick_start/pricing)")
    if pending:
        sample_msgs = build_messages(args.model, ddl_map[pending[0][0]], pending[0][1].get("question", ""))
        print(f"  sample prompt (item {pending[0][0]}, system, truncated):")
        print("    " + sample_msgs[0]["content"].replace("\n", "\n    ")[:400])
        print(f"  sample prompt (item {pending[0][0]}, user, truncated):")
        print("    " + sample_msgs[1]["content"].replace("\n", "\n    ")[:400])

    if args.dry_run:
        print("DRY RUN - no API calls were made. Remove --dry-run to run for real.")
        return 0

    # --- generate -----------------------------------------------------------
    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
    if fail_records:
        write_records(output_path, fail_records)
        print(f"Wrote {len(fail_records)} ddl-missing failure records.")

    if not pending:
        print("Nothing to do (all requested indices already recorded).")
        records: List[Dict[str, Any]] = []
    else:
        print(f"Generating {len(pending)} examples via {args.model} "
              f"(concurrency={args.concurrency})...")
        try:
            records = asyncio.run(
                generate_async(pending, ddl_map, client, args, output_path)
            )
        except KeyboardInterrupt:
            print("\nInterrupted - completed items are already persisted; "
                  "re-run the same command to resume.")
            return 130

    # --- post-run summary & actual cost -------------------------------------
    success = sum(1 for r in records if r.get("success"))
    actual_usd, in_hit, in_miss, out_tok = compute_actual_cost(records, args.model)
    print("=" * 70)
    print(f"DONE: {len(records)} processed, {success} success, "
          f"{len(records) - success} failed (overall success rate incl. resume: "
          f"{(success + len(done_success)) / max(1, len(slice_items)):.1%})")
    print(f"COST REPORT (from API usage of this run):")
    print(f"  prompt tokens    : {in_miss + in_hit:,} (cache hit {in_hit:,} / miss {in_miss:,})")
    print(f"  completion tokens: {out_tok:,}")
    print(f"  cost             : ~${actual_usd:.2f} USD")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
