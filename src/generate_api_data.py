#!/usr/bin/env python3
"""
Generate training data by calling DeepSeek V4 Flash API on Spider train examples.
The API generates reasoning + SQL, which we use for SFT cold-start before GRPO.

Usage:
    export DEEPSEEK_API_KEY=sk-xxx
    python src/generate_api_data.py --num 500 --output data/api_sft_data.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from spider_utils import SpiderLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "api_sft_data.json"

# DeepSeek V4 Flash API endpoint
API_BASE = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"

SYSTEM_PROMPT = """You are an expert Text-to-SQL assistant. Given a database schema and a natural-language question, you MUST:
1. Think step-by-step about which tables, columns, joins, filters, and aggregations are needed.
2. Generate exactly one valid SQLite SQL query.
3. Put the final SQL inside a ```sql code block.

Rules:
- Use only tables and columns that appear in the schema.
- Use the minimum number of tables required.
- Do not join a table merely because a foreign key exists.
- Do not invent database objects."""


def build_prompt(question: str, ddl: str) -> str:
    return f"Database Schema:\n{ddl}\n\nQuestion: {question}\n\nThink step by step and generate the SQL."


def call_api(client: OpenAI, question: str, ddl: str, max_retries: int = 3) -> Dict[str, Any]:
    """Call DeepSeek API and return the reasoning + SQL."""
    prompt = build_prompt(question, ddl)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            content = response.choices[0].message.content
            return {"success": True, "response": content, "error": None}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"success": False, "response": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=500, help="Number of examples")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--spider-dir", type=str, default=str(SPIDER_DIR))
    args = parser.parse_args()

    # API key from env
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: Set DEEPSEEK_API_KEY environment variable")
        return 1

    client = OpenAI(api_key=api_key, base_url=API_BASE)

    # Load Spider train data
    loader = SpiderLoader(args.spider_dir)
    train_path = Path(args.spider_dir) / "train_spider.json"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    train_data = train_data[:args.num]

    print(f"Generating {len(train_data)} examples via DeepSeek V4 Flash...")
    results = []
    success_count = 0

    for i, item in enumerate(train_data):
        question = item["question"]
        gold_sql = item["query"]
        db_id = item["db_id"]

        # Get DDL
        try:
            ddl, _ = loader.get_ddl_with_source(db_id)
        except RuntimeError:
            continue

        api_result = call_api(client, question, ddl)
        status = "✓" if api_result["success"] else "✗"

        if api_result["success"]:
            success_count += 1
            results.append({
                "db_id": db_id,
                "question": question,
                "ddl": ddl,
                "gold_sql": gold_sql,
                "api_response": api_result["response"],
            })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(train_data)}] {status} success_rate={success_count}/{i+1}")
        else:
            print(f"  [{i+1}/{len(train_data)}] {status}", end="\r")

    print(f"\nDone! {success_count}/{len(train_data)} succeeded")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} examples to: {output_path}")


if __name__ == "__main__":
    main()
