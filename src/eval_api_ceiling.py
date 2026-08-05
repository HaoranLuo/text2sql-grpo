#!/usr/bin/env python3
"""Evaluate DeepSeek API as an upper-bound reference on dev-100.

Runs the same 100 Spider dev questions through DeepSeek V4 Flash API,
executes the generated SQL on SQLite, and reports match rate.
No GPU needed — run on the login node.
"""
import json, os, re, sys, time
from pathlib import Path

from openai import OpenAI
from spider_utils import SpiderLoader, DatabaseExecutor, compare_execution_results

PROJECT = Path(__file__).resolve().parent.parent
SPIDER = str(PROJECT / 'data' / 'spider_data')
OUT = PROJECT / 'outputs' / 'eval_api_ceiling'

API_BASE = 'https://api.deepseek.com'
MODEL = 'deepseek-v4-flash'

SYSTEM = """You are an expert Text-to-SQL assistant. Given a database schema and a question, generate exactly one valid SQLite SQL query. Put the final SQL inside a ```sql code block. Use only tables/columns in the schema. Use the minimum number of tables required. Do not invent objects."""


def extract_sql(text: str) -> str:
    m = re.search(r"```sql\s*(.*?)```", text, re.I | re.S)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.S)
    if m: return m.group(1).strip()
    m = re.search(r"\b(SELECT|WITH)\b.*?;", text, re.I | re.S)
    if m: return m.group(0).strip()
    return ""


def main():
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key:
        print("ERROR: Set DEEPSEEK_API_KEY env var")
        return 1
    client = OpenAI(api_key=api_key, base_url=API_BASE)

    loader = SpiderLoader(SPIDER)
    executor = DatabaseExecutor(SPIDER)
    items = loader.load_dev(limit=100, start_index=0)

    results = []
    match_count = 0
    start = time.time()

    for i, item in enumerate(items):
        db_id, question, gold_sql = item['db_id'], item['question'], item['query']
        ddl, _ = loader.get_ddl_with_source(db_id)
        prompt = f"Database Schema:\n{ddl}\n\nQuestion: {question}\n\nGenerate the SQL:"

        ok = False
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}],
                    temperature=0.2, max_tokens=512)
                text = resp.choices[0].message.content
                ok = True
                break
            except Exception as e:
                time.sleep(2 ** attempt)
        if not ok:
            results.append({'di': item['dataset_index'], 'match': False, 'error': 'api_failed'})
            continue

        sql = extract_sql(text)
        if not sql:
            results.append({'di': item['dataset_index'], 'match': False, 'error': 'no_sql'})
            continue

        pred = executor.execute(db_id, sql)
        gold = executor.execute(db_id, gold_sql)
        is_match = (pred['success'] and gold['success'] and
                    compare_execution_results(pred['saved_rows'], gold['saved_rows'], gold_sql=gold_sql)['match'])
        if is_match:
            match_count += 1

        results.append({'di': item['dataset_index'], 'match': is_match, 'error': None if pred['success'] else pred['error']})

        if (i+1) % 10 == 0:
            e = time.time() - start
            print(f"  [{i+1}/100] match={match_count}/{i+1} ({match_count/(i+1):.1%})  {e:.0f}s")

    elapsed = time.time() - start
    rate = match_count / 100
    print(f"\n=== DeepSeek API CEILING on dev-100 ===")
    print(f"Match: {match_count}/100 ({rate:.1%})")
    print(f"Time:  {elapsed:.0f}s")
    print(f"Local 7B zero-shot: 81.0%")
    print(f"API vs 7B:           {rate-0.81:+.1%}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'summary.json', 'w') as f:
        json.dump({'method': 'deepseek_api_ceiling', 'match_count': match_count, 'match_rate': rate,
                   'elapsed_seconds': round(elapsed, 1), 'model': MODEL}, f, indent=2)
    with open(OUT / 'items.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
