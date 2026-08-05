#!/usr/bin/env python3
"""生成高质量 SFT 数据：DeepSeek API 生成 → 执行过滤 → 只保留正确样本"""
import json, os, re, sys, time
from pathlib import Path

from openai import OpenAI
from spider_utils import SpiderLoader, DatabaseExecutor

PROJECT = Path(__file__).resolve().parent.parent
SPIDER = str(PROJECT / 'data' / 'spider_data')
OUT = PROJECT / 'data' / 'sft_filtered.json'

API_BASE = 'https://api.deepseek.com'
MODEL = 'deepseek-v4-flash'

SYSTEM = """You are an expert Text-to-SQL assistant. Given a database schema and a question, generate exactly one valid SQLite SQL query. Put the final SQL inside a ```sql code block. Use only tables/columns in the schema."""


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
        print("ERROR: DEEPSEEK_API_KEY not set")
        return 1
    client = OpenAI(api_key=api_key, base_url=API_BASE)

    loader = SpiderLoader(SPIDER)
    executor = DatabaseExecutor(SPIDER)
    with open(f'{SPIDER}/train_spider.json') as f:
        train = json.load(f)

    # 取前 1000 条生成
    train = train[:1000]
    kept = []
    total = 0
    start = time.time()

    for i, item in enumerate(train):
        question, gold_sql, db_id = item['question'], item['query'], item['db_id']
        try:
            ddl, _ = loader.get_ddl_with_source(db_id)
        except Exception:
            continue

        # API 生成
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"Schema:\n{ddl}\n\nQuestion: {question}\n\nSQL:"}],
                temperature=0.2, max_tokens=512)
            text = resp.choices[0].message.content
        except Exception as e:
            print(f"  API error: {e}")
            continue

        sql = extract_sql(text)
        if not sql:
            continue

        # 执行过滤：预测结果必须与 gold 完全一致才保留
        pred = executor.execute(db_id, sql)
        gold = executor.execute(db_id, gold_sql)
        if pred['success'] and gold['success'] and pred['saved_rows'] == gold['saved_rows']:
            kept.append({
                'text': f"You are an expert Text-to-SQL assistant.\n\nDatabase Schema:\n{ddl}\n\nQuestion: {question}\n\nAnswer: {text}",
                'db_id': db_id, 'question': question, 'gold_sql': gold_sql,
            })
            total += 1

        if (i+1) % 100 == 0:
            print(f"  [{i+1}/1000] 保留 {total} 条 ({total/(i+1):.0%})")

    elapsed = time.time() - start
    print(f"\n=== 完成: 1000 条 → 保留 {len(kept)} 条 ({len(kept)/1000:.0%}) ===")
    print(f"耗时: {elapsed:.0f}s")

    with open(OUT, 'w') as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f"保存到: {OUT}")


if __name__ == '__main__':
    main()
