#!/usr/bin/env python3
"""M1: M-Schema prompt 评估 — 用结构化 schema 格式替代 DDL"""
import json, sys, time, torch
from collections import Counter
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
from reasoning_generator_agent import ReasoningGeneratorAgent
from spider_utils import SpiderLoader, DatabaseExecutor, compare_execution_results

MODEL = str(PROJECT / 'models' / 'qwen2.5-coder-7b-instruct')
SPIDER = str(PROJECT / 'data' / 'spider_data')
OUT = PROJECT / 'outputs' / 'eval_mschema'


def ddl_to_mschema(ddl: str) -> str:
    """把 DDL 转成 M-Schema 风格（更结构化，符合现代 text-to-sql 模型偏好）"""
    lines = []
    current_table = None
    for line in ddl.split('\n'):
        line = line.strip()
        if line.startswith('CREATE TABLE'):
            # "CREATE TABLE name (" -> extract name
            name = line.replace('CREATE TABLE', '').replace('(', '').strip().strip('"').strip('`')
            current_table = name
            lines.append(f"Table {name}:")
        elif current_table and line and not line.startswith(')'):
            # Column definitions
            if 'FOREIGN KEY' in line:
                lines.append(f"  {line.strip().rstrip(',')}")
            else:
                # "col TYPE PRIMARY KEY" or "col TYPE"
                col_def = line.rstrip(',')
                lines.append(f"  - {col_def}")
        elif line == ');':
            current_table = None
    return '\n'.join(lines)


def build_mschema_prompt(question: str, mschema: str) -> str:
    return f"""You are an expert Text-to-SQL assistant. Given the database schema and a question, generate exactly one valid SQLite SQL query.

Database Schema:
{mschema}

Question: {question}

Instructions:
1. Use only tables and columns from the schema.
2. Use the minimum number of tables required.
3. Do not join unless necessary.
4. Put the final SQL inside a ```sql code block."""


def main():
    print("M1: M-Schema prompt evaluation")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={'': 0}, local_files_only=True, trust_remote_code=True)
    model.eval()
    model.config.pad_token_id = tokenizer.eos_token_id

    loader = SpiderLoader(SPIDER)
    executor = DatabaseExecutor(SPIDER)
    items = loader.load_dev(limit=100, start_index=0)

    match_count = 0
    results = []
    start_t = time.time()

    for i, item in enumerate(items):
        db_id, question, gold_sql = item['db_id'], item['question'], item['query']
        ddl, _ = loader.get_ddl_with_source(db_id)
        mschema = ddl_to_mschema(ddl)
        prompt = build_mschema_prompt(question, mschema)
        chat = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)

        enc = tokenizer(chat, return_tensors='pt', truncation=True, max_length=1536).to('cuda:0')
        in_len = enc['input_ids'].shape[1]
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
        parsed = ReasoningGeneratorAgent.extract_sql(text)
        sql = parsed['sql'] if parsed['parse_success'] else ''

        is_match = False
        if sql:
            pred = executor.execute(db_id, sql)
            gold = executor.execute(db_id, gold_sql)
            if pred['success'] and gold['success']:
                is_match = compare_execution_results(pred['saved_rows'], gold['saved_rows'], gold_sql=gold_sql)['match']
        if is_match:
            match_count += 1

        results.append({'di': item['dataset_index'], 'match': is_match, 'parse': bool(sql)})
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/100] match={match_count}/{i+1} ({match_count/(i+1):.1%})")

    elapsed = time.time() - start_t
    rate = match_count / 100
    print(f"\n=== M-SCHEMA RESULT ===")
    print(f"Match: {match_count}/100 ({rate:.1%})")
    print(f"Time:  {elapsed:.0f}s")
    print(f"Baseline (DDL prompt): 81.0%")
    print(f"Delta: {rate-0.81:+.1%}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'summary.json', 'w') as f:
        json.dump({'method': 'mschema_prompt', 'match_count': match_count, 'match_rate': rate,
                   'elapsed_seconds': round(elapsed, 1)}, f, indent=2)


if __name__ == '__main__':
    main()
