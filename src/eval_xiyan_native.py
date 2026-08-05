#!/usr/bin/env python3
"""M4: XiYanSQL-7B 原生格式评估 — 用它的 M-Schema 风格 prompt"""
import json, sys, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
from reasoning_generator_agent import ReasoningGeneratorAgent
from spider_utils import SpiderLoader, DatabaseExecutor, compare_execution_results

MODEL = str(PROJECT / 'models' / 'XiYanSQL-7B')
SPIDER = str(PROJECT / 'data' / 'spider_data')
OUT = PROJECT / 'outputs' / 'eval_xiyan_native'


def build_xiyan_prompt(question: str, ddl: str) -> str:
    """XiYanSQL 官方 prompt 风格（M-Schema 表格描述）"""
    # 转成表格化 schema
    schema_lines = []
    current_table = None
    for line in ddl.split('\n'):
        line = line.strip()
        if line.startswith('CREATE TABLE'):
            name = line.replace('CREATE TABLE', '').replace('(', '').strip().strip('"').strip('`')
            current_table = name
            schema_lines.append(f"### Table: {name}")
        elif current_table and line and not line.startswith(')'):
            col = line.rstrip(',')
            schema_lines.append(f"- {col}")
        elif line == ');':
            current_table = None

    return f"""你是数据库专家，请根据数据库表结构，将用户问题转换为 SQL 查询语句。

数据库表结构：
{schema_lines}

问题：{question}

请只输出 SQL 查询语句，不要输出其他内容。"""


def main():
    print("M4: XiYanSQL-7B native format evaluation")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={'': 0}, local_files_only=True, trust_remote_code=True)
    model.eval()

    loader = SpiderLoader(SPIDER)
    executor = DatabaseExecutor(SPIDER)
    items = loader.load_dev(limit=100, start_index=0)

    match_count = 0
    parse_count = 0
    results = []
    start_t = time.time()

    for i, item in enumerate(items):
        db_id, question, gold_sql = item['db_id'], item['question'], item['query']
        ddl, _ = loader.get_ddl_with_source(db_id)
        prompt = build_xiyan_prompt(question, ddl)

        # XiYanSQL 用纯文本 prompt（不是 chat 格式）
        enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to('cuda:0')
        in_len = enc['input_ids'].shape[1]
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)

        # XiYanSQL 输出纯 SQL（无 ```sql 包装），用宽松提取
        sql = text.strip()
        # 去掉可能的包裹
        if '```' in sql:
            import re
            m = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.S)
            if m:
                sql = m.group(1).strip()
        # 只取第一条语句
        sql = sql.split(';')[0].strip()

        is_parse = bool(sql and sql.upper().startswith(('SELECT', 'WITH')))
        if is_parse:
            parse_count += 1

        is_match = False
        if is_parse:
            pred = executor.execute(db_id, sql)
            gold = executor.execute(db_id, gold_sql)
            if pred['success'] and gold['success']:
                is_match = compare_execution_results(pred['saved_rows'], gold['saved_rows'], gold_sql=gold_sql)['match']
        if is_match:
            match_count += 1

        results.append({'di': item['dataset_index'], 'match': is_match, 'parse': is_parse})
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/100] match={match_count}/{i+1} ({match_count/(i+1):.1%}) parse={parse_count}")

    elapsed = time.time() - start_t
    rate = match_count / 100
    print(f"\n=== XIYANSQL NATIVE RESULT ===")
    print(f"Match: {match_count}/100 ({rate:.1%})")
    print(f"Parse: {parse_count}/100")
    print(f"Time:  {elapsed:.0f}s")
    print(f"Baseline: 81.0%")
    print(f"Delta: {rate-0.81:+.1%}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'summary.json', 'w') as f:
        json.dump({'method': 'xiyan_native', 'match_count': match_count, 'match_rate': rate,
                   'parse_count': parse_count, 'elapsed_seconds': round(elapsed, 1)}, f, indent=2)


if __name__ == '__main__':
    main()
