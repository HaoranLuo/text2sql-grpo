#!/usr/bin/env python3
"""M5: 多 Prompt 变体投票 — 3种prompt各生成1个greedy候选，执行投票"""
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
OUT = PROJECT / 'outputs' / 'eval_multi_prompt'


def build_prompt_variants(question: str, ddl: str) -> list:
    """3 种 prompt 变体"""
    # 变体1: 标准 DDL（当前 81%）
    p1 = ReasoningGeneratorAgent.build_prompt(question=question, ddl_schema=ddl, dialect='sqlite')

    # 变体2: M-Schema 风格（简洁表格）
    schema_lines = []
    current = None
    for line in ddl.split('\n'):
        line = line.strip()
        if line.startswith('CREATE TABLE'):
            name = line.replace('CREATE TABLE', '').replace('(', '').strip().strip('"').strip('`')
            current = name
            schema_lines.append(f"Table {name}:")
        elif current and line and not line.startswith(')'):
            schema_lines.append(f"  - {line.rstrip(',')}")
        elif line == ');':
            current = None
    p2 = f"""Given the database schema and question, generate the SQLite SQL.

Schema:
{chr(10).join(schema_lines)}

Question: {question}

SQL:"""

    # 变体3: 强调"先想再写"
    p3 = f"""Database schema:
{ddl}

Question: {question}

Think carefully about which tables and columns are needed, then output ONLY the SQL query in a ```sql block."""

    return [p1, p2, p3]


def main():
    print("M5: Multi-prompt voting (3 variants, greedy)")
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
        prompts = build_prompt_variants(question, ddl)

        # 每个变体 greedy 生成
        exec_results = []
        parsed_sqls = []
        for p in prompts:
            chat = tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
            enc = tokenizer(chat, return_tensors='pt', truncation=True, max_length=1536).to('cuda:0')
            in_len = enc['input_ids'].shape[1]
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
            parsed = ReasoningGeneratorAgent.extract_sql(text)
            sql = parsed['sql'] if parsed['parse_success'] else ''
            parsed_sqls.append(sql)
            if sql:
                r = executor.execute(db_id, sql)
                if r['success']:
                    exec_results.append(tuple(tuple(row) for row in r['saved_rows']))

        # 投票（含空结果处理：都失败则用第一个可解析的）
        if exec_results:
            mc = Counter(exec_results).most_common(1)[0]
            voted_rows = [list(row) for row in mc[0]]
            vc = mc[1]
        elif any(parsed_sqls):
            # 全部执行失败但至少解析成功——用第一个（尽力而为）
            r = executor.execute(db_id, parsed_sqls[0])
            voted_rows = r['saved_rows'] if r['success'] else []
            vc = 0
        else:
            voted_rows, vc = [], 0

        gold_r = executor.execute(db_id, gold_sql)
        gold_rows = gold_r['saved_rows'] if gold_r['success'] else []
        is_match = gold_r['success'] and compare_execution_results(voted_rows, gold_rows, gold_sql=gold_sql)['match']
        if is_match:
            match_count += 1

        results.append({'di': item['dataset_index'], 'match': is_match, 'votes': vc, 'valid': len(exec_results)})
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/100] match={match_count}/{i+1} ({match_count/(i+1):.1%})")

    elapsed = time.time() - start_t
    rate = match_count / 100
    print(f"\n=== MULTI-PROMPT VOTING RESULT ===")
    print(f"Match: {match_count}/100 ({rate:.1%})")
    print(f"Time:  {elapsed:.0f}s")
    print(f"Baseline: 81.0%")
    print(f"Delta: {rate-0.81:+.1%}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'summary.json', 'w') as f:
        json.dump({'method': 'multi_prompt_vote', 'match_count': match_count, 'match_rate': rate,
                   'elapsed_seconds': round(elapsed, 1)}, f, indent=2)


if __name__ == '__main__':
    main()
