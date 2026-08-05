#!/usr/bin/env python3
"""训练后3B(100条G4=50%) × 5prompt投票：验证能否超65%"""
import json, re, sys, time, torch
from collections import Counter
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
from reasoning_generator_agent import ReasoningGeneratorAgent
from spider_utils import SpiderLoader, DatabaseExecutor, compare_execution_results

BASE_MODEL = str(PROJECT / 'models' / 'Qwen2.5-Coder-3B-Instruct')
LORA = str(PROJECT / 'checkpoints' / 'grpo_3b_3lvl' / 'checkpoint-25')  # 100条G4三级=50%
SPIDER = str(PROJECT / 'data' / 'spider_data')
OUT = PROJECT / 'outputs' / 'eval_5prompt_3b_trained'


def build_prompt_variants(question: str, ddl: str) -> list:
    """5 种 prompt 视角"""
    p1 = ReasoningGeneratorAgent.build_prompt(question=question, ddl_schema=ddl, dialect='sqlite')
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
    p3 = f"""Database schema:
{ddl}

Question: {question}

Think carefully, then output ONLY the SQL query in a ```sql block."""
    p4 = f"""Database schema:
{ddl}

Question: {question}

IMPORTANT: Use the MINIMUM number of tables. If one table suffices, do NOT join others. Output SQL in ```sql block."""
    p5 = f"""Schema:
{ddl}

Question: {question}

Write the SQL query. Only output SQL, nothing else."""
    return [p1, p2, p3, p4, p5]


def main():
    print("5prompt投票 on 训练后3B (100条G4=50%)")
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map={'': 0}, local_files_only=True, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, LORA)
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

        exec_results = []
        for p in prompts:
            chat = tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
            enc = tokenizer(chat, return_tensors='pt', truncation=True, max_length=1536).to('cuda:0')
            in_len = enc['input_ids'].shape[1]
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
            parsed = ReasoningGeneratorAgent.extract_sql(text)
            if parsed['parse_success']:
                r = executor.execute(db_id, parsed['sql'])
                if r['success']:
                    exec_results.append(tuple(tuple(row) for row in r['saved_rows']))

        if exec_results:
            mc = Counter(exec_results).most_common(1)[0]
            voted_rows = [list(row) for row in mc[0]]
            vc = mc[1]
        else:
            voted_rows, vc = [], 0

        gold_r = executor.execute(db_id, gold_sql)
        gold_rows = gold_r['saved_rows'] if gold_r['success'] else []
        is_match = gold_r['success'] and compare_execution_results(voted_rows, gold_rows, gold_sql=gold_sql)['match']
        if is_match:
            match_count += 1

        results.append({'di': item['dataset_index'], 'match': is_match})
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/100] match={match_count}/{i+1} ({match_count/(i+1):.1%})")

    elapsed = time.time() - start_t
    rate = match_count / 100
    print(f"\n=== 5prompt×训练后3B RESULT ===")
    print(f"Match: {match_count}/100 ({rate:.1%})")
    print(f"Time: {elapsed:.0f}s")
    print(f"参考: 3B基线45% | 训练50% | 训练+3prompt投票65%")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'summary.json', 'w') as f:
        json.dump({'method': '5prompt_3b_trained', 'match_rate': rate, 'match_count': match_count,
                   'elapsed_seconds': round(elapsed, 1)}, f, indent=2)


if __name__ == '__main__':
    main()
