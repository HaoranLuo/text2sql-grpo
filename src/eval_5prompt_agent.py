#!/usr/bin/env python3
"""通用 5prompt 投票评估（修复后口径：full_rows 比较 + truncated 检查）

用法:
    python src/eval_5prompt_agent.py --lora-path checkpoints/p2a_500/checkpoint-25 \
        --output-dir outputs/eval_5p_p2a500 [--limit 100]

5 种 prompt 视角 + 执行结果多数投票（与 eval_5prompt_3b_trained 一致）。
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT / 'src'))
from reasoning_generator_agent import ReasoningGeneratorAgent
from spider_utils import SpiderLoader, DatabaseExecutor, compare_execution_results

BASE_MODEL = str(PROJECT / 'models' / 'Qwen2.5-Coder-3B-Instruct')
SPIDER = str(PROJECT / 'data' / 'spider_data')


def build_prompt_variants(question: str, ddl: str) -> list:
    """5/7 种 prompt 视角（p1-p5 为原始 5 视角；p6/p7 用于 7p 验证）"""
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
    # p6/p7: 7p 验证用（第 6、7 视角）
    p6 = f"""Database schema:
{ddl}

Question: {question}

Step by step: 1) which tables and columns are needed; 2) what filters/joins;
3) the SQLite SQL. Output the final SQL in a ```sql block."""
    p7 = f"""Database schema:
{ddl}

Question: {question}

Write the SQL as a SINGLE line without any comments or explanation."""
    return [p1, p2, p3, p4, p5, p6, p7]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora-path', required=True, help='LoRA adapter 目录（含 checkpoint）')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--n-prompts', type=int, default=5, choices=[5, 7],
                        help='投票用多少个 prompt 视角 (5 或 7)')
    args = parser.parse_args()

    lora = args.lora_path
    out_dir = Path(args.output_dir)
    limit = args.limit
    start_index = args.start_index
    n_prompts = args.n_prompts
    print(f"{n_prompts}prompt投票 | LoRA: {lora} | {limit} 条 (start={start_index})")

    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={'': 0},
        local_files_only=True, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, lora)
    model.eval()
    model.config.pad_token_id = tokenizer.eos_token_id

    loader = SpiderLoader(SPIDER)
    executor = DatabaseExecutor(SPIDER)
    items = loader.load_dev(limit=limit, start_index=start_index)

    match_count = 0
    results = []
    start_t = time.time()

    for i, item in enumerate(items):
        db_id, question, gold_sql = item['db_id'], item['question'], item['query']
        ddl, _ = loader.get_ddl_with_source(db_id)
        prompts = build_prompt_variants(question, ddl)[:n_prompts]

        exec_results = []   # (full_rows_tuple, truncated_flag)
        for p in prompts:
            chat = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
            enc = tokenizer(chat, return_tensors='pt', truncation=True, max_length=1536).to('cuda:0')
            in_len = enc['input_ids'].shape[1]
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
            parsed = ReasoningGeneratorAgent.extract_sql(text)
            if parsed['parse_success']:
                r = executor.execute(db_id, parsed['sql'])
                if r['success']:
                    exec_results.append((tuple(tuple(row) for row in r['full_rows']),
                                         r['full_rows_truncated']))

        # 多数投票（修复后口径：full_rows 比较 + truncated 则判负）
        voted_rows, vc, voted_truncated = [], 0, False
        if exec_results:
            mc = Counter(e[0] for e in exec_results).most_common(1)[0]
            voted_rows = [list(row) for row in mc[0]]
            vc = mc[1]
            voted_truncated = any(e[1] for e in exec_results if e[0] == mc[0])

        gold_r = executor.execute(db_id, gold_sql)
        gold_rows = gold_r['full_rows'] if gold_r['success'] else []
        gold_truncated = gold_r.get('full_rows_truncated', False) if gold_r['success'] else False

        if gold_r['success'] and not voted_truncated and not gold_truncated:
            is_match = compare_execution_results(
                voted_rows, gold_rows, gold_sql=gold_sql)['match']
        else:
            is_match = False
        if is_match:
            match_count += 1

        results.append({'di': item['dataset_index'], 'match': is_match,
                        'votes': vc, 'truncated': voted_truncated})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{limit}] match={match_count}/{i+1} ({match_count/(i+1):.1%})")

    elapsed = time.time() - start_t
    rate = match_count / limit
    print(f"\n=== 5prompt投票 RESULT ===")
    print(f"Match: {match_count}/{limit} ({rate:.1%})")
    print(f"Time: {elapsed:.0f}s")
    print(f"LoRA: {lora}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump({'method': '5prompt_vote', 'lora': lora,
                   'match_rate': rate, 'match_count': match_count,
                   'start_index': start_index, 'limit': limit,
                   'elapsed_seconds': round(elapsed, 1)}, f, indent=2)
    with open(out_dir / 'items.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
