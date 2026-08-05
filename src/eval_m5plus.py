#!/usr/bin/env python3
"""M5-Plus: 5 prompt 视角 + 结果分组仲裁（升级版投票）"""
import json, re, sys, time, torch
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
from reasoning_generator_agent import ReasoningGeneratorAgent
from spider_utils import SpiderLoader, DatabaseExecutor, compare_execution_results

MODEL = str(PROJECT / 'models' / 'qwen2.5-coder-7b-instruct')
SPIDER = str(PROJECT / 'data' / 'spider_data')
OUT = PROJECT / 'outputs' / 'eval_m5plus'


def build_prompt_variants(question: str, ddl: str) -> list:
    """5 种 prompt 视角"""
    # 1. 标准 DDL（原81%）
    p1 = ReasoningGeneratorAgent.build_prompt(question=question, ddl_schema=ddl, dialect='sqlite')

    # 2. M-Schema 表格
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

    # 3. 先想再写
    p3 = f"""Database schema:
{ddl}

Question: {question}

Think carefully about which tables and columns are needed, then output ONLY the SQL query in a ```sql block."""

    # 4. 反JOIN提醒（针对我们发现的模型乱JOIN问题）
    p4 = f"""Database schema:
{ddl}

Question: {question}

IMPORTANT: Use the MINIMUM number of tables. If the answer needs only one table, do NOT join others. Output the SQL in a ```sql block."""

    # 5. 一步到位简洁版
    p5 = f"""Schema:
{ddl}

Question: {question}

Write the SQL query. Only output SQL, nothing else. No explanations, no comments."""

    return [p1, p2, p3, p4, p5]


def _sql_quality_score(sql: str) -> float:
    """仲裁用的质量分：表引用数越少越好（最小表原则），长度适中"""
    tables = set(re.findall(r"\bFROM\s+([a-zA-Z_]\w*)", sql, re.I)) | set(re.findall(r"\bJOIN\s+([a-zA-Z_]\w*)", sql, re.I))
    # 表少 + 长度合理 = 好
    return -len(tables) * 2.0 - len(sql) * 0.001


def main():
    print("M5-Plus: 5 prompts + result-group arbitration")
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

        # 生成 5 个候选 + 执行
        candidates = []  # (rows_key, sql, quality_score)
        for p in prompts:
            chat = tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
            enc = tokenizer(chat, return_tensors='pt', truncation=True, max_length=1536).to('cuda:0')
            in_len = enc['input_ids'].shape[1]
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
            parsed = ReasoningGeneratorAgent.extract_sql(text)
            sql = parsed['sql'] if parsed['parse_success'] else ''
            if sql:
                r = executor.execute(db_id, sql)
                if r['success']:
                    key = tuple(tuple(row) for row in r['saved_rows'])
                    candidates.append((key, sql, _sql_quality_score(sql)))

        # 结果分组仲裁
        if candidates:
            # 按执行结果分组
            groups = defaultdict(list)
            for key, sql, q in candidates:
                groups[key].append((sql, q))
            # 组内同结果：取质量分最高的为代表
            group_reps = {}
            for key, members in groups.items():
                best = max(members, key=lambda m: m[1])
                group_reps[key] = (best[0], len(members), best[1])
            # 组间仲裁：票数最多优先，平票则质量分高者胜
            max_votes = max(g[1] for g in group_reps.values())
            top_groups = [(k, v) for k, v in group_reps.items() if v[1] == max_votes]
            if len(top_groups) == 1:
                voted_rows = [list(row) for row in top_groups[0][0]]
            else:
                # 平票：质量分高者胜
                winner = max(top_groups, key=lambda g: g[1][2])
                voted_rows = [list(row) for row in winner[0]]
        else:
            voted_rows = []

        gold_r = executor.execute(db_id, gold_sql)
        gold_rows = gold_r['saved_rows'] if gold_r['success'] else []
        is_match = gold_r['success'] and compare_execution_results(voted_rows, gold_rows, gold_sql=gold_sql)['match']
        if is_match:
            match_count += 1

        results.append({'di': item['dataset_index'], 'match': is_match,
                        'n_groups': len(groups) if candidates else 0,
                        'n_candidates': len(candidates)})
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/100] match={match_count}/{i+1} ({match_count/(i+1):.1%})")

    elapsed = time.time() - start_t
    rate = match_count / 100
    print(f"\n=== M5-PLUS RESULT (5 prompts + arbitration) ===")
    print(f"Match: {match_count}/100 ({rate:.1%})")
    print(f"Time:  {elapsed:.0f}s")
    print(f"M5 (3prompt投票): 85.0% | Baseline: 81.0%")
    print(f"Delta vs M5: {rate-0.85:+.1%}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'summary.json', 'w') as f:
        json.dump({'method': 'm5plus_5prompt_arbitration', 'match_count': match_count, 'match_rate': rate,
                   'elapsed_seconds': round(elapsed, 1)}, f, indent=2)


if __name__ == '__main__':
    main()
