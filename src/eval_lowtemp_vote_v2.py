#!/usr/bin/env python3
"""M2-v2: 低温投票修正版 — 4候选(减半防超时) + 注释剥离提取器"""
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
OUT = PROJECT / 'outputs' / 'eval_lowtemp_vote_v2'

N_CAND = 4
TEMP = 0.2


def main():
    print(f"M2-v2: Low-temp voting ({N_CAND} candidates, temp={TEMP})")
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
        prompt = ReasoningGeneratorAgent.build_prompt(question=question, ddl_schema=ddl, dialect='sqlite')
        chat = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)

        exec_results = []
        for _ in range(N_CAND):
            enc = tokenizer(chat, return_tensors='pt', truncation=True, max_length=1536).to('cuda:0')
            in_len = enc['input_ids'].shape[1]
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=256, do_sample=True, temperature=TEMP,
                                     top_p=0.9, pad_token_id=tokenizer.eos_token_id)
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
        is_match = vc > 0 and gold_r['success'] and compare_execution_results(voted_rows, gold_rows, gold_sql=gold_sql)['match']
        if is_match:
            match_count += 1

        results.append({'di': item['dataset_index'], 'match': is_match, 'votes': vc, 'valid': len(exec_results)})
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/100] match={match_count}/{i+1} ({match_count/(i+1):.1%})")

    elapsed = time.time() - start_t
    rate = match_count / 100
    print(f"\n=== M2-v2 RESULT ===")
    print(f"Match: {match_count}/100 ({rate:.1%})")
    print(f"Time:  {elapsed:.0f}s")
    print(f"Baseline: 81.0% | M5(3prompt): 85.0%")
    print(f"Delta vs base: {rate-0.81:+.1%}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'summary.json', 'w') as f:
        json.dump({'method': f'lowtemp_vote_v2_{N_CAND}_t{TEMP}', 'match_count': match_count, 'match_rate': rate,
                   'elapsed_seconds': round(elapsed, 1)}, f, indent=2)


if __name__ == '__main__':
    main()
