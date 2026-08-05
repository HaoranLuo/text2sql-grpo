#!/usr/bin/env python3
"""Self-consistency: num_return_sequences=5 in one generate() call, then vote."""
import json, sys, time, torch
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
from spider_utils import SpiderLoader, DatabaseExecutor, compare_execution_results
from transformers import AutoModelForCausalLM, AutoTokenizer
from reasoning_generator_agent import ReasoningGeneratorAgent

N = 5  # diverse candidates per prompt
MODEL = str(PROJECT / 'models' / 'qwen2.5-coder-7b-instruct')
SPIDER = str(PROJECT / 'data' / 'spider_data')
OUT = PROJECT / 'outputs' / 'eval_sc_100'

print(f'Self-Consistency v2: num_return_sequences={N} (single forward pass)')

tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={'':0}, local_files_only=True, trust_remote_code=True)
model.eval()
print(f'VRAM: {torch.cuda.memory_allocated(0)/1024**3:.1f} GiB')

loader = SpiderLoader(SPIDER); executor = DatabaseExecutor(SPIDER)
items = loader.load_dev(limit=100, start_index=0)
for it in items:
    ddl, _ = loader.get_ddl_with_source(it['db_id'])
    it['_chat'] = tokenizer.apply_chat_template(
        [{'role':'user','content': ReasoningGeneratorAgent.build_prompt(question=it['question'], ddl_schema=ddl, dialect='sqlite')}],
        tokenize=False, add_generation_prompt=True)

results = []; match_count = 0; start_t = time.time()

for i, item in enumerate(items):
    enc = tokenizer(item['_chat'], return_tensors='pt', truncation=True, max_length=1536).to('cuda:0')
    in_len = enc['input_ids'].shape[1]

    # ONE forward pass → N diverse candidates via sampling
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=256, do_sample=True, temperature=0.8, top_p=0.95,
                             num_return_sequences=N, pad_token_id=tokenizer.eos_token_id)

    exec_results = []
    for k in range(N):
        gen_ids = out[k][in_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        parsed = ReasoningGeneratorAgent.extract_sql(text)
        if parsed['parse_success']:
            r = executor.execute(item['db_id'], parsed['sql'])
            if r['success']:
                exec_results.append(tuple(tuple(row) for row in r['saved_rows']))

    if exec_results:
        mc = Counter(exec_results).most_common(1)[0]
        voted_rows, vc = [list(row) for row in mc[0]], mc[1]
    else:
        voted_rows, vc = [], 0

    gold_r = executor.execute(item['db_id'], item['query'])
    gold_rows = gold_r['saved_rows'] if gold_r['success'] else []
    is_match = vc > 0 and gold_r['success'] and compare_execution_results(voted_rows, gold_rows, gold_sql=item['query'])['match']
    if is_match: match_count += 1

    results.append({'di': item['dataset_index'], 'match': is_match, 'votes': vc, 'valid': len(exec_results)})
    if (i+1) % 10 == 0:
        e = time.time() - start_t
        print(f'  [{i+1}/100] match={match_count}/{i+1} ({match_count/(i+1):.1%})  {e:.0f}s')

elapsed = time.time() - start_t
rate = match_count / 100
print(f'\n=== SELF-CONSISTENCY v2 ({N} candidates) ===')
print(f'Match: {match_count}/100 ({rate:.1%})')
print(f'Time:  {elapsed:.0f}s ({elapsed/100:.1f}s/q)')
print(f'Baseline: 81.0%  |  Delta: {rate-0.81:+.1%}')

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / 'summary.json', 'w') as f:
    json.dump({'method': f'sc_v2_{N}', 'match_count': match_count, 'match_rate': rate, 'elapsed': round(elapsed,1)}, f, indent=2)
