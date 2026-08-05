#!/usr/bin/env python3
"""Self-consistency via beam search: 5 beams per question, majority vote."""
import json, sys, time, torch
from collections import Counter
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
from reasoning_generator_agent import ReasoningGeneratorAgent
from spider_utils import SpiderLoader, DatabaseExecutor, compare_execution_results

N_BEAMS = 5
MODEL_PATH = str(PROJECT / 'models' / 'qwen2.5-coder-7b-instruct')
SPIDER = str(PROJECT / 'data' / 'spider_data')
OUT = PROJECT / 'outputs' / 'eval_sc_100'
BATCH_SIZE = 1  # questions per batch

print(f'Self-Consistency: beam search {N_BEAMS} beams, batch={BATCH_SIZE}')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map={'':0}, local_files_only=True, trust_remote_code=True)
model.eval()

loader = SpiderLoader(SPIDER)
executor = DatabaseExecutor(SPIDER)
items = loader.load_dev(limit=100, start_index=0)

# Pre-build prompts
for item in items:
    ddl, _ = loader.get_ddl_with_source(item['db_id'])
    item['_chat'] = tokenizer.apply_chat_template(
        [{'role':'user','content': ReasoningGeneratorAgent.build_prompt(question=item['question'], ddl_schema=ddl, dialect='sqlite')}],
        tokenize=False, add_generation_prompt=True)

results = []; match_count = 0; start_t = time.time()
orig_side = tokenizer.padding_side

for batch_start in range(0, len(items), BATCH_SIZE):
    batch_items = items[batch_start: batch_start+BATCH_SIZE]
    chats = [it['_chat'] for it in batch_items]

    tokenizer.padding_side = 'left'
    enc = tokenizer(chats, return_tensors='pt', padding=True, truncation=True, max_length=1536).to('cuda:0')
    tokenizer.padding_side = orig_side
    padded_len = enc['input_ids'].shape[1]

    # Beam search: 5 beams per prompt, deterministic, diverse
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=256, num_beams=N_BEAMS, num_return_sequences=N_BEAMS,
                             early_stopping=True, pad_token_id=tokenizer.eos_token_id)

    # Process: output is [batch*N_BEAMS, seq_len]
    for qi, item in enumerate(batch_items):
        db_id = item['db_id']; gold_sql = item['query']
        exec_results = []
        for k in range(N_BEAMS):
            idx = qi * N_BEAMS + k
            gen_ids = out[idx][padded_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            parsed = ReasoningGeneratorAgent.extract_sql(text)
            sql = parsed['sql'] if parsed['parse_success'] else ''
            if sql:
                r = executor.execute(db_id, sql)
                if r['success']:
                    exec_results.append(tuple(tuple(row) for row in r['saved_rows']))

        valid = [e for e in exec_results if e is not None]
        if valid:
            mc = Counter(valid).most_common(1)[0]
            voted_rows, vote_count = [list(row) for row in mc[0]], mc[1]
        else:
            voted_rows, vote_count = [], 0

        gold_r = executor.execute(db_id, gold_sql)
        gold_rows = gold_r['saved_rows'] if gold_r['success'] else []
        if vote_count > 0 and gold_r['success']:
            is_match = compare_execution_results(voted_rows, gold_rows, gold_sql=gold_sql)['match']
        else:
            is_match = False
        if is_match: match_count += 1

        results.append({'dataset_index': item['dataset_index'], 'voted_match': is_match, 'vote_count': vote_count, 'n_valid': len(valid)})

    n = len(results); elapsed = time.time() - start_t
    print(f'  [{n}/100] match={match_count}/{n} ({match_count/n:.1%})  {elapsed:.0f}s')

elapsed = time.time() - start_t
rate = match_count / 100
print(f'\n=== BEAM-SC ({N_BEAMS} beams) ===')
print(f'Match: {match_count}/100 ({rate:.1%})')
print(f'Time:  {elapsed:.0f}s')
print(f'Baseline: 81.0%')
print(f'Delta:    {rate-0.81:+.1%}')

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / 'summary.json', 'w') as f:
    json.dump({'method': f'beam_self_consistency_{N_BEAMS}', 'match_count': match_count, 'match_rate': rate, 'elapsed_seconds': round(elapsed,1)}, f, indent=2)
