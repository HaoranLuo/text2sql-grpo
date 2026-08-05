#!/bin/bash
#SBATCH --job-name=eval_sc
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eval_sc_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eval_sc_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON -c "
import json, time, torch
from collections import Counter
from pathlib import Path

MODEL_PATH = '$BASE/models/qwen2.5-coder-7b-instruct'
SPIDER = '$BASE/data/spider_data'
OUT = '$BASE/outputs/eval_sc_100'
N = 5  # candidates per question

from transformers import AutoModelForCausalLM, AutoTokenizer
from spider_utils import SpiderLoader, DatabaseExecutor, compare_execution_results

print(f'Self-Consistency: {N} candidates via batched sampling')
agent_prompt_builder = None
exec('from reasoning_generator_agent import ReasoningGeneratorAgent; global agent_prompt_builder; agent_prompt_builder = ReasoningGeneratorAgent.build_prompt')

# Load model once
print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map={'':0}, local_files_only=True, trust_remote_code=True)
model.eval()

loader = SpiderLoader(SPIDER)
executor = DatabaseExecutor(SPIDER)
items = loader.load_dev(limit=100, start_index=0)

results = []; match_count = 0; start = time.time()

for i, item in enumerate(items):
    db_id = item['db_id']; question = item['question']; gold_sql = item['query']
    ddl, _ = loader.get_ddl_with_source(db_id)
    prompt = agent_prompt_builder(question=question, ddl_schema=ddl, schema_links=None, evidence=None, dialect='sqlite')
    chat = tokenizer.apply_chat_template([{'role':'user','content':prompt}], tokenize=False, add_generation_prompt=True)

    # Batch: N copies of same prompt for diverse sampling
    chats = [chat] * N
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    enc = tokenizer(chats, return_tensors='pt', padding=True, truncation=True, max_length=1536).to('cuda:0')
    tokenizer.padding_side = orig_side
    padded_len = enc['input_ids'].shape[1]

    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=256, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tokenizer.eos_token_id)

    # Extract SQL from each candidate
    exec_results = []
    for k in range(N):
        gen_ids = out[k][padded_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        parsed = ReasoningGeneratorAgent.extract_sql(text)
        sql = parsed['sql'] if parsed['parse_success'] else ''
        if sql:
            r = executor.execute(db_id, sql)
            if r['success']:
                key = tuple(tuple(row) for row in r['saved_rows'])
                exec_results.append(key)
            else:
                exec_results.append(None)
        else:
            exec_results.append(None)

    # Majority vote
    valid = [e for e in exec_results if e is not None]
    if valid:
        most_common = Counter(valid).most_common(1)[0][0]
        vote_count = Counter(valid).most_common(1)[0][1]
        voted_rows = [list(row) for row in most_common]
    else:
        voted_rows = []; vote_count = 0

    gold_r = executor.execute(db_id, gold_sql)
    gold_rows = gold_r['saved_rows'] if gold_r['success'] else []
    if vote_count > 0 and gold_r['success']:
        cmp = compare_execution_results(voted_rows, gold_rows, gold_sql=gold_sql)
        is_match = cmp['match']
    else:
        is_match = False
    if is_match: match_count += 1

    results.append({'dataset_index': item['dataset_index'], 'db_id': db_id, 'question': question, 'voted_match': is_match, 'vote_count': vote_count, 'n_valid': len(valid)})

    if (i+1) % 10 == 0:
        e = time.time() - start
        print(f'  [{i+1}/100] match={match_count}/{i+1} ({match_count/(i+1):.1%})  {e:.0f}s')

elapsed = time.time() - start
rate = match_count / 100
print(f'\n=== SELF-CONSISTENCY ({N} candidates) ===')
print(f'Match: {match_count}/100 ({rate:.1%})')
print(f'Time:  {elapsed:.0f}s')
print(f'Qwen7B zero-shot: 81.0%')
print(f'Delta:            {rate-0.81:+.1%}')

Path(OUT).mkdir(parents=True, exist_ok=True)
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump({'method': f'self_consistency_{N}', 'model': MODEL_PATH, 'match_count': match_count, 'match_rate': rate, 'elapsed_seconds': round(elapsed,1), 'candidates': N}, f, indent=2)
"
