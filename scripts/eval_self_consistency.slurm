#!/bin/bash
#SBATCH --job-name=eval_sc
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eval_sc_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eval_sc_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON -c "
import json, time, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, '$BASE/src')
from reasoning_generator_agent import ReasoningGeneratorAgent
from spider_utils import SpiderLoader, DatabaseExecutor, compare_execution_results, compute_summary

MODEL = '$BASE/models/qwen2.5-coder-7b-instruct'
SPIDER = '$BASE/data/spider_data'
OUT = '$BASE/outputs/eval_sc_100'
N_CANDIDATES = 5

print(f'Self-Consistency: {N_CANDIDATES} candidates per question')
print(f'Model: {MODEL}')

agent = ReasoningGeneratorAgent(model_path=MODEL, max_new_tokens=512)
loader = SpiderLoader(SPIDER)
executor = DatabaseExecutor(SPIDER)
items = loader.load_dev(limit=100, start_index=0)

results = []
match_count = 0
start = time.time()

for i, item in enumerate(items):
    di = item['dataset_index']
    db_id = item['db_id']
    question = item['question']
    gold_sql = item['query']

    # Get DDL
    ddl, _ = loader.get_ddl_with_source(db_id)

    # Generate N candidates (sequential, same prompt)
    candidates = []
    for _ in range(N_CANDIDATES):
        gen = agent.generate(question=question, ddl_schema=ddl, dialect='sqlite')
        c = gen['candidates'][0]
        if c['parse_success']:
            candidates.append(c['sql'])
        else:
            candidates.append('')

    # Execute each candidate
    exec_results = []
    for sql in candidates:
        if not sql:
            exec_results.append(None)
            continue
        r = executor.execute(db_id, sql)
        if r['success']:
            # Use row tuples as hashable keys
            key = tuple(tuple(row) for row in r['saved_rows'])
            exec_results.append(key)
        else:
            exec_results.append(None)

    # Majority vote among successful executions
    valid = [e for e in exec_results if e is not None]
    if valid:
        most_common = Counter(valid).most_common(1)[0][0]
        vote_count = Counter(valid).most_common(1)[0][1]
        # Convert back to list of lists
        voted_rows = [list(row) for row in most_common]
    else:
        voted_rows = []
        vote_count = 0

    # Execute gold for comparison
    gold_r = executor.execute(db_id, gold_sql)
    gold_rows = gold_r['saved_rows'] if gold_r['success'] else []

    # Compare voted result with gold
    if vote_count > 0 and gold_r['success']:
        cmp = compare_execution_results(voted_rows, gold_rows, gold_sql=gold_sql)
        is_match = cmp['match']
    else:
        is_match = False

    if is_match:
        match_count += 1

    results.append({
        'dataset_index': di, 'db_id': db_id, 'question': question,
        'voted_match': is_match, 'vote_count': vote_count,
        'n_valid': len(valid), 'n_candidates': N_CANDIDATES,
    })

    if (i+1) % 10 == 0:
        elapsed = time.time() - start
        print(f'  [{i+1}/100] match={match_count}/{i+1} ({match_count/(i+1):.1%})  {elapsed:.0f}s')

elapsed = time.time() - start
match_rate = match_count / 100

print(f'')
print(f'=== SELF-CONSISTENCY RESULT ({N_CANDIDATES} candidates) ===')
print(f'Match: {match_count}/100 ({match_rate:.1%})')
print(f'Time:  {elapsed:.0f}s ({elapsed/100:.1f}s per question)')

# Save
Path(OUT).mkdir(parents=True, exist_ok=True)
summary = {
    'method': f'self_consistency_{N_CANDIDATES}',
    'model': MODEL,
    'match_count': match_count,
    'match_rate': match_rate,
    'elapsed_seconds': round(elapsed, 1),
    'candidates_per_question': N_CANDIDATES,
}
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
with open(f'{OUT}/items.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Comparison
baseline = 0.81
diff = match_rate - baseline
print(f'')
print(f'Qwen7B zero-shot: {baseline:.1%}')
print(f'Self-consistency: {match_rate:.1%}')
print(f'Delta:            {diff:+.1%}')
"
