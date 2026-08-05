#!/bin/bash
#SBATCH --job-name=eval_csc
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eval_csc_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eval_csc_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
$BASE/envs/reasoning3b/bin/python $BASE/src/evaluate_after_grpo.py \
    --limit 100 --spider-dir $BASE/data/spider_data \
    --model-path $BASE/models/CscSQL-Grpo-7B \
    --output-dir $BASE/outputs/eval_csc_100 --batch-size 4

echo "=== CSC-SQL Result ==="
$BASE/envs/reasoning3b/bin/python -c "
import json
with open('$BASE/outputs/eval_csc_100/summary.json') as f:
    s = json.load(f)
print(f\"Match: {s['custom_execution_match_rate']:.1%}\")
print(f\"Exec:  {s['prediction_execution_success_rate']:.1%}\")
print(f\"Parse: {s['parse_success_rate']:.1%}\")
"
