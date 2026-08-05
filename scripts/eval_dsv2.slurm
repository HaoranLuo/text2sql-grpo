#!/bin/bash
#SBATCH --job-name=eval_dsv2
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eval_dsv2_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eval_dsv2_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
$BASE/envs/reasoning3b/bin/python $BASE/src/evaluate_after_grpo.py \
    --limit 100 --spider-dir $BASE/data/spider_data \
    --model-path $BASE/models/deepseek-coder-v2-lite \
    --output-dir $BASE/outputs/eval_dsv2_100 --batch-size 2

echo "=== Result ==="
$BASE/envs/reasoning3b/bin/python -c "
import json
with open('$BASE/outputs/eval_dsv2_100/summary.json') as f:
    s = json.load(f)
print(f\"Parse: {s['parse_success_rate']:.1%}\")
print(f\"Exec:  {s['prediction_execution_success_rate']:.1%}\")
print(f\"Match: {s['custom_execution_match_rate']:.1%}\")
"
