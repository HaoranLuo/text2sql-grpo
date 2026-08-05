#!/bin/bash
#SBATCH --job-name=grpo_tuned
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/grpo_tuned_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/grpo_tuned_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/qwen2.5-coder-7b-instruct
SPIDER=$BASE/data/spider_data

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== GRPO Tuned: lower LR, higher beta, three_level reward ==="
echo "lr=1e-6, beta=0.1, 25 steps"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 100 \
    --num-generations 2 \
    --max-steps 25 \
    --reward-type three_level \
    --learning-rate 1e-6 \
    --beta 0.1 \
    --temperature 0.7 \
    --output-dir $BASE/checkpoints/grpo_7b_tuned_25 \
    --spider-dir $SPIDER \
    --model-path $MODEL

echo "Training done: $(date)"

$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/grpo_7b_tuned_25 \
    --limit 100 --spider-dir $SPIDER --model-path $MODEL \
    --output-dir $BASE/outputs/grpo_7b_tuned_25 --batch-size 4

TUNED=$($PYTHON -c "import json; s=json.load(open('$BASE/outputs/grpo_7b_tuned_25/summary.json')); print(s['custom_execution_match_rate'])")
echo "GRPO Tuned result: $TUNED"
echo "Done: $(date)"
