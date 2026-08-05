#!/bin/bash
#SBATCH --job-name=14b_grpo
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/14b_grpo_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/14b_grpo_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-14B
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/grpo_14b

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 14B GRPO Training ==="
echo "Started: $(date) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 100 \
    --num-generations 2 \
    --max-steps 50 \
    --reward-type three_level \
    --learning-rate 5e-6 \
    --beta 0.04 \
    --temperature 0.7 \
    --output-dir "$OUT" \
    --spider-dir "$SPIDER" \
    --model-path "$MODEL"

echo "Training done: $(date)"

$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path "$OUT" --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/grpo_14b_100 --batch-size 1

echo "=== 14B GRPO Result ==="
$PYTHON -c "import json; s=json.load(open('$BASE/outputs/grpo_14b_100/summary.json')); print(f'Match: {s[\"custom_execution_match_rate\"]:.1%}')"
echo "Done: $(date)"
