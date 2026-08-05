#!/bin/bash
#SBATCH --job-name=3b_fixed
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/3b_fixed_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/3b_fixed_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/grpo_3b_fixed

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 3B GRPO with ROOT-CAUSE fixes (chat prompt + list compat) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 100 \
    --num-generations 4 \
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
    --output-dir $BASE/outputs/grpo_3b_fixed --batch-size 8

echo "=== 3B Fixed Result ==="
$PYTHON -c "import json; s=json.load(open('$BASE/outputs/grpo_3b_fixed/summary.json')); print(f'Match: {s[\"custom_execution_match_rate\"]:.1%} (baseline 37%)')"
echo "Done: $(date)"
