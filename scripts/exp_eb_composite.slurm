#!/bin/bash
#SBATCH --job-name=eb_composite
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eb_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eb_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/qwen2.5-coder-7b-instruct
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/grpo_eb

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== E-B: Composite dense reward (G=8, temp=1.0) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 100 \
    --num-generations 8 \
    --max-steps 50 \
    --reward-type partial \
    --learning-rate 3e-6 \
    --beta 0.04 \
    --temperature 1.0 \
    --output-dir "$OUT" \
    --spider-dir "$SPIDER" \
    --model-path "$MODEL"

echo "Training done: $(date)"

$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path "$OUT" --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/grpo_eb --batch-size 4

echo "=== E-B Result ==="
$PYTHON -c "import json; s=json.load(open('$BASE/outputs/grpo_eb/summary.json')); print(f'Match: {s[\"custom_execution_match_rate\"]:.1%} (baseline 81%)')"
echo "Done: $(date)"
