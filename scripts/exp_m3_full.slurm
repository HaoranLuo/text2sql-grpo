#!/bin/bash
#SBATCH --job-name=m3_full
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/m3_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/m3_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/qwen2.5-coder-7b-instruct
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/grpo_m3_full

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== M3: Full-data GRPO (7000 examples, G=8) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 7000 \
    --num-generations 8 \
    --max-steps 100 \
    --reward-type three_level \
    --learning-rate 3e-6 \
    --beta 0.04 \
    --temperature 1.0 \
    --output-dir "$OUT" \
    --spider-dir "$SPIDER" \
    --model-path "$MODEL"

echo "Training done: $(date)"

$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path "$OUT" --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/grpo_m3_full --batch-size 4

echo "=== M3 Result ==="
$PYTHON -c "import json; s=json.load(open('$BASE/outputs/grpo_m3_full/summary.json')); print(f'Match: {s[\"custom_execution_match_rate\"]:.1%} (baseline 81%)')"
echo "Done: $(date)"
