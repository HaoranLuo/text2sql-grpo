#!/bin/bash
#SBATCH --job-name=p2c2_partial
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/p2c2_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/p2c2_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/p2c2_partial

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== C2: 3B + partial奖励 + 100条 + G=8 (与A1控制变量) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 100 --num-generations 8 --max-steps 25 \
    --reward-type partial --learning-rate 3e-6 --beta 0.04 --temperature 1.0 \
    --output-dir "$OUT" --spider-dir "$SPIDER" --model-path "$MODEL"

echo "Training done: $(date)"

$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path "$OUT" --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/p2c2_partial --batch-size 8

echo "=== C2 Result ==="
$PYTHON -c "
import json
with open('$BASE/outputs/p2c2_partial/summary.json') as f:
    s = json.load(f)
print(f'partial奖励: {s[\"custom_execution_match_rate\"]:.1%}  (基线45%, 三级奖励A1=?)')
"
echo "Done: $(date)"
