#!/bin/bash
#SBATCH --job-name=p2d
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/p2d_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/p2d_%j.err

# P2D: 3B + partial奖励 + 500条 + G4 + 25步
# 假设: partial(细粒度信号) × 更多数据 可能优于三级(100条50% / 500条50%)
set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/p2d_partial500

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== P2D: 3B + partial奖励 + 500条 + G=4 + 25步 ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 500 --num-generations 4 --max-steps 25 \
    --reward-type partial --learning-rate 3e-6 --beta 0.04 --temperature 1.0 \
    --train-batch-size 8 \
    --output-dir "$OUT" --spider-dir "$SPIDER" --model-path "$MODEL"

echo "Training done: $(date)"

$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path "$OUT" --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/p2d_partial500 --batch-size 8

echo "=== P2D Result ==="
$PYTHON -c "
import json
with open('$BASE/outputs/p2d_partial500/summary.json') as f:
    s = json.load(f)
print(f'partial500: {s[\"custom_execution_match_rate\"]:.1%}  (三级500=50%, partial100=46%, 基线45%)')
"
echo "Done: $(date)"
