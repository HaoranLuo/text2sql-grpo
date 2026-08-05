#!/bin/bash
#SBATCH --job-name=c3_atomic
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/c3_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/c3_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/c3_atomic

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== C3: 3B + 原子奖励(FINER) + 100条 + G=4 ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 100 --num-generations 4 --max-steps 25 \
    --reward-type atomic --learning-rate 3e-6 --beta 0.04 --temperature 1.0 \
    --output-dir "$OUT" --spider-dir "$SPIDER" --model-path "$MODEL"

echo "Training done: $(date)"

$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path "$OUT" --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/c3_atomic --batch-size 8

echo "=== C3 Result ==="
$PYTHON -c "
import json
with open('$BASE/outputs/c3_atomic/summary.json') as f:
    s = json.load(f)
print(f'原子奖励: {s[\"custom_execution_match_rate\"]:.1%}  (基线45%, 三级G4=50%, 三级G8=47%, partial=46%)')
"
echo "Done: $(date)"
