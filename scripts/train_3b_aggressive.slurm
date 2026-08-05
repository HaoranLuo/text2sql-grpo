#!/bin/bash
#SBATCH --job-name=grpo_3b_xl
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/grpo_3bxl_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/grpo_3bxl_%j.err

set -e

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
OUTPUT_DIR=$BASE/checkpoints/grpo_3b_aggressive
SPIDER_DIR=$BASE/data/spider_data
MODEL_PATH=$BASE/models/Qwen2.5-Coder-3B-Instruct

mkdir -p "$BASE/logs" "$OUTPUT_DIR"
export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "3B Aggressive GRPO Training"
echo "Started: $(date) | Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Config: 1000 samples, 200 steps, 4 generations"
echo "============================================"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 1000 \
    --num-generations 4 \
    --max-steps 200 \
    --learning-rate 5e-6 \
    --beta 0.04 \
    --temperature 0.7 \
    --output-dir "$OUTPUT_DIR" \
    --spider-dir "$SPIDER_DIR" \
    --model-path "$MODEL_PATH"

echo ""
echo "Training done: $(date)"

# Post-training eval
echo "=== Post-Training Evaluation ==="
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path "$OUTPUT_DIR" \
    --limit 100 \
    --spider-dir "$SPIDER_DIR" \
    --model-path "$MODEL_PATH" \
    --output-dir "$BASE/outputs/grpo_3b_aggressive_100" \
    --batch-size 8

# Compare
echo ""
echo "=== Before vs After ==="
BL_SUM="$BASE/outputs/baseline_pretrain_100/summary.json"
GR_SUM="$BASE/outputs/grpo_3b_aggressive_100/summary.json"
if [ -f "$BL_SUM" ] && [ -f "$GR_SUM" ]; then
    $PYTHON -c "
import json
with open('$BL_SUM') as f: bl = json.load(f)
with open('$GR_SUM') as f: gr = json.load(f)
print(f'Baseline: {bl[\"custom_execution_match_rate\"]:.1%}')
print(f'After 200-step GRPO: {gr[\"custom_execution_match_rate\"]:.1%}')
print(f'Delta: {gr[\"custom_execution_match_rate\"]-bl[\"custom_execution_match_rate\"]:+.1%}')
"
fi

echo "Done: $(date)"
