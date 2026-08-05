#!/bin/bash
#SBATCH --job-name=sft_grpo
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/sft_grpo_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/sft_grpo_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/qwen2.5-coder-7b-instruct
SPIDER=$BASE/data/spider_data
API_DATA=$BASE/data/api_sft_data.json

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "SFT + GRPO Pipeline"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "============================================"

# =========================================================================
# Step 1: SFT Cold-Start
# =========================================================================
echo ""
echo "=== Step 1: SFT Training (imitating DeepSeek API) ==="

$PYTHON $BASE/src/train_sft.py \
    --data "$API_DATA" \
    --output $BASE/checkpoints/sft_coldstart \
    --model-path "$MODEL" \
    --epochs 2 \
    --batch-size 1 \
    --lr 2e-5

echo "SFT done: $(date)"

# =========================================================================
# Step 2: GRPO on SFT model
# =========================================================================
echo ""
echo "=== Step 2: GRPO Training (three-level reward) ==="

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 100 \
    --num-generations 2 \
    --max-steps 50 \
    --reward-type three_level \
    --learning-rate 5e-6 \
    --beta 0.04 \
    --temperature 0.7 \
    --output-dir $BASE/checkpoints/grpo_sft_50 \
    --spider-dir "$SPIDER" \
    --model-path "$MODEL"

echo "GRPO done: $(date)"

# =========================================================================
# Step 3: Evaluate
# =========================================================================
echo ""
echo "=== Step 3: Evaluation ==="

# Baseline (SFT model, before GRPO)
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/sft_coldstart \
    --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/eval_sft_100 --batch-size 4

# After GRPO (ckpt-25)
if [ -d $BASE/checkpoints/grpo_sft_50/checkpoint-25 ]; then
    $PYTHON $BASE/src/evaluate_after_grpo.py \
        --lora-path $BASE/checkpoints/grpo_sft_50/checkpoint-25 \
        --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
        --output-dir $BASE/outputs/eval_sft_grpo_25 --batch-size 4
fi

# After GRPO (ckpt-50)
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/grpo_sft_50 \
    --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/eval_sft_grpo_50 --batch-size 4

# =========================================================================
# Summary
# =========================================================================
echo ""
echo "============================================"
echo "FINAL RESULTS"
echo "============================================"
echo "7B zero-shot:        81%"
SFT=$($PYTHON -c "import json; s=json.load(open('$BASE/outputs/eval_sft_100/summary.json')); print(s['custom_execution_match_rate'])")
echo "After SFT:            $SFT"
G25="N/A"
[ -f $BASE/outputs/eval_sft_grpo_25/summary.json ] && G25=$($PYTHON -c "import json; s=json.load(open('$BASE/outputs/eval_sft_grpo_25/summary.json')); print(s['custom_execution_match_rate'])")
G50=$($PYTHON -c "import json; s=json.load(open('$BASE/outputs/eval_sft_grpo_50/summary.json')); print(s['custom_execution_match_rate'])")
echo "After SFT+GRPO 25:   $G25"
echo "After SFT+GRPO 50:   $G50"
echo ""
echo "Done: $(date)"
