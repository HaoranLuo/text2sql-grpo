#!/bin/bash
#SBATCH --job-name=exp_reward
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/exp_reward_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/exp_reward_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/qwen2.5-coder-7b-instruct
SPIDER=$BASE/data/spider_data
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "GPU")

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "Reward Function Experiment"
echo "Node: $(hostname) | GPU: $GPU | $(date)"
echo "============================================"
echo ""
echo "7B Baseline: 81% Spider EX (zero-shot)"
echo ""
echo "Group B: binary (0/1) x 25 steps (CONTROL)"
echo "Group A: three_level (1.0/0.1/0.0) x 50 steps (EXPERIMENT)"
echo ""

# =========================================================================
# GROUP B: Binary reward, 25 steps (CONTROL)
# =========================================================================
echo "============================================"
echo "GROUP B: binary reward x 25 steps"
echo "============================================"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 100 \
    --num-generations 2 \
    --max-steps 25 \
    --reward-type binary \
    --learning-rate 5e-6 \
    --beta 0.04 \
    --temperature 0.7 \
    --output-dir $BASE/checkpoints/grpo_7b_binary_25 \
    --spider-dir $SPIDER \
    --model-path $MODEL

echo "Group B training done: $(date)"

$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/grpo_7b_binary_25 \
    --limit 100 --spider-dir $SPIDER --model-path $MODEL \
    --output-dir $BASE/outputs/grpo_7b_binary_25 --batch-size 4

B_RESULT=$($PYTHON -c "import json; s=json.load(open('$BASE/outputs/grpo_7b_binary_25/summary.json')); print(s['custom_execution_match_rate'])")
echo "Group B result: $B_RESULT"

# =========================================================================
# GROUP A: Three-level reward, 50 steps (EXPERIMENT)
# =========================================================================
echo ""
echo "============================================"
echo "GROUP A: three_level reward x 50 steps"
echo "============================================"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 100 \
    --num-generations 2 \
    --max-steps 50 \
    --reward-type three_level \
    --learning-rate 5e-6 \
    --beta 0.04 \
    --temperature 0.7 \
    --output-dir $BASE/checkpoints/grpo_7b_threelevel_50 \
    --spider-dir $SPIDER \
    --model-path $MODEL

echo "Group A training done: $(date)"

# Evaluate final (step 50)
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/grpo_7b_threelevel_50 \
    --limit 100 --spider-dir $SPIDER --model-path $MODEL \
    --output-dir $BASE/outputs/grpo_7b_threelevel_50 --batch-size 4

# Evaluate checkpoint-25 (early stopping check)
if [ -d $BASE/checkpoints/grpo_7b_threelevel_50/checkpoint-25 ]; then
    $PYTHON $BASE/src/evaluate_after_grpo.py \
        --lora-path $BASE/checkpoints/grpo_7b_threelevel_50/checkpoint-25 \
        --limit 100 --spider-dir $SPIDER --model-path $MODEL \
        --output-dir $BASE/outputs/grpo_7b_threelevel_25 --batch-size 4
fi

# =========================================================================
# SUMMARY
# =========================================================================
echo ""
echo "============================================"
echo "FINAL RESULTS"
echo "============================================"
echo ""
echo "7B Baseline (zero-shot):  81.0%"
echo "Group B (binary x 25):    $B_RESULT"
A50=$($PYTHON -c "import json; s=json.load(open('$BASE/outputs/grpo_7b_threelevel_50/summary.json')); print(s['custom_execution_match_rate'])")
echo "Group A (three_level x50): $A50"
A25="N/A"
if [ -f $BASE/outputs/grpo_7b_threelevel_25/summary.json ]; then
    A25=$($PYTHON -c "import json; s=json.load(open('$BASE/outputs/grpo_7b_threelevel_25/summary.json')); print(s['custom_execution_match_rate'])")
fi
echo "Group A (three_level x25): $A25"
echo ""
echo "Done: $(date)"
