#!/bin/bash
#SBATCH --job-name=exp_partial
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/exp_partial_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/exp_partial_%j.err

set -e

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/qwen2.5-coder-7b-instruct
SPIDER=$BASE/data/spider_data
OUTPUT_DIR=$BASE/checkpoints/grpo_7b_partial_50
EVAL_DIR=$BASE/outputs/grpo_7b_partial_50
BL_EVAL_DIR=$BASE/outputs/baseline_7b_100

mkdir -p "$BASE/logs" "$OUTPUT_DIR" "$EVAL_DIR"
export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "Experiment: partial reward (row-overlap credit)"
echo "7B model on A40 | 100 examples | 50 steps | 2 gens"
echo "lr=5e-6 | beta=0.04 | reward-type=partial"
echo "Started: $(date) | Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================"

# =========================================================================
# Step 1: Baseline (pre-training)
# =========================================================================
echo ""
echo "=== Step 1: Baseline (pre-training) ==="
if [ ! -f "$BL_EVAL_DIR/summary.json" ]; then
    $PYTHON $BASE/src/evaluate_after_grpo.py \
        --limit 100 \
        --spider-dir "$SPIDER" \
        --model-path "$MODEL" \
        --output-dir "$BL_EVAL_DIR" \
        --batch-size 4
else
    echo "Baseline already exists, skipping: $BL_EVAL_DIR"
fi

BL_SUM="$BL_EVAL_DIR/summary.json"
if [ -f "$BL_SUM" ]; then
    $PYTHON -c "
import json
with open('$BL_SUM') as f: s = json.load(f)
print(f'7B Baseline: Parse {s[\"parse_success_rate\"]:.1%}, Exec {s[\"prediction_execution_success_rate\"]:.1%}, Match {s[\"custom_execution_match_rate\"]:.1%}')
"
fi

# =========================================================================
# Step 2: GRPO training with partial reward
# =========================================================================
echo ""
echo "=== Step 2: GRPO Training (partial reward, 50 steps) ==="
$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 100 \
    --num-generations 2 \
    --max-steps 50 \
    --reward-type partial \
    --learning-rate 5e-6 \
    --beta 0.04 \
    --temperature 0.7 \
    --output-dir "$OUTPUT_DIR" \
    --spider-dir "$SPIDER" \
    --model-path "$MODEL"

echo "Training done: $(date)"

# =========================================================================
# Step 3: Post-training evaluation
# =========================================================================
echo ""
echo "=== Step 3: Post-Training Evaluation ==="
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path "$OUTPUT_DIR" \
    --limit 100 \
    --spider-dir "$SPIDER" \
    --model-path "$MODEL" \
    --output-dir "$EVAL_DIR" \
    --batch-size 4

# =========================================================================
# Step 4: Compare before vs after
# =========================================================================
echo ""
echo "=== Step 4: Before vs After ==="
GR_SUM="$EVAL_DIR/summary.json"
if [ -f "$BL_SUM" ] && [ -f "$GR_SUM" ]; then
    $PYTHON -c "
import json
with open('$BL_SUM') as f: bl = json.load(f)
with open('$GR_SUM') as f: gr = json.load(f)
print()
print('┌──────────────────────┬──────────┬──────────┬──────────┐')
print('│ Metric               │  Before  │  After   │  Delta   │')
print('├──────────────────────┼──────────┼──────────┼──────────┤')
print(f'│ Parse success        │ {bl[\"parse_success_rate\"]:>7.1%}  │ {gr[\"parse_success_rate\"]:>7.1%}  │ {gr[\"parse_success_rate\"]-bl[\"parse_success_rate\"]:>+6.1%}  │')
print(f'│ Execution success    │ {bl[\"prediction_execution_success_rate\"]:>7.1%}  │ {gr[\"prediction_execution_success_rate\"]:>7.1%}  │ {gr[\"prediction_execution_success_rate\"]-bl[\"prediction_execution_success_rate\"]:>+6.1%}  │')
print(f'│ Execution match      │ {bl[\"custom_execution_match_rate\"]:>7.1%}  │ {gr[\"custom_execution_match_rate\"]:>7.1%}  │ {gr[\"custom_execution_match_rate\"]-bl[\"custom_execution_match_rate\"]:>+6.1%}  │')
print('└──────────────────────┴──────────┴──────────┴──────────┘')
"
fi

echo ""
echo "Done: $(date)"
