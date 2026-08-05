#!/bin/bash
#SBATCH --job-name=grpo_train
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/grpo_train_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/grpo_train_%j.err

# =========================================================================
# GRPO Training + Evaluation Pipeline
# =========================================================================

set -e

echo "============================================"
echo "GRPO Training Job"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "============================================"

# --- Paths ---
PROJECT_ROOT=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
CONDA_ENV=$PROJECT_ROOT/envs/reasoning3b
PYTHON=$CONDA_ENV/bin/python
LOGS_DIR=$PROJECT_ROOT/logs
OUTPUT_DIR=$PROJECT_ROOT/checkpoints/grpo_lora

mkdir -p "$LOGS_DIR" "$OUTPUT_DIR"

# --- Activate environment ---
echo ""
echo "Activating conda environment..."
source $(dirname $(dirname $CONDA_ENV))/etc/profile.d/conda.sh 2>/dev/null || true
export PATH="$CONDA_ENV/bin:$PATH"

echo "Python: $(which python)"
echo "Python version: $($PYTHON --version)"

# --- GPU check ---
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not available"

# =========================================================================
# Step 1: GRPO Training
# =========================================================================
echo ""
echo "============================================"
echo "Step 1: GRPO Training"
echo "============================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PYTHON $PROJECT_ROOT/src/train_reasoning_grpo.py \
    --num-train 100 \
    --num-generations 2 \
    --max-steps 40 \
    --learning-rate 5e-6 \
    --beta 0.04 \
    --temperature 0.7 \
    --output-dir "$OUTPUT_DIR" \
    --spider-dir "$PROJECT_ROOT/data/spider_data" \
    --model-path "$PROJECT_ROOT/models/Qwen2.5-Coder-3B-Instruct"

echo ""
echo "Training completed at: $(date)"

# =========================================================================
# Step 2: Post-Training Evaluation
# =========================================================================
echo ""
echo "============================================"
echo "Step 2: Post-Training Evaluation"
echo "============================================"

$PYTHON $PROJECT_ROOT/src/evaluate_after_grpo.py \
    --lora-path "$OUTPUT_DIR" \
    --limit 100 \
    --spider-dir "$PROJECT_ROOT/data/spider_data" \
    --model-path "$PROJECT_ROOT/models/Qwen2.5-Coder-3B-Instruct" \
    --output-dir "$PROJECT_ROOT/outputs/grpo_post_train_100"

echo ""
echo "Evaluation completed at: $(date)"

# =========================================================================
# Step 3: Print comparison
# =========================================================================
echo ""
echo "============================================"
echo "Step 3: Before vs After Comparison"
echo "============================================"

BASELINE_SUMMARY=$PROJECT_ROOT/outputs/baseline_pretrain_100/summary.json
GRPO_SUMMARY=$PROJECT_ROOT/outputs/grpo_post_train_100/summary.json

if [ -f "$BASELINE_SUMMARY" ] && [ -f "$GRPO_SUMMARY" ]; then
    echo ""
    echo "Baseline (pre-train):"
    $PYTHON -c "
import json
with open('$BASELINE_SUMMARY') as f:
    s = json.load(f)
print(f\"  Execution match: {s['custom_execution_match_count']}/{s['total_completed']} ({s['custom_execution_match_rate']:.1%})\")
print(f\"  Parse success:   {s['parse_success_rate']:.1%}\")
print(f\"  Exec success:    {s['prediction_execution_success_rate']:.1%}\")
"
    echo ""
    echo "After GRPO:"
    $PYTHON -c "
import json
with open('$GRPO_SUMMARY') as f:
    s = json.load(f)
print(f\"  Execution match: {s['custom_execution_match_count']}/{s['total_completed']} ({s['custom_execution_match_rate']:.1%})\")
print(f\"  Parse success:   {s['parse_success_rate']:.1%}\")
print(f\"  Exec success:    {s['prediction_execution_success_rate']:.1%}\")
"
    echo ""
    echo "Delta:"
    $PYTHON -c "
import json
with open('$BASELINE_SUMMARY') as f:
    bl = json.load(f)
with open('$GRPO_SUMMARY') as f:
    gr = json.load(f)
diff = gr['custom_execution_match_rate'] - bl['custom_execution_match_rate']
print(f\"  Execution match: {diff:+.1%}\")
print(f\"  Parse success:   {gr['parse_success_rate'] - bl['parse_success_rate']:+.1%}\")
print(f\"  Exec success:    {gr['prediction_execution_success_rate'] - bl['prediction_execution_success_rate']:+.1%}\")
"
else
    echo "Cannot compute comparison — one or both summary files missing."
fi

echo ""
echo "============================================"
echo "Job finished at: $(date)"
echo "LoRA adapter: $OUTPUT_DIR"
echo "Results: $PROJECT_ROOT/outputs/grpo_post_train_100/"
echo "============================================"
