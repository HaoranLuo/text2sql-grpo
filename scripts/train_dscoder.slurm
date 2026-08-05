#!/bin/bash
#SBATCH --job-name=dsc_grpo
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/dsc_grpo_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/dsc_grpo_%j.err

set -e

PROJECT_ROOT=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
CONDA_ENV=$PROJECT_ROOT/envs/reasoning3b
PYTHON=$CONDA_ENV/bin/python
LOGS_DIR=$PROJECT_ROOT/logs
MODEL_PATH=$PROJECT_ROOT/models/qwen2.5-coder-7b-instruct
OUTPUT_DIR=$PROJECT_ROOT/checkpoints/grpo_qwen7b
SPIDER_DIR=$PROJECT_ROOT/data/spider_data

mkdir -p "$LOGS_DIR" "$OUTPUT_DIR"

export PATH="$CONDA_ENV/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$PROJECT_ROOT/cache/huggingface

echo "============================================"
echo "DeepSeek-Coder-6.7B GRPO Training"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "============================================"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

# =========================================================================
# Step 1: Baseline (pre-training)
# =========================================================================
echo ""
echo "============================================"
echo "Step 1: Baseline Evaluation (before GRPO)"
echo "============================================"

$PYTHON $PROJECT_ROOT/src/evaluate_after_grpo.py \
    --lora-path "" \
    --limit 100 \
    --spider-dir "$SPIDER_DIR" \
    --model-path "$MODEL_PATH" \
    --output-dir "$PROJECT_ROOT/outputs/baseline_qwen7b_100" \
    --batch-size 8

BASELINE_SUMMARY=$PROJECT_ROOT/outputs/baseline_qwen7b_100/summary.json

if [ -f "$BASELINE_SUMMARY" ]; then
    echo ""
    echo "Baseline result:"
    $PYTHON -c "
import json
with open('$BASELINE_SUMMARY') as f:
    s = json.load(f)
print(f\"  Execution match: {s['custom_execution_match_count']}/{s['total_completed']} ({s['custom_execution_match_rate']:.1%})\")
"
fi

# =========================================================================
# Step 2: GRPO Training
# =========================================================================
echo ""
echo "============================================"
echo "Step 2: GRPO Training"
echo "============================================"

$PYTHON $PROJECT_ROOT/src/train_reasoning_grpo.py \
    --num-train 500 \
    --num-generations 4 \
    --max-steps 100 \
    --learning-rate 5e-6 \
    --beta 0.04 \
    --temperature 0.7 \
    --output-dir "$OUTPUT_DIR" \
    --spider-dir "$SPIDER_DIR" \
    --model-path "$MODEL_PATH"

echo ""
echo "Training completed at: $(date)"

# =========================================================================
# Step 3: Post-Training Evaluation
# =========================================================================
echo ""
echo "============================================"
echo "Step 3: Post-Training Evaluation"
echo "============================================"

$PYTHON $PROJECT_ROOT/src/evaluate_after_grpo.py \
    --lora-path "$OUTPUT_DIR" \
    --limit 100 \
    --spider-dir "$SPIDER_DIR" \
    --model-path "$MODEL_PATH" \
    --output-dir "$PROJECT_ROOT/outputs/grpo_qwen7b_100" \
    --batch-size 8

# =========================================================================
# Step 4: Comparison
# =========================================================================
echo ""
echo "============================================"
echo "Step 4: Before vs After Comparison"
echo "============================================"

GRPO_SUMMARY=$PROJECT_ROOT/outputs/grpo_qwen7b_100/summary.json

if [ -f "$BASELINE_SUMMARY" ] && [ -f "$GRPO_SUMMARY" ]; then
    $PYTHON -c "
import json
with open('$BASELINE_SUMMARY') as f: bl = json.load(f)
with open('$GRPO_SUMMARY') as f: gr = json.load(f)

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
echo "============================================"
echo "Job finished at: $(date)"
echo "LoRA adapter: $OUTPUT_DIR"
echo "Results: $PROJECT_ROOT/outputs/grpo_qwen7b_100/"
echo "============================================"
