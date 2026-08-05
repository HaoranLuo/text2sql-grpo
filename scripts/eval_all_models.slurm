#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eval_all_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eval_all_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
SPIDER=$BASE/data/spider_data

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Evaluating ALL models on Spider 100 dev ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"

# Each model gets evaluated, results go to outputs/eval_<name>/
# --model-path for base model eval, --lora-path for LoRA adapter eval

declare -A MODELS
MODELS[Qwen7B_zero]=""
MODELS[Qwen7B_SFT]="--lora-path $BASE/checkpoints/sft_coldstart"
MODELS[Qwen7B_SFT_GRPO25]="--lora-path $BASE/checkpoints/grpo_sft_50/checkpoint-25"
MODELS[Qwen7B_SFT_GRPO50]="--lora-path $BASE/checkpoints/grpo_sft_50"

# Check if new models exist
if [ -d $BASE/models/XiYanSQL-7B ]; then
    MODELS[XiYanSQL]="--model-path $BASE/models/XiYanSQL-7B"
fi
if [ -d $BASE/models/OmniSQL-7B ]; then
    MODELS[OmniSQL]="--model-path $BASE/models/OmniSQL-7B"
fi

for NAME in "${!MODELS[@]}"; do
    EXTRA="${MODELS[$NAME]}"
    if [ -z "$EXTRA" ]; then
        # Skip Qwen7B baseline (already have it)
        [ "$NAME" == "Qwen7B_zero" ] && continue
        continue
    fi

    echo ""
    echo "=== Evaluating: $NAME ==="

    if [[ "$EXTRA" == *"lora-path"* ]]; then
        MODEL="$BASE/models/qwen2.5-coder-7b-instruct"
        PK="$EXTRA"
    elif [[ "$EXTRA" == *"model-path"* ]]; then
        MODEL=""
        PK="$EXTRA"
    else
        echo "SKIP: $NAME"
        continue
    fi

    # Check if result already exists
    if [ -f $BASE/outputs/eval_$NAME/summary.json ]; then
        echo "Already done, skipping"
        continue
    fi

    if [ -n "$MODEL" ]; then
        $PYTHON $BASE/src/evaluate_after_grpo.py \
            $PK --limit 100 --spider-dir $SPIDER \
            --model-path "$MODEL" --output-dir $BASE/outputs/eval_$NAME --batch-size 4
    else
        $PYTHON $BASE/src/evaluate_after_grpo.py \
            $PK --limit 100 --spider-dir $SPIDER \
            --output-dir $BASE/outputs/eval_$NAME --batch-size 4
    fi
done

# =========================================================================
# FINAL COMPARISON
# =========================================================================
echo ""
echo "============================================"
echo "FINAL COMPARISON"
echo "============================================"
echo ""

$PYTHON -c "
import json, os
base = '$BASE/outputs'
entries = [
    ('eval_Qwen7B_SFT', 'Qwen7B + SFT (API 500条)'),
    ('eval_Qwen7B_SFT_GRPO25', 'Qwen7B + SFT + GRPO 25步'),
    ('eval_Qwen7B_SFT_GRPO50', 'Qwen7B + SFT + GRPO 50步'),
    ('eval_XiYanSQL', 'XiYanSQL-7B (零样本)'),
    ('eval_OmniSQL', 'OmniSQL-7B (零样本)'),
]
print(f\"{'Model':<35} {'Parse':>7} {'Exec':>7} {'Match':>7}\")
print('-' * 60)
print(f\"{'Qwen7B (zero-shot, known)':<35} {'100%':>7} {'97%':>7} {'81%':>7}\")

for folder, label in entries:
    p = os.path.join(base, folder, 'summary.json')
    if os.path.exists(p):
        with open(p) as f: s = json.load(f)
        m = s['custom_execution_match_rate']
        marker = ' ← BEST!' if m >= 0.85 else ''
        print(f\"{label:<35} {s['parse_success_rate']:>6.1%} {s['prediction_execution_success_rate']:>6.1%} {m:>6.1%}{marker}\")
    else:
        print(f\"{label:<35} {'N/A':>7} {'N/A':>7} {'N/A':>7}\")
"
echo ""
echo "Done: $(date)"
