#!/bin/bash
#SBATCH --job-name=7b_500d
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/7b_500d_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/7b_500d_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/qwen2.5-coder-7b-instruct
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/grpo_7b_500d

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 7B GRPO with 500 training examples ==="
echo "Started: $(date) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 500 \
    --num-generations 2 \
    --max-steps 50 \
    --reward-type three_level \
    --learning-rate 5e-6 \
    --beta 0.04 \
    --temperature 0.7 \
    --output-dir "$OUT" \
    --spider-dir "$SPIDER" \
    --model-path "$MODEL"

echo "Training done: $(date)"

# Evaluate ckpt-25 and final
for CK in checkpoint-25 final; do
    LP="$OUT"
    [ "$CK" != "final" ] && LP="$OUT/$CK"
    DEST="grpo_7b_500d_$CK"
    [ "$CK" = "final" ] && DEST="grpo_7b_500d"
    $PYTHON $BASE/src/evaluate_after_grpo.py \
        --lora-path "$LP" --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
        --output-dir $BASE/outputs/$DEST --batch-size 4
done

echo "=== 7B+500d Results ==="
for D in grpo_7b_500d_25 grpo_7b_500d; do
    R=$($PYTHON -c "import json; s=json.load(open('$BASE/outputs/$D/summary.json')); print(f'{s[\"custom_execution_match_rate\"]:.1%}')" 2>/dev/null || echo N/A)
    echo "$D: $R"
done
echo "Done: $(date)"
