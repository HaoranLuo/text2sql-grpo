#!/bin/bash
#SBATCH --job-name=a2_g4
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:58:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/a2g4_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/a2g4_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/a2_g4

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== A2-G4: 500条 + G=4 + 三级奖励 (修正G) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 500 --num-generations 4 --max-steps 25 \
    --reward-type three_level --learning-rate 3e-6 --beta 0.04 --temperature 1.0 \
    --output-dir "$OUT" --spider-dir "$SPIDER" --model-path "$MODEL"

echo "Training done: $(date)"

$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path "$OUT" --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/a2_g4 --batch-size 8

echo "=== A2-G4 Result ==="
$PYTHON -c "
import json
with open('$BASE/outputs/a2_g4/summary.json') as f:
    s = json.load(f)
m = s['custom_execution_match_rate']
print(f'500条G4: {m:.1%}  (100条G4=50%, 500条G8=待出)')
"
echo "Done: $(date)"
