#!/bin/bash
#SBATCH --job-name=p2a_rest
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/p2ar_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/p2ar_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== P2A-续: A2(500)→A3(2000)→A4(7000) 三级G8 ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

for N in 500 2000 7000; do
    echo ""
    echo "=== A($N条) ==="
    $PYTHON $BASE/src/train_reasoning_grpo.py \
        --num-train $N --num-generations 8 --max-steps 25 \
        --reward-type three_level --learning-rate 3e-6 --beta 0.04 --temperature 1.0 \
        --output-dir $BASE/checkpoints/p2a_$N --spider-dir "$SPIDER" --model-path "$MODEL"
    $PYTHON $BASE/src/evaluate_after_grpo.py \
        --lora-path $BASE/checkpoints/p2a_$N --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
        --output-dir $BASE/outputs/p2a_$N --batch-size 8
done

echo ""
echo "=== P2A 汇总 ==="
$PYTHON -c "
import json
base = '$BASE/outputs'
for n in ['100', '500', '2000', '7000']:
    try:
        with open(f'{base}/p2a_{n}/summary.json') as f:
            s = json.load(f)
        print(f'{n}条: {s[\"custom_execution_match_rate\"]:.1%}')
    except Exception as e:
        print(f'{n}条: 失败 - {e}')
"
echo "Done: $(date)"
