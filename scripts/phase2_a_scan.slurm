#!/bin/bash
#SBATCH --job-name=p2a_scan
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/p2a_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/p2a_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Phase2-A: 数据量扫描 (3B + G=8 + 三级奖励) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

# A1: 100条
echo ""
echo "=== A1: 100 examples ==="
$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 100 --num-generations 8 --max-steps 25 \
    --reward-type three_level --learning-rate 3e-6 --beta 0.04 --temperature 1.0 \
    --output-dir $BASE/checkpoints/p2a_100 --spider-dir "$SPIDER" --model-path "$MODEL"
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/p2a_100 --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/p2a_100 --batch-size 8

# A2: 500条
echo ""
echo "=== A2: 500 examples ==="
$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 500 --num-generations 8 --max-steps 25 \
    --reward-type three_level --learning-rate 3e-6 --beta 0.04 --temperature 1.0 \
    --output-dir $BASE/checkpoints/p2a_500 --spider-dir "$SPIDER" --model-path "$MODEL"
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/p2a_500 --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/p2a_500 --batch-size 8

# A3: 2000条
echo ""
echo "=== A3: 2000 examples ==="
$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 2000 --num-generations 8 --max-steps 25 \
    --reward-type three_level --learning-rate 3e-6 --beta 0.04 --temperature 1.0 \
    --output-dir $BASE/checkpoints/p2a_2000 --spider-dir "$SPIDER" --model-path "$MODEL"
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/p2a_2000 --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/p2a_2000 --batch-size 8

# A4: 7000条
echo ""
echo "=== A4: 7000 examples ==="
$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 7000 --num-generations 8 --max-steps 25 \
    --reward-type three_level --learning-rate 3e-6 --beta 0.04 --temperature 1.0 \
    --output-dir $BASE/checkpoints/p2a_7000 --spider-dir "$SPIDER" --model-path "$MODEL"
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/p2a_7000 --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/p2a_7000 --batch-size 8

# 汇总
echo ""
echo "============================================"
echo "=== Phase2-A 数据量扫描结果 ==="
echo "============================================"
$PYTHON -c "
import json
base = '$BASE/outputs'
for n in ['100', '500', '2000', '7000']:
    try:
        with open(f'{base}/p2a_{n}/summary.json') as f:
            s = json.load(f)
        print(f'{n}条: {s[\"custom_execution_match_rate\"]:.1%}  (基线45%)')
    except Exception as e:
        print(f'{n}条: 失败 - {e}')
"
echo ""
echo "Done: $(date)"
