#!/bin/bash
# 干净泛化检验: p2a_500(GRPO 血统) 候选池生成 — ORM 训练数据完全不含此类候选
#SBATCH --job-name=pool_p2a5
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/pool_p2a5_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/pool_p2a5_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

LIMIT=${LIMIT:-1034}
OUTPUT_DIR=${OUTPUT_DIR:-$BASE/outputs/eval_pool_p2a500}

echo "=== p2a_500 候选池（GRPO 血统, ORM 未见过）==="
$PYTHON $BASE/src/eval_pool_b1.py \
    --limit $LIMIT \
    --output-dir $OUTPUT_DIR \
    --loras p2a_500 \
    --only-lora p2a_500 \
    --n 16 --temperature 1.0 --top-p 1.0 --seed 0 \
    --max-new-tokens 2048 \
    --checkpoint-every 25

echo "Done: $(date)"
