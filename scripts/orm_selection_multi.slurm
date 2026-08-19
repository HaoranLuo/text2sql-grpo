#!/bin/bash
# 主战役: 多源池(v1s1+v2s1+v3s0+p2a5s0) + ORM 裁决（3090）
#SBATCH --job-name=orm_multi
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_multi_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_multi_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

if [ ! -f "$BASE/checkpoints/orm_b1/adapter_config.json" ]; then
    echo "FATAL: orm_b1 不存在" >&2
    exit 1
fi

echo "=== ORM 主战役（多源池 64 候选/题）==="
$PYTHON $BASE/src/orm_selection.py \
    --items $BASE/outputs/eval_pool_multi/items.json \
    --out-dir $BASE/outputs/orm_selection_multi \
    --spider-dir $BASE/data/spider_data \
    --orm-checkpoint $BASE/checkpoints/orm_b1 \
    --base-model $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --merge-python $BASE/envs/reasoning3b/bin/python \
    --threads 16

echo "Done: $(date)"
