#!/bin/bash
# ORM 跨池泛化检验：pool_b2 的 v3 候选（ORM 从未见过）→ 四臂选择（gpudebug 3090）
#SBATCH --job-name=orm_xpool
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_xpool_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_xpool_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

if [ ! -f "$BASE/checkpoints/orm_b1/adapter_config.json" ]; then
    echo "FATAL: orm_b1 不存在" >&2
    exit 1
fi

echo "=== ORM 跨池泛化检验（v3-only 候选, 步骤 1/2）==="
$PYTHON $BASE/src/orm_selection.py \
    --items $BASE/outputs/eval_pool_b2_v3only/items.json \
    --out-dir $BASE/outputs/orm_selection_xpool \
    --spider-dir $BASE/data/spider_data \
    --orm-checkpoint $BASE/checkpoints/orm_b1 \
    --base-model $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --merge-python $BASE/envs/reasoning3b/bin/python \
    --threads 16

echo "Done: $(date)"
