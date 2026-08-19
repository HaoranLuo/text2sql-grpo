#!/bin/bash
# ORM 四臂选择：gpudebug 3090 版（步骤 1/2：打分+分组，官方复评另跑 CPU 作业）
# 时间估算: vLLM 打分 5-15min + CPU 分组 ~15min ≈ 20-30min < 50min 墙钟
#SBATCH --job-name=orm_sel_g
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_sel_g_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_sel_g_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/vllmenv/bin/python
OUT_DIR=${OUT_DIR:-$BASE/outputs/orm_selection}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

if [ ! -f "$BASE/checkpoints/orm_b1/adapter_config.json" ]; then
    echo "FATAL: checkpoints/orm_b1 无 adapter_config.json（ORM 未训练完成）" >&2
    exit 1
fi

echo "=== ORM 打分 + 4 臂选择（3090, 步骤 1/2）==="
$PYTHON $BASE/src/orm_selection.py \
    --items $BASE/outputs/eval_pool_b1/items.json \
    --out-dir "$OUT_DIR" \
    --spider-dir $BASE/data/spider_data \
    --orm-checkpoint $BASE/checkpoints/orm_b1 \
    --base-model $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --merge-python $BASE/envs/reasoning3b/bin/python \
    --threads 16

echo "Done: $(date)"
