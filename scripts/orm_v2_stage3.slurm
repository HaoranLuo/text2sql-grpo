#!/bin/bash
# orm_selection_v2 阶段 3：全部臂（复用分数缓存，不再打分）→ 完整 summary + items
# 用法: sbatch --export=ITEMS=...,OUT=...,BASELINE=... scripts/orm_v2_stage3.slurm
#SBATCH --job-name=ormv2_s3
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv2_s3_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv2_s3_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
ITEMS=${ITEMS:-outputs/eval_pool_multi/items.json}
OUT=${OUT:-outputs/orm_selection_v2}
BASELINE=${BASELINE:-outputs/orm_selection_multi/items_arm_orm_grouphead.json}

echo "=== stage3 (all arms, cache only): $ITEMS -> $OUT ==="
envs/reasoning3b/bin/python src/orm_selection_v2.py \
    --items $ITEMS --out-dir $OUT \
    --baseline-items $BASELINE \
    --orm-backend cpu --skip-scoring \
    --arms all \
    --threads 16
echo "Done: $(date)"
