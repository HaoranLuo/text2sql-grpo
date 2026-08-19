#!/bin/bash
# orm_selection_v2 阶段 1：零 ORM 臂（baseline + P0-2 + P0-3 + P0-5），纯 CPU
# 用法: sbatch --export=ITEMS=outputs/eval_pool_multi/items.json,OUT=outputs/orm_selection_v2,BASELINE=outputs/orm_selection_multi/items_arm_orm_grouphead.json,LIMIT= scripts/orm_v2_stage1.slurm
#SBATCH --job-name=ormv2_s1
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv2_s1_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv2_s1_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
ITEMS=${ITEMS:-outputs/eval_pool_multi/items.json}
OUT=${OUT:-outputs/orm_selection_v2}
BASELINE=${BASELINE:-outputs/orm_selection_multi/items_arm_orm_grouphead.json}
LIMIT_ARG=""
if [ -n "$LIMIT" ]; then LIMIT_ARG="--limit $LIMIT"; fi

echo "=== stage1: $ITEMS -> $OUT (arms baseline,p02,p03,p05) ==="
envs/reasoning3b/bin/python src/orm_selection_v2.py \
    --items $ITEMS --out-dir $OUT \
    --baseline-items $BASELINE \
    --orm-backend none \
    --arms baseline,p02,p03,p05 \
    --threads 16 $LIMIT_ARG
echo "Done: $(date)"
