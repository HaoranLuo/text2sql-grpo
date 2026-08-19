#!/bin/bash
# orm_selection_v2 阶段 2（合并版）：一个作业内先打 multi 池分片、再打 unseen 池分片
# 用法: sbatch --export=SHARD=0,NSHARDS=4 scripts/orm_v2_score.slurm
#SBATCH --job-name=ormv2_sc
#SBATCH --partition=cpu8358
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv2_sc_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv2_sc_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
SHARD=${SHARD:-0}
NSHARDS=${NSHARDS:-4}
PY=envs/reasoning3b/bin/python

echo "=== [multi] score shard $SHARD/$NSHARDS ==="
$PY src/orm_selection_v2.py \
    --items outputs/eval_pool_multi/items.json \
    --out-dir outputs/orm_selection_v2 \
    --baseline-items outputs/orm_selection_multi/items_arm_orm_grouphead.json \
    --orm-backend cpu --score-only \
    --score-shard $SHARD $NSHARDS \
    --cpu-threads 32 --score-batch 16 \
    --threads 16

echo "=== [unseen] score shard $SHARD/$NSHARDS ==="
$PY src/orm_selection_v2.py \
    --items outputs/eval_pool_unseen/items.json \
    --out-dir outputs/orm_selection_v2_unseen \
    --baseline-items outputs/orm_selection_unseen/items_arm_orm_grouphead.json \
    --orm-backend cpu --score-only \
    --score-shard $SHARD $NSHARDS \
    --cpu-threads 32 --score-batch 16 \
    --threads 16

echo "Done: $(date)"
