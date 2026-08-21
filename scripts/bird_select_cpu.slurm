#!/bin/bash
# BIRD 管线 CPU 作业：--phase prep（执行分组 + arm_vav + ORM payloads）/ --phase final（裁决 + 官方 EX）
# 冒烟: sbatch --export=ALL,PHASE=prep,LIMIT=10,ITEMS=$BASE/outputs/eval_pool_bird_smoke/items.json,OUT_DIR=$BASE/outputs/bird_select_smoke scripts/bird_select_cpu.slurm
#SBATCH --job-name=bird_sel
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/bird_sel_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/bird_sel_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
cd $BASE

PHASE=${PHASE:-prep}
ITEMS=${ITEMS:-$BASE/outputs/eval_pool_bird/items.json}
OUT_DIR=${OUT_DIR:-$BASE/outputs/bird_select}
EXTRA=""
if [ -n "${LIMIT:-}" ]; then
    EXTRA="--limit $LIMIT"
fi

echo "=== bird_select --phase $PHASE (cpu6348, items=$ITEMS, out=$OUT_DIR) ==="
$PYTHON $BASE/src/bird_select.py --phase $PHASE \
    --items "$ITEMS" \
    --out-dir "$OUT_DIR" \
    --work-dir "$OUT_DIR/work" \
    --db-root $BASE/data/bird/bird_dev/dev_20240627/dev_databases \
    --data-json $BASE/data/bird/bird_dev/dev_20240627/dev.json \
    --ground-truth $BASE/data/bird/bird_dev/dev_20240627/dev.sql \
    --evaluator-py $BASE/tmp_idea_research/finer-sql/evaluation/official_bird_evaluation/evaluation_bird_ex.py \
    --threads 16 --num-cpus 12 \
    $EXTRA

echo "Done: $(date)"
