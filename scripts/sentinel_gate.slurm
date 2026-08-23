#!/bin/bash
# Sentinel Gate（Minority Sentinel 高置信翻转门控）：纯 CPU，只复用 bird_select_ormbird_bird_bal2 缓存
# 提交: sbatch scripts/sentinel_gate.slurm
#SBATCH --job-name=sentinel_gate
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/sentinel_gate_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/sentinel_gate_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
cd $BASE

echo "=== sentinel_gate (cpu6348) ==="
$PYTHON $BASE/src/sentinel_gate.py \
    --base-dir $BASE/outputs/bird_select_ormbird_bird_bal2 \
    --out-dir $BASE/outputs/bird_sentinel_gate \
    --db-root $BASE/data/bird/bird_dev/dev_20240627/dev_databases \
    --data-json $BASE/data/bird/bird_dev/dev_20240627/dev.json \
    --ground-truth $BASE/data/bird/bird_dev/dev_20240627/dev.sql \
    --evaluator-py $BASE/tmp_idea_research/finer-sql/evaluation/official_bird_evaluation/evaluation_bird_ex.py \
    --num-cpus 12

echo "Done: $(date)"
