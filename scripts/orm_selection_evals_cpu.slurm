#!/bin/bash
# ORM 四臂选择：官方复评（步骤 2/2，纯 CPU）
#SBATCH --job-name=orm_evals
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_evals_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_evals_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
OUT_DIR=${OUT_DIR:-$BASE/outputs/orm_selection}
cd $BASE

echo "=== 官方复评（4 臂）==="
for ARM in arm_orm_argmax arm_orm_grouphead arm_orm_tiebreak arm_orm_hybrid; do
    bash $BASE/scripts/eval_official.sh \
        "$OUT_DIR"/items_${ARM}.json \
        $BASE/outputs/official_orm_selection_${ARM}
    echo "--- OFFICIAL $ARM ---"
    grep -A1 'EXECUTION ACCURACY' \
        $BASE/outputs/official_orm_selection_${ARM}/official_result.txt | tail -1
done

echo "Done: $(date)"
