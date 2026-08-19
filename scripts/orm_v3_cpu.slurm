#!/bin/bash
# orm_selection_v3 步骤 3：CPU 重算两池全部臂（读三份分数缓存）+ 官方复评（eval_official.sh）
# 用法: sbatch scripts/orm_v3_cpu.slurm
#SBATCH --job-name=ormv3_cpu
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv3_cpu_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv3_cpu_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=$BASE/envs/reasoning3b/bin/python

echo "=== [multi] 重算（baseline + m8×2 + caps k3/5/10 + caps_full）==="
$PY src/orm_selection_v3.py \
    --items outputs/eval_pool_multi/items.json \
    --out-dir outputs/orm_selection_v3 \
    --baseline-items outputs/orm_selection_multi/items_arm_orm_grouphead.json \
    --full-scores outputs/orm_selection_v2/scores/scores_vllm.json \
    --skip-scoring --threads 16

echo "=== [unseen] 重算 ==="
$PY src/orm_selection_v3.py \
    --items outputs/eval_pool_unseen/items.json \
    --out-dir outputs/orm_selection_v3_unseen \
    --baseline-items outputs/orm_selection_unseen/items_arm_orm_grouphead.json \
    --full-scores outputs/orm_selection_v2_unseen/scores/scores_vllm.json \
    --skip-scoring --threads 16

echo "=== 官方复评（两池全部 items_*.json）==="
for OUT in outputs/orm_selection_v3 outputs/orm_selection_v3_unseen; do
    NAME=$(basename $OUT)
    RESULTS="$OUT/official_ex.json"
    echo "{" > "$RESULTS"
    FIRST=1
    for ITEMS_FILE in "$OUT"/items_*.json; do
        ARM=$(basename "$ITEMS_FILE" .json | sed 's/^items_//')
        EVAL_DIR="$OUT/official_${NAME}_${ARM}"
        bash scripts/eval_official.sh "$ITEMS_FILE" "$EVAL_DIR" > /dev/null 2>&1
        EX=$(grep -A1 'EXECUTION ACCURACY' "$EVAL_DIR/official_result.txt" | tail -1 | awk '{print $NF}')
        EM=$(grep -A1 'EXACT MATCHING ACCURACY' "$EVAL_DIR/official_result.txt" | tail -1 | awk '{print $NF}')
        if [ "$FIRST" -eq 0 ]; then echo "," >> "$RESULTS"; fi
        FIRST=0
        printf '  "%s": {"ex": "%s", "em": "%s"}' "$ARM" "$EX" "$EM" >> "$RESULTS"
        echo "  $NAME/$ARM EX=$EX EM=$EM"
    done
    echo "" >> "$RESULTS"
    echo "}" >> "$RESULTS"
done

echo "Done: $(date)"
