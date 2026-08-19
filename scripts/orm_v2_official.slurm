#!/bin/bash
# orm_selection_v2 官方复评：对 OUT 下所有 items_*.json 跑 eval_official.sh
# 用法: sbatch --export=OUT=outputs/orm_selection_v2 scripts/orm_v2_official.slurm
#SBATCH --job-name=ormv2_off
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv2_off_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv2_off_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
OUT=${OUT:-outputs/orm_selection_v2}
NAME=$(basename $OUT)

echo "=== official re-eval for $OUT ==="
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
    echo "  $ARM EX=$EX EM=$EM"
done
echo "" >> "$RESULTS"
echo "}" >> "$RESULTS"
echo "=== done: $RESULTS ==="
