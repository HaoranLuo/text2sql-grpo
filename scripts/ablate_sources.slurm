#!/bin/bash
# 逐源消融（发文支持）：CPU 重算各子集臂 + 全部官方 EX
#SBATCH --job-name=ablate_src
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ablate_src_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ablate_src_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=$BASE/envs/reasoning3b/bin/python

echo "=== 消融重算（执行 1 次 × 全部子集臂）==="
$PY src/ablate_sources.py \
    --items outputs/eval_pool_multi/items.json \
    --scores outputs/orm_selection_v2/scores/scores_vllm.json \
    --out-dir outputs/ablate_sources \
    --threads 16

echo "=== 官方 EX（全部 arm）==="
RESULTS="$BASE/outputs/ablate_sources/official_ex.json"
echo "{" > "$RESULTS"
FIRST=1
for ITEMS_FILE in "$BASE"/outputs/ablate_sources/items_*.json; do
    ARM=$(basename "$ITEMS_FILE" .json | sed 's/^items_//')
    EVAL_DIR="$BASE/outputs/ablate_sources/official_$ARM"
    bash scripts/eval_official.sh "$ITEMS_FILE" "$EVAL_DIR" > /dev/null 2>&1
    EX=$(grep -A1 'EXECUTION ACCURACY' "$EVAL_DIR/official_result.txt" | tail -1 | awk '{print $NF}')
    if [ "$FIRST" -eq 0 ]; then echo "," >> "$RESULTS"; fi
    FIRST=0
    printf '  "%s": {"ex": "%s"}' "$ARM" "$EX" >> "$RESULTS"
    echo "  $ARM EX=$EX"
done
echo "" >> "$RESULTS"
echo "}" >> "$RESULTS"

echo "Done: $(date)"
