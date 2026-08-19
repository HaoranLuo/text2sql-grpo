#!/bin/bash
# B1 裁决 + 官方复评（纯 CPU）：多实例 vav 四臂 × 三池 → 官方 EX
#SBATCH --job-name=adj_b1
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/adj_b1_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/adj_b1_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE

echo "=== B1 裁决（多实例 vav, 全实例, 16 线程）==="
$BASE/envs/reasoning3b/bin/python $BASE/src/adjudicate_pool.py \
    --items $BASE/outputs/eval_pool_b1/items.json \
    --out-dir $BASE/outputs/adjudicate_b1 \
    --spider-dir $BASE/data/spider_data \
    --threads 16

echo "=== 官方复评（关键臂 × 池）==="
for AP in arm_vav_multi_all_both arm_vav_multi_all_v1_only arm_vav_multi_all_v2_only arm_vav1_both arm_vav_multi2_both arm_maj_both; do
    bash $BASE/scripts/eval_official.sh \
        $BASE/outputs/adjudicate_b1/items_${AP}.json \
        $BASE/outputs/official_adj_${AP}
    echo "--- OFFICIAL $AP ---"
    grep -A3 'EXECUTION ACCURACY' $BASE/outputs/official_adj_${AP}/official_result.txt | head -2
done

echo "Done: $(date)"
