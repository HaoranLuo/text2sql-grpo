#!/bin/bash
# B2 裁决 + 官方复评（纯 CPU）：三模型池（v1+v2+v3）多实例 vav
#SBATCH --job-name=adj_b2
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/adj_b2_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/adj_b2_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE

echo "=== B2 裁决（三模型池, 全实例, 16 线程）==="
$BASE/envs/reasoning3b/bin/python $BASE/src/adjudicate_pool.py \
    --items $BASE/outputs/eval_pool_b2/items.json \
    --out-dir $BASE/outputs/adjudicate_b2 \
    --spider-dir $BASE/data/spider_data \
    --threads 16

echo "=== 官方复评（关键臂）==="
for AP in arm_vav_multi_all_both arm_vav_multi_all_v1_only arm_vav_multi_all_v2_only arm_maj_both; do
    bash $BASE/scripts/eval_official.sh \
        $BASE/outputs/adjudicate_b2/items_${AP}.json \
        $BASE/outputs/official_adj_b2_${AP}
    echo "--- OFFICIAL $AP ---"
    grep -A3 'EXECUTION ACCURACY' $BASE/outputs/official_adj_b2_${AP}/official_result.txt | head -2
done

echo "Done: $(date)"
