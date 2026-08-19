#!/bin/bash
# T1-1/T1-2 组级软排序 + 平票门控裁决升级（纯 CPU）：
# 6 臂（基线 + soft_cross×3 + ladder + gated）离线重算 → 官方 EX 复评
# 历史数据：adjudicate_pool 全量（同量级执行）约 14 分钟；本作业含 group_reps
# 阶段与 6 次官方复评，超时上限 2 小时
#SBATCH --job-name=adj_soft
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/adj_soft_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/adj_soft_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE

echo "=== T1-1/T1-2 软排序裁决（全实例, 16 线程, 含 group_reps 判别力验证）==="
$BASE/envs/reasoning3b/bin/python $BASE/src/adjudicate_soft.py \
    --items $BASE/outputs/eval_pool_b1/items.json \
    --out-dir $BASE/outputs/adjudicate_soft \
    --spider-dir $BASE/data/spider_data \
    --threads 16

echo "=== 官方复评（6 臂）==="
for ARM in arm_vav_multi_all arm_soft_cross_w1 arm_soft_cross_w2 arm_soft_cross_w3 arm_soft_ladder arm_gated_structural; do
    bash $BASE/scripts/eval_official.sh \
        $BASE/outputs/adjudicate_soft/items_${ARM}.json \
        $BASE/outputs/official_adj_soft_${ARM}
    echo "--- OFFICIAL $ARM ---"
    grep -A3 'EXECUTION ACCURACY' $BASE/outputs/official_adj_soft_${ARM}/official_result.txt | head -2
done

echo "Done: $(date)"
