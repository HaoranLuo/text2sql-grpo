#!/bin/bash
# T2 ORM 打标（纯 CPU）：B1 候选池 20596 唯一候选 × 全实例官方 exec-match
# 参照 adjudicate_b1_cpu.slurm 的 CPU 分区惯例；产出 orm_train.json /
# orm_questions.json / orm_label_stats.json，并自动与裁决器 arm_maj 交叉核对
#SBATCH --job-name=orm_label
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_label_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_label_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE

echo "=== T2 ORM 打标（全量 1034 题，16 线程，~20 分钟）==="
$BASE/envs/reasoning3b/bin/python $BASE/src/label_orm_data.py \
    --items $BASE/outputs/eval_pool_b1/items.json \
    --spider-dir $BASE/data/spider_data \
    --threads 16

echo "Done: $(date)"
echo "产物: $BASE/data/orm_train.json / orm_questions.json / orm_label_stats.json"
