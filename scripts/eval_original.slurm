#!/bin/bash
#SBATCH --job-name=orig_eval
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orig_eval_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orig_eval_%j.err

# 原始 Spider 口径评估（论文对齐验证，CPU 即可）
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
envs/reasoning3b/bin/python tools/original_spider_eval/evaluation.py \
    --gold outputs/official_original_vav/gold.sql \
    --pred outputs/official_original_vav/pred.sql \
    --etype exec \
    --db data/spider_data/database \
    --table data/spider_data/tables.json 2>&1 | tee outputs/official_original_vav/original_result.txt
echo "=== DONE ==="
