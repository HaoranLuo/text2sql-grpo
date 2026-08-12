#!/bin/bash
# D5 账单实测: DeepSeek API 50 题实测账单 (CPU 分区, 与 GPU 训练并行)
# DEEPSEEK_API_KEY 由 sbatch --export 传入(不写入脚本/git)
#SBATCH --job-name=d5_billtest
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/d5_billtest_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/d5_billtest_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
envs/reasoning3b/bin/python src/gen_reasoning_data.py \
    --train-file $BASE/data/reasoning_data/d5_subset_50.json \
    --spider-dir $BASE/data/spider_data \
    --output $BASE/data/reasoning_data/d5_billtest_50.jsonl \
    --model deepseek-chat --num-train 50 --concurrency 8
echo "=== D5 BILL TEST DONE ==="
