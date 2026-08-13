#!/bin/bash
# 蒸馏 Step1: 50 题 A/B 实测(flash 非思考 vs flash 思考 vs pro 思考)
# DEEPSEEK_API_KEY 由 sbatch --export 传入(不写入脚本/git)
#SBATCH --job-name=distill_ab
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/distill_ab_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/distill_ab_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=envs/reasoning3b/bin/python

echo "=== A: flash 非思考 25 题 ==="
$PY src/gen_reasoning_data.py \
    --train-file $BASE/data/reasoning_data/ab_flash_nothink_25.json \
    --spider-dir $BASE/data/spider_data \
    --output $BASE/data/reasoning_data/ab_flash_nothink_25.jsonl \
    --model deepseek-v4-flash --thinking disabled \
    --max-tokens 4096 --num-train 25 --concurrency 8

echo "=== B: flash 思考 25 题 ==="
$PY src/gen_reasoning_data.py \
    --train-file $BASE/data/reasoning_data/ab_flash_think_25.json \
    --spider-dir $BASE/data/spider_data \
    --output $BASE/data/reasoning_data/ab_flash_think_25.jsonl \
    --model deepseek-v4-flash --thinking enabled \
    --max-tokens 4096 --num-train 25 --concurrency 8

echo "=== C: pro 思考 5 题 ==="
$PY src/gen_reasoning_data.py \
    --train-file $BASE/data/reasoning_data/ab_pro_think_5.json \
    --spider-dir $BASE/data/spider_data \
    --output $BASE/data/reasoning_data/ab_pro_think_5.jsonl \
    --model deepseek-v4-pro --thinking enabled \
    --max-tokens 4096 --num-train 5 --concurrency 8

echo "=== AB DONE ==="
