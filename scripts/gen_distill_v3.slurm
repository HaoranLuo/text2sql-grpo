#!/bin/bash
# 蒸馏第三轨迹(温度 0.5, 多样性补充; SFT v3 备料, 在剂量响应+2.7pp 基础上继续压边际)
# DEEPSEEK_API_KEY 由 --export 传入
#SBATCH --job-name=distill_3c
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/distill_3c_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/distill_3c_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=envs/reasoning3b/bin/python
OUTDIR=$BASE/data/reasoning_data

echo "=== 第三轨迹(温度 0.5) ==="
$PY src/gen_reasoning_data.py --train-file $OUTDIR/distill_v2_flash_p0.json \
    --spider-dir $BASE/data/spider_data --output $OUTDIR/distill_v3_flash_p0.jsonl \
    --model deepseek-v4-flash --thinking disabled --temperature 0.5 \
    --max-tokens 4096 --concurrency 16 --sleep 0 > logs/distill_v3_p0.log 2>&1 &

$PY src/gen_reasoning_data.py --train-file $OUTDIR/distill_v2_flash_p1.json \
    --spider-dir $BASE/data/spider_data --output $OUTDIR/distill_v3_flash_p1.jsonl \
    --model deepseek-v4-flash --thinking disabled --temperature 0.5 \
    --max-tokens 4096 --concurrency 16 --sleep 0 > logs/distill_v3_p1.log 2>&1 &

$PY src/gen_reasoning_data.py --train-file $OUTDIR/distill_v2_flash_p2.json \
    --spider-dir $BASE/data/spider_data --output $OUTDIR/distill_v3_flash_p2.jsonl \
    --model deepseek-v4-flash --thinking disabled --temperature 0.5 \
    --max-tokens 4096 --concurrency 16 --sleep 0 > logs/distill_v3_p2.log 2>&1 &

wait
echo "=== 第三轨迹完成: $(date) ==="
