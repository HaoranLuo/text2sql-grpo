#!/bin/bash
# 蒸馏 Step3b: 第二条轨迹(温度 1.0, 同题多样性采样; A/B 后 thinking 已弃用, 全部非思考)
# DEEPSEEK_API_KEY 由 --export 传入
#SBATCH --job-name=distill_2b
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/distill_2b_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/distill_2b_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=envs/reasoning3b/bin/python
OUTDIR=$BASE/data/reasoning_data

echo "=== 第二轨迹放量(温度 1.0, 4 进程并行) ==="
echo "start: $(date)"

$PY src/gen_reasoning_data.py --train-file $OUTDIR/distill_v2_flash_p0.json \
    --spider-dir $BASE/data/spider_data --output $OUTDIR/distill_v2b_flash_p0.jsonl \
    --model deepseek-v4-flash --thinking disabled --temperature 1.0 \
    --max-tokens 4096 --concurrency 16 --sleep 0 > logs/distill_v2b_p0.log 2>&1 &

$PY src/gen_reasoning_data.py --train-file $OUTDIR/distill_v2_flash_p1.json \
    --spider-dir $BASE/data/spider_data --output $OUTDIR/distill_v2b_flash_p1.jsonl \
    --model deepseek-v4-flash --thinking disabled --temperature 1.0 \
    --max-tokens 4096 --concurrency 16 --sleep 0 > logs/distill_v2b_p1.log 2>&1 &

$PY src/gen_reasoning_data.py --train-file $OUTDIR/distill_v2_flash_p2.json \
    --spider-dir $BASE/data/spider_data --output $OUTDIR/distill_v2b_flash_p2.jsonl \
    --model deepseek-v4-flash --thinking disabled --temperature 1.0 \
    --max-tokens 4096 --concurrency 16 --sleep 0 > logs/distill_v2b_p2.log 2>&1 &

$PY src/gen_reasoning_data.py --train-file $OUTDIR/distill_v2_pro_hard.json \
    --spider-dir $BASE/data/spider_data --output $OUTDIR/distill_v2b_pro_hard.jsonl \
    --model deepseek-v4-pro --thinking disabled --temperature 1.0 \
    --max-tokens 4096 --concurrency 16 --sleep 0 > logs/distill_v2b_pro.log 2>&1 &

wait
echo "=== 第二轨迹完成: $(date) ==="
