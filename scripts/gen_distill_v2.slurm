#!/bin/bash
# 蒸馏 Step3 放量: flash 非思考 × 3 份(2159 题中常规题) + pro 非思考难题 1 份
# 4 进程并行(总并发 64), 断点续传; DEEPSEEK_API_KEY 由 --export 传入
#SBATCH --job-name=distill_v2
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/distill_v2_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/distill_v2_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=envs/reasoning3b/bin/python
OUTDIR=$BASE/data/reasoning_data

echo "=== 放量开始: 4 进程并行(3×flash非思考 + 1×pro非思考难题) ==="
echo "start: $(date)"

# 4 个后台进程, 各自独立输出文件(断点续传安全)
$PY src/gen_reasoning_data.py --train-file $OUTDIR/distill_v2_flash_p0.json \
    --spider-dir $BASE/data/spider_data --output $OUTDIR/distill_v2_flash_p0.jsonl \
    --model deepseek-v4-flash --thinking disabled --temperature 0.7 \
    --max-tokens 4096 --concurrency 16 --sleep 0 > logs/distill_v2_p0.log 2>&1 &

$PY src/gen_reasoning_data.py --train-file $OUTDIR/distill_v2_flash_p1.json \
    --spider-dir $BASE/data/spider_data --output $OUTDIR/distill_v2_flash_p1.jsonl \
    --model deepseek-v4-flash --thinking disabled --temperature 0.7 \
    --max-tokens 4096 --concurrency 16 --sleep 0 > logs/distill_v2_p1.log 2>&1 &

$PY src/gen_reasoning_data.py --train-file $OUTDIR/distill_v2_flash_p2.json \
    --spider-dir $BASE/data/spider_data --output $OUTDIR/distill_v2_flash_p2.jsonl \
    --model deepseek-v4-flash --thinking disabled --temperature 0.7 \
    --max-tokens 4096 --concurrency 16 --sleep 0 > logs/distill_v2_p2.log 2>&1 &

$PY src/gen_reasoning_data.py --train-file $OUTDIR/distill_v2_pro_hard.json \
    --spider-dir $BASE/data/spider_data --output $OUTDIR/distill_v2_pro_hard.jsonl \
    --model deepseek-v4-pro --thinking disabled --temperature 0.7 \
    --max-tokens 4096 --concurrency 16 --sleep 0 > logs/distill_v2_pro.log 2>&1 &

wait
echo "=== 放量完成: $(date) ==="
for f in $OUTDIR/distill_v2_*.jsonl; do
    lines=$(wc -l < $f)
    echo "$(basename $f): $lines 行"
done
