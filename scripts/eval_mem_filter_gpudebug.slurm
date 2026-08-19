#!/bin/bash
# memory 质心过滤: gpudebug 短作业版(100 条, n=16, 支持断点续跑)
#SBATCH --job-name=mem_filt
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/mem_filt_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/mem_filt_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

echo "=== memory centroid filter (gpudebug, 100 items, n=16) ==="
$PYTHON src/eval_mem_filter.py \
    --lora-path $BASE/checkpoints/sft_v2 \
    --spider-dir $BASE/data/spider_data \
    --output-dir $BASE/outputs/eval_mem_filter \
    --limit 100 --n 16 --drop-ratio 0.2,0.3 --min-votes 3 \
    --checkpoint-every 10

echo "Done: $(date)"
