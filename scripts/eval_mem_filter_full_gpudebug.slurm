#!/bin/bash
# memory 质心过滤全量 1034 — gpudebug 切片版(50min/片, checkpoint 断点续跑)
#SBATCH --job-name=memf_sl
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/memf_sl_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/memf_sl_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

echo "=== mem_filter full 切片 (resume, min-votes=0) ==="
$BASE/envs/reasoning3b/bin/python $BASE/src/eval_mem_filter.py \
    --lora-path $BASE/checkpoints/sft_v2 \
    --spider-dir $BASE/data/spider_data \
    --output-dir $BASE/outputs/eval_mem_filter_full \
    --limit 1034 --n 16 --drop-ratio 0.2,0.3 --min-votes 0 \
    --checkpoint-every 25

echo "Done: $(date)"
