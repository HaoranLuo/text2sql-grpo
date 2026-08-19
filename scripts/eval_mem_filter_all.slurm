#!/bin/bash
# mem_filter 对照臂: min-votes=0 全启用过滤（负结果的严谨性补强, 100 条 gpudebug）
#SBATCH --job-name=mem_filt2
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/mem_filt2_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/mem_filt2_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

echo "=== memory centroid filter v2 (min-votes=0, 全启用对照) ==="
$BASE/envs/reasoning3b/bin/python $BASE/src/eval_mem_filter.py \
    --lora-path $BASE/checkpoints/sft_v2 \
    --spider-dir $BASE/data/spider_data \
    --output-dir $BASE/outputs/eval_mem_filter_all \
    --limit 100 --n 16 --drop-ratio 0.2,0.3 --min-votes 0 \
    --checkpoint-every 10

echo "Done: $(date)"
