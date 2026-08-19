#!/bin/bash
# memory 质心过滤 全量 1034 验证 (min-votes=0 全启用, drop 0.2 主档) — 子集 +4pp 待全量确认
#SBATCH --job-name=memf_full
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/memf_full_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/memf_full_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

echo "=== memory centroid filter FULL 1034 (min-votes=0) ==="
$BASE/envs/reasoning3b/bin/python $BASE/src/eval_mem_filter.py \
    --lora-path $BASE/checkpoints/sft_v2 \
    --spider-dir $BASE/data/spider_data \
    --output-dir $BASE/outputs/eval_mem_filter_full \
    --limit 1034 --n 16 --drop-ratio 0.2,0.3 --min-votes 0 \
    --checkpoint-every 25

echo "Done: $(date)"
