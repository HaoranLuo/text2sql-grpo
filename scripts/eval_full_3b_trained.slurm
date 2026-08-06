#!/bin/bash
#SBATCH --job-name=f3bt
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/f3bt_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/f3bt_%j.err

# 全量 1034 复验：3B 训练后（grpo_3b_3lvl/checkpoint-25）
# v2: gpudebug 50 分钟超时被杀(83%) → 改 A40 2 小时档
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$BASE/envs/reasoning3b/bin/python $BASE/src/evaluate_after_grpo.py \
    --limit 1034 \
    --lora-path $BASE/checkpoints/grpo_3b_3lvl/checkpoint-25 \
    --output-dir $BASE/outputs/eval_full_1034_3b_trained \
    --batch-size 8
