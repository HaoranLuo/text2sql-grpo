#!/bin/bash
#SBATCH --job-name=f1034
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:49:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/f1034_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/f1034_%j.err

# 全量 dev 1034 复验：7B 基线单prompt（官方口径可比的前置）
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$BASE/envs/reasoning3b/bin/python $BASE/src/evaluate_after_grpo.py \
    --limit 1034 \
    --model-path $BASE/models/qwen2.5-coder-7b-instruct \
    --output-dir $BASE/outputs/eval_full_1034_7b \
    --batch-size 4
