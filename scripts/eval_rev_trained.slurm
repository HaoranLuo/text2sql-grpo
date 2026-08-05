#!/bin/bash
#SBATCH --job-name=rev_trained
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/rev_trained_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/rev_trained_%j.err

# 修复后代码重跑 3B 训练后（grpo_3b_3lvl/checkpoint-25）——验证 50% 是否不变
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$BASE/envs/reasoning3b/bin/python $BASE/src/evaluate_after_grpo.py \
    --limit 100 \
    --lora-path $BASE/checkpoints/grpo_3b_3lvl/checkpoint-25 \
    --output-dir $BASE/outputs/eval_rev_trained
