#!/bin/bash
#SBATCH --job-name=rev_base
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/rev_base_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/rev_base_%j.err

# 修复后代码重跑 3B 基线（无 LoRA）——验证 45% 是否不变
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$BASE/envs/reasoning3b/bin/python $BASE/src/evaluate_after_grpo.py \
    --limit 100 \
    --output-dir $BASE/outputs/eval_rev_baseline
