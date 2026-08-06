#!/bin/bash
#SBATCH --job-name=rev7b
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/rev7b_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/rev7b_%j.err

# 修复后代码重跑 7B 基线（无 LoRA）——验证 81% 是否不变
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$BASE/envs/reasoning3b/bin/python $BASE/src/evaluate_after_grpo.py \
    --limit 100 \
    --model-path $BASE/models/qwen2.5-coder-7b-instruct \
    --output-dir $BASE/outputs/eval_rev_7b_base \
    --batch-size 4
