#!/bin/bash
#SBATCH --job-name=5p7b
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5p7b_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5p7b_%j.err

# 7B 零样本 × 5prompt 投票（3p=85%，验证纯5p简单投票）
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$BASE/envs/reasoning3b/bin/python $BASE/src/eval_5prompt_agent.py \
    --base-model $BASE/models/qwen2.5-coder-7b-instruct \
    --output-dir $BASE/outputs/eval_5p_7b_base \
    --n-prompts 5
