#!/bin/bash
#SBATCH --job-name=a3_v5p
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:49:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/a3_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/a3_%j.err

# A3: 投票自蒸馏后模型 × 5prompt 投票（能否超 71%）
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$BASE/envs/reasoning3b/bin/python $BASE/src/eval_5prompt_agent.py \
    --lora-path $BASE/checkpoints/sft_vote \
    --output-dir $BASE/outputs/eval_5p_sft_vote \
    --n-prompts 5
