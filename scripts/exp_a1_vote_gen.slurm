#!/bin/bash
#SBATCH --job-name=a1_vote
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:49:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/a1_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/a1_%j.err

# A1: 投票自蒸馏 - 生成 SFT 数据（200 训练题 × 3prompt 投票）
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$BASE/envs/reasoning3b/bin/python $BASE/src/gen_vote_sft.py \
    --lora-path $BASE/checkpoints/p2a_500/checkpoint-25 \
    --num-train 200 --n-prompts 3 --min-votes 2 \
    --output $BASE/data/vote_sft_data.json
