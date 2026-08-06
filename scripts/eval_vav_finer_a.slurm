#!/bin/bash
#SBATCH --job-name=vav30a
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:30:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/vav30a_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/vav30a_%j.err

# P1 验收：FINER-3B × n=30 采样 + vav 投票，全量前半 (0-519)
# 预期: 官方 EX ≈ 85%（论文数字）
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE
envs/reasoning3b/bin/python finer_port/eval_vav.py \
    --model-path $BASE/models/FINER-SQL-3B-Spider \
    --output-dir $BASE/outputs/eval_vav_finer_a \
    --n-samples 30 --limit 520 --start-index 0 \
    --prompt-style finer --batch-size 4
