#!/bin/bash
#SBATCH --job-name=vav_smk
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:49:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/vav_smk_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/vav_smk_%j.err

# P1 验收冒烟：FINER-SQL-3B-Spider 权重 + finer prompt + n=1（单次生成）
# 验证: 权重可加载 / 管线通 / 单次生成水平
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE
envs/reasoning3b/bin/python finer_port/eval_vav.py \
    --model-path $BASE/models/FINER-SQL-3B-Spider \
    --output-dir $BASE/outputs/eval_vav_finer_smoke \
    --n-samples 1 --limit 100 --prompt-style finer --batch-size 4
