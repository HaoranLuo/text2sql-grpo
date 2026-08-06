#!/bin/bash
#SBATCH --job-name=dl_finer
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:49:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/dl_finer_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/dl_finer_%j.err

# 下载 FINER-SQL-3B-Spider 权重（验证 85% 官方 EX 真实性）
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE
envs/reasoning3b/bin/huggingface-cli download griffith-bigdata/FINER-SQL-3B-Spider \
    --local-dir models/FINER-SQL-3B-Spider 2>&1 | tail -3
echo "DOWNLOAD_DONE"
