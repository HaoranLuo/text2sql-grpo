#!/bin/bash
#SBATCH --job-name=gen_distill
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/gen_distill_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/gen_distill_%j.err

# 推理蒸馏数据生成（DeepSeek API，CPU 即可）
# DEEPSEEK_API_KEY 由 --export 传入（不写入脚本/git）
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
envs/reasoning3b/bin/python src/gen_reasoning_data.py \
    --num-train 1000 --model deepseek-chat --concurrency 8
echo "=== GEN DONE ==="
