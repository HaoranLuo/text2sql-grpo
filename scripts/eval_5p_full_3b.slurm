#!/bin/bash
#SBATCH --job-name=5pf1034
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:30:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5pf1034_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5pf1034_%j.err

# 全量 1034 3B 训练后 × 5prompt 投票（执行分组升级版）——官方口径全景
# 预计: 5170 次生成 × 4.4s ≈ 6.3 小时
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$BASE/envs/reasoning3b/bin/python $BASE/src/eval_5prompt_agent.py \
    --lora-path $BASE/checkpoints/grpo_3b_3lvl/checkpoint-25 \
    --output-dir $BASE/outputs/eval_5p_full_1034_3b \
    --n-prompts 5 --limit 1034
