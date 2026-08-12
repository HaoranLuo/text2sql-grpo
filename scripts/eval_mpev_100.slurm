#!/bin/bash
#SBATCH --job-name=mpev100
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:49:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/mpev100_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/mpev100_%j.err

# L2 MPEV 验证：3B GRPO checkpoint-25 (LoRA) + 标准 5p prompt
#   5 视角 (p1-p5) x K=6 采样 = 30 候选/题 -> 执行分组投票 (vav 口径)
#   100 条子集；输出 outputs/eval_mpev_100_3b/ (checkpoint.json / items.json / summary.json)
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE
envs/reasoning3b/bin/python finer_port/eval_mpev.py \
    --model-path $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --lora-path $BASE/checkpoints/grpo_3b_3lvl/checkpoint-25 \
    --prompt-style 5p \
    --n-perspectives 5 \
    --samples-per-prompt 6 \
    --limit 100 \
    --output-dir $BASE/outputs/eval_mpev_100_3b
