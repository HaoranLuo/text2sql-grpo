#!/bin/bash
# Phase 3: SFT 检查点 5prompt 投票 全量 1034(训练侧终点 60.8% 的投票放大, 预期 68-75%)
#SBATCH --job-name=5p_sft1k
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:30:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5p_sft1k_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5p_sft1k_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$BASE/envs/reasoning3b/bin/python $BASE/src/eval_5prompt_agent.py \
    --lora-path $BASE/checkpoints/sft_phase1 \
    --output-dir $BASE/outputs/eval_5p_sft_phase1 \
    --n-prompts 5 --limit 1034

echo "=== 官方口径 ==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/eval_5p_sft_phase1/items.json \
    $BASE/outputs/official_5p_sft_phase1

echo "Done: $(date)"
