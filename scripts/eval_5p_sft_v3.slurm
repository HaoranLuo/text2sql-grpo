#!/bin/bash
# sft_v3 训练后的 5p 投票放大（排队在 sft_v3 之后自动衔接）
#SBATCH --job-name=5p_sftv3
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:30:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5p_sftv3_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5p_sftv3_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 5p 投票 × sft_v3 全量 1034 ==="
$BASE/envs/reasoning3b/bin/python $BASE/src/eval_5prompt_agent.py \
    --lora-path $BASE/checkpoints/sft_v3 \
    --output-dir $BASE/outputs/eval_5p_sft_v3 \
    --n-prompts 5 --limit 1034

echo "=== 官方口径 ==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/eval_5p_sft_v3/items.json \
    $BASE/outputs/official_5p_sft_v3

echo "Done: $(date)"
