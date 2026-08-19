#!/bin/bash
# E3: 全量 1034 5p 投票补跑（L7 "投票抹平训练差异" 全量闭合）
# 用法: CKPT=p2a_100 sbatch scripts/eval_5p_full_ckpt.slurm
#SBATCH --job-name=5p_e3
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:30:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5p_e3_%x_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5p_e3_%x_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=${CKPT:?usage: CKPT=p2a_100 sbatch scripts/eval_5p_full_ckpt.slurm}

echo "=== E3 全量 5p 投票: checkpoint=$CKPT ==="
$BASE/envs/reasoning3b/bin/python $BASE/src/eval_5prompt_agent.py \
    --lora-path $BASE/checkpoints/$CKPT \
    --output-dir $BASE/outputs/eval_5p_full_$CKPT \
    --n-prompts 5 --limit 1034

echo "=== 官方口径 ==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/eval_5p_full_$CKPT/items.json \
    $BASE/outputs/official_5p_full_$CKPT

echo "Done: $(date)"
