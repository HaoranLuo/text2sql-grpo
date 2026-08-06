#!/bin/bash
#SBATCH --job-name=a2_vsft
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:49:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/a2_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/a2_%j.err

# A2: 投票自蒸馏 SFT 训练 + 单prompt 评估
set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== A2: 投票自蒸馏 SFT ==="
$PYTHON $BASE/src/train_sft.py \
    --data $BASE/data/vote_sft_data.json \
    --output $BASE/checkpoints/sft_vote \
    --model-path "$MODEL" \
    --epochs 3 --batch-size 1 --lr 2e-5

echo "=== A2: 单prompt 评估（关键指标：50% → ?） ==="
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/sft_vote \
    --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/eval_sft_vote --batch-size 8
