#!/bin/bash
# SFT v4-hard: 难题补课(6444 v3 mix + 273/308 难题轨迹 = ~6717/6752 条) — 难题定向加料
# D1 配方不变: r=32/alpha=64, lr=1e-5, 3 epochs, dev 早停, chat 格式与评估端一致
# 数据由 src/prep_sft_v4hard.py 生成(纯 CPU); 轨迹来自 outputs/hard_traj/trajectories.jsonl
#SBATCH --job-name=sft_v4hard
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/sft_v4hard_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/sft_v4hard_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
DATA=$BASE/data/sft_v4hard_mix.json

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== SFT v4-hard (v3 mix + 难题轨迹, ~6717 条) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

$PYTHON $BASE/src/train_sft.py \
    --data "$DATA" \
    --output $BASE/checkpoints/sft_v4hard \
    --model-path "$MODEL" \
    --epochs 3 --lr 1e-5 --lora-r 32 --lora-alpha 64 --eval-split 0.05

echo "SFT v4-hard done: $(date)"

echo "=== 全量 1034 自定义评估 ==="
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/sft_v4hard \
    --model-path "$MODEL" --spider-dir "$SPIDER" \
    --limit 1034 --batch-size 8 \
    --output-dir $BASE/outputs/eval_sft_v4hard_full

echo "=== 官方口径 ==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/eval_sft_v4hard_full/items.json \
    $BASE/outputs/official_sft_v4hard

echo "Done: $(date)"
