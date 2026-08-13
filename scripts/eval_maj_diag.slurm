#!/bin/bash
# 文献弹药 8.3: 投票失败诊断曲线 maj@K —— SFT phase1 检查点 (LoRA)
#   Spider dev 前 100 条, 同 prompt 一次采样 K=32 条, 按前缀 K∈{4,8,16,32}
#   逐档计算 maj@K + Pass@K/Average@K + 执行分组数 + Self-BLEU 近似 (vav 口径)
# 输出: outputs/eval_maj_diag_100/ (checkpoint.json / items.json / summary.json)
# 官方口径: 对 K=最大档(32) 的 items.json 跑 scripts/eval_official.sh
#SBATCH --job-name=maj_diag
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/maj_diag_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/maj_diag_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

echo "=== Step 1: maj@K 投票失败诊断评估 ==="
envs/reasoning3b/bin/python src/eval_maj_diag.py \
    --model-path $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --lora-path $BASE/checkpoints/sft_phase1 \
    --spider-dir $BASE/data/spider_data \
    --limit 100 \
    --k-list 4,8,16,32 \
    --temperature 1.0 \
    --output-dir $BASE/outputs/eval_maj_diag_100

echo "=== Step 2: 官方口径 (K=32 档 items) ==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/eval_maj_diag_100/items.json \
    $BASE/outputs/official_maj_diag_100

echo "Done: $(date)"
