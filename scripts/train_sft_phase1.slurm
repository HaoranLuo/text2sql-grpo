#!/bin/bash
# Phase 1: 蒸馏 SFT 训练 + 全量 1034 评估（自定义 + 官方双口径）
# D1 配方: LoRA r=32 / lr=1e-5 / 3 epochs / dev 早停 / gold 15% mix（数据由 prep_sft_data.py 生成）
# Gate G2: SFT 后单次官方口径 >=55% 才进 GRPO
#SBATCH --job-name=sft_p1
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/sft_p1_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/sft_p1_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
DATA=$BASE/data/sft_phase1_mix.json

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "Phase 1 SFT (D1: r=32, lr=1e-5, 3ep, dev早停)"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "============================================"

echo "=== Step 1: SFT Training ==="
$PYTHON $BASE/src/train_sft.py \
    --data "$DATA" \
    --output $BASE/checkpoints/sft_phase1 \
    --model-path "$MODEL" \
    --epochs 3 --lr 1e-5 --lora-r 32 --lora-alpha 64 --eval-split 0.05

echo "SFT done: $(date)"

echo "=== Step 2: T1.2 全量 1034 自定义口径评估 ==="
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/sft_phase1 \
    --model-path "$MODEL" --spider-dir "$SPIDER" \
    --limit 1034 --batch-size 8 \
    --output-dir $BASE/outputs/eval_sft_phase1_full

echo "=== Step 3: 官方口径评估 (test-suite) ==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/eval_sft_phase1_full/items.json \
    $BASE/outputs/official_sft_phase1

echo "=== FINAL ==="
$PYTHON -c "
import json
s = json.load(open('$BASE/outputs/eval_sft_phase1_full/summary.json'))
print('custom_execution_match_rate:', s.get('custom_execution_match_rate'))
"
tail -n 15 $BASE/outputs/official_sft_phase1/official_result.txt
echo "Done: $(date)"
