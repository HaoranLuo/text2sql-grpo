#!/bin/bash
# Phase 2 T2.1: GRPO three_level 从 SFT 检查点起步
# 数据卫生 G3: --train-file 用 GRPO-manifest 子集(与 SFT-gold 零重叠), 不用 train_spider 前 N 条
# 配方: 三级奖励 1.0/0.1/0.0(3B 唯一验证有效) + lr 3e-6 + beta 0.04 + T 1.0 + G4 + 40 步
#SBATCH --job-name=grpo_t21
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/grpo_t21_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/grpo_t21_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/grpo_t21_sft3lvl

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== T2.1 GRPO three_level from SFT ckpt ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --train-file $BASE/data/grpo_subset_500.json \
    --num-train 500 \
    --num-generations 4 \
    --max-steps 40 \
    --reward-type three_level \
    --learning-rate 3e-6 \
    --beta 0.04 \
    --temperature 1.0 \
    --lora-init $BASE/checkpoints/sft_phase1 \
    --output-dir "$OUT" \
    --spider-dir "$SPIDER" \
    --model-path "$MODEL"

echo "Training done: $(date)"

echo "=== 评估 checkpoints(全量 1034 自定义口径) ==="
for CK in $(ls -d $OUT/checkpoint-* 2>/dev/null); do
    NAME=$(basename $CK)
    echo "--- $NAME ---"
    $PYTHON $BASE/src/evaluate_after_grpo.py \
        --lora-path "$CK" --model-path "$MODEL" --spider-dir "$SPIDER" \
        --limit 1034 --batch-size 8 \
        --output-dir $BASE/outputs/grpo_t21_$NAME
done

echo "--- final ---"
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path "$OUT" --model-path "$MODEL" --spider-dir "$SPIDER" \
    --limit 1034 --batch-size 8 \
    --output-dir $BASE/outputs/grpo_t21_final

echo "=== 官方口径(final) ==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/grpo_t21_final/items.json \
    $BASE/outputs/official_grpo_t21_final

echo "Done: $(date)"
