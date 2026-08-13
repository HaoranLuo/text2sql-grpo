#!/bin/bash
# Phase 2 T2.2: GRPO finer 稠密奖励 从 SFT 检查点起步(对症药)
# T2.1 教训: 三级稀疏奖励回吐 -10pp → 本配方: 非负稠密 finer(1.0 / 0.5+0.5×atomic / 0.0)
# + 更温和优化(β 0.04→0.02, lr 3e-6→2e-6, 防洗掉 SFT 行为)
# G3: --train-file 用 GRPO-manifest 子集(与 SFT-gold 零重叠)
#SBATCH --job-name=grpo_t22
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/grpo_t22_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/grpo_t22_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/grpo_t22_finer

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== T2.2 GRPO finer from SFT ckpt ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --train-file $BASE/data/grpo_subset_500.json \
    --num-train 500 \
    --num-generations 4 \
    --max-steps 40 \
    --reward-type finer \
    --learning-rate 2e-6 \
    --beta 0.02 \
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
        --output-dir $BASE/outputs/grpo_t22_$NAME
done

echo "--- final ---"
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path "$OUT" --model-path "$MODEL" --spider-dir "$SPIDER" \
    --limit 1034 --batch-size 8 \
    --output-dir $BASE/outputs/grpo_t22_final

echo "=== 官方口径(final) ==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/grpo_t22_final/items.json \
    $BASE/outputs/official_grpo_t22_final

echo "Done: $(date)"
