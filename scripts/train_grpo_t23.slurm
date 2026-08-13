#!/bin/bash
# Phase 2 T2.3: GRPO finer + Dr.GRPO 加固(最后一搏)
# T2.1 三级 -10pp / T2.2 finer -9pp → 本配方加: loss_type=dr_grpo(token-mean 去长度归一化,
# Dr.GRPO 论文损失; 监管审查确认 TRL 0.17 支持) + scale_rewards=none(不缩放, bool 语义)
# + lr 2e-6→1e-6(更强锁SFT); 零优势组靠 RewardStdGuard 兜底
# 残余(TRL >=0.18 才有, 记录在案): dapo 零优势组过滤 / clip-higher
# preflight 必须带: PREFLIGHT_SLURM=scripts/train_grpo_t23.slurm PREFLIGHT_EXP=grpo_t23_drgrpo
#SBATCH --job-name=grpo_t23
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/grpo_t23_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/grpo_t23_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/grpo_t23_drgrpo

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== T2.3 GRPO finer+dr_grpo from SFT ckpt ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"
echo "trl: $($PYTHON -c 'import trl; print(trl.__version__)')"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --train-file $BASE/data/grpo_subset_500.json \
    --num-train 500 \
    --num-generations 4 \
    --max-steps 40 \
    --reward-type finer \
    --loss-type dr_grpo \
    --scale-rewards none \
    --learning-rate 1e-6 \
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
        --output-dir $BASE/outputs/grpo_t23_$NAME
done

echo "--- final ---"
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path "$OUT" --model-path "$MODEL" --spider-dir "$SPIDER" \
    --limit 1034 --batch-size 8 \
    --output-dir $BASE/outputs/grpo_t23_final

echo "=== 官方口径(final) ==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/grpo_t23_final/items.json \
    $BASE/outputs/official_grpo_t23_final

echo "Done: $(date)"
