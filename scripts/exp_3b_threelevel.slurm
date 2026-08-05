#!/bin/bash
#SBATCH --job-name=3b_3lvl
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/3b_3lvl_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/3b_3lvl_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/grpo_3b_3lvl

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "3B + three_level reward + A40 + 修复后代码"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"
echo "============================================"

# =============================================
# Step 1: 基线（训练前 100 条）
# =============================================
echo ""
echo "=== Step 1: Baseline (3B zero-shot) ==="
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/3b_3lvl_baseline --batch-size 8

# =============================================
# Step 2: GRPO 训练（three_level, G=4, 75步）
# 75 步是 3B 甜点（旧实验 40-75 步最佳）
# =============================================
echo ""
echo "=== Step 2: GRPO (three_level, G=4, 75 steps) ==="
$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 100 \
    --num-generations 4 \
    --max-steps 75 \
    --reward-type three_level \
    --learning-rate 3e-6 \
    --beta 0.04 \
    --temperature 1.0 \
    --output-dir "$OUT" \
    --spider-dir "$SPIDER" \
    --model-path "$MODEL"

echo "Training done: $(date)"

# =============================================
# Step 3: 评估所有 checkpoint（早停找最佳）
# =============================================
echo ""
echo "=== Step 3: Evaluate checkpoints ==="
for CK in checkpoint-25 checkpoint-50 checkpoint-75; do
    if [ -d "$OUT/$CK" ]; then
        DEST="3b_3lvl_$CK"
        echo "--- $CK ---"
        $PYTHON $BASE/src/evaluate_after_grpo.py \
            --lora-path "$OUT/$CK" --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
            --output-dir $BASE/outputs/$DEST --batch-size 8
    fi
done

# 最终 checkpoint
echo "--- final ---"
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path "$OUT" --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/3b_3lvl_final --batch-size 8

# =============================================
# Step 4: 汇总对比
# =============================================
echo ""
echo "============================================"
echo "=== 3B three_level 实验汇总 ==="
echo "============================================"
$PYTHON -c "
import json, os
base = '$BASE/outputs'
entries = [
    ('3b_3lvl_baseline', '基线(训练前)'),
    ('3b_3lvl_checkpoint-25', '训练25步'),
    ('3b_3lvl_checkpoint-50', '训练50步'),
    ('3b_3lvl_checkpoint-75', '训练75步'),
    ('3b_3lvl_final', '训练75步最终'),
]
print(f\"{'阶段':<20} {'Match':>8}\")
print('-' * 30)
best = None
for folder, label in entries:
    p = os.path.join(base, folder, 'summary.json')
    if os.path.exists(p):
        with open(p) as f:
            s = json.load(f)
        m = s['custom_execution_match_rate']
        print(f\"{label:<20} {m:>7.1%}\")
        if best is None or m > best[1]:
            best = (label, m)
if best:
    print(f\"\n最佳: {best[0]} = {best[1]:.1%}\")
    print(f\"基线: 37.0%\")
    print(f\"提升: {best[1]-0.37:+.1%}\")
"
echo ""
echo "Done: $(date)"
