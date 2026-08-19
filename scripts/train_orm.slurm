#!/bin/bash
# T2 自训 ORM 训练（GradeSQL 式 Yes/No 验证器）：Qwen2.5-Coder-3B-Instruct + LoRA
# 前置：data/orm_train.json + data/orm_questions.json 已由 src/label_orm_data.py
#       全量打标产出（见 scripts/label_orm_cpu.slurm，CPU 分区 ~20 分钟）
# 配方沿用 D1 惯例：r=32/alpha=64, lr 1e-5, 3 epochs；dev 按题切 5%（防同题泄漏），
# 报告 acc / rank_acc / rank_acc_recoverable / rank_acc_maj_wrong（败局题救回率）
#SBATCH --job-name=orm_train
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_train_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_train_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
DATA=$BASE/data/orm_train.json
QUESTIONS=$BASE/data/orm_questions.json

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== T2 ORM 训练（GradeSQL 式 Yes/No, 3B LoRA）==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

if [ ! -f "$DATA" ]; then
    echo "FATAL: 缺少 $DATA —— 先跑 scripts/label_orm_cpu.slurm 打标" >&2
    exit 1
fi

$PYTHON $BASE/src/train_orm.py \
    --data "$DATA" \
    --questions "$QUESTIONS" \
    --output $BASE/checkpoints/orm_b1 \
    --model-path "$MODEL" \
    --epochs 3 --batch-size 2 --grad-accum 4 --lr 1e-5 \
    --lora-r 32 --lora-alpha 64 \
    --dev-frac 0.05 --max-length 2048 --pos-weight 0
# pos-weight 0 = 自动取 neg/pos（实测正率 34.3% → ≈1.92）；ranking 目标下给
# 少数正类加权重，避免被 66% 负类淹没；对照可用 --pos-weight 1.0（GradeSQL 默认）

echo "ORM train done: $(date)"
echo "dev 指标: $BASE/checkpoints/orm_b1/eval_metrics.json"
