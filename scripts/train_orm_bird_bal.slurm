#!/bin/bash
# BIRD 判卷适配 step2: 判卷老师训练（GradeSQL 式 Yes/No ORM，BIRD 域，便宜配方 1 epoch）
# 前置:
#   - data/orm_train_bird_cap12.json + data/orm_questions_bird.json（label_orm_bird_cpu.slurm）
#   - data/bird_orm/orm_train_bal{1,2,3}.json（登录节点跑 prep_orm_balanced.py：
#     envs/reasoning3b/bin/python src/prep_orm_balanced.py \
#         --in data/orm_train_bird_cap12.json --out-dir data/bird_orm \
#         --stats-out data/bird_orm/orm_bal_stats.json）
# DATA env 三档切换:
#   bird_bal2（默认，主配方）   cap12 均衡 60:40，--pos-weight 1.0
#   bird_bal1                  cap12 1:1 对照（GradeSQL 区间下界之外，探测用）
#   bird_bal3                  cap12 纯上采样 ~60:40 对照
#   bird_raw                   cap12 原版分布 + --pos-weight 0（自动 neg/pos）对照
# 与 train_orm_bal.slurm 的差异: 数据/题集换 BIRD、--epochs 1（便宜适配：cap12 样本
#   量≈Spider 单 epoch 规模，且 BIRD prompt 更长——1 epoch ≈ 2.5-4h，3 epoch ≈ 10h+）、
#   dev 仍按题 5%（seed 42；全量/cap/bal 题目集合一致 → 各变体 dev 划分完全相同）。
# 提交: sbatch --export=ALL,DATA=bird_bal2 scripts/train_orm_bird_bal.slurm
#SBATCH --job-name=orm_bird_train
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_bird_train_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_bird_train_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
QUESTIONS=$BASE/data/orm_questions_bird.json

DATA=${DATA:-bird_bal2}
case "$DATA" in
  bird_bal1|bird_bal2|bird_bal3)
    DATA_FILE=$BASE/data/bird_orm/orm_train_${DATA#bird_}.json
    POS_WEIGHT=1.0 ;;
  bird_raw)
    DATA_FILE=$BASE/data/orm_train_bird_cap12.json
    POS_WEIGHT=0 ;;  # 自动 neg/pos（对照 arm，保持原版 34.3% 式口径）
  *)
    echo "FATAL: DATA 必须是 bird_bal1|bird_bal2|bird_bal3|bird_raw（收到 '$DATA'）" >&2
    exit 1 ;;
esac
CKPT=$BASE/checkpoints/orm_bird_${DATA}

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== BIRD ORM 训练 [$DATA]（便宜配方：1 epoch, 按题 5% dev, seed 42）==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"
echo "data: $DATA_FILE"
echo "ckpt: $CKPT"

if [ ! -f "$DATA_FILE" ]; then
    echo "FATAL: 缺少 $DATA_FILE —— 先跑 label_orm_bird_cpu.slurm + prep_orm_balanced.py" >&2
    exit 1
fi

$PYTHON $BASE/src/train_orm.py \
    --data "$DATA_FILE" \
    --questions "$QUESTIONS" \
    --output "$CKPT" \
    --model-path "$MODEL" \
    --epochs 1 --batch-size 2 --grad-accum 4 --lr 1e-5 \
    --lora-r 32 --lora-alpha 64 \
    --dev-frac 0.05 --max-length 2048 --pos-weight "$POS_WEIGHT" \
    --seed 42

echo "BIRD ORM [$DATA] train done: $(date)"
echo "dev 指标: $CKPT/eval_metrics.json"
echo "下一步: sbatch --export=ALL,ORM_CKPT=$CKPT,OUT_DIR=$BASE/outputs/bird_select_ormbird_${DATA} scripts/bird_orm_score_adapted.slurm"
