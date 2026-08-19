#!/bin/bash
# 判卷老师均衡重训（P1 牌一）：三档正负均衡数据训练 GradeSQL 式 Yes/No ORM
# 用法（DATA env 三档切换，默认 bal1）:
#   sbatch --export=DATA=bal1 scripts/train_orm_bal.slurm
#   sbatch --export=DATA=bal2 scripts/train_orm_bal.slurm   # 建议先跑 bal2 单档验证
#   sbatch --export=DATA=bal3 scripts/train_orm_bal.slurm
#   提交时可用 -J orm_train_bal2 区分作业名（日志以 %j 作业号区分）
# 前置：src/prep_orm_balanced.py 已产出 data/orm_train_bal1/2/3.json
#       （HPC 登录节点 CPU 直接跑，几秒级，无需 sbatch）
# 配方克隆 train_orm.slurm（aiaca40/12h, LoRA r32/alpha64, lr 1e-5, 3ep, 按题 5% dev）
# 与 train_orm.slurm 唯一差异:
#   --pos-weight 1.0 —— 均衡后数据正负比已落在 GradeSQL 最优区间(58-69%)，统一 CE；
#     自动模式(--pos-weight 0) 此时算得 neg/pos≈0.64-0.67，会把正类权重压到 1 以下，
#     与均衡目的相反，故显式禁用。对照实验原版(34.3%)保持 --pos-weight 0 自动 1.92。
#SBATCH --job-name=orm_train_bal
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_train_bal_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_train_bal_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
QUESTIONS=$BASE/data/orm_questions.json

# ---- DATA env 三档切换 ----
DATA=${DATA:-bal1}
case "$DATA" in
  bal1|bal2|bal3) ;;
  *) echo "FATAL: DATA 必须是 bal1|bal2|bal3（收到 '$DATA'）" >&2; exit 1 ;;
esac
DATA_FILE=$BASE/data/orm_train_${DATA}.json
CKPT=$BASE/checkpoints/orm_${DATA}

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== ORM 均衡重训 [$DATA]（P1 牌一）==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"
echo "data: $DATA_FILE"
echo "ckpt: $CKPT"

if [ ! -f "$DATA_FILE" ]; then
    echo "FATAL: 缺少 $DATA_FILE —— 先在 HPC 登录节点跑 src/prep_orm_balanced.py" >&2
    exit 1
fi

$PYTHON $BASE/src/train_orm.py \
    --data "$DATA_FILE" \
    --questions "$QUESTIONS" \
    --output "$CKPT" \
    --model-path "$MODEL" \
    --epochs 3 --batch-size 2 --grad-accum 4 --lr 1e-5 \
    --lora-r 32 --lora-alpha 64 \
    --dev-frac 0.05 --max-length 2048 --pos-weight 1.0

echo "ORM [$DATA] train done: $(date)"
echo "dev 指标: $CKPT/eval_metrics.json"
