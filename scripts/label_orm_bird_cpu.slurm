#!/bin/bash
# BIRD 判卷适配 step1: 候选池自动打标（纯 CPU，官方 set-of-tuples 语义）
# 前置: outputs/eval_pool_bird/items.json（1534 题 × 64 候选）已生成
# 预估: ~80k 唯一 (sql, db) 执行（实测 prep 同规模 577s）+ JSON 落盘 ≈ 15-25 min
# 产物: data/orm_train_bird.json + data/orm_train_bird_cap12.json +
#       data/orm_questions_bird.json + data/orm_label_stats_bird.json
# 冒烟: sbatch --export=ALL,LIMIT=30,OUT=/tmp/orm_train_bird_smoke.json,STATS=/tmp/orm_label_stats_bird_smoke.json scripts/label_orm_bird_cpu.slurm
#SBATCH --job-name=orm_bird_label
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_bird_label_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_bird_label_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
cd $BASE

OUT=${OUT:-$BASE/data/orm_train_bird.json}
QUESTIONS=${QUESTIONS:-$BASE/data/orm_questions_bird.json}
STATS=${STATS:-$BASE/data/orm_label_stats_bird.json}
CAP_OUT=${CAP_OUT:-$BASE/data/orm_train_bird_cap12.json}
CAP=${CAP:-12}

echo "=== BIRD ORM 打标（官方 set-of-tuples 语义，16 线程）==="
EXTRA=""
if [ -n "${LIMIT:-}" ]; then
    EXTRA="--limit $LIMIT"
fi

$PYTHON $BASE/src/label_orm_bird.py \
    --items $BASE/outputs/eval_pool_bird/items.json \
    --db-root $BASE/data/bird/bird_dev/dev_20240627/dev_databases \
    --out "$OUT" \
    --questions-out "$QUESTIONS" \
    --stats-out "$STATS" \
    --max-per-question "$CAP" \
    --cap-out "$CAP_OUT" \
    --vav-items $BASE/outputs/bird_select/items_arm_vav.json \
    --threads 16 \
    $EXTRA

echo "Done: $(date)"
echo "产物: $OUT / $QUESTIONS / $STATS / $CAP_OUT"
echo "下一步: 登录节点跑 src/prep_orm_balanced.py（--in $CAP_OUT --out-dir data/bird_orm）"
