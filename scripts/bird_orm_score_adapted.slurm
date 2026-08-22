#!/bin/bash
# BIRD 判卷适配 step3: 新判卷组代表打分（克隆 bird_orm_score.slurm，唯一差异 =
# ORM_CKPT / OUT_DIR 走 env；打分对象不变 = outputs/bird_select 同口径 rankable 组代表）
# 预估: 引擎 ~4min + 打分 10711 组代表 ≈ 3-5 min（实测 orm_b1 同规模 4:39）< 50min
# 前置: ORM_CKPT 已训练（adapter_config.json 存在）+ OUT_DIR/work/prep.json + orm_payloads.json
#       （由 bird_select --phase prep 产出的新目录）
# 提交: sbatch --export=ALL,ORM_CKPT=$BASE/checkpoints/orm_bird_bird_bal2,OUT_DIR=$BASE/outputs/bird_select_ormbird_bird_bal2 scripts/bird_orm_score_adapted.slurm
#SBATCH --job-name=bird_score_orm
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/bird_score_orm_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/bird_score_orm_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

ORM_CKPT=${ORM_CKPT:-$BASE/checkpoints/orm_bird_bird_bal2}
OUT_DIR=${OUT_DIR:-$BASE/outputs/bird_select_ormbird_bird_bal2}

if [ ! -f "$ORM_CKPT/adapter_config.json" ]; then
    echo "FATAL: $ORM_CKPT 无 adapter_config.json（ORM 未就绪）" >&2
    exit 1
fi
if [ ! -f "$OUT_DIR/work/orm_payloads.json" ]; then
    echo "FATAL: 缺少 $OUT_DIR/work/orm_payloads.json —— 先跑 bird_select --phase prep" >&2
    exit 1
fi

echo "=== bird_select --phase score（新判卷）==="
echo "orm checkpoint: $ORM_CKPT"
echo "out dir: $OUT_DIR"
$PYTHON $BASE/src/bird_select.py --phase score \
    --out-dir "$OUT_DIR" \
    --work-dir "$OUT_DIR/work" \
    --orm-checkpoint "$ORM_CKPT" \
    --base-model $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --merge-python $BASE/envs/reasoning3b/bin/python \
    --logprobs-topk 20 --max-length 2048 --chunk-size 512

echo "Done: $(date)"
echo "下一步: sbatch --export=ALL,PHASE=final,OUT_DIR=$OUT_DIR scripts/bird_select_cpu.slurm"
