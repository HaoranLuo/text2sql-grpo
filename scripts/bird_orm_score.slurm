#!/bin/bash
# BIRD 管线 step: 组代表 ORM 打分（gpudebug 3090；VllmScorer, orm_b1, logprobs=20）
# 预估: 引擎 ~4min + 打分 ~15-30min（~9k 组代表 × 2048 token）< 50min
#SBATCH --job-name=bird_score
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/bird_score_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/bird_score_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

OUT_DIR=${OUT_DIR:-$BASE/outputs/bird_select}

if [ ! -f "$BASE/checkpoints/orm_b1/adapter_config.json" ]; then
    echo "FATAL: checkpoints/orm_b1 无 adapter_config.json（ORM 未就绪）" >&2
    exit 1
fi

echo "=== bird_select --phase score (3090) ==="
$PYTHON $BASE/src/bird_select.py --phase score \
    --out-dir "$OUT_DIR" \
    --work-dir "$OUT_DIR/work" \
    --orm-checkpoint $BASE/checkpoints/orm_b1 \
    --base-model $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --merge-python $BASE/envs/reasoning3b/bin/python \
    --logprobs-topk 20 --max-length 2048 --chunk-size 512

echo "Done: $(date)"
