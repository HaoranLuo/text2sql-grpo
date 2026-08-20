#!/bin/bash
# orm_bal2 训练后 dev 权威评估（--eval-only 加载已训练适配器）
# 背景：训练本体已完成（8295 步全跑完），仅训练后 evaluate_dev 崩溃（datasets 迭代怪癖，已修复）
#SBATCH --job-name=ormbal2_eval
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormbal2_eval_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormbal2_eval_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== orm_bal2 --eval-only（dev 权威指标）==="
$PYTHON $BASE/src/train_orm.py \
    --data $BASE/data/orm_train_bal2.json \
    --questions $BASE/data/orm_questions.json \
    --output $BASE/checkpoints/orm_bal2 \
    --model-path "$MODEL" \
    --eval-only \
    --batch-size 2 --dev-frac 0.05 --max-length 2048 --pos-weight 1.0

echo "Done: $(date)"
