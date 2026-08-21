#!/bin/bash
# RetrySQL CPT 冒烟（3090）：--limit 100 验证数据路径+collator+8bit adam，几秒~几分钟
#SBATCH --job-name=retry_smoke
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/retry_smoke_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/retry_smoke_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=$BASE/envs/reasoning3b/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PY src/train_retry_cpt.py \
    --data data/retry_cpt_train.json \
    --model-path checkpoints/sft_v3_merged \
    --output checkpoints/retry_cpt_smoke \
    --epochs 1 --batch-size 2 --grad-accum 32 --lr 5e-5 \
    --weight-decay 0.1 --save-steps 999999 --seed 42 --limit 100

echo "Done: $(date)"
