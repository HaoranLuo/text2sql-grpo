#!/bin/bash
# RetrySQL 全参短 CPT（P1-3）：sft_v3_merged 起点，错误注入数据，8bit adam 全参
#SBATCH --job-name=retry_cpt
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/retry_cpt_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/retry_cpt_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=$BASE/envs/reasoning3b/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "$BASE/checkpoints/sft_v3_merged/config.json" ]; then
    echo "FATAL: checkpoints/sft_v3_merged 不存在（依赖 merge_sft_v3 作业）" >&2
    exit 1
fi
if [ ! -f "$BASE/data/retry_cpt_train.json" ]; then
    echo "FATAL: data/retry_cpt_train.json 不存在（先跑 src/gen_retry_data.py）" >&2
    exit 1
fi

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"
$PY src/train_retry_cpt.py \
    --data data/retry_cpt_train.json \
    --model-path checkpoints/sft_v3_merged \
    --output checkpoints/retry_cpt \
    --epochs 2 --batch-size 2 --grad-accum 32 --lr 5e-5 \
    --weight-decay 0.1 --save-steps 50 --seed 42

echo "Done: $(date)"
