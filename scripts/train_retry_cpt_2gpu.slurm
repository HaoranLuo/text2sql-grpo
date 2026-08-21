#!/bin/bash
# RetrySQL 全参短 CPT（P1-3）双 A40 DDP 版：2×A40 并行，配方不变（有效 batch 64 = 每卡 2×accum16×2 卡）
#SBATCH --job-name=retry_cpt
#SBATCH --partition=aiaca40
#SBATCH --qos=2a40
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/retry_cpt_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/retry_cpt_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=$BASE/envs/reasoning3b/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

if [ ! -f "$BASE/checkpoints/sft_v3_merged/config.json" ]; then
    echo "FATAL: checkpoints/sft_v3_merged 不存在" >&2
    exit 1
fi

echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ') | $(date)"
$PY -m torch.distributed.run --nproc_per_node=2 \
    src/train_retry_cpt.py \
    --data data/retry_cpt_train.json \
    --model-path checkpoints/sft_v3_merged \
    --output checkpoints/retry_cpt \
    --epochs 2 --batch-size 2 --grad-accum 16 --lr 5e-5 \
    --weight-decay 0.1 --save-steps 50 --seed 42

echo "Done: $(date)"
