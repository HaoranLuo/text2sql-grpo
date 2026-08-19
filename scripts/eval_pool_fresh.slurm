#!/bin/bash
# 主战役多源池: v1/v2 新种子(seed=1)重采 — 与 ORM 训练样本(seed=0)错开
#SBATCH --job-name=pool_fresh
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/pool_fresh_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/pool_fresh_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

LIMIT=${LIMIT:-1034}
OUTPUT_DIR=${OUTPUT_DIR:-$BASE/outputs/eval_pool_fresh}

echo "=== v1/v2 新种子重采 (seed=1, 与 ORM 训练 seed=0 错开) ==="
for LORA in sft_phase1 sft_v2; do
  echo "--- pass: $LORA ---"
  $PYTHON $BASE/src/eval_pool_b1.py \
      --limit $LIMIT \
      --output-dir $OUTPUT_DIR \
      --loras sft_phase1,sft_v2 \
      --only-lora $LORA \
      --n 16 --temperature 1.0 --top-p 1.0 --seed 1 \
      --max-new-tokens 2048 \
      --checkpoint-every 25
done

echo "Done: $(date)"
