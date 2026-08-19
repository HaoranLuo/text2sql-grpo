#!/bin/bash
# B2 三模型池生成: Spider dev 1034 × {sft_phase1, sft_v2, sft_v3} × n=16 采样(vLLM)
# gpudebug 50min 切片 + checkpoint 断点续跑(与 eval_pool_b1 同款机制)
#SBATCH --job-name=pool_b2
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/pool_b2_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/pool_b2_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

LIMIT=${LIMIT:-1034}
OUTPUT_DIR=${OUTPUT_DIR:-$BASE/outputs/eval_pool_b2}

echo "=== eval_pool_b2 (limit=$LIMIT, 3-lora pool, out=$OUTPUT_DIR) ==="
for LORA in sft_phase1 sft_v2 sft_v3; do
  echo "--- pass: $LORA ---"
  $PYTHON $BASE/src/eval_pool_b1.py \
      --limit $LIMIT \
      --output-dir $OUTPUT_DIR \
      --loras sft_phase1,sft_v2,sft_v3 \
      --only-lora $LORA \
      --n 16 --temperature 1.0 --top-p 1.0 --seed 0 \
      --max-new-tokens 2048 \
      --checkpoint-every 25
done

echo "Done: $(date)"
