#!/bin/bash
#SBATCH --job-name=eval_7b
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eval_7b_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eval_7b_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python

for ckpt in checkpoint-25 checkpoint-50 checkpoint-75; do
  echo "=== $ckpt ==="
  $PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/grpo_7b/$ckpt \
    --limit 100 \
    --spider-dir $BASE/data/spider_data \
    --model-path $BASE/models/qwen2.5-coder-7b-instruct \
    --output-dir $BASE/outputs/grpo_7b_$ckpt \
    --batch-size 4
done

echo "=== Done ==="
