#!/bin/bash
# GO/NO-GO 门 步骤 1/2：单次贪心生成（GPU 3090，仅生成 items.json）
# 用法: sbatch --export=MODEL=...,OUT=... scripts/gate_single_shot.slurm
# 官方 EX 评估由 gate_official.slurm（CPU）接续
#SBATCH --job-name=gate_ss
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/gate_ss_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/gate_ss_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=$BASE/envs/vllmenv/bin/python
MODEL=${MODEL:?需 MODEL env}
OUT=${OUT:?需 OUT env}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 单次贪心生成: $MODEL -> $OUT ==="
$PY src/gen_single_shot.py --model-path "$MODEL" --output-dir "$OUT" --limit 1034

echo "Done: $(date)"
