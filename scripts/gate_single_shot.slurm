#!/bin/bash
# GO/NO-GO 门：单次贪心生成 + 官方 EX（Spider dev 1034 题）
# 用法: sbatch --export=MODEL=...,OUT=... scripts/gate_single_shot.slurm
# MODEL = plain HF 模型目录（合并后 sft_v3 或 retry_cpt/checkpoint-N）
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

echo "=== 单次贪心评估: $MODEL -> $OUT ==="
$PY src/gen_single_shot.py --model-path "$MODEL" --output-dir "$OUT" --limit 1034

echo "=== 官方 EX ==="
bash scripts/eval_official.sh "$OUT/items.json" "$OUT/official"

echo "Done: $(date)"
