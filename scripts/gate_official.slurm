#!/bin/bash
# GO/NO-GO 门 步骤 2/2：官方 EX（CPU，eval_official.sh）
# 用法: sbatch --export=OUT=... scripts/gate_official.slurm
#SBATCH --job-name=gate_off
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/gate_off_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/gate_off_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
OUT=${OUT:?需 OUT env}

if [ ! -f "$OUT/items.json" ]; then
    echo "FATAL: $OUT/items.json 不存在（先跑 gate_single_shot）" >&2
    exit 1
fi

echo "=== 官方 EX: $OUT ==="
bash scripts/eval_official.sh "$OUT/items.json" "$OUT/official"

echo "Done: $(date)"
