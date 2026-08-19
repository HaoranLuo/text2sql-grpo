#!/bin/bash
# orm_selection_v3 步骤 1：payload 落盘（分组 + 组代表 + 部分证据化 + prompt/token 计数）
# 纯 CPU（reasoning3b 有 sqlglot 30.x → 部分证据化走 sqlglot 重组）。
# 用法: sbatch scripts/orm_v3_payloads.slurm
#SBATCH --job-name=ormv3_dump
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv3_dump_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv3_dump_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=$BASE/envs/reasoning3b/bin/python

echo "=== [multi] payload dump（部分证据化，sqlglot）==="
$PY src/orm_selection_v3.py \
    --items outputs/eval_pool_multi/items.json \
    --out-dir outputs/orm_selection_v3 \
    --dump-payloads --threads 16

echo "=== [unseen] payload dump ==="
$PY src/orm_selection_v3.py \
    --items outputs/eval_pool_unseen/items.json \
    --out-dir outputs/orm_selection_v3_unseen \
    --dump-payloads --threads 16

echo "Done: $(date)"
