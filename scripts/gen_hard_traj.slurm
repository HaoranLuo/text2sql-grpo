#!/bin/bash
# 难题定向补课轨迹合成（RevDecomp-SFT 路线 B）——DeepSeek API 生成 + 执行验证
# 用法:
#   pilot:  sbatch --export=ALL,PILOT_LIMIT=20,N_SHARDS=1,CONCURRENCY=4,MODEL=deepseek-v4-pro,THINKING=1,REPAIR_ROUNDS=2,MAX_TOKENS=8192 scripts/gen_hard_traj.slurm
#   全量:   sbatch --export=ALL --array=1-4 scripts/gen_hard_traj.slurm   # 4 进程 stride 分片
#   DEEPSEEK_API_KEY 由 --export=ALL 随提交 shell 环境传入（不硬编码）
#SBATCH --job-name=hard_traj
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/hard_traj_%A_%a.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/hard_traj_%A_%a.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PY=$BASE/envs/reasoning3b/bin/python

SHARD=${SLURM_ARRAY_TASK_ID:-1}
N_SHARDS=${N_SHARDS:-4}
LIMIT=${PILOT_LIMIT:-0}
MODEL=${MODEL:-deepseek-v4-flash}
TEMPERATURE=${TEMPERATURE:-0.5}
CONCURRENCY=${CONCURRENCY:-8}
MAX_TOKENS=${MAX_TOKENS:-4096}
REPAIR=${REPAIR_ROUNDS:-0}
ATTEMPTS=${ATTEMPTS:-1}
ROUTE=${ROUTE:-free}
THINKING_FLAG=""
[ "${THINKING:-0}" = "1" ] && THINKING_FLAG="--thinking-enabled"
THINKING_ALL_FLAG=""
[ "${THINKING_ALL_ROUNDS:-0}" = "1" ] && THINKING_ALL_FLAG="--thinking-all-rounds"
EXPLAIN_FLAG=""
[ "$ROUTE" = "explain" ] && EXPLAIN_FLAG="--explain"
RETRY_FLAG=""
[ "${RETRY:-0}" = "1" ] && RETRY_FLAG="--retry-failed"

echo "=== hard_traj: route=$ROUTE shard=$SHARD/$N_SHARDS limit=$LIMIT model=$MODEL thinking=${THINKING:-0} repair=$REPAIR attempts=$ATTEMPTS ==="
echo "start: $(date)"

$PY $BASE/src/gen_hard_trajectories.py \
    --spider-dir $BASE/data/spider_data \
    --group-correctness $BASE/outputs/adjudicate_soft/group_level_correctness.json \
    --out-dir $BASE/outputs/hard_traj \
    --model $MODEL --temperature $TEMPERATURE \
    --max-tokens $MAX_TOKENS --concurrency $CONCURRENCY --sleep 0 \
    --repair-rounds $REPAIR --attempts $ATTEMPTS $THINKING_FLAG $THINKING_ALL_FLAG \
    $EXPLAIN_FLAG $RETRY_FLAG --shard $SHARD --n-shards $N_SHARDS --limit $LIMIT

echo "=== hard_traj shard done: $(date) ==="
