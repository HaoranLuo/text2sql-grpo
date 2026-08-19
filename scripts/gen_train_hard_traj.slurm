#!/bin/bash
# train 集 hard/extra 讲解式轨迹生成（路线 B 修订：无 dev 泄漏）——DeepSeek API + 执行验证
# 用法:
#   1) 构建目标清单(纯 CPU, login node):
#        python src/gen_train_hard_traj.py --build-targets
#   2) 4 进程 stride 分片并行生成:
#        sbatch --array=1-4 scripts/gen_train_hard_traj.slurm
#   3) 全部完成后合并(纯 CPU):
#        python src/gen_train_hard_traj.py --merge-trains
#   DEEPSEEK_API_KEY 从 ~/.bashrc 读取(项目惯例: key 不进脚本/仓库)
#SBATCH --job-name=gen_tr_hard
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/gen_tr_hard_%A_%a.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/gen_tr_hard_%A_%a.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PY=$BASE/envs/reasoning3b/bin/python

source ~/.bashrc   # DEEPSEEK_API_KEY（绝不硬编码到仓库文件）

SHARD=${SLURM_ARRAY_TASK_ID:-1}
N_SHARDS=${N_SHARDS:-4}
MODEL=${MODEL:-deepseek-v4-flash}
CONCURRENCY=${CONCURRENCY:-8}
REPAIR=${REPAIR_ROUNDS:-2}
MAX_TOKENS=${MAX_TOKENS:-4096}
RETRY_FLAG=""
[ "${RETRY:-0}" = "1" ] && RETRY_FLAG="--retry-failed"

echo "=== gen_train_hard_traj: shard=$SHARD/$N_SHARDS model=$MODEL concurrency=$CONCURRENCY repair=$REPAIR retry=${RETRY:-0} ==="
echo "start: $(date)"

$PY $BASE/src/gen_train_hard_traj.py \
    --spider-dir $BASE/data/spider_data \
    --mix $BASE/data/sft_v3_mix.json \
    --out-dir $BASE/outputs/hard_traj \
    --model $MODEL --temperature 0.3 \
    --max-tokens $MAX_TOKENS --concurrency $CONCURRENCY --sleep 0 \
    --repair-rounds $REPAIR \
    --max-qlen 120 --n-targets 360 --out-n 340 \
    --shard $SHARD --n-shards $N_SHARDS $RETRY_FLAG

echo "=== gen_train_hard_traj shard done: $(date) ==="
