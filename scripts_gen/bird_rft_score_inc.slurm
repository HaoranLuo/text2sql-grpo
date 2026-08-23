#!/bin/bash
# 5 源池 ORM 新增组代表打分（gpudebug，vllmenv）：
#   4 源组代表分已由 bird_score_merge.py --phase partition 复用；
#   本作业只打 work/orm_scores_need.json 中的新增组代表（RFT 引入）。
# 提交: sbatch --export=ALL,NEW_WORK=$BASE/outputs/bird_select_5src/work scripts/bird_rft_score_inc.slurm
#SBATCH --job-name=bird_rft_score
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/bird_rft_score_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/bird_rft_score_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

NEW_WORK=${NEW_WORK:-$BASE/outputs/bird_select_5src/work}

if [ ! -f "$NEW_WORK/orm_scores_need.json" ]; then
    echo "FATAL: 缺少 $NEW_WORK/orm_scores_need.json —— 先跑 bird_score_merge.py --phase partition" >&2
    exit 1
fi

echo "=== bird_score_merge --phase score-gpu（新增组代表打分）==="
$PYTHON $BASE/src/bird_score_merge.py --phase score-gpu \
    --new-work "$NEW_WORK" \
    --orm-checkpoint $BASE/checkpoints/orm_bird_bird_bal2 \
    --base-model $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --merge-python $BASE/envs/reasoning3b/bin/python \
    --logprobs-topk 20 --max-length 2048 --chunk-size 512

echo "Done: $(date)"
