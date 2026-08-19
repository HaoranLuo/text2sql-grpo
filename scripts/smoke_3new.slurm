#!/bin/bash
# 三脚本端到端冒烟（gpudebug, 每个 --limit 2，全部应在 ~20 分钟内完成）
#SBATCH --job-name=smoke3
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/smoke3_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/smoke3_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

echo "=== [1/3] smoke: eval_mix_vote (limit=2) ==="
$PYTHON src/eval_mix_vote.py \
    --lora-a $BASE/checkpoints/sft_phase1 \
    --lora-b $BASE/checkpoints/sft_v2 \
    --output-dir $BASE/outputs/smoke_mix_vote \
    --limit 2 --temperature 0.0

echo "=== [2/3] smoke: eval_mem_filter (limit=2) ==="
$PYTHON src/eval_mem_filter.py \
    --lora-path $BASE/checkpoints/sft_v2 \
    --spider-dir $BASE/data/spider_data \
    --output-dir $BASE/outputs/smoke_mem_filter \
    --limit 2 --n 16 --drop-ratio 0.2,0.3 --min-votes 3

echo "=== [3/3] smoke: eval_maj_temp_grid (n=2) ==="
$PYTHON src/eval_maj_temp_grid.py \
    --lora-path $BASE/checkpoints/sft_v2 \
    --base-model $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --spider-dir $BASE/data/spider_data \
    --n 2 \
    --grid "8,16:0.7,1.0" \
    --output-dir $BASE/outputs/smoke_maj_tgrid

echo "=== ALL SMOKE DONE ==="
