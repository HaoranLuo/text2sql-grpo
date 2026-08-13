#!/bin/bash
# T3.1 MPEV 官方口径复测: SFT 检查点 × 5视角(p1-p5) × K=6 = 30候选 → vav 投票
# (T0.4 归因后: eval_official.sh 注释折叠 bug 已修; 100 条子集先验, 全量后补)
#SBATCH --job-name=mpev_sft
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/mpev_sft_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/mpev_sft_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE
envs/reasoning3b/bin/python finer_port/eval_mpev.py \
    --model-path $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --lora-path $BASE/checkpoints/sft_phase1 \
    --prompt-style 5p \
    --n-perspectives 5 \
    --samples-per-prompt 6 \
    --limit 100 \
    --output-dir $BASE/outputs/eval_mpev_sft_100

echo "=== 官方口径(修复后) ==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/eval_mpev_sft_100/items.json \
    $BASE/outputs/official_mpev_sft_100

echo "Done: $(date)"
