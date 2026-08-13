#!/bin/bash
# Phase 3: SFT 妫€鏌ョ偣 5prompt 鎶曠エ 鍏ㄩ噺 1034(璁粌渚х粓鐐?60.8% 鐨勬姇绁ㄦ斁澶? 棰勬湡 68-75%)
#SBATCH --job-name=5p_sftv2
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:30:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5p_sftv2_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5p_sftv2_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$BASE/envs/reasoning3b/bin/python $BASE/src/eval_5prompt_agent.py \
    --lora-path $BASE/checkpoints/sft_v2 \
    --output-dir $BASE/outputs/eval_5p_sft_v2 \
    --n-prompts 5 --limit 1034

echo "=== 瀹樻柟鍙ｅ緞 ==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/eval_5p_sft_v2/items.json \
    $BASE/outputs/official_5p_sft_v2

echo "Done: $(date)"

