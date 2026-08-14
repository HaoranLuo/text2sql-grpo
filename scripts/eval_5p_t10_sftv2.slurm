#!/bin/bash
# 瀹為獙 1锛氶珮娓?5p 鎶曠エ锛圱=1.0锛夆€斺€?SFT v2 妫€鏌ョ偣锛屽叏閲?1034 鏉?# 鐩殑锛氭楠屾姇绁ㄥ鐩婄缉姘寸殑鍊欓€夊鏍锋€у亣璁撅紙v1 +9.3pp vs v2 +5.2pp锛宮aj@K 4 绁ㄩケ鍜岋級
# 瀵规瘮鍩虹嚎锛歟val_5p_sft_v2锛堣椽蹇冭В鐮侊級
#SBATCH --job-name=5p_t10_sftv2
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5p_t10_sftv2_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5p_t10_sftv2_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$BASE/envs/reasoning3b/bin/python $BASE/src/eval_5p_t10.py \
    --lora-path $BASE/checkpoints/sft_v2 \
    --output-dir $BASE/outputs/eval_5p_t10_sft_v2 \
    --n-prompts 5 --limit 1034 --temperature 1.0

echo "=== 瀹樻柟鍙ｅ緞 ==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/eval_5p_t10_sft_v2/items.json \
    $BASE/outputs/official_5p_t10_sft_v2

echo "Done: $(date)"
