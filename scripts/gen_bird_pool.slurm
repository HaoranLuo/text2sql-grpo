#!/bin/bash
# BIRD dev 大候选池生成：4 checkpoint × 16 采样（vLLM 单引擎 max_loras=4，50min 切片）
# 生成预算 --max-gen-seconds 2400 内处理，超预算落 checkpoint 正常退出（exit 0），
# 链脚本 scripts/bird_chain.sh 重提续跑直至 fully_done。
#SBATCH --job-name=bird_pool
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/bird_pool_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/bird_pool_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

# 冒烟: sbatch --export=ALL,LIMIT=10,OUTPUT_DIR=$BASE/outputs/eval_pool_bird_smoke scripts/gen_bird_pool.slurm
LIMIT=${LIMIT:-1534}
OUTPUT_DIR=${OUTPUT_DIR:-$BASE/outputs/eval_pool_bird}

echo "=== gen_bird_pool (limit=$LIMIT, out=$OUTPUT_DIR, slice=00:50:00) ==="
$PYTHON $BASE/src/gen_bird_pool.py \
    --limit $LIMIT \
    --output-dir $OUTPUT_DIR \
    --loras sft_phase1,sft_v2,sft_v3,p2a_500 \
    --n 16 --temperature 1.0 --top-p 1.0 --seed 0 \
    --max-new-tokens 2048 \
    --max-prompt-tokens 3072 \
    --checkpoint-every 25 \
    --max-gen-seconds 2400

echo "Done: $(date)"
