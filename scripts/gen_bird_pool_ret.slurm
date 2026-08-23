#!/bin/bash
# BIRD dev 大候选池生成（检索器 top-8 表裁剪版）：4 checkpoint × 16 采样
# （vLLM 单引擎 max_loras=4，50min 切片）。与 scripts/gen_bird_pool.slurm 同口径，
# 唯一差异 = 追加 --retriever-tables（离线 top-8 清单）+ --retriever-k 8。
# 生成预算 --max-gen-seconds 2400 内处理，超预算落 checkpoint 正常退出（exit 0），
# 链脚本 scripts/bird_ret_chain.sh 重提续跑直至 fully_done。
# 冒烟: sbatch --export=ALL,LIMIT=10,OUTPUT_DIR=$BASE/outputs/eval_pool_bird_ret_smoke scripts/gen_bird_pool_ret.slurm
#SBATCH --job-name=bird_pool_ret
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/bird_pool_ret_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/bird_pool_ret_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

LIMIT=${LIMIT:-1534}
OUTPUT_DIR=${OUTPUT_DIR:-$BASE/outputs/eval_pool_bird_ret}
RETRIEVER_TABLES=${RETRIEVER_TABLES:-$BASE/outputs/schema_retriever_infer/bird_dev_topk_k8.json}
RETRIEVER_K=${RETRIEVER_K:-8}

echo "=== gen_bird_pool RET (limit=$LIMIT, out=$OUTPUT_DIR, slice=00:50:00) ==="
echo "retriever-tables=$RETRIEVER_TABLES retriever-k=$RETRIEVER_K"
$PYTHON $BASE/src/gen_bird_pool.py \
    --limit $LIMIT \
    --output-dir $OUTPUT_DIR \
    --loras sft_phase1,sft_v2,sft_v3,p2a_500 \
    --n 16 --temperature 1.0 --top-p 1.0 --seed 0 \
    --max-new-tokens 2048 \
    --max-prompt-tokens 3072 \
    --checkpoint-every 25 \
    --max-gen-seconds 2400 \
    --retriever-tables $RETRIEVER_TABLES \
    --retriever-k $RETRIEVER_K

echo "Done: $(date)"
