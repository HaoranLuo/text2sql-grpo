#!/bin/bash
# P1-2: fine-tune Qwen3-Embedding-0.6B schema retriever (HN-SupCon, table-level)
# then evaluate recall@k / precision@k on BIRD dev 100 + Spider dev.
# A40 单卡, 预计训练 1.5~3h (0.6B, bs32, 3 epochs) + 评测 ~10min。
# QOS 1a40 对本用户 MaxJobsPU=1 强制串行, 无需依赖声明。
#SBATCH --job-name=retriever
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/train_retriever_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/train_retriever_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$BASE/cache/huggingface

cd $BASE

echo "=== [0/4] baseline eval: untrained Qwen3-Embedding-0.6B (BIRD dev 100) ==="
$PYTHON src/eval_schema_retriever.py \
    --model models/Qwen3-Embedding-0.6B \
    --eval-data data/retriever_eval_bird.jsonl --n 100 --seed 42 \
    --out-dir outputs/schema_retriever_eval_bird100_base

echo "=== [1/4] train: HN-SupCon bi-encoder, table-level schema items ==="
$PYTHON src/train_schema_retriever.py \
    --data-paths data/retriever_bird.jsonl data/retriever_spider.jsonl \
    --model-name-or-path models/Qwen3-Embedding-0.6B \
    --out-dir checkpoints/schema_retriever \
    --epochs 3 --batch-size 32 --n-neg 10 --lr 3e-5

echo "=== [2/4] eval: trained model, BIRD dev 100 ==="
$PYTHON src/eval_schema_retriever.py \
    --model checkpoints/schema_retriever/final \
    --eval-data data/retriever_eval_bird.jsonl --n 100 --seed 42 \
    --out-dir outputs/schema_retriever_eval_bird100

echo "=== [2b/4] eval: trained model, BIRD dev full 1534 ==="
$PYTHON src/eval_schema_retriever.py \
    --model checkpoints/schema_retriever/final \
    --eval-data data/retriever_eval_bird.jsonl \
    --out-dir outputs/schema_retriever_eval_birdfull

echo "=== [3/4] eval: trained model, Spider dev (all 1034) ==="
$PYTHON src/eval_schema_retriever.py \
    --model checkpoints/schema_retriever/final \
    --eval-data data/retriever_eval_spider.jsonl \
    --out-dir outputs/schema_retriever_eval_spiderdev

echo "=== [4/4] baseline for Spider dev (untrained) ==="
$PYTHON src/eval_schema_retriever.py \
    --model models/Qwen3-Embedding-0.6B \
    --eval-data data/retriever_eval_spider.jsonl \
    --out-dir outputs/schema_retriever_eval_spiderdev_base

echo "=== report: tmp_idea_research/retriever_report.md ==="
$PYTHON src/write_retriever_report.py

echo "Done: $(date)"
