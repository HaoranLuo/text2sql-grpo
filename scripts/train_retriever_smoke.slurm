#!/bin/bash
# Smoke test for the schema retriever training + eval pipeline (P1-2).
# Runs on gpudebug (1x3090, short): 30 training steps + a 20-question eval on
# the untrained model. Real training goes to aiaca40 via train_retriever.slurm.
#SBATCH --job-name=retriever_smoke
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:49:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/retriever_smoke_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/retriever_smoke_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$BASE/cache/huggingface

cd $BASE

echo "=== smoke: 30 training steps (bs16, no epoch save overhead) ==="
$PYTHON src/train_schema_retriever.py \
    --data-paths data/retriever_bird.jsonl data/retriever_spider.jsonl \
    --model-name-or-path models/Qwen3-Embedding-0.6B \
    --out-dir checkpoints/schema_retriever_smoke \
    --epochs 1 --batch-size 16 --n-neg 6 --lr 3e-5 \
    --steps-limit 30 --log-every 10

echo "=== smoke: eval untrained model, BIRD dev 20 questions ==="
$PYTHON src/eval_schema_retriever.py \
    --model models/Qwen3-Embedding-0.6B \
    --eval-data data/retriever_eval_bird.jsonl --n 20 --seed 42 \
    --out-dir outputs/schema_retriever_eval_smoke

echo "=== smoke: eval the 30-step checkpoint, BIRD dev 20 questions ==="
$PYTHON src/eval_schema_retriever.py \
    --model checkpoints/schema_retriever_smoke/final \
    --eval-data data/retriever_eval_bird.jsonl --n 20 --seed 42 \
    --out-dir outputs/schema_retriever_eval_smoke_ckpt

echo "SMOKE_DONE $(date)"
