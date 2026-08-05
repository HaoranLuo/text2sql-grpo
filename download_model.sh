#!/bin/bash
#SBATCH --job-name=download_qwen3b
#SBATCH --partition=cpu8358
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/download_qwen3b_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/download_qwen3b_%j.err

set -e

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python

export HF_HOME=$BASE/cache/huggingface
export TMPDIR=$BASE/tmp
export MODEL_DIR=$BASE/models/Qwen2.5-Coder-3B-Instruct

mkdir -p "$HF_HOME" "$TMPDIR" "$MODEL_DIR" "$BASE/logs"

echo "Download started at: $(date)"
echo "Node: $(hostname)"
echo "Model directory: $MODEL_DIR"

"$PYTHON" -c "import os; from huggingface_hub import snapshot_download; path=snapshot_download(repo_id='Qwen/Qwen2.5-Coder-3B-Instruct', local_dir=os.environ['MODEL_DIR']); print('Downloaded to:', path)"

echo "Download finished at: $(date)"
