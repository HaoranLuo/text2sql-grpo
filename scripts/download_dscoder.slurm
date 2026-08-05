#!/bin/bash
#SBATCH --job-name=download_dsc
#SBATCH --partition=cpu8358
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/download_dsc_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/download_dsc_%j.err

set -e

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL_DIR=$BASE/models/deepseek-coder-6.7b-instruct
HF_HOME=$BASE/cache/huggingface

mkdir -p "$HF_HOME" "$MODEL_DIR" "$BASE/logs"

export HF_HOME
export HF_ENDPOINT=https://hf-mirror.com

echo "Downloading DeepSeek-Coder-6.7B-Instruct..."
echo "Started at: $(date)"

$PYTHON -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='deepseek-ai/deepseek-coder-6.7b-instruct',
    local_dir='$MODEL_DIR',
    max_workers=8,
)
print(f'Downloaded to: {path}')
"

echo "Download finished at: $(date)"
echo "Model size:"
du -sh "$MODEL_DIR"
echo "File count:"
ls "$MODEL_DIR" | wc -l
