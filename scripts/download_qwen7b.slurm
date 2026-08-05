#!/bin/bash
#SBATCH --job-name=dl_qwen7b
#SBATCH --partition=cpu8358
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/dl_qwen7b_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/dl_qwen7b_%j.err

set -e

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL_DIR=$BASE/models/qwen2.5-coder-7b-instruct

export HF_ENDPOINT=https://hf-mirror.com

mkdir -p "$MODEL_DIR" "$BASE/logs"

echo "Downloading Qwen2.5-Coder-7B-Instruct..."
echo "Started: $(date)"

$PYTHON -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    'Qwen/Qwen2.5-Coder-7B-Instruct',
    local_dir='$MODEL_DIR',
    max_workers=4,
    resume_download=True,
)
print(f'Done: {path}')
"

echo "Finished: $(date)"
echo "Size:"
du -sh "$MODEL_DIR"
echo "Files:"
ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l
