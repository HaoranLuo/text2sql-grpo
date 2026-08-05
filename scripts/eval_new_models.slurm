#!/bin/bash
#SBATCH --job-name=eval_new
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eval_new_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/eval_new_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
SPIDER=$BASE/data/spider_data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Eval Qwen3-8B
echo "=== Qwen3-8B ==="
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --limit 100 --spider-dir $SPIDER \
    --model-path $BASE/models/Qwen3-8B \
    --output-dir $BASE/outputs/eval_qwen3_100 --batch-size 2

# Eval Qwen2.5-Coder-14B
echo "=== Qwen2.5-Coder-14B ==="
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --limit 100 --spider-dir $SPIDER \
    --model-path $BASE/models/Qwen2.5-Coder-14B \
    --output-dir $BASE/outputs/eval_qwen14_100 --batch-size 1

# Compare
echo ""
echo "=== COMPARISON ==="
echo "Qwen7B baseline: 81%"
for M in qwen3_100 qwen14_100; do
    R=$($PYTHON -c "import json; s=json.load(open('$BASE/outputs/eval_$M/summary.json')); print(f\"{s['custom_execution_match_rate']:.1%}\")" 2>/dev/null || echo "N/A")
    echo "$M: $R"
done
echo "Done: $(date)"
