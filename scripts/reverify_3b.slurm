#!/bin/bash
#SBATCH --job-name=rev_3b
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/rev3b_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/rev3b_%j.err

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
SPIDER=$BASE/data/spider_data
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 用修复后提取器重测 3B 关键结果 ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 1. 真实基线（修复提取器后）
echo ""
echo "--- 真实基线（修复后） ---"
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/3b_baseline_v2 --batch-size 8

# 2. 训练后 25 步 checkpoint（修复提取器后）
echo ""
echo "--- 训练25步 checkpoint-25（修复后） ---"
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/grpo_3b_3lvl/checkpoint-25 \
    --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/3b_c25_v2 --batch-size 8

# 3. 汇总
echo ""
echo "============================================"
echo "=== 3B 真实结果（修复提取器后） ==="
echo "============================================"
$PYTHON -c "
import json
base = '$BASE/outputs'
def get(folder):
    with open(f'{base}/{folder}/summary.json') as f:
        s = json.load(f)
    return s['custom_execution_match_rate']

b = get('3b_baseline_v2')
c25 = get('3b_c25_v2')
print(f'真实基线:        {b:.1%}')
print(f'训练25步(真实):  {c25:.1%}')
print(f'真实提升:        {c25-b:+.1%}')
print()
print('对比（带bug提取器）:')
print(f'  旧基线: 37.0% (旧环境) / 34.0% (新环境带bug)')
print(f'  训练25步: 40.0% (带bug)')
"
echo ""
echo "Done: $(date)"
