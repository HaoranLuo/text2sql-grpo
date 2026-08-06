#!/bin/bash
#SBATCH --job-name=b2_grpo
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/b2_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/b2_%j.err

# B2: 续跑 B（SFT 已完成=47%，修复 lora-init 梯度后重跑 GRPO 阶段）
set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== B4: 在 SFT LoRA 上继续 GRPO（修复 requires_grad） ==="
$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 500 --num-generations 4 --max-steps 25 \
    --reward-type three_level --learning-rate 3e-6 --beta 0.04 --temperature 1.0 \
    --train-batch-size 8 --filter-gold \
    --lora-init $BASE/checkpoints/sft_gold500 \
    --output-dir $BASE/checkpoints/grpo_sft_gold500 \
    --spider-dir "$SPIDER" --model-path "$MODEL"

echo "=== B5: 最终单prompt 评估 ==="
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/grpo_sft_gold500 \
    --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/grpo_sft_gold500 --batch-size 8

echo "=== B6: 5prompt 投票 ==="
$PYTHON $BASE/src/eval_5prompt_agent.py \
    --lora-path $BASE/checkpoints/grpo_sft_gold500 \
    --output-dir $BASE/outputs/eval_5p_grpo_sft_gold500 \
    --n-prompts 5

echo "=== B 结果汇总 ==="
$PYTHON -c "
import json
for name, path in [('SFT(47%)', '$BASE/outputs/eval_sft_gold500/summary.json'),
                   ('SFT+GRPO(单prompt)', '$BASE/outputs/grpo_sft_gold500/summary.json'),
                   ('SFT+GRPO+5p投票', '$BASE/outputs/eval_5p_grpo_sft_gold500/summary.json')]:
    try:
        s = json.load(open(path))
        rate = s.get('custom_execution_match_rate', s.get('match_rate'))
        print(f'{name}: {rate:.1%}')
    except Exception as e:
        print(f'{name}: 读取失败')
print('参考: 3B基线45% | SFT冷启动47% | 三级GRPO 50% | 三级+5p投票 71%')
"
echo "Done: $(date)"
