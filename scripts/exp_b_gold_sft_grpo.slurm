#!/bin/bash
#SBATCH --job-name=b_gsg
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/b_gsg_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/b_gsg_%j.err

# B: 真正的两阶段：gold SQL SFT 冷启动 → 在 SFT LoRA 上继续 GRPO（--lora-init）
set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== B1: 构建 gold SQL SFT 数据（train 前 500 条） ==="
$PYTHON -c "
import json, sys
sys.path.insert(0, '$BASE/src')
from reasoning_generator_agent import ReasoningGeneratorAgent
from spider_utils import SpiderLoader

loader = SpiderLoader('$SPIDER')
with open('$SPIDER/train_spider.json') as f:
    train = json.load(f)

records = []
for item in train[:500]:
    try:
        ddl, _ = loader.get_ddl_with_source(item['db_id'])
    except Exception:
        continue
    prompt = ReasoningGeneratorAgent.build_prompt(question=item['question'], ddl_schema=ddl, dialect='sqlite')
    completion = '```sql\n' + item['query'].strip().rstrip(';') + '\n```'
    records.append({'text': prompt + '\n\n' + completion})

print(f'SFT data: {len(records)} samples')
with open('$BASE/data/sft_gold500.json', 'w') as f:
    json.dump(records, f, ensure_ascii=False)
"

echo "=== B2: SFT 训练（gold SQL 冷启动） ==="
$PYTHON $BASE/src/train_sft.py \
    --data $BASE/data/sft_gold500.json \
    --output $BASE/checkpoints/sft_gold500 \
    --model-path "$MODEL" \
    --epochs 3 --batch-size 1 --lr 2e-5

echo "=== B3: SFT 后单prompt 评估 ==="
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/sft_gold500 \
    --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/eval_sft_gold500 --batch-size 8

echo "=== B4: 在 SFT LoRA 上继续 GRPO（--lora-init + 过滤干净数据） ==="
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

echo ""
echo "=== B 结果汇总 ==="
$PYTHON -c "
import json
for name, path in [('SFT(单prompt)', '$BASE/outputs/eval_sft_gold500/summary.json'),
                   ('SFT+GRPO(单prompt)', '$BASE/outputs/grpo_sft_gold500/summary.json'),
                   ('SFT+GRPO+5p投票', '$BASE/outputs/eval_5p_grpo_sft_gold500/summary.json')]:
    try:
        s = json.load(open(path))
        rate = s.get('custom_execution_match_rate', s.get('match_rate'))
        print(f'{name}: {rate:.1%}')
    except Exception as e:
        print(f'{name}: 读取失败 {e}')
print('参考: 3B基线45% | 三级GRPO 50% | 三级+5p投票 71%')
"
echo "Done: $(date)"
