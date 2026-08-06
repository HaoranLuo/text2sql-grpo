#!/bin/bash
#SBATCH --job-name=a2_vsft
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:49:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/a2_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/a2_%j.err

# A2: 投票自蒸馏 SFT 训练 + 单prompt 评估
# 调研优化（RESEARCH_NOTES 4/6）：混合 10-20% gold seed 数据防 model collapse
set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== A2-0: 混合投票蒸馏数据 + gold seed 数据（防 model collapse） ==="
$PYTHON -c "
import json, sys
sys.path.insert(0, '$BASE/src')
from reasoning_generator_agent import ReasoningGeneratorAgent
from spider_utils import SpiderLoader

vote_data = json.load(open('$BASE/data/vote_sft_data.json'))
print(f'投票蒸馏数据: {len(vote_data)} 条')

# 附加 gold seed 数据（训练集前 100 条，与 B 实验同构建方式）
loader = SpiderLoader('$SPIDER')
train = json.load(open('$SPIDER/train_spider.json'))[:100]
seed = []
for item in train:
    try:
        ddl, _ = loader.get_ddl_with_source(item['db_id'])
    except Exception:
        continue
    prompt = ReasoningGeneratorAgent.build_prompt(question=item['question'], ddl_schema=ddl, dialect='sqlite')
    completion = '\`\`\`sql\n' + item['query'].strip().rstrip(';') + '\n\`\`\`'
    seed.append({'text': prompt + '\n\n' + completion})
print(f'gold seed 数据: {len(seed)} 条')

mixed = vote_data + seed
print(f'混合后: {len(mixed)} 条（蒸馏:{len(vote_data)} + seed:{len(seed)}）')
with open('$BASE/data/vote_sft_mix.json', 'w') as f:
    json.dump(mixed, f, ensure_ascii=False)
"

echo "=== A2: 投票自蒸馏 SFT（混合数据） ==="
$PYTHON $BASE/src/train_sft.py \
    --data $BASE/data/vote_sft_mix.json \
    --output $BASE/checkpoints/sft_vote \
    --model-path "$MODEL" \
    --epochs 3 --batch-size 1 --lr 2e-5

echo "=== A2: 单prompt 评估（关键指标：50% → ?） ==="
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/sft_vote \
    --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/eval_sft_vote --batch-size 8
