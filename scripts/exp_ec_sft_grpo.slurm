#!/bin/bash
#SBATCH --job-name=ec_sft
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ec_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ec_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/qwen2.5-coder-7b-instruct
SPIDER=$BASE/data/spider_data

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== E-C: SFT cold-start from Spider gold SQL + GRPO ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

# Step 1: 用 Spider train 的 gold SQL 构建 SFT 数据（只保留正确SQL）
# 直接用 train_spider.json 的 query 作为目标，不做推理链（简单可靠）
$PYTHON -c "
import json, sys
sys.path.insert(0, '$BASE/src')
from datasets import Dataset
from reasoning_generator_agent import ReasoningGeneratorAgent
from spider_utils import SpiderLoader

loader = SpiderLoader('$SPIDER')
with open('$SPIDER/train_spider.json') as f:
    train = json.load(f)

records = []
for item in train[:2000]:
    try:
        ddl, _ = loader.get_ddl_with_source(item['db_id'])
    except Exception:
        continue
    prompt = ReasoningGeneratorAgent.build_prompt(question=item['question'], ddl_schema=ddl, dialect='sqlite')
    # SFT 目标：gold SQL（人工标注，保证正确）
    completion = item['query']
    text = prompt + '\n\n' + '```sql\n' + completion + '\n```'
    records.append({'text': text})

print(f'SFT data: {len(records)} samples')
with open('$BASE/data/sft_spider.json', 'w') as f:
    json.dump(records, f, ensure_ascii=False)
"

# Step 2: SFT 训练
echo ""
echo "=== Step 2: SFT ==="
$PYTHON $BASE/src/train_sft.py \
    --data $BASE/data/sft_spider.json \
    --output $BASE/checkpoints/sft_spider \
    --model-path "$MODEL" \
    --epochs 2 \
    --batch-size 1 \
    --lr 2e-5

echo "SFT done: $(date)"

# Step 3: SFT 后评估（应为显著提升或至少不降）
echo ""
echo "=== Step 3: SFT eval ==="
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/sft_spider \
    --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/eval_sft_spider --batch-size 4

# Step 4: GRPO on SFT model (G=8)
echo ""
echo "=== Step 4: GRPO on SFT (G=8) ==="
$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 100 \
    --num-generations 8 \
    --max-steps 50 \
    --reward-type three_level \
    --learning-rate 3e-6 \
    --beta 0.04 \
    --temperature 1.0 \
    --output-dir $BASE/checkpoints/grpo_ec \
    --spider-dir "$SPIDER" \
    --model-path "$MODEL"

echo "GRPO done: $(date)"

# Step 5: 最终评估
echo ""
echo "=== Step 5: Final eval ==="
$PYTHON $BASE/src/evaluate_after_grpo.py \
    --lora-path $BASE/checkpoints/grpo_ec \
    --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
    --output-dir $BASE/outputs/grpo_ec --batch-size 4

echo ""
echo "=== E-C Results ==="
echo "Baseline: 81%"
$PYTHON -c "
import json
s = json.load(open('$BASE/outputs/eval_sft_spider/summary.json'))
print(f'SFT: {s[\"custom_execution_match_rate\"]:.1%}')
s2 = json.load(open('$BASE/outputs/grpo_ec/summary.json'))
print(f'SFT+GRPO: {s2[\"custom_execution_match_rate\"]:.1%}')
"
echo "Done: $(date)"
