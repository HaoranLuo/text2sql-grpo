#!/bin/bash
# 官方 test-suite 评估器集成：用已有 items.json 的预测重评
# 用法: bash scripts/eval_official.sh <items.json> <output_dir>
set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
SPIDER=$BASE/data/spider_data
EVAL=$BASE/tools/test_suite_eval/evaluation.py
ITEMS=${1:?items.json 路径}
OUT=${2:?输出目录}

mkdir -p "$OUT"

# 1. 从 items.json 提取预测 + 对齐 gold 子集（官方格式: 每行 SQL，pred 每行一条）
#    按 dataset_index 精确对齐；parse 失败的空预测跳过（pred/gold 同步）
$PYTHON -c "
import json, sys
items = json.load(open('$ITEMS'))
dev = json.load(open('$SPIDER/dev.json'))

pred_lines, gold_lines = [], []
for it in items:
    sql = (it.get('predicted_sql') or '').strip().rstrip(';')
    sql = ' '.join(sql.split())  # 折叠空白为单空格
    if not sql:
        continue  # parse 失败 → 跳过（同步跳过 gold）
    # 兼容两种字段名：eval_5prompt_agent 用 'di'，其他用 'dataset_index'
    idx = it.get('dataset_index', it.get('di'))
    if idx is None:
        continue
    d = dev[idx]
    pred_lines.append(sql)
    gold_lines.append(f\"{d['query'].strip().rstrip(';')}\t{d['db_id']}\")

with open('$OUT/pred.sql', 'w') as f:
    f.write('\n'.join(pred_lines) + '\n')
with open('$OUT/gold.sql', 'w') as f:
    f.write('\n'.join(gold_lines) + '\n')
print(f'对齐 {len(pred_lines)} 条（跳过 {len(items)-len(pred_lines)} 条空预测）')
"

# 2. 官方评估（exec = test-suite 准确率，match = 官方 EM）
$PYTHON "$EVAL" \
    --gold "$OUT/gold.sql" \
    --pred "$OUT/pred.sql" \
    --db "$SPIDER/database" \
    --table "$SPIDER/tables.json" \
    --etype all 2>&1 | tee "$OUT/official_result.txt"

echo "=== 官方评估完成: $OUT ==="
