#!/bin/bash
# 原始 Spider 口径评估（论文 Official EX 对齐版）
# 用法: bash scripts/eval_original.sh <items.json> <output_dir>
# 差异: 原始评估器 + --etype exec + 原始单实例库 + 空预测写 SELECT 1
set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
SPIDER=$BASE/data/spider_data
EVAL=$BASE/tools/original_spider_eval/evaluation.py
ITEMS=${1:?items.json 路径}
OUT=${2:?输出目录}

mkdir -p "$OUT"

# 1. 生成 pred/gold（空预测写 SELECT 1，保持 1034 行对齐）
$PYTHON -c "
import json
items = json.load(open('$ITEMS'))
dev = json.load(open('$SPIDER/dev.json'))

pred_lines, gold_lines = [], []
for it in items:
    sql = (it.get('predicted_sql') or '').strip().rstrip(';')
    sql = ' '.join(sql.split())
    if not sql:
        sql = 'SELECT 1'  # 空预测占位（计错，与 FINER majority_voting 行为一致）
    idx = it.get('dataset_index', it.get('di'))
    d = dev[idx]
    pred_lines.append(sql)
    gold_lines.append(f\"{d['query'].strip().rstrip(';')}\t{d['db_id']}\")

with open('$OUT/pred.sql', 'w') as f:
    f.write('\n'.join(pred_lines) + '\n')
with open('$OUT/gold.sql', 'w') as f:
    f.write('\n'.join(gold_lines) + '\n')
print(f'对齐 {len(pred_lines)} 条（含空预测占位）')
"

# 2. 原始 Spider 评估（--etype exec，原始单实例库）
$PYTHON "$EVAL" \
    --gold "$OUT/gold.sql" \
    --pred "$OUT/pred.sql" \
    --etype exec \
    --db "$SPIDER/database" \
    --table "$SPIDER/tables.json" 2>&1 | tee "$OUT/original_result.txt"

echo "=== 原始口径评估完成: $OUT ==="
