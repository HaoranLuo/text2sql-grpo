#!/bin/bash
# 修复官方评估：跳过空行（parse 失败的预测），并过滤对应 gold 行
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
SPIDER=$BASE/data/spider_data

for RUN in official_3b_base official_3b_trained; do
    OUT=$BASE/outputs/$RUN
    # 过滤 pred 空行 + 对应 gold 行（保持对齐）
    $PYTHON -c "
import json
items = json.load(open('$BASE/outputs/$(echo $RUN | sed 's/official_/eval_rev_/')/items.json'))
items = sorted(items, key=lambda x: x['dataset_index'])
dev = json.load(open('$SPIDER/dev.json'))
idx_set = {it['dataset_index'] for it in items}

pred_lines, gold_lines = [], []
for it in items:
    sql = (it.get('predicted_sql') or '').strip().rstrip(';')
    sql = ' '.join(sql.split())
    if not sql:
        continue
    pred_lines.append(sql)
    gold_lines.append(f\"{dev[it['dataset_index']]['query'].strip().rstrip(';')}\t{dev[it['dataset_index']]['db_id']}\")

with open('$OUT/pred.sql', 'w') as f:
    f.write('\n'.join(pred_lines) + '\n')
with open('$OUT/gold.sql', 'w') as f:
    f.write('\n'.join(gold_lines) + '\n')
print('$RUN aligned:', len(pred_lines))
"
done
