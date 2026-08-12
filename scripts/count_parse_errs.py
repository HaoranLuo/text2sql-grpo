"""统计剩余解析错误的类型分布"""
import sys
sys.path.insert(0, "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/tools/original_spider_eval")
from process_sql import get_sql, Schema, get_schema
import sqlite3, json
from collections import Counter

base = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data"
dev = json.load(open(f"{base}/dev.json"))
pred = open("/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/outputs/official_original_vav/pred.sql").read().splitlines()

errs = Counter()
for p, d in zip(pred, dev):
    db = d["db_id"]
    try:
        conn = sqlite3.connect(f"{base}/database/{db}/{db}.sqlite")
        schema = Schema(get_schema(f"{base}/database/{db}/{db}.sqlite"))
        get_sql(schema, p)
    except Exception as e:
        errs[type(e).__name__ + ":" + str(e)[:70]] += 1

print("TOTAL:", sum(errs.values()))
for k, v in errs.most_common(14):
    print(f"{v:4d}  {k}")
