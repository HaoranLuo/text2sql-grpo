"""诊断 pred 解析失败：打印前 10 个失败样例 + 共同模式"""
import sys

sys.path.insert(0, "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/tools/original_spider_eval")
from process_sql import get_sql, Schema, get_schema
import sqlite3, os

base = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data"
dev = json = __import__("json").load(open(f"{base}/dev.json"))

pred_lines = open("/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/outputs/official_original_vav/pred.sql").read().splitlines()

fails = []
for i, (p, d) in enumerate(zip(pred_lines, dev)):
    db = d["db_id"]
    db_path = f"{base}/database/{db}/{db}.sqlite"
    try:
        conn = sqlite3.connect(db_path)
        schema = Schema(get_schema(db_path))
        get_sql(schema, p)
    except Exception as e:
        fails.append((i, db, p[:150], str(e)[:80]))

print(f"TOTAL FAILS: {len(fails)}")
for i, db, p, e in fails[:10]:
    print(f"\n[{i}] db={db}\n  SQL: {p}\n  ERR: {e}")
