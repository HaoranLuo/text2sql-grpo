"""打印前 5 个 IndexError 样例"""
import sys, traceback
sys.path.insert(0, "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/tools/original_spider_eval")
from process_sql import get_sql, Schema, get_schema
import sqlite3, json

base = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data"
dev = json.load(open(f"{base}/dev.json"))
pred = open("/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/outputs/official_original_vav/pred.sql").read().splitlines()

shown = 0
for i, (p, d) in enumerate(zip(pred, dev)):
    db = d["db_id"]
    try:
        conn = sqlite3.connect(f"{base}/database/{db}/{db}.sqlite")
        schema = Schema(get_schema(f"{base}/database/{db}/{db}.sqlite"))
        get_sql(schema, p)
    except IndexError as e:
        if shown < 5:
            print(f"[{i}] db={db}\n  SQL: {p[:200]}")
            traceback.print_exc(limit=3)
            print("---")
            shown += 1
