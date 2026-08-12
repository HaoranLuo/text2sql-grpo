"""采集所有解析错误的样例（index, db, sql, error_type, error_msg）到 JSON"""
import sys, json
sys.path.insert(0, "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/tools/original_spider_eval")
from process_sql import get_sql, Schema, get_schema
import sqlite3

base = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data"
dev = json.load(open(f"{base}/dev.json"))
pred = open("/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/outputs/official_original_vav/pred.sql").read().splitlines()

errors = []
for i, (p, d) in enumerate(zip(pred, dev)):
    db = d["db_id"]
    try:
        conn = sqlite3.connect(f"{base}/database/{db}/{db}.sqlite")
        schema = Schema(get_schema(f"{base}/database/{db}/{db}.sqlite"))
        get_sql(schema, p)
    except Exception as e:
        errors.append({
            "index": i, "db": db, "sql": p[:300],
            "error_type": type(e).__name__,
            "error_msg": str(e)[:100],
        })

print(f"TOTAL_ERRORS: {len(errors)}")
with open("/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/outputs/official_original_vav/parse_errors.json", "w") as f:
    json.dump(errors, f, ensure_ascii=False, indent=1)
print("saved: parse_errors.json")
