"""定位评估器报错对应的 db（dev.json 索引与 db_id）"""
import json, os

base = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data"
dev = json.load(open(f"{base}/dev.json"))

# 检查所有 db 的 singer 表是否含 last_name（找评估器报错的库）
for db in sorted(set(d["db_id"] for d in dev)):
    p = f"{base}/database/{db}/{db}.sqlite"
    if not os.path.exists(p):
        print(f"{db}: NO FILE")
        continue
    import sqlite3
    conn = sqlite3.connect(p)
    try:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        for t in tables:
            cols = conn.execute(f"PRAGMA table_info({t[0]})").fetchall()
            if any(c[1].lower() == "last_name" for c in cols):
                print(f"{db}/{t[0]}: HAS last_name {[c[1] for c in cols][:10]}")
    except Exception as e:
        print(f"{db}: ERR {str(e)[:60]}")
    conn.close()
print("---dev[237]---")
print(dev[237]["db_id"], dev[237]["question"][:50])
