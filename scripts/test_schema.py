"""查看 concert_singer 库的表结构"""
import sqlite3

DB = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data/database/concert_singer/concert_singer.sqlite"
conn = sqlite3.connect(DB)
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("TABLES:", [t[0] for t in tables])
for t in tables:
    cols = conn.execute(f"PRAGMA table_info({t[0]})").fetchall()
    print(f"  {t[0]}: {[c[1] for c in cols][:8]}")
