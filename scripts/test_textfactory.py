"""验证 text_factory 在 singer 库上的行为"""
import sqlite3

print("PY:", __import__("sys").version.split()[0])

# 方法1: 模块级设置
sqlite3.text_factory = lambda b: b.decode("utf-8", errors="replace")
conn = sqlite3.connect("/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data/database/singer/singer.sqlite")
try:
    r = conn.execute("SELECT last_name FROM singer LIMIT 5").fetchall()
    print("METHOD1_OK:", r[:2])
except Exception as e:
    print("METHOD1_FAIL:", type(e).__name__, str(e)[:100])

# 方法2: 连接级设置（备用方案）
conn2 = sqlite3.connect("/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data/database/singer/singer.sqlite")
conn2.text_factory = lambda b: b.decode("utf-8", errors="replace")
try:
    r2 = conn2.execute("SELECT last_name FROM singer LIMIT 5").fetchall()
    print("METHOD2_OK:", r2[:2])
except Exception as e:
    print("METHOD2_FAIL:", type(e).__name__, str(e)[:100])
