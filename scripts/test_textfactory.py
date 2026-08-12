"""验证 text_factory 在 concert_singer 库（评估器报错的场景）"""
import sqlite3
import sys

print("PY:", sys.version.split()[0])

DB = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data/database/concert_singer/concert_singer.sqlite"

# 方法1: 模块级设置（评估器采用的方式）
sqlite3.text_factory = lambda b: b.decode("utf-8", errors="replace")
try:
    conn = sqlite3.connect(DB)
    r = conn.execute("SELECT last_name FROM singer LIMIT 5").fetchall()
    print("METHOD1_OK:", r)
except Exception as e:
    print("METHOD1_FAIL:", type(e).__name__, str(e)[:120])

# 方法3: 关键测试——复制评估器的完整调用（import evaluation 后执行）
sys.path.insert(0, "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/tools/original_spider_eval")
import evaluation as ev
try:
    conn3 = sqlite3.connect(DB)
    conn3.text_factory = sqlite3.text_factory
    r3 = conn3.execute("SELECT last_name FROM singer LIMIT 5").fetchall()
    print("METHOD3_OK:", r3)
except Exception as e:
    print("METHOD3_FAIL:", type(e).__name__, str(e)[:120])
