"""验证 text_factory 在问题库（wta_1 last_name）的行为"""
import sqlite3
import sys

sys.path.insert(0, "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/tools/original_spider_eval")
print("PY:", sys.version.split()[0])

DB = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data/database/wta_1/wta_1.sqlite"

# 不设置 text_factory（复现评估器报错）
try:
    conn = sqlite3.connect(DB)
    r = conn.execute("SELECT last_name FROM players LIMIT 5").fetchall()
    print("DEFAULT_OK:", r)
except Exception as e:
    print("DEFAULT_FAIL:", type(e).__name__, str(e)[:120])

# 设置模块级 text_factory
sqlite3.text_factory = lambda b: b.decode("utf-8", errors="replace")
try:
    conn2 = sqlite3.connect(DB)
    r2 = conn2.execute("SELECT last_name FROM players LIMIT 5").fetchall()
    print("FACTORY_OK:", r2)
except Exception as e:
    print("FACTORY_FAIL:", type(e).__name__, str(e)[:120])

# 检查 evaluation.py 是否真的设置了（import 后检查）
import evaluation as ev
print("EVAL_TEXT_FACTORY_SET:", ev.sqlite3.text_factory is not sqlite3.text_factory or str(ev.sqlite3.text_factory)[:60])
