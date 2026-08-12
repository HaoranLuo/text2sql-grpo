"""楠岃瘉 text_factory 鍦?singer 搴撲笂鐨勮涓?""
import sqlite3

print("PY:", __import__("sys").version.split()[0])

# 鏂规硶1: 妯″潡绾ц缃?sqlite3.text_factory = lambda b: b.decode("utf-8", errors="replace")
conn = sqlite3.connect("/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data/database/singer/concert_singer.sqlite")
try:
    r = conn.execute("SELECT last_name FROM singer LIMIT 5").fetchall()
    print("METHOD1_OK:", r[:2])
except Exception as e:
    print("METHOD1_FAIL:", type(e).__name__, str(e)[:100])

# 鏂规硶2: 杩炴帴绾ц缃紙澶囩敤鏂规锛?conn2 = sqlite3.connect("/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data/database/singer/concert_singer.sqlite")
conn2.text_factory = lambda b: b.decode("utf-8", errors="replace")
try:
    r2 = conn2.execute("SELECT last_name FROM singer LIMIT 5").fetchall()
    print("METHOD2_OK:", r2[:2])
except Exception as e:
    print("METHOD2_FAIL:", type(e).__name__, str(e)[:100])
