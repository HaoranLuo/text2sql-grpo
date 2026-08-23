#!/usr/bin/env python3
"""诊断检索器池 38.07 的异常：检查 items 字段与打分键对齐。"""
import json

base = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b"

items = json.load(open(f"{base}/outputs/eval_pool_bird_ret/items.json"))
it = items[0]
print("ret item keys:", list(it.keys()))
print("retriever_tables:", it.get("retriever_tables"))
print("retriever_is_full:", it.get("retriever_is_full"))
print("retriever_gold_missing:", it.get("retriever_gold_missing"))
c = it["candidates"][0]
print("cand keys:", list(c.keys()))
print("sql sample:", str(c.get("sql"))[:80])
n_gm = sum(1 for x in items if x.get("retriever_gold_missing"))
print(f"gold_missing 题数: {n_gm}/{len(items)}")

# 打分键
try:
    s = json.load(open(f"{base}/outputs/bird_select_bird_ret/work/orm_scores.json"))
    print("score entries:", len(s))
    ks = list(s.keys())[:3]
    print("score keys sample:", ks)
except Exception as e:
    print("scores load fail:", e)

# 对照原池
orig = json.load(open(f"{base}/outputs/eval_pool_bird/items.json"))
print("orig item keys:", list(orig[0].keys()))
print("orig sql sample:", str(orig[0]["candidates"][0].get("sql"))[:80])
