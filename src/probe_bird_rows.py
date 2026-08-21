#!/usr/bin/env python3
"""取 birdsql/bird_sql_dev_20251106 的前 2 行，看字段结构。"""
import json
import sys
import urllib.request

url = ("https://datasets-server.huggingface.co/rows?dataset="
       "birdsql/bird_sql_dev_20251106&config=default&split=dev_20251106"
       "&offset=0&length=2")
with urllib.request.urlopen(url, timeout=60) as r:
    d = json.load(r)

for row in d.get("rows", []):
    r = row["row"]
    print("keys:", list(r.keys()))
    for k, v in r.items():
        if k in ("sqlite", "db", "database", "db_bytes"):
            print(f"  {k}: <bytes len={len(v) if isinstance(v, (bytes, str)) else v}>")
        else:
            s = str(v)
            print(f"  {k}: {s[:120]}")
    print("---")
    break
