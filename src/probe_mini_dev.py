#!/usr/bin/env python3
"""取 mini_dev_sqlite 前 2 行看字段。"""
import json
import urllib.request

url = ("https://datasets-server.huggingface.co/rows?dataset="
       "birdsql/bird_mini_dev&config=default&split=mini_dev_sqlite"
       "&offset=0&length=2")
with urllib.request.urlopen(url, timeout=60) as r:
    d = json.load(r)

for row in d.get("rows", []):
    r = row["row"]
    print("keys:", list(r.keys()))
    for k, v in r.items():
        if isinstance(v, (bytes, str)) and len(str(v)) > 300:
            print(f"  {k}: <long len={len(str(v))}>")
        else:
            print(f"  {k}: {str(v)[:150]}")
    print("---")
