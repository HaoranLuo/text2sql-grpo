#!/usr/bin/env python3
"""取 bird-critic-1.0-sqlite 前 2 行看字段（找 sqlite 库文件列）。"""
import json
import urllib.request

url = ("https://datasets-server.huggingface.co/rows?dataset="
       "birdsql/bird-critic-1.0-sqlite&config=default&split=train"
       "&offset=0&length=2")
with urllib.request.urlopen(url, timeout=60) as r:
    d = json.load(r)

for row in d.get("rows", []):
    r = row["row"]
    print("keys:", list(r.keys()))
    for k, v in r.items():
        if isinstance(v, (bytes, str)) and len(str(v)) > 200:
            print(f"  {k}: <long len={len(str(v))} head={str(v)[:80]}>")
        else:
            print(f"  {k}: {str(v)[:120]}")
    print("---")
