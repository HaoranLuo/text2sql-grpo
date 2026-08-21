#!/usr/bin/env python3
"""列出 ModelScope huybery/BIRD-bench 的文件树。"""
import json
import urllib.request

url = ("https://modelscope.cn/api/v1/datasets/huybery/BIRD-bench/"
       "repo/tree?Revision=master&Recursive=true")
with urllib.request.urlopen(url, timeout=40) as r:
    d = json.load(r)

files = d.get("Data", {}).get("Files", [])
print("total files:", len(files))
for f in files[:30]:
    print(f.get("Path"))
print("...")
db = [f.get("Path") for f in files
      if "database" in (f.get("Path") or "").lower()]
print("db-ish:", db[:12], "... total", len(db))
