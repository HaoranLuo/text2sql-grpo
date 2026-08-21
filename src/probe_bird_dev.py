#!/usr/bin/env python3
"""探测 birdsql/bird_sql_dev_20251106 的结构（configs/splits/columns/size）。"""
import json
import sys
import urllib.request

url = "https://datasets-server.huggingface.co/info?dataset=birdsql/bird_sql_dev_20251106"
with urllib.request.urlopen(url, timeout=30) as r:
    d = json.load(r)

if "dataset_info" not in d:
    print("RAW:", json.dumps(d)[:300])
    sys.exit(1)

info = d["dataset_info"]
print("configs:", list(info.keys()))
for cname, cfg in info.items():
    for split, s in cfg.get("splits", {}).items():
        print(f"{cname}/{split}: {s['num_examples']} rows, "
              f"size {s['num_bytes']/1e9:.2f} GB")
        cols = s["columns"]
        print("cols:", [(x["name"], x["type"]) for x in cols][:14])
