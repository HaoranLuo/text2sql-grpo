#!/usr/bin/env python3
"""探测 bird-critic-1.0-sqlite 结构：找含 sqlite 数据库文件的配置。"""
import json
import sys
import urllib.request

url = "https://datasets-server.huggingface.co/info?dataset=birdsql/bird-critic-1.0-sqlite"
with urllib.request.urlopen(url, timeout=30) as r:
    d = json.load(r)

if "dataset_info" not in d:
    print("RAW:", json.dumps(d)[:300])
    sys.exit(1)

for cn, cfg in d["dataset_info"].items():
    for sp, s in cfg.get("splits", {}).items():
        cols = s.get("columns", [])
        print(f"{cn}/{sp}: rows={s.get('num_examples')}, "
              f"size={ (s.get('num_bytes') or 0)/1e9:.2f} GB")
        print("  cols:", [(c["name"], c["type"]) for c in cols][:15])
