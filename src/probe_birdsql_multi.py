#!/usr/bin/env python3
"""探测 birdsql 旗下疑似带 sqlite 库的数据集结构。"""
import json
import sys
import urllib.request

for ds in ["birdsql/bird_mini_dev",
           "birdsql/livesqlbench-base-lite-sqlite",
           "birdsql/bird-critic-1.0-sqlite"]:
    print("====", ds)
    try:
        url = f"https://datasets-server.huggingface.co/info?dataset={ds}"
        with urllib.request.urlopen(url, timeout=40) as r:
            info = json.load(r)
        if "dataset_info" not in info:
            print("  RAW:", json.dumps(info)[:200])
            continue
        for cn, cfg in info["dataset_info"].items():
            for sp, s in cfg.get("splits", {}).items():
                print(f"  {cn}/{sp}: rows={s.get('num_examples')}, "
                      f"size={(s.get('num_bytes') or 0)/1e9:.2f} GB")
                cols = s.get("columns", [])
                if cols:
                    print("  cols:", [(c["name"], c["type"]) for c in cols][:16])
    except Exception as e:
        print("  ERR:", str(e)[:150])
