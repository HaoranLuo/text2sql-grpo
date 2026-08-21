#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BIRD dev 下载（HuggingFace bird-bench/BIRD, config=text2sql）→ data/bird/。

产物：
  data/bird/dev.json                （1534 行，不含 sqlite bytes）
  data/bird/dev_databases/<db_id>/<db_id>.sqlite
防御性处理列名（SQL/query 两种 gold 列名）与 sqlite 列缺失情况。
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

OUT = "data/bird"
DBS = os.path.join(OUT, "dev_databases")
os.makedirs(DBS, exist_ok=True)

t0 = time.time()
from datasets import load_dataset  # noqa: E402

print("[dl-bird] loading bird-bench/BIRD (text2sql, dev) ...", flush=True)
ds = load_dataset("bird-bench/BIRD", "text2sql", split="dev")
print(f"[dl-bird] rows: {len(ds)} | cols: {ds.column_names}", flush=True)

cols = set(ds.column_names)
rows = []
n_db = 0
for i, row in enumerate(ds):
    row = dict(row)
    db_id = str(row["db_id"])
    # 提取 sqlite 数据库文件
    blob = row.pop("sqlite", None)
    if blob is not None:
        if isinstance(blob, str):
            blob = blob.encode("latin-1")
        db_dir = os.path.join(DBS, db_id)
        os.makedirs(db_dir, exist_ok=True)
        with open(os.path.join(db_dir, f"{db_id}.sqlite"), "wb") as f:
            f.write(blob)
        n_db += 1
    # 清理不可序列化字段
    for k in list(row.keys()):
        if isinstance(row[k], bytes):
            row.pop(k)
    rows.append(row)
    if (i + 1) % 200 == 0:
        print(f"[dl-bird] {i+1}/{len(ds)} ...", flush=True)

json.dump(rows, open(os.path.join(OUT, "dev.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"[dl-bird] done: {len(rows)} rows, {n_db} dbs, {time.time()-t0:.1f}s",
      flush=True)
