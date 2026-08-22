#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spider dev 全量自动 evidence（SEED 式三件套，纯 CPU 离线，无 LLM）。

Spider 无官方 evidence 概念（SEED 仓库做法见 README run_make_evidence_spider_dev.sh：
拿 BIRD train few-shot 给 Spider 生成 evidence）。本脚本以确定性规则生成同款
「库结构摘要 + 样例执行 + 关键词映射」提示，供 B1 系 Spider 生成/评估管线把
evidence 并入 prompt（对齐 R3 建议）。

数据布局（与 scripts/check_dbs.py 的 HPC 布局一致）：
  data/spider_data/dev.json                          （1034 题）
  data/spider_data/tables.json                       （PK/FK/类型元数据）
  data/spider_data/database/<db_id>/<db_id>.sqlite   （官方 dev 20 库）

用法：
  python src/gen_spider_evidence.py
  python src/gen_spider_evidence.py --limit 20 --print-samples 3

产物 data/auto_evidence_spider.json：
  {"<dataset_index 0..1033>": "<evidence 文本>"}  # key 对齐管线 dataset_index

只读 sqlite（mode=ro），不写库、不联网、不占 GPU。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import gen_evidence_core as core  # noqa: E402

DEFAULT_DATA_JSON = str(_PROJECT / "data" / "spider_data" / "dev.json")
DEFAULT_TABLES_JSON = str(_PROJECT / "data" / "spider_data" / "tables.json")
DEFAULT_DB_ROOT = str(_PROJECT / "data" / "spider_data" / "database")
DEFAULT_OUT = str(_PROJECT / "data" / "auto_evidence_spider.json")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Spider dev 全量自动 evidence（SEED 式三件套，纯 CPU 离线）")
    p.add_argument("--data-json", default=DEFAULT_DATA_JSON)
    p.add_argument("--tables-json", default=DEFAULT_TABLES_JSON)
    p.add_argument("--db-root", default=DEFAULT_DB_ROOT)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--db-filter", default="",
                   help="逗号分隔 db_id 白名单（冒烟用，空=全量 dev 20 库）")
    p.add_argument("--limit", type=int, default=0,
                   help="最多处理 N 题（0=全量 1034，冒烟用）")
    p.add_argument("--print-samples", type=int, default=0,
                   help="打印前 N 条 evidence 文本（人工质检用）")
    return p.parse_args(argv)


def load_spider_meta(tables_json: str) -> Dict[str, Dict[str, Any]]:
    """tables.json → {db_id: 单库元数据}（列名/类型/PK/FK）。"""
    with open(tables_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {d["db_id"]: d for d in data}


def main() -> None:
    args = parse_args()
    db_filter = {s.strip() for s in args.db_filter.split(",") if s.strip()}
    with open(args.data_json, "r", encoding="utf-8") as fh:
        items = json.load(fh)
    meta = load_spider_meta(args.tables_json)
    if db_filter:
        items = [it for it in items if it["db_id"] in db_filter]
    if args.limit:
        items = items[: args.limit]
    print(f"Spider auto-evidence | {len(items)} questions | "
          f"db_filter={sorted(db_filter) or 'all'} | out={args.out}")

    bundle_cache: Dict[str, core.DbBundle] = {}
    evidences: Dict[str, str] = {}
    skipped: List[int] = []
    t0 = time.perf_counter()

    for i, it in enumerate(items):
        db_id = it["db_id"]
        if db_id not in bundle_cache:
            t_db = time.perf_counter()
            db_path = Path(args.db_root) / db_id / f"{db_id}.sqlite"
            if not db_path.is_file():
                print(f"[WARN] db missing: {db_path}")
                skipped.append(i)
                continue
            bundle = core.build_bundle_from_sqlite(
                str(db_path), tables_meta=meta.get(db_id))
            bundle_cache[db_id] = bundle
            print(f"[db:{db_id}] bundle built in {time.perf_counter() - t_db:.1f}s "
                  f"({len(bundle.tables)} tables, "
                  f"{len(bundle.value_exact)} distinct value keys)")
        bundle = bundle_cache[db_id]
        evidences[str(i)] = core.build_evidence(it["question"], bundle)
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(items)} done "
                  f"({time.perf_counter() - t0:.0f}s elapsed)")

    for i in skipped:
        evidences.setdefault(str(i), "")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(evidences, fh, ensure_ascii=False, indent=1)

    stats = core.evidence_stats(evidences)
    print("\n" + "=" * 64)
    print("  SPIDER AUTO-EVIDENCE SUMMARY")
    print("=" * 64)
    print(f"  questions: {stats['n']} | empty: {stats['empty']} | "
          f"skipped_db_missing: {len(skipped)}")
    print(f"  chars min/med/max: {stats['chars_min']}/{stats['chars_med']}/"
          f"{stats['chars_max']}")
    print(f"  total wall: {time.perf_counter() - t0:.0f}s")
    print(f"  saved: {args.out}")

    if args.print_samples:
        print("\n" + "-" * 64)
        shown = 0
        for idx, ev in evidences.items():
            if shown >= args.print_samples:
                break
            if not ev:
                continue
            it = items[int(idx)]
            print(f"\n### idx={idx} db={it['db_id']}")
            print(f"Q: {it['question']}")
            print(f"gold: {it['query']}")
            print(f"AUTO:\n{ev}")
            shown += 1


if __name__ == "__main__":
    main()
