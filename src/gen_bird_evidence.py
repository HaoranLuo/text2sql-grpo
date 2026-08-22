#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BIRD dev 全量自动 evidence 生成（SEED 式三件套，纯 CPU 离线，无 LLM）。

对应 tmp_idea_research/bird_gen_scan_report.md §8 R3：SEED 用 3 次 LLM 调用
生成 per-question evidence；本脚本以确定性规则复刻其三件套（见
gen_evidence_core.py 模块 docstring），对 BIRD dev 11 库离线生成，输出
question_id → evidence 文本。

数据布局（HPC/本地同构，与 src/gen_bird_pool.py 的 DEFAULT_DATA_JSON /
DEFAULT_DB_ROOT 完全一致）：
  data/bird/bird_dev/dev_20240627/dev.json            （1534 题）
  data/bird/bird_dev/dev_20240627/dev_databases/<db_id>/<db_id>.sqlite
  .../dev_databases/<db_id>/database_description/*.csv（列注释，可选）

用法：
  python src/gen_bird_evidence.py                       # 全量
  python src/gen_bird_evidence.py --db-filter superhero,student_club,toxicology
  python src/gen_bird_evidence.py --limit 10 --out data/bird/auto_evidence_smoke.json

产物 data/bird/auto_evidence.json：
  {"<question_id>": "<evidence 文本>", ...}  # question_id 为 dev.json 原值 str()

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

DEFAULT_DATA_JSON = str(
    _PROJECT / "data" / "bird" / "bird_dev" / "dev_20240627" / "dev.json")
DEFAULT_DB_ROOT = str(
    _PROJECT / "data" / "bird" / "bird_dev" / "dev_20240627" / "dev_databases")
DEFAULT_OUT = str(_PROJECT / "data" / "bird" / "auto_evidence.json")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BIRD dev 全量自动 evidence（SEED 式三件套，纯 CPU 离线）")
    p.add_argument("--data-json", default=DEFAULT_DATA_JSON)
    p.add_argument("--db-root", default=DEFAULT_DB_ROOT)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--db-filter", default="",
                   help="逗号分隔 db_id 白名单（冒烟用，空=全量 11 库）")
    p.add_argument("--limit", type=int, default=0,
                   help="最多处理 N 题（0=全量，冒烟用）")
    p.add_argument("--print-samples", type=int, default=0,
                   help="打印前 N 条 evidence 文本（人工质检用）")
    return p.parse_args(argv)


def load_items(data_json: str) -> List[Dict[str, Any]]:
    with open(data_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"dev.json 结构异常（期望 list）: {data_json}")
    return data


def main() -> None:
    args = parse_args()
    db_filter = {s.strip() for s in args.db_filter.split(",") if s.strip()}
    items = load_items(args.data_json)
    if db_filter:
        items = [it for it in items if it["db_id"] in db_filter]
    if args.limit:
        items = items[: args.limit]
    print(f"BIRD auto-evidence | {len(items)} questions | "
          f"db_filter={sorted(db_filter) or 'all'} | out={args.out}")

    bundle_cache: Dict[str, core.DbBundle] = {}
    evidences: Dict[str, str] = {}
    skipped: List[str] = []
    t0 = time.perf_counter()

    for i, it in enumerate(items):
        db_id = it["db_id"]
        qid = str(it["question_id"])
        if db_id not in bundle_cache:
            t_db = time.perf_counter()
            db_path = Path(args.db_root) / db_id / f"{db_id}.sqlite"
            if not db_path.is_file():
                print(f"[WARN] db missing: {db_path}")
                skipped.append(qid)
                continue
            desc_dir = Path(args.db_root) / db_id / "database_description"
            comments = core.read_bird_comments(str(desc_dir), str(db_path))
            bundle = core.build_bundle_from_sqlite(str(db_path), comments=comments)
            bundle_cache[db_id] = bundle
            print(f"[db:{db_id}] bundle built in {time.perf_counter() - t_db:.1f}s "
                  f"({len(bundle.tables)} tables, "
                  f"{len(bundle.value_exact)} distinct value keys)")
        bundle = bundle_cache[db_id]
        ev = core.build_evidence(it["question"], bundle)
        evidences[qid] = ev
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(items)} done "
                  f"({time.perf_counter() - t0:.0f}s elapsed)")

    # 缺失库的题：evidence 留空占位（不吞题）
    for qid in skipped:
        evidences.setdefault(qid, "")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(evidences, fh, ensure_ascii=False, indent=1)

    stats = core.evidence_stats(evidences)
    print("\n" + "=" * 64)
    print("  BIRD AUTO-EVIDENCE SUMMARY")
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
        for qid, ev in evidences.items():
            if not ev or shown >= args.print_samples:
                if not ev:
                    continue
                if shown >= args.print_samples:
                    break
            it = next((x for x in items if str(x["question_id"]) == qid), None)
            print(f"\n### qid={qid} db={it['db_id'] if it else '?'}")
            print(f"Q: {it['question'] if it else '?'}")
            print(f"official evidence: {(it.get('evidence') or '') if it else '?'}")
            print(f"AUTO:\n{ev}")
            shown += 1


if __name__ == "__main__":
    main()
