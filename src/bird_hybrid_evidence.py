#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""混合证据拼池（BIRD 冲刺第 1 招，零生成成本）：原池 × 证据池按题合并。

规则（按官方 dev.json 的 evidence 字段，口径与证据战役 agent 一致）：
  - 官方 evidence 非空 → 取原池（outputs/eval_pool_bird）该题的完整 item（candidates 原样）
  - 官方 evidence 为空/缺失 → 取证据池（outputs/eval_pool_bird_ev，规则版 auto evidence）该题的完整 item

输出 outputs/eval_pool_bird_hybrid/items.json（结构与原池/证据池完全一致，
dataset_index = question_id，题目顺序与 dev.json 一致）+ summary.json（拼池统计）。

本脚本只做 JSON 拼池，不执行任何 SQL、不改任何已有产物（原池/证据池只读）。

用法：
  envs/reasoning3b/bin/python src/bird_hybrid_evidence.py \
      --orig-pool outputs/eval_pool_bird/items.json \
      --ev-pool outputs/eval_pool_bird_ev/items.json \
      --dev-json data/bird/bird_dev/dev_20240627/dev.json \
      --out-dir outputs/eval_pool_bird_hybrid
"""
import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层不是 list")
    return data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orig-pool", default="outputs/eval_pool_bird/items.json")
    ap.add_argument("--ev-pool", default="outputs/eval_pool_bird_ev/items.json")
    ap.add_argument("--dev-json",
                    default="data/bird/bird_dev/dev_20240627/dev.json")
    ap.add_argument("--out-dir", default="outputs/eval_pool_bird_hybrid")
    args = ap.parse_args(argv)

    orig_path = Path(args.orig_pool)
    ev_path = Path(args.ev_pool)
    dev_path = Path(args.dev_json)
    out_dir = Path(args.out_dir)
    if not orig_path.is_absolute():
        orig_path = PROJECT / orig_path
    if not ev_path.is_absolute():
        ev_path = PROJECT / ev_path
    if not dev_path.is_absolute():
        dev_path = PROJECT / dev_path
    if not out_dir.is_absolute():
        out_dir = PROJECT / out_dir

    # ---- 官方 evidence 空/非空清单（以 dev.json 实际数据为准）----
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    if not isinstance(dev, list):
        raise ValueError(f"{dev_path} 顶层不是 list")
    evidence_empty: Dict[int, bool] = {}
    for x in dev:
        qid = int(x.get("question_id"))
        ev = x.get("evidence")
        evidence_empty[qid] = (ev is None
                               or (isinstance(ev, str) and not ev.strip()))
    n_dev = len(dev)
    n_nonempty = sum(1 for v in evidence_empty.values() if not v)
    n_empty = n_dev - n_nonempty
    print(f"[hybrid] dev.json: {n_dev} 题 | 官方 evidence 非空 {n_nonempty} "
          f"| 空 {n_empty}", file=sys.stderr)

    # ---- 两池加载 + 对齐校验 ----
    orig = _load(orig_path)
    ev = _load(ev_path)
    assert len(orig) == len(ev) == n_dev, \
        f"池长度不一致: orig={len(orig)} ev={len(ev)} dev={n_dev}"

    def item_identity(it: Dict[str, Any]):
        return (it.get("dataset_index", it.get("di")),
                it.get("db_id", ""), it.get("question", ""))

    assert [item_identity(x)[0] for x in orig] == list(range(n_dev)), \
        "原池 dataset_index 必须为 0..N-1 且与 dev.json 顺序一致"
    assert [item_identity(x)[0] for x in ev] == list(range(n_dev)), \
        "证据池 dataset_index 必须为 0..N-1 且与 dev.json 顺序一致"
    for i in range(n_dev):
        if item_identity(orig[i]) != item_identity(ev[i]):
            raise ValueError(
                f"两池第 {i} 题不对齐: orig={item_identity(orig[i])} "
                f"ev={item_identity(ev[i])}")
    print("[hybrid] 两池逐题对齐校验通过（dataset_index/db_id/question 全一致）",
          file=sys.stderr)

    # ---- 拼池 ----
    hybrid: List[Dict[str, Any]] = []
    n_from_orig = n_from_ev = 0
    cand_from_orig = cand_from_ev = 0
    for i in range(n_dev):
        qid = i
        take_ev = evidence_empty[qid]   # 官方 evidence 缺失 → 证据池
        src = ev[i] if take_ev else orig[i]
        hybrid.append(copy.deepcopy(src))
        nc = len(src.get("candidates") or [])
        if take_ev:
            n_from_ev += 1
            cand_from_ev += nc
        else:
            n_from_orig += 1
            cand_from_orig += nc

    # 结构一致性自检：hybrid item 的键集合必须与原池/证据池完全一致
    key_ok = all(set(x.keys()) == set(orig[0].keys()) for x in hybrid)
    assert key_ok, "hybrid item 键集合与原池不一致"
    assert [item_identity(x)[0] for x in hybrid] == list(range(n_dev)), \
        "hybrid dataset_index 顺序异常"

    # ---- 落盘 ----
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.json"
    with open(items_path, "w", encoding="utf-8") as fh:
        json.dump(hybrid, fh, ensure_ascii=False, indent=2)

    cand_total = sum(len(x.get("candidates") or []) for x in hybrid)
    summary = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": "src/bird_hybrid_evidence.py",
            "method": ("混合证据拼池（零生成成本）：官方 evidence 非空题取原池候选、"
                       "官方 evidence 缺失题取证据池候选；item 原样复制（deepcopy），"
                       "dataset_index = question_id，顺序与 dev.json 一致。"),
            "orig_pool": str(orig_path),
            "ev_pool": str(ev_path),
            "dev_json": str(dev_path),
            "n_questions": n_dev,
        },
        "evidence_split": {
            "nonempty_official_evidence": n_nonempty,
            "empty_official_evidence": n_empty,
        },
        "pool_usage": {
            "from_orig_pool": {"n_questions": n_from_orig,
                               "n_candidates": cand_from_orig},
            "from_ev_pool": {"n_questions": n_from_ev,
                             "n_candidates": cand_from_ev},
        },
        "candidates": {
            "total": cand_total,
            "avg_per_question": round(cand_total / n_dev, 2),
            "orig_pool_total": sum(len(x.get("candidates") or []) for x in orig),
            "ev_pool_total": sum(len(x.get("candidates") or []) for x in ev),
        },
        "empty_evidence_question_ids": sorted(
            qid for qid, v in evidence_empty.items() if v),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"[hybrid] DONE: {n_dev} 题 | 原池 {n_from_orig} 题 "
          f"({cand_from_orig} 候选) + 证据池 {n_from_ev} 题 "
          f"({cand_from_ev} 候选) = {cand_total} 候选", file=sys.stderr)
    print(f"[hybrid] items -> {items_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
