#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拼 5 源 BIRD 候选池：主池 4 源（outputs/eval_pool_bird）原样在前 + RFT
（outputs/eval_pool_bird_rft）追加 → outputs/eval_pool_bird_5src/items.json。

规则：
  - 主池 1534 题的 64 候选/题按字节级原样保留（不重排、不改字段）；
  - RFT 16 候选/题（model="rft_bird_v3"）追加，按 (model, sample_idx) 排序后
    落在 4 源之后（"rft_bird_v3" > "sft_v3" 字典序）；
  - 两池 dataset_index 顺序必须一致（都按 question_id 升序 0..1533）；
  - 每题的 RFT 候选数必须 = 16（或该题在 RFT 池中有 error 字段 → 记 0 并告警）；
  - 输出 summary.json 记录合并口径与校验结果。

用法：envs/reasoning3b/bin/python src/merge_pool_rft.py
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_MAIN = _PROJECT / "outputs" / "eval_pool_bird" / "items.json"
DEFAULT_RFT = _PROJECT / "outputs" / "eval_pool_bird_rft" / "items.json"
DEFAULT_OUT = _PROJECT / "outputs" / "eval_pool_bird_5src"

MAIN_MODELS = ["p2a_500", "sft_phase1", "sft_v2", "sft_v3"]
RFT_MODEL = "rft_bird_v3"
N_PER_MODEL = 16


def main() -> None:
    ap = argparse.ArgumentParser(description="合并主池 4 源 + RFT 源 → 5 源池")
    ap.add_argument("--main-items", type=Path, default=DEFAULT_MAIN)
    ap.add_argument("--rft-items", type=Path, default=DEFAULT_RFT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=N_PER_MODEL)
    args = ap.parse_args()

    main_items = json.loads(args.main_items.read_text(encoding="utf-8"))
    rft_items = json.loads(args.rft_items.read_text(encoding="utf-8"))

    # ---- 顺序与 id 对齐校验 ----
    main_ids = [int(it["dataset_index"]) for it in main_items]
    rft_ids = [int(it["dataset_index"]) for it in rft_items]
    if main_ids != list(range(len(main_ids))):
        raise SystemExit(f"主池 dataset_index 非 0..N-1 升序（首尾 "
                         f"{main_ids[:3]}...{main_ids[-3:]}）")
    if rft_ids != main_ids:
        raise SystemExit(f"RFT 池与主池题目顺序/集合不一致 "
                         f"（len main={len(main_ids)}, len rft={len(rft_ids)}）")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = []
    n_main_cands = 0
    n_rft_cands = 0
    n_rft_error_items = 0
    n_rft_incomplete = 0
    for m_it, r_it in zip(main_items, rft_items):
        # ---- 主池原样保留 ----
        main_cands = list(m_it.get("candidates") or [])
        main_by_key = {(c.get("model"), c.get("sample_idx")): c for c in main_cands}
        if len(main_cands) != len(MAIN_MODELS) * args.n:
            raise SystemExit(f"item {m_it['dataset_index']}: 主池候选数 "
                             f"{len(main_cands)} != {len(MAIN_MODELS) * args.n}")
        for m in MAIN_MODELS:
            for j in range(args.n):
                c = main_by_key.get((m, j))
                if c is None or c.get("model") != m or c.get("sample_idx") != j:
                    raise SystemExit(f"item {m_it['dataset_index']}: 主池缺 "
                                     f"({m},{j}) 候选")
        n_main_cands += len(main_cands)

        # ---- RFT 追加 ----
        rft_cands = [c for c in (r_it.get("candidates") or [])
                     if c.get("model") == RFT_MODEL]
        rft_cands = sorted(rft_cands, key=lambda c: c.get("sample_idx", -1))
        if r_it.get("error"):
            n_rft_error_items += 1
        if len(rft_cands) < args.n:
            n_rft_incomplete += 1
            print(f"[WARN] item {m_it['dataset_index']}: RFT 候选 "
                  f"{len(rft_cands)}/{args.n}（error={r_it.get('error')}）")
        n_rft_cands += len(rft_cands)

        item = dict(m_it)  # 保留主池所有字段
        item["candidates"] = main_cands + rft_cands
        item["candidates"].sort(key=lambda c: (c.get("model", ""), c.get("sample_idx", -1)))
        merged.append(item)

    expected_total = len(main_ids) * (len(MAIN_MODELS) + 1) * args.n
    actual_total = n_main_cands + n_rft_cands
    out_items = out_dir / "items.json"
    out_items.write_text(json.dumps(merged, ensure_ascii=False, indent=1),
                         encoding="utf-8")

    summary = {
        "note": (
            "5 源 BIRD 候选池：主池 4 源（eval_pool_bird，字节级原样）在前 + "
            "rft_bird_v3 16 采样/题追加在后；每题目标 80 候选。裁决 = "
            "bird_select.py 原管线（prep → ORM 打分 → final 官方 EX）。"
        ),
        "n_questions": len(merged),
        "main_items_file": str(args.main_items),
        "rft_items_file": str(args.rft_items),
        "n_main_candidates": n_main_cands,
        "n_rft_candidates": n_rft_cands,
        "expected_total": expected_total,
        "actual_total": actual_total,
        "n_rft_error_items": n_rft_error_items,
        "n_rft_incomplete_items": n_rft_incomplete,
        "merged_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"merged {len(merged)} 题: main={n_main_cands} + rft={n_rft_cands} "
          f"= {actual_total} 候选 (期望 {expected_total})")
    print(f"rft error items={n_rft_error_items}, incomplete={n_rft_incomplete}")
    print(f"-> {out_items}")
    print(f"-> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    sys.exit(main())
