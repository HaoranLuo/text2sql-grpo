#!/usr/bin/env python3
"""P1#4 数据物化: GRPO manifest(规范索引) → 训练 JSON (新文件, 不改任何现有文件).

背景 (见 data_hygiene/split_record.md):
  规范索引 0..6999 = data/spider_data/train_spider.json; 7000..8658 = train_others.json。
  GRPO 5500 / SFT-gold 1500 / 未分配池 1659 三集不相交, 并集 = 8659 (seed=42 确定性划分)。
  src/train_reasoning_grpo.build_dataset() 只接受 JSON list ({question, query, db_id}),
  本脚本按 manifest 的规范索引从 HPC 原始数据物化出该子集, 并保证:
  1) sha256 校验原始文件 == split_record.md 记录值 (数据卫生 G3, 不一致立即退出);
  2) 每条 manifest 的 (db_id, question) 与原始数据逐条核对;
  3) 与 --sft-manifest 断言交集为空 (构造性不相交的运行时复验);
  4) 幂等: 输出已存在且与 manifest 一致则跳过; --check-only 只校验不写(准备阶段用)。
用法 (HPC, 只写新文件 data/grpo_train_5500.json + .sha256):
  python scripts/make_grpo_train_json.py \
      --manifests data_hygiene/grpo_5500_manifest.jsonl \
      [--manifests data_hygiene/unassigned_pool_1659_manifest.jsonl] \
      --sft-manifest data_hygiene/sft_gold_1500_manifest.jsonl \
      --spider-dir data/spider_data --out data/grpo_train_5500.json
注: 追加 unassigned 1659 池 = "更大"选项(7159 条, +6 个新库 academic/geo/imdb/
    restaurants/scholar/yelp); 该池在卫生划分中被显式保留为"未分配", 吸收它需先做
    重新分配决策, 故默认只用 grpo_5500。
"""
import argparse
import hashlib
import json
import sys

# 与 data_hygiene/split_record.md §4 一致 (2026-08-13 记录, 勿改;
# 原始数据若被替换必须先重跑卫生划分并更新记录)
EXPECTED_SHA256 = {
    "train_spider.json": "c43d0d72e59e1a9e1a60837da9bf70d5a6277226bdb7f634d544f380646f527a",
    "train_others.json": "7adb04af470b3c9be653504e03c9a36c1b963a861f308ecf25d436472284e10f",
}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True,
                    help="GRPO manifest jsonl (规范索引), 可传多个做并集")
    ap.add_argument("--sft-manifest", default=None,
                    help="SFT-gold manifest, 用于不相交断言")
    ap.add_argument("--spider-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-only", action="store_true",
                    help="只校验不写文件 (提交前准备阶段)")
    args = ap.parse_args()

    spider = args.spider_dir
    # 1) sha256 校验原始数据 (G3)
    for fname, want in EXPECTED_SHA256.items():
        p = f"{spider}/{fname}"
        got = file_sha256(p)
        if got != want:
            sys.exit(f"[FAIL] sha256 不匹配 {p}\n  got  {got}\n  want {want}\n"
                     f"  与 split_record.md 记录不一致, 停止!")
        print(f"[sha256 OK] {fname}")

    ts = json.load(open(f"{spider}/train_spider.json", encoding="utf-8"))
    to = json.load(open(f"{spider}/train_others.json", encoding="utf-8"))
    print(f"[data] train_spider={len(ts)} train_others={len(to)} total={len(ts) + len(to)}")

    def resolve(idx):
        if 0 <= idx < len(ts):
            return ts[idx]
        if len(ts) <= idx < len(ts) + len(to):
            return to[idx - len(ts)]
        sys.exit(f"[FAIL] 规范索引越界: {idx}")

    rows, seen = [], set()
    for mp in args.manifests:
        n_skip = 0
        for e in load_jsonl(mp):
            idx = e["idx"]
            if idx in seen:
                n_skip += 1
                continue
            seen.add(idx)
            src = resolve(idx)
            if src["db_id"] != e["db_id"] or src["question"].strip() != e["question"].strip():
                sys.exit(f"[FAIL] manifest 与实际数据不一致 idx={idx}: "
                         f"manifest=({e['db_id']}, {e['question'][:40]!r}) "
                         f"data=({src['db_id']}, {src['question'][:40]!r})")
            rows.append({"idx": idx, "db_id": src["db_id"],
                         "question": src["question"], "query": src["query"]})
        if n_skip:
            print(f"[info] {mp}: 与其它 manifest 去重跳过 {n_skip} 条")
    print(f"[manifest] 合计 {len(rows)} 条, 覆盖 {len(set(r['db_id'] for r in rows))} 个 db")

    if args.sft_manifest:
        sft_idx = {e["idx"] for e in load_jsonl(args.sft_manifest)}
        overlap = sft_idx & seen
        if overlap:
            sys.exit(f"[FAIL] 与 SFT-gold 重叠 {len(overlap)} 条: {sorted(overlap)[:20]}")
        print(f"[disjoint OK] 与 SFT-gold ({len(sft_idx)} 条) 交集为空")

    if args.check_only:
        print("[check-only] 校验通过, 未写文件")
        return

    try:
        existing = json.load(open(args.out, encoding="utf-8"))
        if (len(existing) == len(rows)
                and existing[0]["idx"] == rows[0]["idx"]
                and existing[-1]["idx"] == rows[-1]["idx"]):
            print(f"[skip] {args.out} 已存在且与 manifest 一致 ({len(existing)} 条), 不重写")
            return
        print(f"[rewrite] {args.out} 已存在但不一致, 覆盖重写")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    h = file_sha256(args.out)
    with open(args.out + ".sha256", "w", encoding="utf-8") as f:
        f.write(f"{h}  {args.out}\n")
    print(f"[written] {args.out} ({len(rows)} 条) sha256={h}")


if __name__ == "__main__":
    main()
