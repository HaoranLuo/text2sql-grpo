#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐源消融（发文支持实验）：单源/双源/三源/四源 × {MI-VAV, ORM grouphead}。

- 执行结果：每题的每个唯一候选 SQL 只执行一次（本进程内缓存），所有子集复用
- ORM 分数：读 v2 打分缓存（键 = "<题序>\\t<归一化SQL>"，缺分按 0 计并统计）
- 子集按模型标签的首次出现顺序定义：single×4、(0,1)、(2,3)、(0,1,2)、all
- 组级分 = size × P(Yes)（grouphead 同款）；MI-VAV = arm_baseline
输出 outputs/ablate_sources/items_<subset>_<arm>.json（eval_official.sh 兼容）+ summary.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

import adjudicate_pool as AP  # noqa: E402
import adjudicate_soft as AS  # noqa: E402


def size_x_score_key(kg: Tuple[Any, Dict[str, Any]], entries, s_of):
    g = kg[1]
    rep = AS._group_rep(entries, g)
    ei = next(i for i, e in enumerate(entries) if e["key"] == rep["key"])
    s = s_of(ei)
    return (g["size"] * (s if s is not None else 0.0), g["size"], str(kg[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="outputs/eval_pool_multi/items.json")
    ap.add_argument("--scores",
                    default="outputs/orm_selection_v2/scores/scores_vllm.json")
    ap.add_argument("--out-dir", default="outputs/ablate_sources")
    ap.add_argument("--spider-dir", default="data/spider_data")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--query-timeout", type=float, default=30.0)
    ap.add_argument("--max-instances", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = AP._load_items(Path(args.items))
    if args.limit:
        items = items[:args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ORM 分数（v2 缓存）
    raw_scores = json.load(open(args.scores, encoding="utf-8"))
    score_map = {}
    for k, v in raw_scores.items():
        parts = k.split("\t", 1)
        if len(parts) == 2:
            score_map[(int(parts[0]), parts[1])] = float(v)

    # 每题去重 + 模型标签发现（按首次出现顺序）
    entries_by_q = [AP._dedupe(item.get("candidates") or []) for item in items]
    tag_order: List[str] = []
    for q in entries_by_q:
        for e in q:
            for m in sorted(e.get("models") or []):
                if m not in tag_order:
                    tag_order.append(m)
    print(f"[ablate] 题数 {len(items)} | 模型标签顺序: {tag_order}",
          file=sys.stderr)
    assert len(tag_order) >= 2, "模型标签不足 2 个，无法消融"

    t = tag_order
    sft_tags = sorted([x for x in t if not x.lower().startswith("p2a")])
    alien_tags = sorted([x for x in t if x.lower().startswith("p2a")])
    subsets: List[Tuple[str, set]] = [(f"single_{x}", {x}) for x in t]
    if len(sft_tags) >= 2:
        subsets.append((f"pair_sameline_{sft_tags[0]}_{sft_tags[1]}",
                        {sft_tags[0], sft_tags[1]}))
    if sft_tags and alien_tags:
        subsets.append((f"pair_unseen_{sft_tags[-1]}_{alien_tags[0]}",
                        {sft_tags[-1], alien_tags[0]}))
    if len(sft_tags) >= 3:
        subsets.append((f"triple_sft_{'+'.join(sft_tags)}", set(sft_tags)))
    subsets.append(("all", set(t)))

    # 执行引擎 + 每题实例 + 全量签名（一次执行）
    database_dir = Path(args.spider_dir) / "database"
    engine = AP.ExecutionEngine(args.threads, args.query_timeout,
                                5_000_000, 100_000)
    db_cache: Dict[str, List[str]] = {}

    def instances_for(db_id):
        if db_id not in db_cache:
            db_cache[db_id] = AP.list_instances(str(database_dir / db_id), db_id,
                                                args.max_instances)
        return db_cache[db_id]

    per_q = []
    for qi, item in enumerate(items):
        entries = entries_by_q[qi]
        insts = instances_for(item.get("db_id", ""))
        per_q.append({"item": item, "entries": entries, "insts": insts})

    # 收集全部 (sql, db_path) 任务 → 批量执行（与 v3 phase1 同口径）
    phase1_tasks = []
    for qi, q in enumerate(per_q):
        for e in q["entries"]:
            text = (e["sql_text"] or "").strip()
            if not text:
                continue
            for inst in q["insts"]:
                phase1_tasks.append((text, inst))
    phase1_tasks = list(set(phase1_tasks))
    print(f"[ablate] phase1: {len(phase1_tasks)} 唯一任务，开始执行 ...",
          file=sys.stderr)
    engine.run(phase1_tasks, phase="grouping")

    for qi, q in enumerate(per_q):
        entries = q["entries"]
        insts = q["insts"]
        sigs = []
        for e in entries:
            text = (e["sql_text"] or "").strip()
            if not text:
                sigs.append([AP.ERROR_SIG] * len(insts))
            else:
                sigs.append([AP.outcome_signature(engine.get(text, inst))
                             for inst in insts])
        votes_full: Dict[int, int] = Counter()
        for c in q["item"].get("candidates") or []:
            ck = AP.normalize_for_dedup(c.get("sql"))
            for ei, e in enumerate(entries):
                if ck == e["key"]:
                    votes_full[ei] += 1
                    break
        q["sigs"] = sigs
        q["votes_full"] = votes_full

    n_missing_score = 0
    summary: Dict[str, Any] = {"subsets": {}, "tag_order": tag_order}
    for subset_name, subset_tags in subsets:
        print(f"[ablate] subset {subset_name} ...", file=sys.stderr)
        for arm in ("vav", "orm"):
            out_items = []
            for qi, q in enumerate(per_q):
                entries = q["entries"]
                subset_eis = [ei for ei, e in enumerate(entries)
                              if e.get("models") and e["models"] & subset_tags]
                sub_entries = [entries[ei] for ei in subset_eis]
                sub_sigs = [q["sigs"][ei] for ei in subset_eis]
                sub_votes = {new_i: q["votes_full"][ei]
                             for new_i, ei in enumerate(subset_eis)}
                insts = q["insts"]
                joins_cache: Dict[str, Tuple[int, str]] = {}
                rec = None
                if arm == "vav":
                    if sub_entries:
                        rec = AS.arm_baseline(sub_entries, sub_sigs, sub_votes,
                                              insts, joins_cache)
                else:
                    groups, grouped, excluded = AS.build_groups(
                        sub_entries, sub_sigs, sub_votes, len(insts))
                    ranked = AS.rankable_groups(groups)
                    if ranked:
                        di = q["item"].get("dataset_index", qi)

                        def s_of(ei2, _di=di):
                            e = sub_entries[ei2]
                            v = score_map.get((qi, e["key"]))
                            if v is None:
                                v = score_map.get((_di, e["key"]))
                            return v

                        chosen_key, chosen_g = max(
                            ranked, key=lambda kg: size_x_score_key(
                                kg, sub_entries, s_of))
                        rec = AS._base_record(sub_entries, chosen_key, chosen_g,
                                              f"orm_{subset_name}", len(insts),
                                              grouped, excluded, joins_cache)
                        rep = AS._group_rep(sub_entries, chosen_g)
                        ei = next(i for i, e in enumerate(sub_entries)
                                  if e["key"] == rep["key"])
                        v = s_of(ei)
                        if v is None:
                            n_missing_score += 1
                        rec["orm_score"] = v
                if rec is None:
                    rec = AS._fallback_record(sub_entries, sub_votes,
                                              len(insts), 0, len(sub_entries))
                item = q["item"]
                out_items.append({
                    "dataset_index": item.get("dataset_index", qi),
                    "di": item.get("dataset_index", qi),
                    "db_id": item.get("db_id", ""),
                    "question": item.get("question", ""),
                    "gold_sql": item.get("gold_sql") or "",
                    "predicted_sql": rec.get("text", ""),
                    "empty_winner": rec.get("empty_winner", False),
                    "winner_source": rec["source"],
                })
            fname = out_dir / f"items_{subset_name}_{arm}.json"
            fname.write_text(json.dumps(out_items, ensure_ascii=False,
                                        indent=1), encoding="utf-8")
            summary["subsets"].setdefault(subset_name, {})[arm] = str(fname)
    summary["n_missing_orm_scores"] = n_missing_score
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[ablate] done | 缺 ORM 分数候选数: {n_missing_score}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
