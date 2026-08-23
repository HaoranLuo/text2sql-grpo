#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pool_pass_oracle 对照：4 源池 vs 5 源池的天花板诊断（签名相等口径）。

口径与 src/bird_coevolve.py 相同：pool_pass_oracle = 池内存在与 gold 执行签名
一致的候选（sig == gold_sig 且 sig != ERROR_SIG）的题目占比。

数据来源（均为 bird_select.py prep 产物，无需重新执行）：
  - 4 源：outputs/bird_select_ormbird_bird_bal2/work/prep.json（sigs_per_entry）
  - 5 源：outputs/bird_select_5src/work/prep.json
  - gold 签名：优先复用 outputs/bird_select_ormbird_bird_bal2/work/gold_sigs.json
    （与 dev.json 顺序对齐的 1534 个 gold 执行签名，coevolve 同款计算）；若缺失
    或长度不符则用 AP.ExecutionEngine 现场重算。

输出：outputs/bird_select_5src/work/pool_oracle_4src_vs_5src.json +
per-question 明细（5 源新增 pass 但 4 源未 pass 的题）。

用法：envs/reasoning3b/bin/python src/bird_pool_oracle.py
"""
import json
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
for _p in (str(_PROJECT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adjudicate_pool as AP  # noqa: E402

PREP_4 = _PROJECT / "outputs" / "bird_select_ormbird_bird_bal2" / "work" / "prep.json"
PREP_5 = _PROJECT / "outputs" / "bird_select_5src" / "work" / "prep.json"
GOLD_SIGS_4 = _PROJECT / "outputs" / "bird_select_ormbird_bird_bal2" / "work" / "gold_sigs.json"
DB_ROOT = _PROJECT / "data" / "bird" / "bird_dev" / "dev_20240627" / "dev_databases"
OUT = _PROJECT / "outputs" / "bird_select_5src" / "work" / "pool_oracle_4src_vs_5src.json"


def _is_correct(sig, gold_sig):
    if sig is None or gold_sig is None:
        return False
    return sig == gold_sig and sig != AP.ERROR_SIG


def load_gold_sigs(n_questions):
    if GOLD_SIGS_4.exists():
        data = json.loads(GOLD_SIGS_4.read_text(encoding="utf-8"))
        sigs = data.get("sigs") or []
        if len(sigs) == n_questions:
            print(f"[oracle] 复用 gold_sigs.json ({len(sigs)} 条)")
            return sigs
        print(f"[oracle] gold_sigs.json 长度 {len(sigs)} != {n_questions}，重算")
    # 现场重算（coevolve _load_gold_sigs 同款）
    dev = json.loads((_PROJECT / "data" / "bird" / "bird_dev" / "dev_20240627"
                      / "dev.json").read_text(encoding="utf-8"))
    if n_questions < len(dev):
        dev = dev[:n_questions]
    engine = AP.ExecutionEngine(16, 30.0, 5_000_000, 100_000)
    tasks = []
    for d in dev:
        sql = (d.get("SQL") or "").strip()
        if not sql:
            tasks.append((None, ""))
            continue
        db_id = d["db_id"]
        inst = DB_ROOT / db_id / f"{db_id}.sqlite"
        tasks.append((sql, str(inst)))
    engine.run([t for t in tasks if t[0]], phase="gold")
    sigs = []
    for sql, inst in tasks:
        sigs.append(None if not sql else AP.outcome_signature(engine.get(sql, inst)))
    print(f"[oracle] gold sigs 重算完成 ({len(sigs)} 条)")
    return sigs


def pool_oracle(prep_file, gold_sigs):
    prep = json.loads(prep_file.read_text(encoding="utf-8"))
    items = prep["items"]
    assert len(items) == len(gold_sigs), \
        f"{prep_file}: items={len(items)} != gold_sigs={len(gold_sigs)}"
    n_pass = 0
    per_q = {}
    for qc, gold_sig in zip(items, gold_sigs):
        hit = any(_is_correct(sigs[0] if sigs else None, gold_sig)
                  for sigs in qc.get("sigs_per_entry") or [])
        per_q[qc["dataset_index"]] = bool(hit)
        n_pass += int(hit)
    return round(n_pass / len(items) * 100, 2), per_q, len(items)


def main():
    prep5 = json.loads(PREP_5.read_text(encoding="utf-8"))
    n_q = len(prep5["items"])
    gold_sigs = load_gold_sigs(n_q)

    acc4, per4, n4 = pool_oracle(PREP_4, gold_sigs)
    acc5, per5, n5 = pool_oracle(PREP_5, gold_sigs)
    assert n4 == n5 == n_q

    new_pass = [q for q in per5 if per5[q] and not per4[q]]
    lost_pass = [q for q in per5 if not per5[q] and per4[q]]

    out = {
        "pool_pass_oracle_4src": acc4,
        "pool_pass_oracle_5src": acc5,
        "delta_pp": round(acc5 - acc4, 2),
        "n_questions": n_q,
        "n_new_pass_5src": len(new_pass),
        "n_lost_pass_5src": len(lost_pass),
        "new_pass_qids": new_pass,
        "lost_pass_qids": lost_pass,
        "note": ("签名相等口径（与 bird_coevolve.py pool_pass_oracle 同款）：池内存在"
                 "与 gold 执行签名一致的候选的题目占比；仅诊断用途，官方 EX 以官方"
                 "评估器为准。"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"pool_pass_oracle: 4src={acc4}% -> 5src={acc5}% "
          f"(delta={out['delta_pp']:+.2f}pp) | new_pass={len(new_pass)} "
          f"lost_pass={len(lost_pass)}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
