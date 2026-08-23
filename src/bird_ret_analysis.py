#!/usr/bin/env python3
"""BIRD 检索器裁剪全量跑分析（flat8 ret 池 vs 60.37 基线主池）。

对比口径：官方 test-suite 逐题结果 eval_result_dev.json（res 0/1），
arm = arm_orm_grouphead（适配判卷组代表）与 arm_vav（对照）。
输出 JSON 供 tmp_idea_research/retriever_fullrun_report.md 引用。

用法（HPC 登录节点，系统 python3 即可）：
  python3 src/bird_ret_analysis.py \
      --baseline-dir outputs/bird_select_ormbird_bird_bal2 \
      --ret-dir outputs/bird_select_bird_ret \
      --manifest outputs/schema_retriever_infer/bird_dev_topk_k8.json \
      --out outputs/bird_select_bird_ret/ret_analysis.json
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path


def load_rows(path: str) -> dict:
    rows = json.load(open(path, encoding="utf-8"))
    out = {}
    for r in rows:
        out[int(r["question_id"])] = bool(r.get("res"))
    return out


def mcnemar(a: dict, b: dict, qids) -> dict:
    qids = [q for q in qids if q in a and q in b]
    fixed = sum(1 for q in qids if (not a[q]) and b[q])
    broken = sum(1 for q in qids if a[q] and (not b[q]))
    n = fixed + broken
    if n == 0:
        return {"n": 0, "fixed": 0, "broken": 0, "p": 1.0, "n_checked": len(qids)}
    p_upper = sum(math.comb(n, k) * 0.5 ** n for k in range(0, fixed + 1))
    p_two = 2 * min(p_upper, 1 - p_upper)
    return {"n": n, "fixed": fixed, "broken": broken,
            "p": round(p_two, 6), "n_checked": len(qids)}


def acc(a: dict, qids) -> float:
    n = len(qids)
    return round(sum(a[q] for q in qids) / n * 100, 2) if n else 0.0


def subset_stats(a: dict, b: dict, qids, difficulty_map=None) -> dict:
    qids = [q for q in qids if q in a and q in b]
    st = {"n": len(qids), "acc_base": acc(a, qids), "acc_ret": acc(b, qids)}
    st["mcnemar"] = mcnemar(a, b, qids)
    if difficulty_map:
        st["by_difficulty"] = {}
        for d in ("simple", "moderate", "challenging"):
            sub = [q for q in qids if difficulty_map.get(q) == d]
            if sub:
                st["by_difficulty"][d] = {
                    "n": len(sub), "acc_base": acc(a, sub), "acc_ret": acc(b, sub)}
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--ret-dir", required=True)
    ap.add_argument("--manifest", required=True,
                    help="top-k 清单 JSON（bird_dev_topk_k8.json）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    qmeta = manifest["questions"]
    gold_missing_qids = [int(q) for q, e in qmeta.items() if e.get("gold_missing")]
    pruned_qids = [int(q) for q, e in qmeta.items() if not e.get("is_full")]
    full_qids = [int(q) for q, e in qmeta.items() if e.get("is_full")]
    all_qids = sorted(int(q) for q in qmeta)

    out = {
        "manifest": manifest.get("meta"),
        "n_gold_missing": len(gold_missing_qids),
        "n_pruned": len(pruned_qids),
        "n_full": len(full_qids),
        "arms": {},
    }

    base_dir = Path(args.baseline_dir)
    ret_dir = Path(args.ret_dir)

    # 难度表（从基线结果行取，两池一致）
    diffs = {}
    for r in json.load(open(base_dir / "arm_orm_grouphead" / "eval_result_dev.json",
                            encoding="utf-8")):
        diffs[int(r["question_id"])] = r.get("difficulty", "")

    for arm in ("arm_orm_grouphead", "arm_vav"):
        a = load_rows(base_dir / arm / "eval_result_dev.json")
        b = load_rows(ret_dir / arm / "eval_result_dev.json")
        entry = {
            "overall": subset_stats(a, b, all_qids, diffs),
            "gold_missing": subset_stats(a, b, gold_missing_qids, diffs),
            "pruned": subset_stats(a, b, pruned_qids, diffs),
            "is_full": subset_stats(a, b, full_qids, diffs),
            "gold_missing_by_db": {},
        }
        by_db = Counter(qmeta[str(q)]["db_id"] for q in gold_missing_qids)
        for db in sorted(by_db):
            qids_db = [q for q in gold_missing_qids if qmeta[str(q)]["db_id"] == db]
            entry["gold_missing_by_db"][db] = subset_stats(a, b, qids_db)
        # 逐题明细（报告附录用）：仅 gold_missing 题
        entry["gold_missing_rows"] = [
            {"question_id": q, "db_id": qmeta[str(q)]["db_id"],
             "difficulty": diffs.get(q),
             "dropped_tables": qmeta[str(q)]["gold_missing"],
             "base_res": int(a.get(q, 0)), "ret_res": int(b.get(q, 0))}
            for q in sorted(gold_missing_qids)]
        out["arms"][arm] = entry

    # pred 一致性 sanity：is_full 题两池 pred 是否逐字符一致（同 seed 应一致）
    a_rows = {int(r["question_id"]): r.get("pred")
              for r in json.load(open(
                  base_dir / "arm_orm_grouphead" / "eval_result_dev.json",
                  encoding="utf-8"))}
    b_rows = {int(r["question_id"]): r.get("pred")
              for r in json.load(open(
                  ret_dir / "arm_orm_grouphead" / "eval_result_dev.json",
                  encoding="utf-8"))}
    same_full = sum(1 for q in full_qids
                    if q in a_rows and q in b_rows and a_rows[q] == b_rows[q])
    out["pred_identical_on_is_full"] = {
        "n": len(full_qids), "same": same_full,
        "same_pct": round(same_full / len(full_qids) * 100, 2)}

    # token 节省（ret 池实测 + k_selection 理论对比）
    ts_path = ret_dir.parent / "eval_pool_bird_ret" / "retriever_token_stats.json"
    if ts_path.is_file():
        ts = json.load(open(ts_path, encoding="utf-8"))
        out["token_stats"] = {
            "n_items": ts.get("n_items"),
            "n_pruned": ts.get("n_pruned"),
            "n_fallback_full": ts.get("n_fallback_full"),
            "n_gold_missing": ts.get("n_gold_missing"),
            "full_mean": ts["full"]["mean"], "pruned_mean": ts["pruned"]["mean"],
            "full_sum": ts["full"]["sum"], "pruned_sum": ts["pruned"]["sum"],
            "saved_pct": ts.get("saved_pct"),
        }

    json.dump(out, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    # 控制台摘要
    o = out["arms"]["arm_orm_grouphead"]
    print("=== arm_orm_grouphead ===")
    print("overall: base=%.2f ret=%.2f (n=%d)" % (
        o["overall"]["acc_base"], o["overall"]["acc_ret"], o["overall"]["n"]))
    gm = o["gold_missing"]
    print("gold_missing: base=%.2f ret=%.2f (n=%d) fixed=%d broken=%d p=%.4g" % (
        gm["acc_base"], gm["acc_ret"], gm["n"],
        gm["mcnemar"]["fixed"], gm["mcnemar"]["broken"], gm["mcnemar"]["p"]))
    m = o["overall"]["mcnemar"]
    print("overall McNemar: fixed=%d broken=%d n=%d p=%.4g" % (
        m["fixed"], m["broken"], m["n"], m["p"]))
    print("pred identical on is_full: %s" % out["pred_identical_on_is_full"])
    if out.get("token_stats"):
        print("token saved: %.2f%%" % out["token_stats"]["saved_pct"])
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
