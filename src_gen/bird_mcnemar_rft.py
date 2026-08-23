#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RFT 5 源池对照 McNemar：4 源 vs 5 源 的官方逐题结果配对检验。

对比对（每对均基于 eval_result_dev.json 逐题 res，question_id 对齐，n=1534）：
  - arm_orm_grouphead: 4 源（outputs/bird_select_ormbird_bird_bal2）vs
    5 源（outputs/bird_select_5src）——主检验（H1 显著性）；
  - arm_vav: 同池对比——次要描述（H3）。

口径与 src/bird_mcnemar.py 相同（双尾精确 McNemar，binom 精确 p）。

用法：envs/reasoning3b/bin/python src/bird_mcnemar_rft.py
"""
import json
import math
import sys

BASE_4 = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/outputs/bird_select_ormbird_bird_bal2"
BASE_5 = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/outputs/bird_select_5src"


def load(p):
    rows = json.load(open(p))
    out = {}
    for r in rows:
        qid = r.get("question_id", r.get("qid"))
        res = None
        for k in ("res", "execution", "correct", "result"):
            if k in r:
                res = r[k]
                break
        out[qid] = bool(res)
    return out


def mcnemar(a, b):
    fixed = sum(1 for q in a if (not a[q]) and b[q])
    broken = sum(1 for q in a if a[q] and (not b[q]))
    n = fixed + broken
    p_upper = sum(math.comb(n, k) * 0.5 ** n for k in range(0, fixed + 1))
    p_two = 2 * min(p_upper, 1 - p_upper)
    n_a = sum(bool(v) for v in a.values())
    n_b = sum(bool(v) for v in b.values())
    return {
        "n": len(a), "correct_a": n_a, "correct_b": n_b,
        "acc_a": round(n_a / len(a) * 100, 2),
        "acc_b": round(n_b / len(b) * 100, 2),
        "fixed": fixed, "broken": broken, "discordant": n,
        "mcnemar_two_sided_p": p_two,
    }


def main():
    out = {}
    for arm in ("arm_orm_grouphead", "arm_vav"):
        a = load(f"{BASE_4}/{arm}/eval_result_dev.json")
        b = load(f"{BASE_5}/{arm}/eval_result_dev.json")
        if len(a) != len(b) or set(a) != set(b):
            print(f"[FATAL] {arm}: 题目集合不一致 (4src={len(a)}, 5src={len(b)})",
                  file=sys.stderr)
            sys.exit(1)
        r = mcnemar(a, b)
        out[arm] = r
        print(f"{arm}: 4src={r['acc_a']} ({r['correct_a']}/{r['n']}) -> "
              f"5src={r['acc_b']} ({r['correct_b']}/{r['n']}) | "
              f"fixed={r['fixed']} broken={r['broken']} discordant={r['discordant']} "
              f"McNemar_two_sided_p={r['mcnemar_two_sided_p']:.6g}")
    json.dump(out, open(f"{BASE_5}/work/mcnemar_4src_vs_5src.json", "w"),
              ensure_ascii=False, indent=2)
    print(f"-> {BASE_5}/work/mcnemar_4src_vs_5src.json")


if __name__ == "__main__":
    main()
