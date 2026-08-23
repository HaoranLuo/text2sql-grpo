#!/usr/bin/env python3
"""McNemar 检验：arm_vav vs arm_orm_grouphead 的 BIRD 官方逐题结果。"""
import json
import math
import sys

BASE = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/outputs/bird_select_ormbird_bird_bal2"


def load(p):
    rows = json.load(open(p))
    out = {}
    r0 = rows[0]
    print(f"{p.split('/')[-1]}: {len(rows)} rows, keys={list(r0.keys())}")
    for r in rows:
        qid = r.get("question_id", r.get("qid"))
        # 官方评估器逐题结果字段探测
        res = None
        for k in ("res", "execution", "correct", "result"):
            if k in r:
                res = r[k]
                break
        out[qid] = bool(res)
    return out


a = load(f"{BASE}/arm_vav/eval_result_dev.json")
b = load(f"{BASE}/arm_orm_grouphead/eval_result_dev.json")

fixed = sum(1 for q in a if (not a[q]) and b[q])
broken = sum(1 for q in a if a[q] and (not b[q]))
n = fixed + broken
p_upper = sum(math.comb(n, k) * 0.5 ** n for k in range(0, fixed + 1))
p_two = 2 * min(p_upper, 1 - p_upper)
print(f"fixed={fixed} broken={broken} discordant={n} "
      f"McNemar_two_sided_p={p_two:.6g}")
