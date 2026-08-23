#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""McNemar 配对检验：任意两个 BIRD 官方逐题结果文件（跨池/跨臂）。

基于 src/bird_mcnemar.py 泛化：不再固定 BASE 目录与臂名，改为任意
--path-a / --path-b（官方评估器产物 eval_result_dev.json，逐题 res 字段）。

用法：
  envs/reasoning3b/bin/python src/bird_mcnemar_ev.py \
      --path-a outputs/bird_select/arm_vav/eval_result_dev.json \
      --path-b outputs/bird_select_bird_ev/arm_vav/eval_result_dev.json \
      --label-a "原池 vav" --label-b "证据池 vav"
"""
import argparse
import json
import math


def load(p):
    rows = json.load(open(p))
    out = {}
    r0 = rows[0]
    print(f"{p.split('/')[-1]}: {len(rows)} rows, keys={list(r0.keys())}")
    for r in rows:
        qid = r.get("question_id", r.get("qid"))
        # 官方评估器逐题结果字段探测（与 bird_mcnemar.py 同款）
        res = None
        for k in ("res", "execution", "correct", "result"):
            if k in r:
                res = r[k]
                break
        out[qid] = bool(res)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="McNemar：两文件逐题官方执行结果配对检验")
    ap.add_argument("--path-a", required=True)
    ap.add_argument("--path-b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    a = load(args.path_a)
    b = load(args.path_b)
    common = sorted(set(a) & set(b))
    fixed = sum(1 for q in common if (not a[q]) and b[q])   # A 错 B 对
    broken = sum(1 for q in common if a[q] and (not b[q]))  # A 对 B 错
    n = fixed + broken
    acc_a = sum(1 for q in common if a[q]) / len(common) if common else 0.0
    acc_b = sum(1 for q in common if b[q]) / len(common) if common else 0.0
    if n == 0:
        p_two = 1.0
    else:
        p_upper = sum(math.comb(n, k) * 0.5 ** n for k in range(0, fixed + 1))
        p_two = 2 * min(p_upper, 1 - p_upper)
    print(f"common questions: {len(common)}")
    print(f"acc {args.label_a}: {acc_a:.4f} ({sum(1 for q in common if a[q])}/{len(common)})")
    print(f"acc {args.label_b}: {acc_b:.4f} ({sum(1 for q in common if b[q])}/{len(common)})")
    print(f"diff: {acc_b - acc_a:+.4f}")
    print(f"{args.label_b} 修复 {args.label_a} 的错题 (fixed): {fixed}")
    print(f"{args.label_b} 弄坏 {args.label_a} 的对题 (broken): {broken}")
    print(f"discordant={n}  McNemar_two_sided_p={p_two:.6g}")
    if p_two < 0.05:
        print("=> significant at p<0.05")


if __name__ == "__main__":
    main()
