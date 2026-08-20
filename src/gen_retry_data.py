#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RetrySQL 式错误注入数据生成（P1-3 牌一）。

配方（RetrySQL arXiv:2507.02529，经训练侧调研报告核实）：
- 对每条 <think> 推理按句切步；对第 i 步以 p 概率在其前插入「{未来步} [BACK]」
  （FS=仅未来步作损坏源，单次损坏；论文最优 FS p=0.2）
- multiply_factor：每条轨迹出 M 个损坏变体（不同 seed）
- 保留我们自己的 chat 格式（user=DDL+问题，assistant=<think>...</think>+```sql```），
  与 RetrySQL 特殊 token 格式的差异如实记录（格式是外壳；机制=错步+[BACK]+全参 CPT 不变）
输出 data/retry_cpt_train.json（list of {"messages": [...]}），并打印统计。
纯标准库，CPU 秒级。
"""
import argparse
import json
import random
import re
import sys

BACK = "[BACK]"
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_steps(think: str):
    """按句号/换行切步，过滤过短片段；空 think 返回空列表。"""
    raw = re.split(r"(?<=[。.?!])\s+|\n+", think.strip())
    return [s.strip() for s in raw if len(s.strip()) >= 4]


def corrupt_steps(steps, p, rng):
    """RetrySQL FS 单次损坏：每步以 p 概率在其前插入「未来步 [BACK]」。"""
    out = []
    n_corrupted = 0
    for i, s in enumerate(steps):
        future = steps[i + 1:]
        if future and rng.random() < p:
            wrong = rng.choice(future)
            out.append(f"{wrong} {BACK}")
            n_corrupted += 1
        out.append(s)
    return out, n_corrupted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_v4hard_mix.json")
    ap.add_argument("--out", default="data/retry_cpt_train.json")
    ap.add_argument("--corrupt-prob", type=float, default=0.2)
    ap.add_argument("--multiply", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-steps", type=int, default=3,
                    help="think 切步数不足此值的条目跳过（无法做 FS 损坏）")
    args = ap.parse_args()

    raw = json.load(open(args.data, encoding="utf-8"))
    if not isinstance(raw, list):
        print(f"[retry-data] FATAL: 期望 list，实际 {type(raw)}", file=sys.stderr)
        sys.exit(1)

    src = []
    for e in raw:
        msgs = e.get("messages") or []
        if len(msgs) < 2:
            continue
        m = THINK_RE.search(msgs[1]["content"] or "")
        if not m:
            continue
        steps = split_steps(m.group(1))
        if len(steps) < args.min_steps:
            continue
        src.append((msgs, steps, m.span()))

    rng = random.Random(args.seed)
    out = []
    n_steps_tot = n_corrupted_tot = 0
    for msgs, steps, span in src:
        for v in range(args.multiply):
            vrng = random.Random(args.seed * 10007 + v)
            corrupted, nc = corrupt_steps(steps, args.corrupt_prob, vrng)
            a = msgs[1]["content"]
            new_think = " ".join(corrupted)
            new_a = a[:span[0]] + "<think>\n" + new_think + "\n</think>" + a[span[1]:]
            out.append({"messages": [msgs[0], {"role": "assistant", "content": new_a}]})
            n_steps_tot += len(steps)
            n_corrupted_tot += nc

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[retry-data] 源轨迹(含 think 且步数>={args.min_steps}): {len(src)}")
    print(f"[retry-data] 输出条目: {len(out)}（multiply={args.multiply}）")
    print(f"[retry-data] 总步数: {n_steps_tot} 注入损坏: {n_corrupted_tot} "
          f"实测损坏率: {n_corrupted_tot / max(1, n_steps_tot):.4f}（目标 p={args.corrupt_prob}）")
    # 抽样示例
    print("[retry-data] 样例（前 300 字符）:")
    print(out[0]["messages"][1]["content"][:300].replace("\n", " | "))


if __name__ == "__main__":
    main()
