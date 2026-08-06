#!/usr/bin/env python3
"""从训练日志画 GRPO 训练曲线（reward / reward_std / kl / completion_length vs step）

用法:
    python scripts/plot_training_curves.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

for _f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]:
    try:
        plt.rcParams["font.sans-serif"] = [_f]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

PROJECT = Path(__file__).resolve().parent.parent
LOGS = PROJECT / "logs"
CHARTS = PROJECT / "charts"

# 训练日志 → 图例名
RUNS = {
    "3b_3lvl_1639580.out": "三级奖励 75步 (checkpoint-25/50/75)",
    "p2d_1647296.out": "partial奖励 500条 25步 (P2D)",
}


def parse_log(path: Path):
    steps, rewards, r_stds, kls, lengths, grad_norms = [], [], [], [], [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"\{'loss'.*\}", line)
        if not m:
            continue
        try:
            d = json.loads(m.group(0).replace("'", '"'))
        except Exception:
            continue
        steps.append(len(steps) + 1)
        rewards.append(d.get("reward", 0.0))
        r_stds.append(d.get("reward_std", 0.0))
        kls.append(d.get("kl", 0.0))
        lengths.append(d.get("completion_length", 0.0))
        grad_norms.append(d.get("grad_norm", 0.0))
    return steps, rewards, r_stds, kls, lengths, grad_norms


def main():
    CHARTS.mkdir(exist_ok=True)
    parsed = {}
    for fname, label in RUNS.items():
        p = LOGS / fname
        if not p.exists():
            print(f"skip {fname} (not found)")
            continue
        parsed[label] = parse_log(p)
        print(f"{fname}: {len(parsed[label][0])} 个记录点")

    if not parsed:
        print("无可用日志")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1) 平均奖励
    ax = axes[0][0]
    for label, (s, r, *_rest) in parsed.items():
        ax.plot(s, r, marker="o", ms=3, label=label)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("平均奖励 (reward)")
    ax.set_title("GRPO 平均奖励曲线")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2) 奖励标准差（多样性信号）
    ax = axes[0][1]
    for label, (s, r, rs, *_rest) in parsed.items():
        ax.plot(s, rs, marker="o", ms=3, label=label)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("奖励标准差")
    ax.set_title("奖励标准差（组内多样性）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3) KL 散度（偏离基座模型程度）
    ax = axes[1][0]
    for label, (s, r, rs, kl, *_rest) in parsed.items():
        ax.plot(s, kl, marker="o", ms=3, label=label)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("KL")
    ax.set_title("KL 散度（离基座距离，过大=灾难遗忘）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4) 生成长度
    ax = axes[1][1]
    for label, (s, r, rs, kl, ln, *_rest) in parsed.items():
        ax.plot(s, ln, marker="o", ms=3, label=label)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("平均生成长度 (token)")
    ax.set_title("生成长度（越稳定越好）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("GRPO 训练过程曲线（来自训练日志）", fontsize=13)
    plt.tight_layout()
    out = CHARTS / "training_curves.png"
    plt.savefig(out, dpi=150)
    print(f"✅ 训练曲线: {out}")

    # 5) 单图: 奖励对比（汇报用）
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for label, (s, r, *_rest) in parsed.items():
        ax2.plot(s, r, marker="o", ms=4, linewidth=2, label=label)
    ax2.set_xlabel("训练步数")
    ax2.set_ylabel("平均奖励")
    ax2.set_title("GRPO 训练奖励对比（三级 vs partial）")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    out2 = CHARTS / "training_reward_compare.png"
    plt.savefig(out2, dpi=150)
    print(f"✅ 奖励对比图: {out2}")


if __name__ == "__main__":
    main()
