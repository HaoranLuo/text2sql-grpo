#!/usr/bin/env python3
"""美化版图表（论文级样式）：
1. training_reward_compare_v2.png  - 训练奖励曲线（含基线标注）
2. experiment_summary_v2.png      - 关键实验对比（含基线/最佳标注）
3. voting_curve_v2.png            - 投票视角数 vs 成绩曲线（3p/5p/7p）
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 论文级样式
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.edgecolor": "#cccccc",
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "grid.color": "#dddddd",
    "grid.alpha": 0.6,
})

PROJECT = Path(__file__).resolve().parent.parent
CHARTS = PROJECT / "charts"

# 调色板（区分度高）
C_3B = "#1f77b4"    # 蓝
C_7B = "#ff7f0e"    # 橙
C_EXT = "#2ca02c"   # 绿
C_NEG = "#d62728"   # 红
C_ACC = "#9467bd"   # 紫


def plot_reward_compare():
    """训练奖励曲线：三级 vs partial（带基线参考）"""
    parsed = {}
    for fname in ["3b_3lvl_1639580.out", "p2d_1647296.out"]:
        p = PROJECT / "logs" / fname
        if not p.exists():
            continue
        steps, rewards = [], []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.search(r"\{'loss'.*\}", line)
            if not m:
                continue
            try:
                d = json.loads(m.group(0).replace("'", '"'))
            except Exception:
                continue
            steps.append(len(steps) + 1)
            rewards.append(d.get("reward", 0.0))
        parsed[fname] = (steps, rewards)
    if not parsed:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    styles = {
        "3b_3lvl_1639580.out": (C_3B, "三级奖励 · 100条 · 75步", "o"),
        "p2d_1647296.out": (C_NEG, "partial奖励 · 500条 · 25步", "s"),
    }
    for fname, (steps, rewards) in parsed.items():
        color, label, marker = styles.get(fname, (C_ACC, fname, "o"))
        ax.plot(steps, rewards, marker=marker, ms=6, lw=2.2,
                color=color, label=label, zorder=3)

    # 基线参考线
    ax.axhline(0.5, color="#888888", ls="--", lw=1.2, alpha=0.7)
    ax.annotate("三级奖励的稳态均值 ~0.5（≈执行匹配率 50%）",
                xy=(1.02, 0.5), fontsize=9, color="#666666")

    ax.set_xlabel("训练步数 (step)")
    ax.set_ylabel("平均奖励 (reward)")
    ax.set_title("GRPO 训练奖励曲线：三级 vs partial 奖励", pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_ylim(0, 0.75)
    ax.spines[["top", "right"]].set_visible(False)

    # 标注要点
    ax.annotate("partial 信号卡在 0.1~0.2\n（学习信号不足 → 最终 45% = 基线）",
                xy=(4.5, 0.13), xytext=(2.5, 0.28),
                fontsize=9, color=C_NEG,
                arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.2))

    plt.tight_layout()
    out = CHARTS / "training_reward_compare_v2.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {out.name}")


def plot_experiment_summary():
    """关键实验对比：3B 路径 + 7B 路径（阶梯图）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- 左：3B 路径阶梯 ----
    labels = ["3B 基线", "GRPO训练\n(三级25步)", "+投票×3", "+投票×5\n(500条)"]
    values = [45, 50, 65, 71]
    colors = ["#bbbbbb", C_3B, C_3B, C_3B]
    ax1.bar(range(len(values)), values, color=colors, width=0.58, zorder=3,
            edgecolor="white")
    for i, v in enumerate(values):
        ax1.text(i, v + 1.5, f"{v}%", ha="center", fontsize=12,
                 fontweight="bold", color="#333333")
    ax1.annotate("+26 pp", xy=(3, 71), xytext=(2.2, 76),
                 fontsize=13, fontweight="bold", color=C_NEG,
                 arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.4))
    ax1.set_ylim(0, 88)
    ax1.set_ylabel("执行匹配率 (%)")
    ax1.set_title("3B 完整提升路径", fontsize=13, pad=10)
    ax1.set_xticks(range(len(values)))
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.grid(axis="y", ls=":", alpha=0.5)
    ax1.spines[["top", "right"]].set_visible(False)

    # ---- 右：7B 路径 ----
    labels2 = ["7B 基线", "+投票×3", "+投票×5", "+投票×5+仲裁"]
    values2 = [81, 85, 85, 85]
    colors2 = ["#bbbbbb", C_7B, C_7B, C_7B]
    ax2.bar(range(len(values2)), values2, color=colors2, width=0.58, zorder=3,
            edgecolor="white")
    for i, v in enumerate(values2):
        ax2.text(i, v + 1.5, f"{v}%", ha="center", fontsize=12,
                 fontweight="bold", color="#333333")
    ax2.axhline(85, color=C_7B, ls="--", lw=1.0, alpha=0.6)
    ax2.annotate("投票上界 = 85%\n（3种方案全部相同）", xy=(1.2, 85.5),
                 xytext=(0.1, 78), fontsize=9, color="#8c5a00",
                 arrowprops=dict(arrowstyle="->", color="#8c5a00", lw=1.2))
    ax2.set_ylim(0, 95)
    ax2.set_title("7B 投票方案对比（零训练成本）", fontsize=13, pad=10)
    ax2.set_xticks(range(len(labels2)))
    ax2.set_xticklabels(labels2, fontsize=9.5)
    ax2.grid(axis="y", ls=":", alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Text-to-SQL GRPO 项目核心成果", fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()
    out = CHARTS / "experiment_summary_v2.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {out.name}")


def plot_voting_curve():
    """投票视角数 vs 成绩（3p/5p/7p）：展示 5p 甜点"""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = [1, 3, 5, 7]
    y = [50, 65, 71, 71]  # 训练后 3B × 投票数
    ax.plot(x, y, marker="o", ms=9, lw=2.5, color=C_3B, zorder=3,
            markerfacecolor="white", markeredgewidth=2.2)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi}%", (xi, yi), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=11,
                    fontweight="bold", color=C_3B)
    # 甜点标注
    ax.axvline(5, color=C_NEG, ls="--", lw=1.2, alpha=0.7)
    ax.annotate("5p 甜点：再增视角零提升\n（投票已收敛，成本却线性增长）",
                xy=(5, 71), xytext=(4.6, 78),
                fontsize=10, color=C_NEG,
                arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.3))
    ax.set_xticks(x)
    ax.set_xticklabels(["1p\n(单prompt)", "3p", "5p", "7p"], fontsize=11)
    ax.set_xlabel("投票 prompt 视角数")
    ax.set_ylabel("执行匹配率 (%)")
    ax.set_title("投票视角数 vs 成绩（训练后 3B）", fontsize=13, pad=10)
    ax.set_ylim(40, 85)
    ax.grid(ls=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = CHARTS / "voting_curve_v2.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {out.name}")


if __name__ == "__main__":
    CHARTS.mkdir(exist_ok=True)
    plot_reward_compare()
    plot_experiment_summary()
    plot_voting_curve()
