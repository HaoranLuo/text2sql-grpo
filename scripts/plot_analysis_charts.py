#!/usr/bin/env python3
"""分析级图表（实验细节证据链）：
1. analysis_vote_agreement.png - 投票一致度 vs 正确率（投票有效性核心证据）
2. analysis_data_volume.png    - 训练数据量 × (单prompt vs 5p投票) 双线
3. analysis_overfit.png        - checkpoint 步数过拟合曲线
4. analysis_training_panel.png - 训练过程 6 面板全景
5. analysis_full_matrix.png    - 全部实验矩阵热力图
"""
import json
import re
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
})

PROJECT = Path(__file__).resolve().parent.parent
CHARTS = PROJECT / "charts"
C_3B, C_7B, C_ACC, C_NEG = "#1f77b4", "#ff7f0e", "#9467bd", "#d62728"


def plot_vote_agreement():
    """投票一致度 vs 正确率——投票为什么有效的直接证据"""
    items = json.loads((PROJECT / "outputs/eval_5p_p2a500/items.json").read_text())
    c = Counter(i["votes"] for i in items)
    rate = {v: sum(1 for i in items if i["votes"] == v and i["match"]) / n
            for v, n in c.items()}

    xs = sorted(rate)
    ys = [rate[x] * 100 for x in xs]
    ns = [c[x] for x in xs]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(xs, ys, color=[plt.cm.RdYlGn(0.15 + 0.85 * (y / 100)) for y in ys],
                  width=0.66, zorder=3, edgecolor="white")
    for x, y, n in zip(xs, ys, ns):
        ax.text(x, y + 2.5, f"{y:.0f}%\n(n={n})", ha="center", fontsize=10,
                fontweight="bold")

    ax.set_xlabel("5 个 prompt 中执行结果一致的个数（一致度）")
    ax.set_ylabel("该组题目的正确率 (%)")
    ax.set_title("投票一致度 vs 正确率（500条训练模型 × 5prompt，n=100 题）",
                 fontsize=13, pad=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x} 票" for x in xs])
    ax.set_ylim(0, 110)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax.annotate("全票一致 (5/5) → 100% 正确\n0 票一致 → 0% 正确",
                xy=(5, 100), xytext=(3.1, 82),
                fontsize=10, color="#333333",
                arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.text(0.02, 0.02,
            "证据链：模型对同一题的多视角输出越一致，\n该题答对的概率越高——多数投票因此有效。",
            transform=ax.transAxes, fontsize=9, color="#666666",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", alpha=0.8))
    plt.tight_layout()
    out = CHARTS / "analysis_vote_agreement.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {out.name}")


def plot_data_volume():
    """训练数据量 × 评估方式 双线"""
    vols = ["100", "500", "2000", "7000"]
    single = [50, 50, 43, 45]   # 单prompt（修复后口径）
    vote5 = [70, 71, 71, 70]    # 5p投票

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(range(4), single, marker="o", ms=8, lw=2.4, color=C_NEG,
            label="单 prompt 评估", zorder=3)
    ax.plot(range(4), vote5, marker="s", ms=8, lw=2.4, color=C_3B,
            label="5prompt 投票", zorder=3)
    for i, (s, v) in enumerate(zip(single, vote5)):
        ax.text(i, s + 1.5, f"{s}%", ha="center", fontsize=10, color=C_NEG,
                fontweight="bold")
        ax.text(i, v + 1.5, f"{v}%", ha="center", fontsize=10, color=C_3B,
                fontweight="bold")
        ax.annotate("", xy=(i, v - 2), xytext=(i, s + 4),
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8, alpha=0.5))

    ax.axhline(45, color="#888888", ls="--", lw=1.0, alpha=0.6)
    ax.text(3.02, 45, "3B 零样本基线 45%", fontsize=8.5, color="#666666")

    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{v} 条" for v in vols])
    ax.set_xlabel("GRPO 训练数据量")
    ax.set_ylabel("执行匹配率 (%)")
    ax.set_title("训练数据量 vs 评估方式（3B 训练后模型，100 题）", pad=10)
    ax.legend(loc="lower right")
    ax.set_ylim(30, 82)
    ax.grid(ls=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax.annotate("投票抹平数据量差异：\n四种数据量投票后全部收敛 70~71%",
                xy=(1, 71), xytext=(2.35, 60),
                fontsize=10, color="#1a5e8a",
                arrowprops=dict(arrowstyle="->", color="#1a5e8a", lw=1.3))
    plt.tight_layout()
    out = CHARTS / "analysis_data_volume.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {out.name}")


def plot_overfit():
    """checkpoint 步数过拟合曲线"""
    steps = [25, 50, 75]
    acc = [50, 34, 36]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(steps, acc, marker="o", ms=9, lw=2.5, color=C_ACC, zorder=3)
    for s, a in zip(steps, acc):
        ax.text(s, a + 2, f"{a}%", ha="center", fontsize=11,
                fontweight="bold", color=C_ACC)
    ax.axhline(50, color=C_3B, ls="--", lw=1.0, alpha=0.5)
    ax.text(26, 50.8, "25 步甜点", fontsize=9, color=C_3B)
    ax.annotate("过拟合区：训练继续推离先验\n记忆训练查询 → 泛化下降",
                xy=(50, 34), xytext=(42, 44),
                fontsize=10, color=C_NEG,
                arrowprops=dict(arrowstyle="->", color=C_NEG, lw=1.3))
    ax.set_xlabel("GRPO 训练步数 (max_steps)")
    ax.set_ylabel("执行匹配率 (%)")
    ax.set_title("3B 训练步数 vs 成绩（checkpoint 早停实验）", pad=10)
    ax.set_xticks(steps)
    ax.set_ylim(25, 60)
    ax.grid(ls=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = CHARTS / "analysis_overfit.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {out.name}")


def plot_training_panel():
    """训练过程 6 面板全景（三级 vs partial）"""
    runs = {}
    for fname, label, color in [("3b_3lvl_1639580.out", "三级奖励", C_3B),
                                ("p2d_1647296.out", "partial奖励", C_NEG)]:
        p = PROJECT / "logs" / fname
        if not p.exists():
            continue
        steps, rw, rs, kl, ln, gn = [], [], [], [], [], []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.search(r"\{'loss'.*\}", line)
            if not m:
                continue
            try:
                d = json.loads(m.group(0).replace("'", '"'))
            except Exception:
                continue
            steps.append(len(steps) + 1)
            rw.append(d.get("reward", 0))
            rs.append(d.get("reward_std", 0))
            kl.append(d.get("kl", 0))
            ln.append(d.get("completion_length", 0))
            gn.append(d.get("grad_norm", 0))
        runs[label] = (steps, rw, rs, kl, ln, gn, color)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    titles = [("平均奖励 (reward)", "模型学到的平均回报"),
              ("奖励标准差", "组内多样性：高=还在探索"),
              ("KL 散度", "偏离基座：过大=灾难遗忘"),
              ("平均生成长度", "稳定=输出收敛"),
              ("梯度范数", "学习信号强度"),
              ("学习率 (scheduler)", "线性衰减")]
    for j, (label, (steps, rw, rs, kl, ln, gn, color)) in enumerate(runs.items()):
        datas = [rw, rs, kl, ln, gn]
        labels_short = ["reward", "std", "KL", "len", "grad"]
        # 学习率: 从日志单独取
        lrs = []
        p = PROJECT / "logs" / ("3b_3lvl_1639580.out" if "三级" in label else "p2d_1647296.out")
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.search(r"\{'loss'.*\}", line)
            if m:
                try:
                    lrs.append(json.loads(m.group(0).replace("'", '"')).get("learning_rate", 0))
                except Exception:
                    pass
        datas.append(lrs)

        for k, (ax, (title, _sub)) in enumerate(zip(axes.flat, titles)):
            d = datas[k]
            if len(d) != len(steps):
                d = d[:len(steps)]
            ax.plot(steps, d, marker="o", ms=3.5, lw=1.8, color=color, label=label)
            if k == 0:
                ax.legend(fontsize=8.5)
            ax.set_title(f"{title}", fontsize=11)
            ax.set_xlabel("步数", fontsize=8.5)
            ax.grid(ls=":", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("GRPO 训练过程全景对比（三级 vs partial 奖励）", fontsize=14,
                 fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = CHARTS / "analysis_training_panel.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {out.name}")


def plot_full_matrix():
    """全部实验矩阵热力图（模型×方法 → 成绩）"""
    rec_file = PROJECT / "records" / "experiments.jsonl"
    if not rec_file.exists():
        print("skip: 无 records")
        return
    rows = []
    for line in rec_file.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    # 关键实验: (模型族, 方法, 成绩)
    fam_order = ["3B", "7B", "外部"]
    data = {}
    for r in rows:
        fam, meth, score = r.get("family"), r.get("method"), r.get("match_rate")
        if fam in fam_order and isinstance(score, (int, float)) and score > 0:
            data.setdefault(fam, []).append((meth, score * 100))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palettes = {"3B": plt.cm.Blues, "7B": plt.cm.Oranges, "外部": plt.cm.Greens}
    for ax, fam in zip(axes, fam_order):
        items = sorted(data.get(fam, []), key=lambda x: x[1])
        if not items:
            continue
        names = [m for m, _ in items]
        scores = [s for _, s in items]
        ax.barh(range(len(items)), scores, color=[palettes[fam](0.35 + 0.6 * s / 100)
                                                  for s in scores],
                edgecolor="white", height=0.62)
        for i, s in enumerate(scores):
            ax.text(s + 1, i, f"{s:.0f}%", va="center", fontsize=9,
                    fontweight="bold")
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([f"{n[:24]}…" if len(n) > 24 else n for n in names],
                           fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_title(f"{fam} 系列实验成绩分布", fontsize=12, fontweight="bold")
        ax.grid(axis="x", ls=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("全部实验成绩分布（按模型系列）", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = CHARTS / "analysis_full_matrix.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {out.name}")


if __name__ == "__main__":
    CHARTS.mkdir(exist_ok=True)
    plot_vote_agreement()
    plot_data_volume()
    plot_overfit()
    plot_training_panel()
    plot_full_matrix()
