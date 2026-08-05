#!/usr/bin/env python3
"""实验自动记录系统：每次实验后生成结构化记录 + 图表

用法:
    python scripts/record_experiment.py --summary /path/to/summary.json \
        --name "实验名" --method "方法描述" --notes "备注"
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RECORDS_DIR = PROJECT / "records"
RECORDS_FILE = RECORDS_DIR / "experiments.jsonl"
CHARTS_DIR = PROJECT / "charts"

KNOWN_EXPERIMENTS = {
    "3b_baseline_v2": {"model": "Qwen2.5-Coder-3B", "method": "零样本基线", "family": "3B"},
    "3b_c25_v2": {"model": "Qwen2.5-Coder-3B", "method": "GRPO三级奖励25步", "family": "3B"},
    "p2a_100": {"model": "Qwen2.5-Coder-3B", "method": "GRPO 100条G8", "family": "3B"},
    "p2a_500": {"model": "Qwen2.5-Coder-3B", "method": "GRPO 500条G8", "family": "3B"},
    "p2a_2000": {"model": "Qwen2.5-Coder-3B", "method": "GRPO 2000条G8", "family": "3B"},
    "p2a_7000": {"model": "Qwen2.5-Coder-3B", "method": "GRPO 7000条G8", "family": "3B"},
    "p2c2_partial": {"model": "Qwen2.5-Coder-3B", "method": "GRPO partial奖励", "family": "3B"},
    "eval_m5_3b_trained": {"model": "Qwen2.5-Coder-3B", "method": "训练后3prompt投票", "family": "3B"},
    "eval_mschema": {"model": "Qwen2.5-Coder-7B", "method": "M-Schema单prompt", "family": "7B"},
    "eval_multi_prompt": {"model": "Qwen2.5-Coder-7B", "method": "3prompt投票", "family": "7B"},
    "eval_m5plus": {"model": "Qwen2.5-Coder-7B", "method": "5prompt+仲裁", "family": "7B"},
    "eval_m5_7b_trained": {"model": "Qwen2.5-Coder-7B", "method": "训练后3prompt投票", "family": "7B"},
    "eval_lowtemp_vote_v2": {"model": "Qwen2.5-Coder-7B", "method": "低温采样投票", "family": "7B"},
    "eval_xiyan_native": {"model": "XiYanSQL-7B", "method": "原生格式", "family": "外部"},
    "eval_omnisql": {"model": "OmniSQL-7B", "method": "零样本", "family": "外部"},
    "eval_dsv2_100": {"model": "DeepSeek-V2-Lite", "method": "零样本", "family": "外部"},
    "eval_api_ceiling": {"model": "DeepSeek-V4-Flash-API", "method": "API基准", "family": "外部"},
    "p2a_100": {"model": "Qwen2.5-Coder-3B", "method": "GRPO 100条G8三级", "family": "3B"},
    "p2a_500": {"model": "Qwen2.5-Coder-3B", "method": "GRPO 500条G8三级", "family": "3B"},
    "p2a_2000": {"model": "Qwen2.5-Coder-3B", "method": "GRPO 2000条G8三级", "family": "3B"},
    "p2a_7000": {"model": "Qwen2.5-Coder-3B", "method": "GRPO 7000条G8三级", "family": "3B"},
    "p2c2_partial": {"model": "Qwen2.5-Coder-3B", "method": "GRPO 100条G8partial", "family": "3B"},
    "c3_atomic": {"model": "Qwen2.5-Coder-3B", "method": "GRPO 100条G4原子奖励", "family": "3B"},
    "eval_5prompt_3b_trained": {"model": "Qwen2.5-Coder-3B", "method": "训练后3B 5prompt投票", "family": "3B"},
}


def record(summary_path: str, name: str, method: str = "", notes: str = ""):
    with open(summary_path) as f:
        s = json.load(f)

    meta = KNOWN_EXPERIMENTS.get(name, {"model": "?", "method": method, "family": "?"})

    entry = {
        "timestamp": datetime.now().isoformat(),
        "name": name,
        "model": meta.get("model", "?"),
        "method": method or meta.get("method", "?"),
        "family": meta.get("family", "?"),
        "match_rate": (s.get("custom_execution_match_rate")
                       if s.get("custom_execution_match_rate") is not None
                       else s.get("match_rate")),
        "parse_rate": s.get("parse_success_rate"),
        "exec_rate": s.get("prediction_execution_success_rate"),
        "elapsed_seconds": (s.get("elapsed_seconds")
                            if s.get("elapsed_seconds") is not None
                            else s.get("total_wall_seconds")),
        "notes": notes,
    }

    RECORDS_DIR.mkdir(exist_ok=True)
    with open(RECORDS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✅ 已记录: {name} | {entry['model']} | {entry['method']} | {entry['match_rate']}")


def generate_charts():
    """从 records 生成对比图表"""
    if not RECORDS_FILE.exists():
        print("无记录可绘图")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    # 中文字体支持（Windows: SimHei/Microsoft YaHei; Linux: Noto/WQY）
    for _f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]:
        try:
            plt.rcParams["font.sans-serif"] = [_f]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    records = []
    with open(RECORDS_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    if not records:
        return

    CHARTS_DIR.mkdir(exist_ok=True)

    # 图表1: 按模型分组的柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    families = {}
    for r in records:
        families.setdefault(r["family"], []).append(r)

    x_pos = 0
    labels, values, colors = [], [], []
    palette = {"3B": "#1f77b4", "7B": "#ff7f0e", "外部": "#2ca02c"}
    for fam, rs in families.items():
        for r in sorted(rs, key=lambda x: x.get("match_rate") or 0, reverse=True):
            labels.append(f"{r['name']}\n{r['method']}")
            values.append(r.get("match_rate") or 0)
            colors.append(palette.get(fam, "#999"))
            x_pos += 1

    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Match Rate")
    ax.set_title("Text-to-SQL 实验对比（按模型分组着色）")
    ax.axhline(0.81, color="red", linestyle="--", alpha=0.5, label="7B基线81%")
    ax.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "experiment_comparison.png", dpi=150)
    plt.close()
    print(f"📊 图表已生成: {CHARTS_DIR / 'experiment_comparison.png'}")

    # 图表2: 训练进度（如果记录里有多个阶段）
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for fam in ["3B", "7B"]:
        fr = [r for r in records if r["family"] == fam and r.get("match_rate")]
        if fr:
            stages = [r["method"] for r in fr]
            vals = [r["match_rate"] for r in fr]
            ax2.plot(range(len(vals)), vals, marker="o", label=f"{fam}系列")
    ax2.set_xticks([])
    ax2.set_ylabel("Match Rate")
    ax2.set_title("各系列实验进展")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "progress_trend.png", dpi=150)
    plt.close()
    print(f"📊 图表已生成: {CHARTS_DIR / 'progress_trend.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="summary.json 路径")
    parser.add_argument("--name", required=True, help="实验名（对应 output 目录名）")
    parser.add_argument("--method", default="", help="方法描述")
    parser.add_argument("--notes", default="", help="备注")
    parser.add_argument("--charts", action="store_true", help="同时生成图表")
    args = parser.parse_args()

    record(args.summary, args.name, args.method, args.notes)
    if args.charts:
        generate_charts()
