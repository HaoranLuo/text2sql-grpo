"""Generate tmp_idea_research/retriever_report.md from eval/train artifacts (P1-2).

Reads eval_summary.json files (baseline + fine-tuned) and the training metrics
jsonl, then writes the schema-retriever report. Tolerates missing trained-model
artifacts (marks them as pending) so it can also be run pre-training.

Run on the HPC from $BASE:
    envs/reasoning3b/bin/python src/write_retriever_report.py
"""

import argparse
import json
import os
import sys

HPC_BASE = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b"
KS = [3, 5, 8]


def load_summary(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def fmt_row(s, tag):
    if not s:
        return f"| {tag} | (pending) | - | - | - | - | - | - | - |"
    cells = [tag, str(s.get("n_questions", "?"))]
    for k in KS:
        cells.append(f"{s.get(f'recall@{k}', 0):.3f}")
    for k in KS:
        cells.append(f"{s.get(f'p@{k}', 0):.3f}")
    cells.append(f"{s.get('mrr', 0):.3f}")
    return "| " + " | ".join(cells) + " |"


def train_stats(metrics_path):
    if not os.path.exists(metrics_path):
        return None
    losses, steps = [], []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            losses.append(d["loss"])
            steps.append(d["global_step"])
    if not losses:
        return None
    return {
        "n_logs": len(losses),
        "steps": max(steps),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
    }


def data_stats():
    stats = {}
    for name, path in [("bird", f"{HPC_BASE}/data/retriever_bird.jsonl"),
                       ("spider", f"{HPC_BASE}/data/retriever_spider.jsonl")]:
        n = 0
        pos = 0
        tables = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    n += 1
                    pos += len(r["related_tables"])
                    tables += len(r["schema_items"])
            stats[name] = {"records": n, "pos": pos, "tables": tables,
                           "avg_pos": pos / max(n, 1), "avg_tables": tables / max(n, 1)}
        except Exception:
            stats[name] = None
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{HPC_BASE}/tmp_idea_research/retriever_report.md")
    ap.add_argument("--base-dir", default=f"{HPC_BASE}/outputs")
    ap.add_argument("--ckpt-dir", default=f"{HPC_BASE}/checkpoints/schema_retriever")
    args = ap.parse_args()

    bird100_base = load_summary(f"{args.base_dir}/schema_retriever_eval_bird100_base/eval_summary.json")
    bird100_ft = load_summary(f"{args.base_dir}/schema_retriever_eval_bird100/eval_summary.json")
    birdfull_base = load_summary(f"{args.base_dir}/schema_retriever_eval_birdfull_base/eval_summary.json")
    spider_base = load_summary(f"{args.base_dir}/schema_retriever_eval_spiderdev_base/eval_summary.json")
    spider_ft = load_summary(f"{args.base_dir}/schema_retriever_eval_spiderdev/eval_summary.json")
    tstats = train_stats(f"{args.ckpt_dir}/train_metrics.jsonl")
    dstats = data_stats()

    bird100_ft = bird100_ft or load_summary(f"{args.base_dir}/schema_retriever_eval_bird100/eval_summary.json")

    header = "| split | n | recall@3 | recall@5 | recall@8 | P@3 | P@5 | P@8 | MRR |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|"
    rows = [
        fmt_row(bird100_base, "BIRD dev 100 - base (untrained)"),
        fmt_row(bird100_ft, "BIRD dev 100 - fine-tuned"),
        fmt_row(birdfull_base, "BIRD dev full 1534 - base"),
        fmt_row(spider_base, "Spider dev 1034 - base"),
        fmt_row(spider_ft, "Spider dev 1034 - fine-tuned"),
    ]

    tsec = ""
    if tstats:
        tsec = (
            f"- logs: {tstats['n_logs']} (last global step {tstats['steps']})\n"
            f"- loss first {tstats['loss_first']:.4f} -> last {tstats['loss_last']:.4f} "
            f"(min {tstats['loss_min']:.4f})\n"
        )
    else:
        tsec = "- (training metrics not found yet)\n"

    dsec = []
    for name in ("bird", "spider"):
        s = dstats[name]
        if s:
            dsec.append(f"- {name.upper()}: {s['records']} records, "
                        f"avg related tables {s['avg_pos']:.2f}, avg db tables {s['avg_tables']:.2f}")
        else:
            dsec.append(f"- {name.upper()}: (data not found)")
    dsec = "\n".join(dsec)

    report = f"""# Schema Retriever (P1-2) 报告 — 0.6B 双塔 HN-SupCon

> 目标：生成时只喂相关表结构（BIRD/Spider），替换全库 schema 直喂。
> 方法基线：LitE-SQL schema_retriever（HN-SupCon 难负对比 + Qwen3-Embedding-0.6B），
> 本线改为 **表级粒度**（LitE-SQL 原为列级）。
> 报告由 `src/write_retriever_report.py` 自动生成，人工修订版见文件尾注。

## 1. 结论速览

- 训练数据：BIRD train 9428 题（69 库）+ Spider train 8659 题（166 库），
  正例 = gold SQL 引用表（sqlglot 解析 + 表名字典匹配，别名/大小写安全），
  难负 = 同库其余表。
- 微调：Qwen3-Embedding-0.6B 全参，HN-SupCon（temp 0.07, hard-neg threshold 0.1,
  too_hard=True），3 epochs，bs 32，lr 3e-5，A40 单卡。
- 评测：BIRD dev 100 题（seed 42 抽样）+ BIRD dev 全量 1534 + Spider dev 1034，
  gold 表集合 recall@k / P@k, k=3/5/8。

## 2. 评测结果

{header}
{chr(10).join(rows)}

（recall@8 在 BIRD dev 上受“平均每库 7 表”天花板影响：k>=库表数时恒为 1.0，
该指标主要看 Spider 与 BIRD 大库。）

## 3. 训练数据构造

{dsec}

- schema item 文本 = `<table>表名</table> <columns>列(类型),...</columns>
  <column descriptions>列: 注释;...</column descriptions>`（BIRD 注释取自
  database_description；Spider 无注释仅类型）。
- 展开样本 = 每个 (题目, 相关表) 一条，negatives = 同库全部非相关表（collate 时
  每条随机采 ≤10 个，LitE-SQL 同款）。
- 输出：`data/retriever_bird.jsonl`, `data/retriever_spider.jsonl`，
  评测集 `data/retriever_eval_bird.jsonl`, `data/retriever_eval_spider.jsonl`。

## 4. 训练

- loss = LitE-SQL `HardNegativeSuperConLoss` 逐行复刻：正例 sim 进 logsumexp 分子；
  难负 = sim ≥ sim_pos − 0.1 的负例进 logsumexp 分母，无难负时退化为最硬负例。
- 池化 = last-token pool（左 padding 感知），bf16 autocast + fp32 主权重 +
  gradient checkpointing，全参微调（lm_head 冻结）。
- 耗时预算：~31.4k 样本 × 3 epochs ≈ 2.9k steps，A40 预计 1-2.5h。
{tsec}

## 5. 生成时推理接入方案（未实现，待训练验收后接线）

1. **表级检索**：对每道题，用微调模型编码 query（指令格式同训练）与所在库全部
   表级 schema item，余弦排序取 top-k（k=8 起步，按库表数自适应，min(k, n_tables)）。
2. **喂生成器**：把 top-k 表的 DDL/列清单替换现有“全库 schema 直喂”块；保留
   表注释；必要时附 “related tables: [...]” 提示行。改动点 = 现有
   `bird_select.py` / eval 链的 schema 组装函数，先加开关（`--schema-mode
   full|retriever_topk`）A/B。
3. **降级策略**：生成 SQL 若引用未检索到的表或执行报 missing table，自动重试
   一次全库 schema 喂入（召回兜底，成本低）。
4. **缓存**：每库表向量离线预编码存 `outputs/schema_retriever_emb/<db_id>.pt`，
   推理进程只算 query 编码（~ms 级）。
5. **验收口径**：BIRD dev 官方 EX 全量 1034 对比（retriever top-k vs 全库），
   期望 EX 持平或略升、prompt 长度显著下降；另记录 recall@k 与 EX 的相关性。

## 6. 文件清单

- 数据：`src/prep_schema_retriever_data.py` -> `data/retriever_*.jsonl`
- 训练：`src/train_schema_retriever.py` (+ `src/retriever_common.py`)
- 评测：`src/eval_schema_retriever.py`
- 作业：`scripts/train_retriever.slurm`（aiaca40/1a40, 12h），
  `scripts/train_retriever_smoke.slurm`（gpudebug 冒烟）
- 权重：`checkpoints/schema_retriever/final`；基座 `models/Qwen3-Embedding-0.6B`
- 评测产物：`outputs/schema_retriever_eval_*`

---
*人工尾注（待补）：与基座对比提升幅度；异常样本复核；正式接入生成的 EX 对比。*
"""
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[report] wrote {args.out}")


if __name__ == "__main__":
    main()
