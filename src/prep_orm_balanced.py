#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/prep_orm_balanced.py — P1 牌一：判卷老师均衡重训数据准备（纯 CPU，仅标准库）。

背景：data/orm_train.json（src/label_orm_data.py 产物，20,415 条，正率 34.3%）
负样本占比过高。GradeSQL（arXiv:2509.01308，ACL 2026）实验给出 ORM 最优正负比
58-69%，当前 34.3% 明显低于该区间。本脚本产出三档均衡变体，供
scripts/train_orm_bal.slurm 用 DATA env 三档切换训练：

  bal1  1:1     负样本降采样到与正样本等量（seed 42）
  bal2  ≈60:40  正样本【整题复制 ×2】+ 负样本降采样（GradeSQL 区间内）
  bal3  ≈60:40  正样本【整题复制 ×3】，负样本全保留——纯上采样实现，对比 bal2 的
                「上采样+降采样」差异：不丢任何负样本，实际正率由原始负样本量决定
                （≈61.1%，仍在 58-69% 区间）

按题不泄漏保证：
  1. 上采样只做【整题复制】：同题全部正样本作为一组一起复制 m 份，复制体与原样本
     question_id 完全一致 → train_orm.py 的 question-level dev 划分（按 question_id
     切 5%）下，同题样本（含复制体）必然同侧，不可能跨 train/dev。
  2. 降采样只删样本、不移动样本，删除不会制造跨侧同题样本。
  3. 负样本降采样保底：每题至少保留 1 条负样本（若该题原有负样本）→ 三档变体的
     题目集合与源数据完全一致（1026 题），train_orm.py --seed 42 下三档 dev 划分
     相同，eval_metrics 可直接横向对比。
  4. 样本字段原样保留（messages/label/candidate_sql/...），仅增删条目。
  5. 落盘前内置校验（verify_variant）：正样本逐题倍率一致、负样本只减不增、
     无凭空出现的 (question_id, candidate_sql)。

用法（HPC 登录节点 CPU，几秒级）：
  envs/reasoning3b/bin/python src/prep_orm_balanced.py \
      --in data/orm_train.json --out-dir data \
      --stats-out data/orm_bal_stats.json --seed 42

产物：data/orm_train_bal1.json / bal2.json / bal3.json + 统计 JSON + 控制台对照表。
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = PROJECT_ROOT / "data" / "orm_train.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data"
DEFAULT_STATS_OUT = PROJECT_ROOT / "data" / "orm_bal_stats.json"

# variant -> (pos_mult, target_pos_frac)
# pos_mult: 正样本整题复制倍数；target_pos_frac: 目标正样本占比（60% = 60:40）。
# bal3 目标 60% 所需负样本 14020 > 现有 13405 → 负样本全保留，实际 ≈61.1%。
VARIANTS: Dict[str, Tuple[int, float]] = {
    "bal1": (1, 0.50),
    "bal2": (2, 0.60),
    "bal3": (3, 0.60),
}


def load_samples(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    for s in samples:
        label = int(s["label"])
        if label not in (0, 1):
            raise ValueError(
                f"非法 label={s['label']!r}（question_id={s.get('question_id')}），"
                f"数据疑似损坏，拒绝继续")
        s["label"] = label
    return samples


def upsample_pos_by_question(pos_samples: List[Dict[str, Any]],
                             pos_mult: int) -> List[Dict[str, Any]]:
    """整题复制：同题全部正样本作为一组一起复制 pos_mult 份（含原件 1 份）。

    复制体复用原 dict 引用（仅 JSON 落盘，无后续原地修改），保证复制体与原件
    question_id/candidate_sql/label 完全一致——这是「同题样本仍同侧」的结构保证。
    """
    if pos_mult <= 0:
        raise ValueError(f"pos_mult 必须 >=1，实际 {pos_mult}")
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for s in pos_samples:
        groups[int(s["question_id"])].append(s)
    out: List[Dict[str, Any]] = []
    for qid in sorted(groups):
        g = groups[qid]
        for _ in range(pos_mult):
            out.extend(g)
    return out


def downsample_neg_with_floor(neg_samples: List[Dict[str, Any]], n_keep: int,
                              rng: random.Random) -> List[Dict[str, Any]]:
    """负样本全局随机降采样，保底每题至少保留 1 条（该题原有负样本时）。

    保底意义：题目集合不变 → 三档变体与源数据题目完全一致 → train_orm.py 按题
    5% dev（seed 42）在三档上划分出相同 dev 集，eval_metrics 可直接对比。
    """
    if n_keep >= len(neg_samples):
        return list(neg_samples)
    by_q: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for s in neg_samples:
        by_q[int(s["question_id"])].append(s)
    qids = sorted(by_q)
    if n_keep < len(qids):
        raise RuntimeError(
            f"负样本预算 {n_keep} 不足以覆盖保底 {len(qids)} 题 ×1 条，"
            f"正负比目标无法实现")
    keep: List[Dict[str, Any]] = [by_q[q].pop() for q in qids]  # 保底部分
    pool = [s for g in by_q.values() for s in g]                # 剩余池
    keep.extend(rng.sample(pool, n_keep - len(qids)))
    return keep


def verify_variant(source: List[Dict[str, Any]], variant: List[Dict[str, Any]],
                   pos_mult: int) -> str:
    """落盘前校验（返回 "pass"，违反则抛 RuntimeError）：
    1. 变体每个样本都能在源数据找到同 (question_id, candidate_sql)（无凭空样本）；
    2. 正样本按题整倍复制：每 (qid, sql) 正样本数 == 源数 × pos_mult（逐题一致）；
    3. 负样本只减不增：每 (qid, sql) 负样本数 <= 源数。
    """
    def key(s: Dict[str, Any]):
        return (int(s["question_id"]), str(s["candidate_sql"]))

    src_pos: Counter = Counter()
    src_neg: Counter = Counter()
    for s in source:
        (src_pos if int(s["label"]) == 1 else src_neg)[key(s)] += 1
    var: Counter = Counter(key(s) for s in variant)

    for k in var:
        if k not in src_pos and k not in src_neg:
            raise RuntimeError(f"[verify] 凭空样本（源数据不存在）: {k}")
    for k, c in src_pos.items():
        if var[k] != c * pos_mult:
            raise RuntimeError(
                f"[verify] 正样本倍率不一致 {k}: 源 {c} → 变体 {var[k]}，"
                f"应为 ×{pos_mult} = {c * pos_mult}")
    for k, c in src_neg.items():
        if var[k] > c:
            raise RuntimeError(
                f"[verify] 负样本凭空增加 {k}: 源 {c} → 变体 {var[k]}")
    # 题目集合一致性（降采样保底 + 正样本全保留 ⇒ 题目集合不变）
    src_q = {int(s["question_id"]) for s in source}
    var_q = {int(s["question_id"]) for s in variant}
    if src_q != var_q:
        raise RuntimeError(
            f"[verify] 题目集合不一致: 源 {len(src_q)} 题 vs 变体 {len(var_q)} 题 "
            f"（差 {sorted(src_q ^ var_q)[:5]}{'...' if len(src_q ^ var_q) > 5 else ''}）")
    return "pass"


def build_variant(samples: List[Dict[str, Any]], pos_mult: int,
                  target_pos_frac: float,
                  rng: random.Random) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pos = [s for s in samples if int(s["label"]) == 1]
    neg = [s for s in samples if int(s["label"]) == 0]

    pos_final = upsample_pos_by_question(pos, pos_mult)
    neg_target = round(len(pos_final) * (1.0 - target_pos_frac) / target_pos_frac)
    neg_final = downsample_neg_with_floor(neg, neg_target, rng)

    out = pos_final + neg_final
    rng.shuffle(out)  # 打散复制块（split 按 question_id，shuffle 不影响不泄漏性）

    check = verify_variant(samples, out, pos_mult)
    n_q = len({int(s["question_id"]) for s in out})
    n_pos = len(pos_final)
    n_neg = len(neg_final)
    return out, {
        "check": check,
        "n_samples": len(out),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_ratio": round(n_pos / len(out), 4),
        "pos_mult": pos_mult,
        "target_pos_frac": target_pos_frac,
        "neg_target": neg_target,
        "neg_kept": n_neg,
        "neg_available": len(neg),
        "neg_dropped": len(neg) - n_neg,
        "n_questions": n_q,
    }


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(
        description="P1 牌一：ORM 均衡重训数据准备（bal1/bal2/bal3 三档，纯 CPU）")
    ap.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--stats-out", default=str(DEFAULT_STATS_OUT))
    ap.add_argument("--variants", default="bal1,bal2,bal3",
                    help="逗号分隔，默认全量 bal1,bal2,bal3")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            print(f"FATAL: 未知变体 '{v}'，可选 {sorted(VARIANTS)}", file=sys.stderr)
            return 2

    samples = load_samples(in_path)
    n = len(samples)
    pos0 = sum(1 for s in samples if int(s["label"]) == 1)
    neg0 = n - pos0
    q0 = len({int(s["question_id"]) for s in samples})
    print(f"[prep] 源数据 {in_path}: 样本 {n} | 正 {pos0} / 负 {neg0} | "
          f"正率 {pos0 / n * 100:.1f}% | 题目 {q0}")

    # 每档独立 rng（seed + 序号），单档可复现、互不干扰
    stats: Dict[str, Any] = {
        "meta": {
            "created_by": "src/prep_orm_balanced.py",
            "source": str(in_path),
            "seed": args.seed,
            "source_samples": n,
            "source_pos": pos0,
            "source_neg": neg0,
            "source_pos_ratio": round(pos0 / n, 4),
            "source_questions": q0,
            "leakage_note": ("上采样按题整块复制（同题正样本组一起复制 m 份，"
                             "question_id 不变→按题 dev 划分必然同侧）；降采样只删"
                             "不增且每题保底 1 条负样本→三档题目集合一致、dev 划分"
                             "一致（train_orm.py --seed 42）"),
        },
        "variants": {},
    }

    print("-" * 88)
    print(f"{'variant':<8}{'样本数':>8}{'正/负':>16}{'正率':>8}{'正×':>5}"
          f"{'负保留/可用':>14}{'题目':>7}{'校验':>6}")
    for i, vname in enumerate(variants):
        rng = random.Random(args.seed + i)
        out, vstat = build_variant(samples, VARIANTS[vname][0],
                                   VARIANTS[vname][1], rng)
        out_file = out_dir / f"orm_train_{vname}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        vstat["file"] = str(out_file)
        stats["variants"][vname] = vstat
        print(f"{vname:<8}{vstat['n_samples']:>8}"
              f"{str(vstat['n_pos']) + '/' + str(vstat['n_neg']):>16}"
              f"{vstat['pos_ratio'] * 100:>7.1f}%{vstat['pos_mult']:>4}×"
              f"{str(vstat['neg_kept']) + '/' + str(vstat['neg_available']):>14}"
              f"{vstat['n_questions']:>7}{vstat['check']:>6}")
    print("-" * 88)

    stats_path = Path(args.stats_out)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    produced = ", ".join(str(out_dir / ("orm_train_" + v + ".json"))
                         for v in variants)
    print(f"[prep] 完成：{produced}")
    print(f"[prep] 统计：{stats_path}")
    print("[prep] 提交训练：sbatch --export=DATA=bal1|bal2|bal3 "
          "scripts/train_orm_bal.slurm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
