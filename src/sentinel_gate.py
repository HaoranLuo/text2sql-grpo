#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""sentinel_gate.py —— Minority Sentinel 式高置信翻转门控（纯 CPU，只复用缓存）。

依据：Minority Sentinel（arXiv 2606.29270）——flip safety > recovery volume；
DPC 经验：多数派错误的 56 题只挽回 12 题（21.4%）——只有高置信翻转才值得保留。

机制：
  1) 翻转事件 = arm_orm_grouphead（判卷终裁）选了非最大组的题
     （胜者组大小 < 该题全场最大组大小）。
  2) 置信度 conf(g) = P(Yes)_g × log(1 + size_g)。
     Δ = conf(判卷胜者组) − conf(MI-VAV 最大组)。
  3) 门控：Δ > τ 才保留翻转，否则回退到 MI-VAV 的大组胜者（arm_vav）。
     τ ∈ {0.5, 1.0, 1.5, 2.0} 网格扫描。

输入（全部为已有缓存，不重新生成、不动判卷模型）：
  outputs/bird_select_ormbird_bird_bal2/work/prep.json         分组 + arm_vav + 执行签名
  outputs/bird_select_ormbird_bird_bal2/work/orm_scores.json   组代表 → P(Yes)
  outputs/bird_select_ormbird_bird_bal2/items_arm_orm_grouphead.json  判卷终裁逐题（sanity）
  outputs/bird_select_ormbird_bird_bal2/{arm_vav,arm_orm_grouphead}/eval_result_dev.json  逐题官方 EX

输出：
  <out_dir>/items_sentinel_tau{τ}.json            逐题（含门控元数据）
  <out_dir>/sentinel_tau{τ}/predict_dev.json      官方格式预测
  <out_dir>/sentinel_tau{τ}/eval_result_dev.json  官方评估器逐题结果
  <out_dir>/work/official_sentinel_tau{τ}.log     评估器日志
  <out_dir>/sentinel_gate_analysis.json           翻转统计 + τ 扫描 + 精度/恢复量曲线

用法（HPC，cpu6348）：
  python src/sentinel_gate.py \
      --base-dir outputs/bird_select_ormbird_bird_bal2 \
      --out-dir outputs/bird_sentinel_gate \
      --db-root data/bird/bird_dev/dev_20240627/dev_databases \
      --data-json data/bird/bird_dev/dev_20240627/dev.json \
      --ground-truth data/bird/bird_dev/dev_20240627/dev.sql \
      --evaluator-py tmp_idea_research/finer-sql/evaluation/official_bird_evaluation/evaluation_bird_ex.py \
      --num-cpus 12
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TAUS_DEFAULT = [0.5, 1.0, 1.5, 2.0]

# ---------------------------------------------------------------------------
# 预注册（先于任何结果查看写入分析文件）：
#   PRIMARY   : 最优 τ 的官方 EX total > 60.37（arm_orm_grouphead 官方分）判正向。
#   SECONDARY : 最优 τ 下翻转精度（保留翻转题中判卷胜者判对比例）目标 > 85%。
#   τ 选择    : 官方 EX total 最大者；平票取更大 τ（更保守）。
# ---------------------------------------------------------------------------
PREREGISTRATION = {
    "primary": "max over tau of official EX total > 60.37 -> positive",
    "secondary": "flip precision at chosen tau > 0.85",
    "tau_selection": "argmax official EX total; tie -> larger tau (more conservative)",
    "baseline_orm_grouphead_ex": 60.37,
    "baseline_vav_ex": 56.26,
    "registered_before": "any sentinel-gate evaluation run",
}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def conf(size: int, score: float) -> float:
    """门控置信度：P(Yes) × log(1 + size)。"""
    return float(score) * math.log1p(int(size))


def run_official_evaluator(predict_path: Path, args: argparse.Namespace,
                           work: Path, arm: str) -> Dict[str, Any]:
    """调用 FINER 官方评估器并解析 accuracy 表（与 bird_select.run_official_evaluator 同口径）。"""
    db_root = str(Path(args.db_root).resolve()).rstrip("/") + "/"
    cmd = [
        sys.executable, str(args.evaluator_py),
        "--db_root_path", db_root,
        "--predicted_sql_json_path", str(predict_path),
        "--data_mode", "dev",
        "--ground_truth_sql_path", str(args.ground_truth),
        "--num_cpus", str(args.num_cpus),
        "--mode_predict", "gpt",
        "--diff_json_path", str(args.data_json),
        "--meta_time_out", str(args.meta_time_out),
    ]
    print(f"[sentinel] official eval [{arm}]: {' '.join(cmd)}", file=sys.stderr)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT,
                          timeout=7200)
    log_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path = work / f"official_{arm}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"官方评估器退出码 {proc.returncode} [{arm}]，日志: {log_path}")
    m_acc = re.search(r"accuracy\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", log_text)
    m_cnt = re.search(r"count\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", log_text)
    if not m_acc:
        raise RuntimeError(f"官方评估器输出中未找到 accuracy 表 [{arm}]，日志: {log_path}")
    out = {
        "arm": arm,
        "returncode": proc.returncode,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "simple": float(m_acc.group(1)), "moderate": float(m_acc.group(2)),
        "challenging": float(m_acc.group(3)), "total": float(m_acc.group(4)),
        "counts": {"simple": int(m_cnt.group(1)), "moderate": int(m_cnt.group(2)),
                   "challenging": int(m_cnt.group(3)), "total": int(m_cnt.group(4))}
        if m_cnt else None,
        "log": str(log_path),
        "eval_result_json": str(predict_path.parent / "eval_result_dev.json"),
    }
    print(f"[sentinel] official [{arm}]: simple={out['simple']} "
          f"moderate={out['moderate']} challenging={out['challenging']} "
          f"total={out['total']}", file=sys.stderr)
    return out


def load_per_question_res(eval_result_json: Path) -> List[int]:
    """官方评估器逐题 res（0/1），按 question_id 排序。"""
    data = json.loads(eval_result_json.read_text(encoding="utf-8"))
    data = sorted(data, key=lambda r: int(r["question_id"]))
    return [int(r["res"]) for r in data]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Minority Sentinel 高置信翻转门控（CPU）")
    ap.add_argument("--base-dir", default="outputs/bird_select_ormbird_bird_bal2")
    ap.add_argument("--out-dir", default="outputs/bird_sentinel_gate")
    ap.add_argument("--db-root", default="data/bird/bird_dev/dev_20240627/dev_databases")
    ap.add_argument("--data-json", default="data/bird/bird_dev/dev_20240627/dev.json")
    ap.add_argument("--ground-truth", default="data/bird/bird_dev/dev_20240627/dev.sql")
    ap.add_argument("--evaluator-py",
                    default="tmp_idea_research/finer-sql/evaluation/official_bird_evaluation/evaluation_bird_ex.py")
    ap.add_argument("--num-cpus", type=int, default=12)
    ap.add_argument("--meta-time-out", type=float, default=30.0)
    ap.add_argument("--taus", type=float, nargs="+", default=TAUS_DEFAULT)
    ap.add_argument("--skip-eval", action="store_true",
                    help="只重建 items/分析，不跑官方评估器（复用已有 eval_result_dev.json）")
    args = ap.parse_args(argv)

    base = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    prep = json.loads((base / "work" / "prep.json").read_text(encoding="utf-8"))
    qcs = prep["items"]
    n_questions = len(qcs)
    score_entries = json.loads((base / "work" / "orm_scores.json").read_text(
        encoding="utf-8"))["entries"]
    score_map: Dict[Tuple[int, int], float] = {
        (int(e["qi"]), int(e["ei"])): float(e["score"]) for e in score_entries}
    items_orm = json.loads((base / "items_arm_orm_grouphead.json").read_text(
        encoding="utf-8"))
    assert len(items_orm) == n_questions

    res_vav = load_per_question_res(base / "arm_vav" / "eval_result_dev.json")
    res_orm = load_per_question_res(base / "arm_orm_grouphead" / "eval_result_dev.json")
    assert len(res_vav) == len(res_orm) == n_questions
    acc_vav = 100.0 * sum(res_vav) / n_questions
    acc_orm = 100.0 * sum(res_orm) / n_questions
    print(f"[sentinel] 逐题官方 EX 复核: arm_vav={acc_vav:.2f} (官方 56.26) "
          f"arm_orm_grouphead={acc_orm:.2f} (官方 60.37)", file=sys.stderr)

    # ------------------------------------------- 重建判卷胜者 + 识别翻转事件
    n_sanity_mismatch = 0
    n_vav_not_ranked = 0
    n_vav_not_max = 0
    gate_rows: List[Dict[str, Any]] = []
    for qc in qcs:
        qi = qc["qi"]
        di = int(qc["dataset_index"])
        entries = qc["entries"]
        ranked = qc["groups_meta"]["ranked"]
        rec_vav = qc["arm_vav"]
        it = items_orm[di]
        assert int(it["dataset_index"]) == di

        if not ranked:
            # 判卷回退题（全池 1 题 fallback_maj）：无组可门控，原样保留
            gate_rows.append({
                "qi": qi, "dataset_index": di, "db_id": qc["db_id"],
                "question": qc["question"], "gold_sql": qc["gold_sql"],
                "difficulty": qc["difficulty"],
                "orm_text": it["predicted_sql"], "vav_text": rec_vav.get("text") or "SELECT 1",
                "orm_group_size": 0, "orm_score": None, "max_group_size": 0,
                "vav_group_size": 0, "vav_score": None,
                "conf_w": None, "conf_m": None, "delta": None,
                "is_flip_event": False, "ungated_reason": "orm_fallback_no_ranked_groups",
                "res_orm": res_orm[di], "res_vav": res_vav[di],
            })
            continue

        # 与 bird_select.phase_final 完全同口径重建 arm_orm_grouphead 胜者：
        # argmax (size × P(Yes), size, str(key))
        best, best_key, best_score = None, None, None
        scored: Dict[str, float] = {}
        for kg in ranked:
            s = score_map.get((qi, int(kg["rep_ei"])), 0.0)
            scored[kg["key"]] = s
            k = (float(kg["size"]) * s, int(kg["size"]), kg["key"])
            if best_key is None or k > best_key:
                best, best_key, best_score = kg, k, s
        orm_text = entries[int(best["rep_ei"])]["sql_text"]
        if orm_text != it["predicted_sql"]:
            n_sanity_mismatch += 1

        max_size = max(int(kg["size"]) for kg in ranked)
        orm_size = int(best["size"])
        is_flip = orm_size < max_size

        # 多数派 = MI-VAV 胜者组（按 group_key 对齐 ranked 取判卷分）
        vav_key = rec_vav.get("group_key")
        vav_group = next((kg for kg in ranked if kg["key"] == vav_key), None)
        if vav_group is None:
            n_vav_not_ranked += 1
            vav_size = int(rec_vav.get("group_size") or 0)
            vav_score = 0.0  # 无判卷分（如 excluded 签名）→ 置信度下界 0
        else:
            vav_size = int(vav_group["size"])
            vav_score = score_map.get((qi, int(vav_group["rep_ei"])), 0.0)
            if vav_size != max_size:
                n_vav_not_max += 1

        conf_w = conf(orm_size, best_score)
        conf_m = conf(vav_size, vav_score)
        gate_rows.append({
            "qi": qi, "dataset_index": di, "db_id": qc["db_id"],
            "question": qc["question"], "gold_sql": qc["gold_sql"],
            "difficulty": qc["difficulty"],
            "orm_text": orm_text, "vav_text": rec_vav.get("text") or "SELECT 1",
            "orm_group_key": best["key"], "vav_group_key": vav_key,
            "orm_group_size": orm_size, "orm_score": best_score,
            "max_group_size": max_size,
            "vav_group_size": vav_size, "vav_score": vav_score,
            "conf_w": round(conf_w, 6), "conf_m": round(conf_m, 6),
            "delta": round(conf_w - conf_m, 6),
            "is_flip_event": bool(is_flip),
            "res_orm": res_orm[di], "res_vav": res_vav[di],
        })

    if n_sanity_mismatch:
        print(f"[sentinel] WARN 重建胜者与 items_arm_orm_grouphead 不一致: "
              f"{n_sanity_mismatch} 题", file=sys.stderr)
    flips = [r for r in gate_rows if r["is_flip_event"]]
    print(f"[sentinel] 翻转事件: {len(flips)}/{n_questions} "
          f"(vav_not_ranked={n_vav_not_ranked}, vav_not_max={n_vav_not_max})",
          file=sys.stderr)

    # 翻转事件四象限（按逐题官方 EX）
    quad = Counter()
    for r in flips:
        quad[(r["res_vav"], r["res_orm"])] += 1
    flip_stats = {
        "n_flip_events": len(flips),
        "quadrant_(res_vav,res_orm)": {f"{a},{b}": c for (a, b), c in sorted(quad.items())},
        "recovery_if_all_kept": quad.get((0, 1), 0),
        "harm_if_all_kept": quad.get((1, 0), 0),
        "neutral_both_wrong": quad.get((0, 0), 0),
        "harmless_both_right": quad.get((1, 1), 0),
        "orm_acc_on_flips_if_all_kept": (
            round(sum(r["res_orm"] for r in flips) / len(flips), 6) if flips else None),
    }

    # ------------------------------------------------------------ τ 扫描裁决
    per_tau_rows: Dict[float, List[Dict[str, Any]]] = {}
    for tau in args.taus:
        rows: List[Dict[str, Any]] = []
        for r in gate_rows:
            di = r["dataset_index"]
            it = items_orm[di]
            if r["is_flip_event"] and not (r["delta"] > tau):
                predicted = r["vav_text"]
                decision = "revert_to_vav"
                winner_source = "sentinel_revert"
                w_size, w_score = r["vav_group_size"], r["vav_score"]
            else:
                predicted = r["orm_text"]
                decision = ("keep_flip" if r["is_flip_event"] else "no_flip")
                winner_source = ("sentinel_keep" if r["is_flip_event"]
                                 else it.get("winner_source"))
                w_size, w_score = r["orm_group_size"], r["orm_score"]
            if not predicted:
                predicted = "SELECT 1"
            row = {k: it[k] for k in (
                "dataset_index", "db_id", "question", "gold_sql", "difficulty",
                "num_candidates", "num_unique_candidates", "num_instances") if k in it}
            row.update({
                "predicted_sql": predicted,
                "empty_winner": bool(it.get("empty_winner")),
                "winner_source": winner_source,
                "winner_votes": w_size or 0,
                "winner_group_size": w_size or 0,
                "orm_score": w_score,
                "gate_tau": tau,
                "gate_is_flip_event": bool(r["is_flip_event"]),
                "gate_conf_w": r["conf_w"], "gate_conf_m": r["conf_m"],
                "gate_delta": r["delta"], "gate_decision": decision,
            })
            rows.append(row)
        per_tau_rows[tau] = rows

    # ------------------------------------------------- 写预测 + 跑官方评估器
    official: Dict[str, Any] = {}
    per_tau_res: Dict[float, List[int]] = {}
    for tau in args.taus:
        arm = f"sentinel_tau{tau:g}"
        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / f"items_{arm}.json", per_tau_rows[tau])
        pred_list = [[r["question"], f"{r['predicted_sql']}\t----- bird -----\t{r['db_id']}"]
                     for r in per_tau_rows[tau]]
        _write_json(arm_dir / "predict_dev.json", pred_list)
        if not args.skip_eval:
            official[arm] = run_official_evaluator(arm_dir / "predict_dev.json", args, work, arm)
        per_tau_res[tau] = load_per_question_res(arm_dir / "eval_result_dev.json")

    # ------------------------------------------------------------------ 分析
    # 逐 τ：保留翻转数、翻转精度、挽回/误翻、净增益（逐题官方 res 精确合成 + 官方总分复核）
    tau_table: List[Dict[str, Any]] = []
    for tau in args.taus:
        kept = [r for r in flips if r["delta"] > tau]
        reverted = [r for r in flips if not (r["delta"] > tau)]
        n_kept = len(kept)
        rec_kept = sum(1 for r in kept if r["res_vav"] == 0 and r["res_orm"] == 1)
        harm_kept = sum(1 for r in kept if r["res_vav"] == 1 and r["res_orm"] == 0)
        orm_right_kept = sum(r["res_orm"] for r in kept)
        prec = (orm_right_kept / n_kept) if n_kept else None
        net = rec_kept - harm_kept
        composed_total = 100.0 * sum(per_tau_res[tau]) / n_questions
        arm = f"sentinel_tau{tau:g}"
        entry = {
            "tau": tau, "arm": arm,
            "kept_flips": n_kept, "reverted_flips": len(reverted),
            "flip_precision_orm_correct": round(prec, 6) if prec is not None else None,
            "recoveries_kept": rec_kept, "harms_kept": harm_kept,
            "net_gain_questions": net,
            "composed_ex_total": round(composed_total, 4),
            "composed_ex_delta_vs_60.37": round(composed_total - 60.37, 4),
        }
        if arm in official:
            entry["official_ex"] = {k: official[arm][k] for k in
                                    ("simple", "moderate", "challenging", "total")}
            entry["official_ex_delta_vs_60.37"] = round(official[arm]["total"] - 60.37, 4)
            # 一致性校验：逐题合成 EX 必须等于官方 EX
            assert abs(composed_total - official[arm]["total"]) < 0.05, (
                f"τ={tau}: 合成 {composed_total} != 官方 {official[arm]['total']}")
        tau_table.append(entry)

    # 精度-恢复量曲线（全阈值扫描，逐题 res 精确）
    # 组合恒等式：EX(th) = acc_orm + (100/n) * Σ_{回退题} (res_vav − res_orm)
    # （非翻转题与保留翻转题都用判卷胜者文本，回退题用 MI-VAV 文本）
    deltas = sorted({r["delta"] for r in flips})
    curve: List[Dict[str, Any]] = []
    for th in [float("-inf")] + deltas:
        kept = [r for r in flips if r["delta"] > th]
        reverted = [r for r in flips if not (r["delta"] > th)]
        ex = acc_orm + 100.0 * sum(r["res_vav"] - r["res_orm"] for r in reverted) / n_questions
        if not kept:
            curve.append({"tau": (round(th, 6) if th != float("-inf") else None),
                          "kept": 0, "precision": None, "recoveries": 0, "harms": 0,
                          "net": 0, "ex_total": round(ex, 4)})
            continue
        rec = sum(1 for r in kept if r["res_vav"] == 0 and r["res_orm"] == 1)
        harm = sum(1 for r in kept if r["res_vav"] == 1 and r["res_orm"] == 0)
        prec = sum(r["res_orm"] for r in kept) / len(kept)
        curve.append({"tau": (round(th, 6) if th != float("-inf") else None),
                      "kept": len(kept), "precision": round(prec, 6),
                      "recoveries": rec, "harms": harm, "net": rec - harm,
                      "ex_total": round(ex, 4)})

    # 选 τ：官方（或合成）EX 最大；平票取大 τ
    def _ex_of(e: Dict[str, Any]) -> float:
        return e.get("official_ex", {}).get("total", e["composed_ex_total"])
    best_ex = max(_ex_of(e) for e in tau_table)
    best_entry = max((e for e in tau_table if abs(_ex_of(e) - best_ex) < 1e-9),
                     key=lambda e: e["tau"])
    verdict = {
        "chosen_tau": best_entry["tau"],
        "chosen_ex_total": _ex_of(best_entry),
        "primary_positive": bool(_ex_of(best_entry) > 60.37),
        "secondary_flip_precision_target_0.85": (
            best_entry["flip_precision_orm_correct"] is not None
            and best_entry["flip_precision_orm_correct"] > 0.85),
    }

    analysis = {
        "preregistration": PREREGISTRATION,
        "method": ("conf(g)=P(Yes)*log(1+size); 翻转事件=判卷胜者组大小<全场最大组; "
                   "Δ=conf(判卷胜者)-conf(MI-VAV大组); Δ>τ 保留翻转否则回退 arm_vav"),
        "n_questions": n_questions,
        "sanity": {
            "winner_reconstruct_mismatch": n_sanity_mismatch,
            "vav_group_not_in_ranked": n_vav_not_ranked,
            "vav_group_not_max_size": n_vav_not_max,
            "per_question_acc_vav": round(acc_vav, 4),
            "per_question_acc_orm": round(acc_orm, 4),
        },
        "flip_stats": flip_stats,
        "tau_table": tau_table,
        "precision_volume_curve": curve,
        "verdict": verdict,
        "paths": {
            "out_dir": str(out_dir),
            "base_dir": str(base),
        },
    }
    _write_json(out_dir / "sentinel_gate_analysis.json", analysis)

    print("\n=== Sentinel Gate τ 扫描（官方 EX）===", file=sys.stderr)
    for e in tau_table:
        print(json.dumps(e, ensure_ascii=False), file=sys.stderr)
    print(f"[sentinel] verdict: {json.dumps(verdict, ensure_ascii=False)}", file=sys.stderr)
    print(f"[sentinel] DONE -> {out_dir / 'sentinel_gate_analysis.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
