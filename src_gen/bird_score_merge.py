#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5 源池 ORM 组代表分合并：4 源组代表分按 (qi, prompt) 复用 + RFT 新组代表新打分。

背景：bird_select.py 的 score 阶段会全量重打所有 rankable 组代表（无增量逻辑）。
4 源池已打分产物 = outputs/bird_select_ormbird_bird_bal2/work/{orm_payloads.json,
orm_scores.json}。5 源池 prep 后组代表 prompt 中，与 4 源同 (question_id, prompt)
者分数必然相同（build_orm_prompt 确定性 + ORM 打分贪心 logprobs 确定性
（T=1.0, seed=0, max_tokens=1）），可直接复用；仅新增组代表（RFT 引入的新组/
新 rep 文本）需 GPU 新打分。本脚本分两个 phase：

  --phase partition（登录节点 CPU，reasoning3b）：
      读旧 orm_payloads/orm_scores（zip 对齐）建 (qi, prompt) → score 映射；
      读新 work/orm_payloads.json，划分 reuse / need；
      写 work/orm_scores_reuse.json、work/orm_scores_need.json、
      work/orm_reuse_stats.json；need 为空时直接写完整 work/orm_scores.json。

  --phase score-gpu（gpudebug，vllmenv，slurm 内）：
      读 work/orm_scores_need.json，用 orm_selection.VllmScorer 同参数打分
      （与 bird_select.py --phase score 完全一致的 Namespace）；
      与 reuse 合并后按新 orm_payloads.json 顺序写完整 work/orm_scores.json。

用法：
  envs/reasoning3b/bin/python src/bird_score_merge.py --phase partition \
      --old-work outputs/bird_select_ormbird_bird_bal2/work \
      --new-work outputs/bird_select_5src/work
  envs/vllmenv/bin/python src/bird_score_merge.py --phase score-gpu \
      --new-work outputs/bird_select_5src/work \
      --orm-checkpoint checkpoints/orm_bird_bird_bal2 \
      --base-model models/Qwen2.5-Coder-3B-Instruct \
      --merge-python envs/reasoning3b/bin/python
"""
import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT = Path(__file__).resolve().parent.parent
for _p in (str(_PROJECT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_OLD_WORK = _PROJECT / "outputs" / "bird_select_ormbird_bird_bal2" / "work"
DEFAULT_NEW_WORK = _PROJECT / "outputs" / "bird_select_5src" / "work"
DEFAULT_ORM_CKPT = _PROJECT / "checkpoints" / "orm_bird_bird_bal2"
DEFAULT_BASE_MODEL = _PROJECT / "models" / "Qwen2.5-Coder-3B-Instruct"
DEFAULT_MERGE_PYTHON = _PROJECT / "envs" / "reasoning3b" / "bin" / "python"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _load_old_map(old_work: Path) -> Dict[Tuple[int, str], float]:
    """旧 4 源 (qi, prompt) → score 映射（orm_payloads 与 orm_scores zip 对齐）。"""
    payloads = json.loads((old_work / "orm_payloads.json").read_text(encoding="utf-8"))
    scores = json.loads((old_work / "orm_scores.json").read_text(encoding="utf-8"))
    entries = scores["entries"]
    if len(payloads) != len(entries):
        raise SystemExit(f"旧 payloads({len(payloads)}) 与 scores({len(entries)}) "
                         f"长度不一致，无法 zip 对齐复用")
    old_map: Dict[Tuple[int, str], float] = {}
    for p, e in zip(payloads, entries):
        if int(p["qi"]) != int(e["qi"]) or int(p["ei"]) != int(e["ei"]):
            raise SystemExit(f"旧 payloads/scores 顺序不对齐: ({p['qi']},{p['ei']}) "
                             f"vs ({e['qi']},{e['ei']})")
        key = (int(p["qi"]), str(p["prompt"]))
        if key in old_map:
            raise SystemExit(f"旧 payloads 重复 prompt key: {key}")
        old_map[key] = float(e["score"])
    return old_map


def phase_partition(args: argparse.Namespace) -> None:
    old_work = Path(args.old_work)
    new_work = Path(args.new_work)
    if old_work.resolve() == new_work.resolve():
        raise SystemExit("old-work 与 new-work 相同——禁止原地覆盖基线产物")
    old_map = _load_old_map(old_work)
    new_payloads = json.loads(
        (new_work / "orm_payloads.json").read_text(encoding="utf-8"))
    print(f"[partition] old scores={len(old_map)}, new payloads={len(new_payloads)}")

    reuse_entries: List[Dict[str, Any]] = []
    need_payloads: List[Dict[str, Any]] = []
    n_dup = 0
    seen: set = set()
    for p in new_payloads:
        qi, ei = int(p["qi"]), int(p["ei"])
        key = (qi, str(p["prompt"]))
        if (qi, ei) in seen:
            n_dup += 1
            continue
        seen.add((qi, ei))
        if key in old_map:
            reuse_entries.append({"qi": qi, "ei": ei, "score": old_map[key]})
        else:
            need_payloads.append(p)

    _write_json(new_work / "orm_scores_reuse.json", {"entries": reuse_entries})
    _write_json(new_work / "orm_scores_need.json", {"payloads": need_payloads})
    stats = {
        "n_new_payloads": len(new_payloads),
        "n_reuse": len(reuse_entries),
        "n_need_score": len(need_payloads),
        "n_dup_qi_ei": n_dup,
        "old_work": str(old_work),
    }
    _write_json(new_work / "orm_reuse_stats.json", stats)
    print(f"[partition] reuse={len(reuse_entries)} need={len(need_payloads)} "
          f"dup={n_dup}")

    if not need_payloads:
        _write_merged_scores(new_work, reuse_entries, stats, None)
        print("[partition] 无新增组代表 → 已直接写完整 orm_scores.json（无需 GPU 打分）")
    else:
        print("[partition] 存在新增组代表 → 提交 bird_rft_score_inc.slurm 后 "
              "运行 --phase score-gpu")


def phase_score_gpu(args: argparse.Namespace) -> None:
    new_work = Path(args.new_work)
    need_file = new_work / "orm_scores_need.json"
    reuse_file = new_work / "orm_scores_reuse.json"
    stats_file = new_work / "orm_reuse_stats.json"
    if not need_file.exists() or not reuse_file.exists():
        raise SystemExit(f"缺少 {need_file}/{reuse_file}——先跑 --phase partition")
    need_payloads = json.loads(need_file.read_text(encoding="utf-8"))["payloads"]
    reuse_entries = json.loads(reuse_file.read_text(encoding="utf-8"))["entries"]
    stats = json.loads(stats_file.read_text(encoding="utf-8"))
    if not need_payloads:
        raise SystemExit("need 列表为空——无需 GPU 打分（partition 阶段应已直接写全量）")

    from orm_selection import VllmScorer  # noqa: E402

    ns = Namespace(
        base_model=str(args.base_model),
        orm_checkpoint=Path(args.orm_checkpoint),
        merge_python=str(args.merge_python),
        max_length=args.max_length,
        logprobs_topk=args.logprobs_topk,
        chunk_size=args.chunk_size,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
    )
    scorer = VllmScorer(ns)
    import time
    t0 = time.perf_counter()
    scores = scorer.score(
        [(p["qi"], p["ei"], p["prompt"]) for p in need_payloads])
    print(f"[score-gpu] {len(need_payloads)} 个新增组代表打分完成 "
          f"({time.perf_counter() - t0:.1f}s, mode={scorer.stats['mode']})")
    n_missing = sum(1 for s in scores if s is None)
    if n_missing:
        raise SystemExit(f"[score-gpu] {n_missing} 个新增组代表缺分（内部错误）")

    need_entries = [{"qi": int(p["qi"]), "ei": int(p["ei"]), "score": float(s)}
                    for p, s in zip(need_payloads, scores)]
    _write_merged_scores(new_work, reuse_entries, stats, need_entries)
    scorer_stats = dict(scorer.stats)
    scorer_stats["n_reuse_from_4src"] = stats["n_reuse"]
    scorer_stats["n_new_scored"] = len(need_entries)
    # 更新已写 orm_scores.json 的 stats 字段
    out = json.loads((new_work / "orm_scores.json").read_text(encoding="utf-8"))
    out["stats"] = scorer_stats
    _write_json(new_work / "orm_scores.json", out)
    print(f"[score-gpu] DONE -> {new_work / 'orm_scores.json'} "
          f"(reuse={stats['n_reuse']} + new={len(need_entries)})")


def _write_merged_scores(new_work: Path, reuse_entries: List[Dict[str, Any]],
                         stats: Dict[str, Any],
                         need_entries: Optional[List[Dict[str, Any]]]) -> None:
    """按新 orm_payloads.json 顺序重组完整 orm_scores.json（final 阶段直接消费）。"""
    new_payloads = json.loads(
        (new_work / "orm_payloads.json").read_text(encoding="utf-8"))
    reuse_map = {(int(e["qi"]), int(e["ei"])): float(e["score"])
                 for e in reuse_entries}
    need_map = {(int(e["qi"]), int(e["ei"])): float(e["score"])
                for e in (need_entries or [])}
    entries = []
    n_missing = 0
    for p in new_payloads:
        key = (int(p["qi"]), int(p["ei"]))
        if key in need_map:
            entries.append({"qi": key[0], "ei": key[1], "score": need_map[key]})
        elif key in reuse_map:
            entries.append({"qi": key[0], "ei": key[1], "score": reuse_map[key]})
        else:
            n_missing += 1
            print(f"[WARN] payload {key} 无分（reuse/need 均缺）")
    if n_missing:
        raise SystemExit(f"[merge] {n_missing} 个 payload 缺分，拒绝写出")
    _write_json(new_work / "orm_scores.json", {
        "entries": entries,
        "stats": {
            "mode": "merge-reuse",
            "n_entries": len(entries),
            "n_reuse_from_4src": stats["n_reuse"],
            "n_new_scored": len(need_entries or []),
            "old_work": stats["old_work"],
        },
    })


def main() -> int:
    ap = argparse.ArgumentParser(description="5 源 ORM 组代表分合并（复用旧分 + 新打分）")
    ap.add_argument("--phase", choices=["partition", "score-gpu"], required=True)
    ap.add_argument("--old-work", type=Path, default=DEFAULT_OLD_WORK)
    ap.add_argument("--new-work", type=Path, default=DEFAULT_NEW_WORK)
    ap.add_argument("--orm-checkpoint", type=Path, default=DEFAULT_ORM_CKPT)
    ap.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    ap.add_argument("--merge-python", default=str(DEFAULT_MERGE_PYTHON))
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--logprobs-topk", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--max-num-seqs", type=int, default=None)
    args = ap.parse_args()

    if args.phase == "partition":
        phase_partition(args)
    else:
        phase_score_gpu(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
