#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/prep_sft_v4hard.py — SFT v4-hard 数据准备（纯 CPU，不调 API / 不用 GPU）。

读 outputs/hard_traj/trajectories_train.jsonl（train 集 hard/extra 讲解式轨迹，
~340 条、与 dev 零重叠、已执行验证；由 src/gen_train_hard_traj.py 生成）与
data/sft_v3_mix.json（6444 条 chat 格式），产出 data/sft_v4hard_mix.json：

  - sft_v3_mix 全部 6444 条原样保留
  - train 讲解式轨迹全量混入（--include-free 仅对旧 dev 版 trajectories.jsonl
    的 free 路由有意义；train 版轨迹无 free，开关默认关闭即全量混入）
  - 难题轨迹比例：340/6784 ≈ 5.0%
  - 格式与 sft_v3_mix 完全一致：[{"messages": [user, assistant]}, ...]，
    user prompt 截断口径照抄 src/prep_sft_data.py::truncate_user_prompt
    （chat 模板化后 <=1536 token，与推理端窗口一致）。
    轨迹 user 直接用轨迹文件自带 messages[0].content（生成时即 canonical
    prompt，与评估端 build_prompt 同源），不重建，保证训练/推理一致。
  - 去重决策：轨迹 question 与 mix 中同 question（精确文本）条目 → 轨迹版
    原地替换旧蒸馏版（轨迹已执行验证且为难题定向讲解，质量更高，避免同题
    两版打架）；替换条数记入 manifest。轨迹内部同 question 只保留第一条。
  - sha256 数据卫生：输入/输出文件哈希写入 data/sft_v4hard_mix.manifest.json

注意（2026-08-18 修订）：首版混入了 dev 集轨迹（outputs/hard_traj/
trajectories.jsonl，308 条）造成 dev 泄漏，已废弃；v4hard 一律使用
trajectories_train.jsonl（train 集，dev 零重叠）。

用法:
    python src/prep_sft_v4hard.py                    # mix 6444 + train 轨迹 340
    python src/prep_sft_v4hard.py --output data/xxx.json --manifest data/xxx.manifest.json
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT = Path(__file__).resolve().parent.parent

try:
    from transformers import AutoTokenizer
    _HAS_TOKENIZER = True
except Exception:  # pragma: no cover - 环境缺失时报错提示
    AutoTokenizer = None
    _HAS_TOKENIZER = False

MAX_PROMPT_TOKENS = 1536  # 与 prep_sft_data / 推理端 prompt 截断窗口一致
DEFAULT_TOKENIZER_PATH = (
    "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/"
    "models/Qwen2.5-Coder-3B-Instruct"
)

# canonical prompt 中 Question 段提取（build_prompt 模板: "Question:\n{q}\n\nOptional Schema Links:"）
_QUESTION_RE = re.compile(r"Question:\s*\n(.*?)\n\nOptional Schema Links:", re.DOTALL)
_THINK_RE = re.compile(r"<think>", re.IGNORECASE)
_SQL_BLOCK_RE = re.compile(r"```sql", re.IGNORECASE)


# ---------------------------------------------------------------------------
# IO / 哈希
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_mix(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


# ---------------------------------------------------------------------------
# prompt 截断（照抄 src/prep_sft_data.py::truncate_user_prompt，同款口径）
# ---------------------------------------------------------------------------

def truncate_user_prompt(user: str, tokenizer: Any,
                         max_prompt_tokens: int = MAX_PROMPT_TOKENS) -> str:
    """把 user 内容截断到模板化后 <= max_prompt_tokens，与推理端窗口一致。"""
    def templated_len(content: str) -> int:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True)
        return len(ids)

    if templated_len(user) <= max_prompt_tokens:
        return user
    lo, hi = 0, len(user)
    while lo < hi:
        mid = (lo + hi) // 2
        if templated_len(user[:mid]) <= max_prompt_tokens:
            lo = mid + 1
        else:
            hi = mid
    return user[: max(0, lo - 1)].rstrip()


# ---------------------------------------------------------------------------
# 轨迹读取 / 校验
# ---------------------------------------------------------------------------

def _valid_messages(msgs: Any) -> bool:
    if not isinstance(msgs, list) or len(msgs) != 2:
        return False
    return (msgs[0].get("role") == "user" and msgs[1].get("role") == "assistant"
            and isinstance(msgs[0].get("content"), str)
            and bool(msgs[0]["content"].strip())
            and isinstance(msgs[1].get("content"), str)
            and bool(msgs[1]["content"].strip()))


def load_trajectories(path: Path, include_free: bool
                      ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """返回 (选中的轨迹记录, 统计)。只保留 success=True 且 messages 合法的
    explain（+ free 若 include_free）。轨迹内部同 question 只保留第一条。"""
    lines = load_jsonl(path)
    route_cnt = {"explain": 0, "free": 0, "other": 0}
    seen_questions = set()
    kept: List[Dict[str, Any]] = []
    skipped = {"not_success": 0, "bad_messages": 0,
               "route_other": 0, "dup_question": 0}
    for rec in lines:
        route = rec.get("route")
        route_cnt[route if route in route_cnt else "other"] += 1
        if route not in ("explain", "free"):
            skipped["route_other"] += 1
            continue
        if rec.get("success") is not True:
            skipped["not_success"] += 1
            continue
        if not _valid_messages(rec.get("messages")):
            skipped["bad_messages"] += 1
            continue
        if route == "free" and not include_free:
            continue
        q = (rec.get("question") or "").strip()
        if q and q in seen_questions:
            skipped["dup_question"] += 1
            continue
        if q:
            seen_questions.add(q)
        kept.append(rec)
    stats = {"lines": len(lines), "route_counts": route_cnt,
             "kept": len(kept), "skipped": skipped}
    return kept, stats


def extract_question_from_prompt(prompt: str) -> Optional[str]:
    m = _QUESTION_RE.search(prompt or "")
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="SFT v4-hard 数据准备（纯 CPU）")
    ap.add_argument("--trajectories",
                    default=str(PROJECT / "outputs" / "hard_traj" / "trajectories_train.jsonl"),
                    help="train 集讲解式轨迹（默认 trajectories_train.jsonl；"
                         "旧 dev 版 trajectories.jsonl 已废弃[dev 泄漏]）")
    ap.add_argument("--mix", default=str(PROJECT / "data" / "sft_v3_mix.json"))
    ap.add_argument("--include-free", action="store_true",
                    help="混入自由生成 35 条（默认不混）")
    ap.add_argument("--output", default=str(PROJECT / "data" / "sft_v4hard_mix.json"))
    ap.add_argument("--manifest",
                    default=str(PROJECT / "data" / "sft_v4hard_mix.manifest.json"))
    ap.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    ap.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    args = ap.parse_args()

    if not _HAS_TOKENIZER:
        print("ERROR: transformers not available for prompt truncation")
        return 1
    print(f"loading tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                              local_files_only=True)

    traj_path = Path(args.trajectories)
    mix_path = Path(args.mix)
    out_path = Path(args.output)
    manifest_path = Path(args.manifest)

    trajs, tstats = load_trajectories(traj_path, args.include_free)
    print(f"[traj] {tstats['lines']} lines, route={tstats['route_counts']}, "
          f"kept={tstats['kept']}, skipped={tstats['skipped']}")

    mix = load_mix(mix_path)
    print(f"[mix] {len(mix)} items")

    # mix question 提取 + 同 question 索引映射
    mix_questions: List[Optional[str]] = []
    q_to_idx: Dict[str, List[int]] = {}
    for i, item in enumerate(mix):
        msgs = item.get("messages")
        q = None
        if isinstance(msgs, list) and msgs and isinstance(msgs[0].get("content"), str):
            q = extract_question_from_prompt(msgs[0]["content"])
        mix_questions.append(q)
        if q:
            q_to_idx.setdefault(q, []).append(i)
    n_no_q = sum(1 for q in mix_questions if q is None)
    dup_groups = sum(1 for idxs in q_to_idx.values() if len(idxs) > 1)
    print(f"[mix] question 提取失败={n_no_q}, mix 内部同 question 重复组={dup_groups}")

    # 去重替换：轨迹 question 命中 mix 同题 → 原地替换（全部同题条目）
    # 轨迹条目 messages 先做 prompt 截断（user 侧同款口径）
    mix = [dict(item) for item in mix]
    replaced_idx: List[int] = []
    replaced_questions: List[str] = []
    appended: List[Dict[str, Any]] = []
    n_trunc = 0
    for traj in trajs:
        msgs = traj["messages"]
        user = msgs[0]["content"]
        trunc_user = truncate_user_prompt(user, tokenizer, args.max_prompt_tokens)
        if trunc_user != user:
            n_trunc += 1
        new_msgs = [{"role": "user", "content": trunc_user},
                    {"role": "assistant", "content": msgs[1]["content"].strip()}]
        q = (traj.get("question") or "").strip()
        idxs = q_to_idx.get(q, [])
        if q and idxs:
            for i in idxs:
                mix[i] = {"messages": new_msgs}
                replaced_idx.append(i)
            replaced_questions.append(q)
        else:
            appended.append({"messages": new_msgs})

    # 轨迹完整性统计（仅统计，不丢弃：数据已执行验证）
    n_no_think = n_no_sql = 0
    for traj in trajs:
        a = traj["messages"][1]["content"]
        if not _THINK_RE.search(a):
            n_no_think += 1
        if not _SQL_BLOCK_RE.search(a):
            n_no_sql += 1
    print(f"[traj] 截断(>{args.max_prompt_tokens} tok): {n_trunc}/{len(trajs)} 条; "
          f"缺 <think>={n_no_think}, 缺 ```sql={n_no_sql}（仅统计，不丢弃）")

    records = mix + appended
    final_total = len(records)
    n_traj_in = len(trajs)
    replaced_mix_items = len(replaced_idx)
    ratio = n_traj_in / final_total if final_total else 0.0
    explain_n = sum(1 for t in trajs if t.get("route") == "explain")
    free_n = n_traj_in - explain_n
    print(f"[dedup] 轨迹命中 mix 同题 {len(replaced_questions)} 题, "
          f"原地替换 mix 条目 {replaced_mix_items} 条; 追加 {len(appended)} 条")
    print(f"[final] mix={len(mix)} + traj_in={n_traj_in} "
          f"(explain={explain_n}, free={free_n}) - replaced={replaced_mix_items} "
          f"= {final_total} 条")
    print(f"[ratio] 难题轨迹占比 = {n_traj_in}/{final_total} = {ratio:.4%} "
          f"(目标参考区间 5-7% 附近)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    out_sha = sha256_file(out_path)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "src/prep_sft_v4hard.py",
        "decisions": {
            "dedup_policy": (
                "轨迹 question 与 mix 同 question（精确文本）→ 轨迹版原地替换"
                "旧蒸馏版（轨迹已执行验证且为难题定向讲解）；轨迹内部同 question "
                "只保留第一条"),
            "include_free": bool(args.include_free),
            "prompt_truncation": (
                f"user prompt chat 模板化后 > {args.max_prompt_tokens} token 时"
                "二分截断（照抄 src/prep_sft_data.py::truncate_user_prompt）"),
        },
        "inputs": {
            "trajectories": {"path": str(traj_path), "sha256": sha256_file(traj_path),
                             "lines": tstats["lines"],
                             "route_counts": tstats["route_counts"]},
            "mix": {"path": str(mix_path), "sha256": sha256_file(mix_path),
                    "items": len(mix_questions)},
        },
        "counts": {
            "mix_base": len(mix_questions),
            "traj_kept": n_traj_in,
            "traj_explain": explain_n,
            "traj_free": free_n,
            "dedup_replaced_mix_items": replaced_mix_items,
            "dedup_replaced_questions": len(replaced_questions),
            "final_total": final_total,
            "traj_ratio": round(ratio, 6),
            "traj_ratio_pct": round(ratio * 100, 3),
        },
        "truncation": {"max_prompt_tokens": args.max_prompt_tokens,
                       "n_truncated": n_trunc},
        "output": {"path": str(out_path), "items": final_total, "sha256": out_sha},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"[manifest] {manifest_path}")
    print(f"[sha256] output={out_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
