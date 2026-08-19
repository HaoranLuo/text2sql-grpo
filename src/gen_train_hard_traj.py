#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/gen_train_hard_traj.py — train 集 hard/extra 讲解式轨迹生成（路线 B 修订：无 dev 泄漏）。

背景：hard_traj v1 的 308 条轨迹全部来自 dev 集，混入 SFT 会造成 dev 评估泄漏。
本脚本改用 train 集生成讲解式轨迹：

目标清单（纯 CPU，--build-targets）：
  - train_spider.json(7000) + train_others.json(1659) = 8659 题
  - canonical eval_hardness（照抄 src/gen_hard_trajectories.py）分类，取 hard/extra
  - question 长度 <= --max-qlen（默认 120 字符，用户建议口径）
  - question 与 dev.json 1034 题全文精确比对 → 强制排除重叠（官方 split 存在
    极少数 train/dev 同题，实测 6 条，全部踢出）
  - question 与 data/sft_v3_mix.json 同题 → 默认排除（省 API 且避免 prep 阶段
    替换浪费）；候选不足时可 --allow-mix-overlap 放行（prep 会替换）
  - database/<db_id>/<db_id>.sqlite 必须存在（gold 可执行前提）
  - --seed 洗牌取 --n-targets 条（默认 360，含缓冲；最终取 verified 前 --out-n=340）

轨迹 = 讲解式 explain（与 dev 版 explain 同款契约）：
  - target SQL = gold SQL（官方答案，其正确性由官方保证，讲解式零风险）
  - user = canonical prompt（干净，不含 target SQL）；assistant = 五段式
    <think> 讲解，最终 SQL 必须与 gold 文本对齐（_norm_eq）
  - success 判据 = 文本对齐 + 轨迹自洽（不提检查器/反馈）；执行比对
    （DatabaseExecutor + compare）仅作审计记录——部分 train 库实例化数据与
    gold 不一致或执行超时（环境问题），不否决"官方 gold 文本正确"的轨迹
  - 未对齐走修复轮（--repair-rounds）

API 调用模式照 gen_hard_trajectories：AsyncOpenAI + deepseek-v4-flash（默认，
讲解式是格式对齐任务，flash 够用且便宜）、thinking disabled、指数退避、
DEEPSEEK_API_KEY 从环境变量读取（绝不硬编码）。

分片：--shard k --n-shards N 按 targets 顺序 stride 取片，各自写
records_train_shard<k>.jsonl；全部完成后 --merge-trains 汇总：
trajectories_train.jsonl（verified 前 --out-n 条，按 targets 顺序）+
summary_train.json（通过率/成本/dev 重叠数）。

用法：
  python src/gen_train_hard_traj.py --build-targets          # 纯 CPU 清单
  python src/gen_train_hard_traj.py --dry-run                # 预检
  sbatch --array=1-4 scripts/gen_train_hard_traj.slurm       # 4 进程并行生成
  python src/gen_train_hard_traj.py --merge-trains           # 合并 + 汇总
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spider_utils import DatabaseExecutor, SpiderLoader  # noqa: E402
from gen_hard_trajectories import (  # noqa: E402
    CNY_PER_USD,
    EXPLAIN_SYSTEM,
    PRICING_USD_PER_M,
    _accumulate_usage,
    _norm_eq,
    build_canonical_prompt,
    build_polish_feedback,
    call_deepseek,
    compute_cost_usd,
    eval_hardness,
    estimate_tokens,
    extract_final_sql,
    load_done,
    make_base_record,
    trajectory_is_self_contained,
    verify_final_sql,
)

try:
    from openai import AsyncOpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    AsyncOpenAI = None
    _OPENAI_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPIDER_DIR = (
    "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data"
)
DEFAULT_MIX = str(PROJECT_ROOT / "data" / "sft_v3_mix.json")
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "hard_traj"
DEFAULT_BASE_URL = "https://api.deepseek.com"

_QUESTION_RE = re.compile(r"Question:\s*\n(.*?)\n\nOptional Schema Links:", re.DOTALL)


# ---------------------------------------------------------------------------
# 目标清单
# ---------------------------------------------------------------------------

def load_train_items(spider_dir: str) -> List[Dict[str, Any]]:
    """train_spider + train_others = 8659 题；di = 全局索引（spider 在前）。"""
    spider = json.loads(
        (Path(spider_dir) / "train_spider.json").read_text(encoding="utf-8"))
    others = json.loads(
        (Path(spider_dir) / "train_others.json").read_text(encoding="utf-8"))
    items: List[Dict[str, Any]] = []
    for src, lst in (("spider", spider), ("others", others)):
        for i, it in enumerate(lst):
            rec = dict(it)
            rec["_src"] = src
            rec["_src_idx"] = i
            items.append(rec)
    return items


def load_dev_questions(spider_dir: str) -> Set[str]:
    dev = json.loads((Path(spider_dir) / "dev.json").read_text(encoding="utf-8"))
    return {it["question"].strip() for it in dev}


def load_mix_questions(mix_path: str) -> Set[str]:
    mix = json.loads(Path(mix_path).read_text(encoding="utf-8"))
    qs: Set[str] = set()
    for m in mix:
        mm = _QUESTION_RE.search(m["messages"][0]["content"])
        if mm:
            qs.add(mm.group(1).strip())
    return qs


def build_targets(spider_dir: str, mix_path: str, args: argparse.Namespace
                  ) -> List[Dict[str, Any]]:
    items = load_train_items(spider_dir)
    dev_q = load_dev_questions(spider_dir)
    mix_q = load_mix_questions(mix_path)
    dbdir = Path(spider_dir) / "database"

    n_dev_overlap = 0
    n_mix_overlap = 0
    n_db_missing = 0
    n_qlen = 0
    n_not_hard = 0
    pool: List[Dict[str, Any]] = []
    for di, it in enumerate(items):
        tree = it.get("sql")
        if not isinstance(tree, dict):
            continue
        d = eval_hardness(tree)
        if d not in ("hard", "extra"):
            n_not_hard += 1
            continue
        q = it["question"].strip()
        if len(q) > args.max_qlen:
            n_qlen += 1
            continue
        if q in dev_q:  # 硬性排除：dev 零重叠契约
            n_dev_overlap += 1
            continue
        if q in mix_q and not args.allow_mix_overlap:
            n_mix_overlap += 1
            continue
        db_id = it["db_id"]
        if not (dbdir / db_id / f"{db_id}.sqlite").exists():
            n_db_missing += 1
            continue
        pool.append({
            "di": di,
            "src": it["_src"],
            "src_idx": it["_src_idx"],
            "db_id": db_id,
            "question": q,
            "gold_sql": it["query"],
            "difficulty": d,
            "source": ["train_hardness"],
        })

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    selected = pool[: args.n_targets]

    print(f"[targets] train total={len(items)}  hard/extra 且 qlen<={args.max_qlen}"
          f" 候选池={len(pool) + n_dev_overlap + n_mix_overlap + n_db_missing + n_qlen}"
          f" (排除: 非hard/extra={n_not_hard} qlen={n_qlen} dev重叠={n_dev_overlap} "
          f"mix同题={n_mix_overlap} 缺库={n_db_missing})", flush=True)
    print(f"[targets] 池内={len(pool)} -> seed={args.seed} 洗牌选 {len(selected)}",
          flush=True)
    diff_cnt = Counter(t["difficulty"] for t in selected)
    print(f"[targets] selected 难度分布: {dict(diff_cnt)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "definition": ("train 集 hard/extra 讲解式目标：canonical eval_hardness "
                       f"分类（train_spider+train_others 共 {len(items)} 题）、"
                       f"question<={args.max_qlen} 字符、dev 零重叠（全文精确比对）、"
                       "mix 同题排除（可 --allow-mix-overlap 放行）、gold 库存在；"
                       "target SQL = gold SQL"),
        "seed": args.seed,
        "counts": {
            "train_total": len(items),
            "pool_after_filter": len(pool),
            "excluded_dev_overlap": n_dev_overlap,
            "excluded_mix_overlap": n_mix_overlap,
            "excluded_db_missing": n_db_missing,
            "excluded_qlen": n_qlen,
            "excluded_not_hard": n_not_hard,
            "selected": len(selected),
        },
        "targets": selected,
    }
    (out_dir / "targets_train.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[targets] wrote {len(selected)} -> {out_dir / 'targets_train.json'}",
          flush=True)
    return selected


# ---------------------------------------------------------------------------
# 单题讲解式生成（target SQL = gold SQL）
# ---------------------------------------------------------------------------

async def process_one(target: Dict[str, Any], ddl: str, client: Any,
                      sem: asyncio.Semaphore, args: argparse.Namespace,
                      executor: DatabaseExecutor,
                      gold_cache: Dict[Any, Dict[str, Any]],
                      ) -> Dict[str, Any]:
    rec = make_base_record(target, args.model, args.temperature)
    rec["route"] = "explain"
    rec["train_source"] = target["src"]
    rec["target_sql"] = target["gold_sql"]
    user_prompt = build_canonical_prompt(target["question"], ddl)
    gen_user = (user_prompt + "\n\nTarget SQL (the correct answer to explain):\n"
                "```sql\n" + target["gold_sql"] + "\n```")
    convo: List[Dict[str, str]] = [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user", "content": gen_user},
    ]

    # gold 执行缓存按 (db_id, gold_sql) 键 —— 修复 gen_hard_trajectories
    # verify_final_sql 的 gold_cache 按 db_id 缓存导致同库多题结果串库的问题
    # （flight_4/college_2 等失败均源于此）。
    gold_key = (target["db_id"], target["gold_sql"])
    if gold_key not in gold_cache:
        gold_cache[gold_key] = executor.execute(target["db_id"], target["gold_sql"])

    async with sem:
        if args.sleep > 0:
            await asyncio.sleep(args.sleep)
        last_text = None
        final_ver: Optional[Dict[str, Any]] = None
        for rnd in range(1 + args.repair_rounds):
            text, reasoning, usage, err = await call_deepseek(
                client, args.model, convo, args.temperature, args.max_tokens,
                args.max_retries, args.timeout, thinking_enabled=False)
            if err is not None:
                rec["error"] = f"api_error: {type(err).__name__}: {err}"[:500]
                return rec
            rec["n_calls"] += 1
            _accumulate_usage(rec, usage)
            convo.append({"role": "assistant", "content": text or ""})
            last_text = text
            final_sql = extract_final_sql(text or "")
            aligned = _norm_eq(final_sql, target["gold_sql"])
            self_c = trajectory_is_self_contained(text or "")
            # 执行比对仅作审计（单元素缓存避免串库）：
            # target = 官方 gold SQL，其正确性由官方保证（"gold 一定执行通过"），
            # 文本对齐 + 自洽即为成功；部分 train 库实例化差异/超时不影响判定。
            final_ver = verify_final_sql(
                target, final_sql, executor,
                {target["db_id"]: gold_cache[gold_key]})
            if aligned and self_c:
                rec["final_matches_target"] = True
                rec["conversation"] = convo
                rec["audit"] = {
                    "exec_match": final_ver.get("match"),
                    "exec_error": final_ver.get("exec_error"),
                    "note": "aligned+self-contained；执行比对仅审计（库环境差异不否决）",
                }
                break
            if rnd < args.repair_rounds:
                if not aligned:
                    fb = ("Your 最终 SQL differs from the target SQL. Rewrite the "
                          "complete trajectory with the 最终 SQL exactly equal to:\n"
                          "```sql\n" + target["gold_sql"] + "\n```")
                else:
                    fb = build_polish_feedback()
                convo.append({"role": "user", "content": fb})

    final_sql_last = extract_final_sql(last_text or "") or None
    rec["response"] = last_text
    rec["final_sql"] = final_sql_last
    rec["verification"] = final_ver
    aligned_final = _norm_eq(final_sql_last, target["gold_sql"])
    self_c_final = trajectory_is_self_contained(last_text or "")
    rec["final_matches_target"] = aligned_final
    rec["messages"] = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": (last_text or "").strip()},
    ]
    rec["generation_user_prompt"] = gen_user  # 审计（含 target SQL）
    if aligned_final and self_c_final:
        rec["success"] = True
    else:
        if rec.get("conversation") is None:
            rec["conversation"] = convo
        if not aligned_final:
            rec["error"] = "verify_failed: stage=target_sql_misaligned"
        else:
            rec["error"] = "verify_failed: stage=not_self_contained"
    return rec


# ---------------------------------------------------------------------------
# 生成 / 合并
# ---------------------------------------------------------------------------

def gen_main(args: argparse.Namespace, targets: List[Dict[str, Any]]) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = SpiderLoader(args.spider_dir)
    ddl_cache: Dict[int, str] = {}
    ddl_missing = 0
    for t in targets:
        if t["di"] in ddl_cache:
            continue
        try:
            ddl_cache[t["di"]] = loader.format_ddl(t["db_id"])
        except Exception:
            ddl_missing += 1
    pending = [t for t in targets if t["di"] in ddl_cache]
    print(f"[gen] ddl ok={len(pending)}/{len(targets)} (missing={ddl_missing})",
          flush=True)

    if args.n_shards > 1:
        pending = [t for i, t in enumerate(pending)
                   if i % args.n_shards == args.shard - 1]
        print(f"[gen] shard {args.shard}/{args.n_shards}: {len(pending)} targets",
              flush=True)
    if args.limit > 0:
        pending = pending[: args.limit]
        print(f"[gen] limit={args.limit} -> {len(pending)} targets", flush=True)

    records_path = out_dir / f"records_train_shard{args.shard}.jsonl"
    done = load_done(records_path)
    if args.retry_failed:
        failed: Set[int] = set()
        if records_path.exists():
            for line in records_path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec.get("dataset_index"), int) and not rec.get("success"):
                    failed.add(rec["dataset_index"])
        pending = [t for t in pending if t["di"] in failed]
    else:
        pending = [t for t in pending if t["di"] not in done]
    print(f"[gen] to process: {len(pending)} (already done: "
          f"{sum(1 for t in targets if t['di'] in done)})", flush=True)

    est_in = sum(estimate_tokens(build_canonical_prompt(t["question"], ddl_cache[t["di"]]))
                 + estimate_tokens(EXPLAIN_SYSTEM) for t in pending)
    est_out = 2500 * len(pending) * (1 + args.repair_rounds)
    p = PRICING_USD_PER_M[args.model]
    est_usd = est_in / 1e6 * p["input_miss"] + est_out / 1e6 * p["output"]
    print(f"[gen] cost estimate (pre-run): ~${est_usd:.2f} USD ≈ "
          f"¥{est_usd * CNY_PER_USD:.2f}", flush=True)

    if args.dry_run:
        print("DRY RUN - no API calls made.")
        return 0
    if not pending:
        print("Nothing to do.")
        return 0

    if not _OPENAI_AVAILABLE:
        print("ERROR: 'openai' package required.")
        return 1
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set in environment.")
        return 1

    executor = DatabaseExecutor(args.spider_dir)
    gold_cache: Dict[Any, Dict[str, Any]] = {}  # key=(db_id, gold_sql), 防同库多题串库
    client = AsyncOpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)
    sem = asyncio.Semaphore(args.concurrency)

    async def _guarded(coro: Any, di: int) -> Dict[str, Any]:
        try:
            return await coro
        except Exception as exc:  # 单条崩溃不炸整个分片
            return {
                "dataset_index": di, "success": False,
                "error": f"worker_exception: {type(exc).__name__}: {exc}"[:500],
            }

    async def run_all() -> List[Dict[str, Any]]:
        tasks = [
            asyncio.create_task(_guarded(
                process_one(t, ddl_cache[t["di"]], client, sem, args,
                            executor, gold_cache), t["di"]))
            for t in pending
        ]
        records: List[Dict[str, Any]] = []
        done_n = 0
        with open(records_path, "a", encoding="utf-8") as fh:
            for coro in asyncio.as_completed(tasks):
                rec = await coro
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                records.append(rec)
                done_n += 1
                if done_n % 5 == 0 or done_n == len(tasks):
                    ok = sum(1 for r in records if r.get("success"))
                    print(f"  [{done_n}/{len(tasks)}] verified={ok}", flush=True)
        return records

    print(f"[gen] generating {len(pending)} trajectories via {args.model} "
          f"(concurrency={args.concurrency}, repair={args.repair_rounds})...",
          flush=True)
    records = asyncio.run(run_all())
    verified = sum(1 for r in records if r.get("success"))
    usd, in_hit, in_miss, out = compute_cost_usd(records, args.model)
    print(f"[gen] shard done: verified={verified}/{len(records)}, "
          f"cost=${usd:.4f}", flush=True)
    return 0


def merge_trains(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    targets_path = out_dir / "targets_train.json"
    if not targets_path.exists():
        print("ERROR: targets_train.json not found; run --build-targets first.")
        return 1
    payload = json.loads(targets_path.read_text(encoding="utf-8"))
    targets = payload.get("targets", [])
    order = {t["di"]: i for i, t in enumerate(targets)}

    rec_paths = sorted(out_dir.glob("records_train_shard*.jsonl"))
    if not rec_paths:
        print("ERROR: no records_train_shard*.jsonl found.")
        return 1
    by_di: Dict[int, Dict[str, Any]] = {}
    all_lines: List[Dict[str, Any]] = []
    for p in rec_paths:
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            all_lines.append(rec)
            if isinstance(rec.get("dataset_index"), int):
                by_di[rec["dataset_index"]] = rec
    ordered = sorted(by_di.values(), key=lambda r: order.get(r["dataset_index"], 10**9))
    verified = [r for r in ordered if r.get("success")]
    selected = verified[: args.out_n]

    # dev 零重叠契约复核（question 全文精确比对）
    dev_q = load_dev_questions(args.spider_dir)
    overlap = [(r["dataset_index"], r["question"])
               for r in selected if r["question"].strip() in dev_q]
    print(f"[merge] verified={len(verified)}/{len(all_lines)} -> trajectories "
          f"取前 {len(selected)}; dev 重叠 = {len(overlap)}", flush=True)
    if overlap:
        print("ERROR: dev overlap detected:", overlap[:10])
        return 1

    traj_path = out_dir / "trajectories_train.jsonl"
    with open(traj_path, "w", encoding="utf-8") as fh:
        for r in selected:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    model = all_lines[0]["model"] if all_lines else args.model
    usd, in_hit, in_miss, out = compute_cost_usd(all_lines, model)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "train 集 hard/extra 讲解式轨迹（路线 B 修订：无 dev 泄漏）",
        "model": model,
        "temperature": (all_lines[0]["temperature"] if all_lines else None),
        "targets_selected": len(targets),
        "attempted": len(all_lines),
        "verified": len(verified),
        "written": len(selected),
        "out_n": args.out_n,
        "dev_overlap_count": len(overlap),
        "fail_breakdown": {
            "api_error": sum(1 for r in all_lines if (r.get("error") or "").startswith("api_error")),
            "parse_fail": sum(1 for r in all_lines if r.get("verification", {}).get("exec_error") == "no_sql_block_parsed"),
            "exec_fail": sum(1 for r in all_lines if r.get("verification", {}).get("exec_error") not in (None, "no_sql_block_parsed")),
            "mismatch": sum(1 for r in all_lines if r.get("verification", {}).get("exec_success")
                            and not r.get("verification", {}).get("match")),
        },
        "tokens": {"prompt_total": in_miss + in_hit, "prompt_hit": in_hit,
                   "prompt_miss": in_miss, "completion": out},
        "cost": {
            "usd": round(usd, 4),
            "cny": round(usd * CNY_PER_USD, 4),
            "pricing_usd_per_m": PRICING_USD_PER_M[model],
        },
        "files": {"trajectories": str(traj_path)},
    }
    (out_dir / "summary_train.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge] wrote {len(selected)} -> {traj_path}; "
          f"summary -> {out_dir / 'summary_train.json'}", flush=True)
    print(f"[merge] cost ${usd:.4f} ≈ ¥{usd * CNY_PER_USD:.4f}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="train 集 hard/extra 讲解式轨迹生成")
    ap.add_argument("--spider-dir", default=DEFAULT_SPIDER_DIR)
    ap.add_argument("--mix", default=DEFAULT_MIX)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--build-targets", action="store_true",
                    help="只构建目标清单 targets_train.json，不调 API")
    ap.add_argument("--merge-trains", action="store_true",
                    help="合并 records_train_shard*.jsonl -> trajectories_train.jsonl")
    ap.add_argument("--allow-mix-overlap", action="store_true",
                    help="允许与 sft_v3_mix 同题的候选（prep 阶段会替换）")
    ap.add_argument("--max-qlen", type=int, default=120,
                    help="question 长度上限（字符，默认 120）")
    ap.add_argument("--n-targets", type=int, default=360,
                    help="目标清单条数（含缓冲；最终取 verified 前 --out-n）")
    ap.add_argument("--out-n", type=int, default=340,
                    help="trajectories_train.jsonl 条数（≈5% 混入比例）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理分片内前 N 题（冒烟测试；0=全部）")
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--model", choices=("deepseek-v4-flash", "deepseek-v4-pro"),
                    default="deepseek-v4-flash")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--repair-rounds", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_trains:
        return merge_trains(args)

    targets_path = out_dir / "targets_train.json"
    if args.build_targets or not targets_path.exists():
        targets = build_targets(args.spider_dir, args.mix, args)
    else:
        payload = json.loads(targets_path.read_text(encoding="utf-8"))
        targets = payload.get("targets", [])
        print(f"[targets] loaded {len(targets)} from {targets_path}", flush=True)
    if args.build_targets:
        print("[targets] done (--build-targets, no API calls).")
        return 0

    return gen_main(args, targets)


if __name__ == "__main__":
    sys.exit(main())
