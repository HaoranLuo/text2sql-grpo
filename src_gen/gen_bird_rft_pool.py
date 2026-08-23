#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BIRD dev RFT 候选池生成：rft_bird_v3/checkpoint-50（全参 plain 模型）× 1534 题 × n=16 采样。

第 5 源池生成（src/gen_bird_pool.py 的 plain-model 单引擎变体，本脚本独立存在、
不改动现有 pipeline）：

  - prompt 与 gen_bird_pool.py 逐字一致：sqlite_master DDL（表名排序拼接）+
    question + evidence（有值才加，ReasoningGeneratorAgent.build_prompt
    canonical 模板，dialect=sqlite）+ chat template（add_generation_prompt=True），
    截断 3072 token；evidence 仅 dev.json 官方（与原池同口径，无 auto evidence、
    无 retriever 裁剪）。
  - 采样口径与主池一致：T=1.0, top_p=1.0, seed=0, n=16 单请求共享 prefill,
    max_new_tokens=2048。
  - 引擎：plain 全参模型（RFT 为全参 checkpoint，无 LoRA adapter）。
  - 解析 = finer_port/sampler.py VavSampler.extract_sql（与主池同口径）。
  - 输出 outputs/eval_pool_bird_rft/items.json：dataset_index=question_id、
    db_id、question、gold_sql、difficulty、candidates[{model="rft_bird_v3",
    sql, parse_success, sample_idx}]（按 sample_idx 排序；不去重——裁决阶段做）。
  - 切片与续跑协议与 gen_bird_pool.py 相同：checkpoint.json（run_config 混配
    保护）+ --max-gen-seconds 预算超时落盘正常退出，由链脚本重提续跑。

用法：
    envs/vllmenv/bin/python src/gen_bird_rft_pool.py --limit 1534 \
        --model-path checkpoints/rft_bird_v3/checkpoint-50 \
        --output-dir outputs/eval_pool_bird_rft
"""
import argparse
import gc
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer  # noqa: E402

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT = _SRC_DIR.parent
_FINER_DIR = _PROJECT / "finer_port"
for _p in (str(_SRC_DIR), str(_FINER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reasoning_generator_agent import ReasoningGeneratorAgent  # noqa: E402
from sampler import VavSampler  # noqa: E402

DEFAULT_DATA_JSON = str(
    _PROJECT / "data" / "bird" / "bird_dev" / "dev_20240627" / "dev.json")
DEFAULT_DB_ROOT = str(
    _PROJECT / "data" / "bird" / "bird_dev" / "dev_20240627" / "dev_databases")
DEFAULT_MODEL_PATH = str(_PROJECT / "checkpoints" / "rft_bird_v3" / "checkpoint-50")
DEFAULT_OUTPUT_DIR = str(_PROJECT / "outputs" / "eval_pool_bird_rft")
MODEL_NAME = "rft_bird_v3"
EVALUATOR_TYPE = "bird_candidate_pool"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BIRD dev RFT 候选池生成：checkpoint-50 plain 模型 × n=16 采样")
    p.add_argument("--data-json", default=DEFAULT_DATA_JSON)
    p.add_argument("--db-root", default=DEFAULT_DB_ROOT)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--limit", type=int, required=True,
                   help="评估条数（全量 1534；冒烟 10）——必填，防误触全量")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--n", type=int, default=16, help="每题采样候选数")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-prompt-tokens", type=int, default=3072)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--max-gen-seconds", type=int, default=2400)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-num-seqs", type=int, default=None)
    return p.parse_args(argv)


def load_bird_items(data_json: str, limit: int, start_index: int) -> List[Dict[str, Any]]:
    with open(data_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"dev.json 结构异常（期望 list）: {data_json}")
    items = []
    for it in data[start_index:start_index + limit]:
        items.append({
            "dataset_index": int(it["question_id"]),
            "di": int(it["question_id"]),
            "db_id": it["db_id"],
            "question": it["question"],
            "evidence": it.get("evidence") or "",
            "gold_sql": it.get("SQL") or "",
            "difficulty": it.get("difficulty", ""),
        })
    return items


def read_ddl(db_root: str, db_id: str) -> str:
    """BIRD schema DDL：sqlite_master 的 CREATE TABLE 语句按表名排序拼接。
    （与 src/gen_bird_pool.py read_ddl 逐字同款。）"""
    db_path = Path(db_root) / db_id / f"{db_id}.sqlite"
    if not db_path.is_file():
        return ""
    con = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY name").fetchall()
    finally:
        con.close()
    return "\n".join(r[0] for r in rows)


# ---------------------------------------------------------------------------
# 题级状态 + checkpoint（协议与 gen_bird_pool.py 相同）
# ---------------------------------------------------------------------------

def build_run_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "evaluator_type": EVALUATOR_TYPE,
        "data_json": str(Path(args.data_json).resolve()),
        "db_root": str(Path(args.db_root).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "start_index": args.start_index,
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "checkpoint_every": args.checkpoint_every,
        "model_name": MODEL_NAME,
    }


def validate_resume_config(stored: Dict[str, Any], current: Dict[str, Any]) -> None:
    mismatches = []
    for key in sorted(set(list(stored.keys()) + list(current.keys()))):
        if stored.get(key) != current.get(key):
            mismatches.append(
                f"  {key}: checkpoint={stored.get(key)!r} vs current={current.get(key)!r}")
    if mismatches:
        print("ERROR: resume 参数与 checkpoint 不一致，拒绝混配：")
        for m in mismatches:
            print(m)
        sys.exit(1)


def save_cp(output_dir: Path, items_by_qid: Dict[int, Dict[str, Any]],
            run_config: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_config": run_config,
        "items": [items_by_qid[q] for q in sorted(items_by_qid)],
    }
    tmp = output_dir / "checkpoint.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, output_dir / "checkpoint.json")


def load_cp(output_dir: Path) -> tuple:
    cp_path = output_dir / "checkpoint.json"
    if not cp_path.exists():
        return {}, None
    with open(cp_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    items_by_qid = {}
    for it in data.get("items", []):
        items_by_qid[int(it["dataset_index"])] = it
    return items_by_qid, data.get("run_config")


def _new_item(src: Dict[str, Any], error: Optional[str] = None) -> Dict[str, Any]:
    item = {
        "dataset_index": src["dataset_index"],
        "di": src["di"],
        "db_id": src["db_id"],
        "question": src["question"],
        "gold_sql": src["gold_sql"],
        "difficulty": src.get("difficulty", ""),
        "candidates": [],
    }
    if error:
        item["error"] = error
    return item


def _count_model(item: Dict[str, Any], model: str) -> int:
    return sum(1 for c in item["candidates"] if c["model"] == model)


def _teardown_engine(llm: Any) -> None:
    if llm is None:
        return
    try:
        del llm
    except Exception:  # pragma: no cover
        pass
    gc.collect()


def main() -> None:
    args = parse_args()
    if args.n < 1:
        raise SystemExit("--n must be >= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"BIRD RFT 候选池 | model={args.model_path} | n={args.n} | "
          f"T={args.temperature} seed={args.seed} top_p={args.top_p} | "
          f"limit={args.limit} (start={args.start_index}) | "
          f"checkpoint-every={args.checkpoint_every} | "
          f"gen budget={args.max_gen_seconds}s")

    # ---- tokenizer（RFT checkpoint 自带；已核验与 base 词表/chat_template 一致）----
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True,
                                              trust_remote_code=True)

    items = load_bird_items(args.data_json, args.limit, args.start_index)
    requested_order = [it["dataset_index"] for it in items]
    requested_set = set(requested_order)
    src_by_qid = {it["dataset_index"]: it for it in items}
    print(f"Loaded {len(items)} BIRD items (start={args.start_index}, limit={args.limit})")

    # ---- 预构造 prompt（与 gen_bird_pool.py 逐字一致）----
    ddl_cache: Dict[str, str] = {}
    ids_by_qid: Dict[int, Optional[List[int]]] = {}
    t0 = time.perf_counter()
    for it in items:
        qid = it["dataset_index"]
        db_id = it["db_id"]
        try:
            if db_id not in ddl_cache:
                ddl_cache[db_id] = read_ddl(args.db_root, db_id)
            ddl = ddl_cache[db_id]
            prompt_text = ReasoningGeneratorAgent.build_prompt(
                question=it["question"], ddl_schema=ddl, schema_links=None,
                evidence=(it["evidence"] or None), dialect="sqlite")
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}], tokenize=False,
                add_generation_prompt=True)
            ids = tokenizer(chat, truncation=True,
                            max_length=args.max_prompt_tokens)["input_ids"]
            ids_by_qid[qid] = ids
        except Exception as exc:
            ids_by_qid[qid] = None
            print(f"[WARN] item {qid} db_id={db_id} prompt build failed: {exc}")
    print(f"prompt build done in {time.perf_counter() - t0:.1f}s")

    # ---- checkpoint / resume ----
    run_config = build_run_config(args)
    items_by_qid, stored_cfg = load_cp(output_dir)
    if stored_cfg is not None:
        validate_resume_config(stored_cfg, run_config)
    for qid in requested_order:
        if qid not in items_by_qid:
            items_by_qid[qid] = _new_item(src_by_qid[qid])
    for qid in requested_order:
        if ids_by_qid[qid] is None and not items_by_qid[qid].get("candidates") and \
                not items_by_qid[qid].get("error"):
            items_by_qid[qid] = _new_item(src_by_qid[qid], error="ddl_load_failed")
            print(f"[WARN] item {qid} recorded as ddl_load_failed")
    for qid in list(items_by_qid):
        if qid not in requested_set:
            del items_by_qid[qid]

    def pending_for() -> List[int]:
        out = []
        for qid in requested_order:
            item = items_by_qid[qid]
            if item.get("error"):
                continue
            if _count_model(item, MODEL_NAME) < args.n:
                out.append(qid)
        return out

    n_done = sum(1 for qid in requested_order
                 if not items_by_qid[qid].get("error")
                 and _count_model(items_by_qid[qid], MODEL_NAME) >= args.n)
    print(f"Resume: {len(items_by_qid)}/{len(items)} items touched "
          f"({n_done} fully done)")

    # ---- vLLM 采样参数（与 gen_bird_pool.py 一致）----
    from vllm import SamplingParams  # noqa: E402
    sampling_params = SamplingParams(n=args.n, temperature=args.temperature,
                                     top_p=args.top_p, seed=args.seed,
                                     max_tokens=args.max_new_tokens)

    gen_start = time.perf_counter()
    deadline = gen_start + args.max_gen_seconds
    MIN_BUDGET_FOR_ENGINE = 120.0

    def budget_left() -> float:
        return deadline - time.perf_counter()

    def run_model(llm: Any) -> bool:
        """处理 pending 直至完成或预算耗尽。返回 True=全部完成。"""
        pending = pending_for()
        while pending:
            if budget_left() <= 0:
                return False
            chunk = pending[:args.checkpoint_every]
            reqs = [{"prompt_token_ids": ids_by_qid[di]} for di in chunk]
            t0c = time.perf_counter()
            try:
                outputs = llm.generate(reqs, sampling_params)
            except Exception as exc:
                print(f"[pass:{MODEL_NAME}] generate failed on chunk starting at qid="
                      f"{chunk[0]}: {exc}")
                save_cp(output_dir, items_by_qid, run_config)
                raise
            dt = time.perf_counter() - t0c
            by_prompt = {tuple(o.prompt_token_ids): o for o in outputs}
            if len(by_prompt) != len(reqs):
                print(f"[WARN] matched {len(by_prompt)}/{len(reqs)} outputs by prompt ids")
            n_tokens = 0
            n_applied = 0
            lens = sorted(len(x.token_ids) for o in outputs for x in o.outputs)
            for di in chunk:
                o = by_prompt.get(tuple(ids_by_qid[di]))
                item = items_by_qid[di]
                if item.get("error"):
                    continue
                if o is None or len(o.outputs) < args.n:
                    print(f"[WARN] item {di}: {len(o.outputs) if o else 0}/{args.n} "
                          f"outputs, retry next slice")
                    continue
                for j in range(args.n):
                    text = tokenizer.decode(o.outputs[j].token_ids,
                                            skip_special_tokens=True)
                    parsed = VavSampler.extract_sql(text)
                    item["candidates"].append({
                        "model": MODEL_NAME,
                        "sql": parsed["sql"] if parsed["parse_success"] else "",
                        "parse_success": bool(parsed["parse_success"]),
                        "sample_idx": j,
                    })
                item["candidates"].sort(key=lambda c: (c["model"], c["sample_idx"]))
                n_applied += 1
            for o in outputs:
                for x in o.outputs:
                    n_tokens += len(x.token_ids)
            pending = pending[len(chunk):]
            save_cp(output_dir, items_by_qid, run_config)
            med = lens[len(lens) // 2] if lens else 0
            print(f"  [pass:{MODEL_NAME}] {n_applied}/{len(chunk)} items in {dt:.1f}s "
                  f"({n_tokens / max(dt, 1e-9):.0f} tok/s) | seq lens "
                  f"min/med/max = {lens[0] if lens else 0}/{med}/{lens[-1] if lens else 0} "
                  f"| remaining {len(pending)} | budget {budget_left():.0f}s")
        return True

    # ---- plain 引擎（RFT 全参模型）----
    engine_kwargs = dict(dtype="bfloat16", trust_remote_code=True, seed=0,
                         max_model_len=args.max_prompt_tokens + args.max_new_tokens)
    if args.enforce_eager:
        engine_kwargs["enforce_eager"] = True
    if args.max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs

    done = False
    pending = pending_for()
    if pending and budget_left() > MIN_BUDGET_FOR_ENGINE:
        from vllm import LLM  # noqa: E402
        print(f"[serve] building plain engine for {args.model_path}")
        t0e = time.perf_counter()
        llm = LLM(model=args.model_path, **engine_kwargs)
        print(f"[serve] engine ready in {time.perf_counter() - t0e:.1f}s")
        # warmup：单请求强制完成加载（同 gen_bird_pool._warmup 口径）
        sp_w = SamplingParams(n=1, temperature=1.0, top_p=1.0, seed=0, max_tokens=16)
        try:
            out = llm.generate([{"prompt_token_ids": ids_by_qid[pending[0]]}], sp_w)
            text = (out[0].outputs[0].text if out and out[0].outputs else "")
            print(f"[serve] warmup ok (first 60 chars): {text[:60]!r}")
        except Exception as exc:
            _teardown_engine(llm)
            save_cp(output_dir, items_by_qid, run_config)
            raise RuntimeError(f"[serve] warmup failed: {exc}") from exc
        done = run_model(llm)
        _teardown_engine(llm)
        save_cp(output_dir, items_by_qid, run_config)

    # ---- 收尾：checkpoint + items.json + summary.json ----
    save_cp(output_dir, items_by_qid, run_config)
    fully_done = all(
        (items_by_qid[qid].get("error") or
         _count_model(items_by_qid[qid], MODEL_NAME) >= args.n)
        for qid in requested_order)
    n_done = sum(1 for qid in requested_order
                 if not items_by_qid[qid].get("error")
                 and _count_model(items_by_qid[qid], MODEL_NAME) >= args.n)
    n_error = sum(1 for qid in requested_order if items_by_qid[qid].get("error"))
    final_items = [items_by_qid[qid] for qid in sorted(items_by_qid)
                   if qid in requested_set]
    with open(output_dir / "items.json", "w", encoding="utf-8") as fh:
        json.dump(final_items, fh, ensure_ascii=False, indent=1)

    cands = [c for qid in requested_order
             for c in items_by_qid[qid].get("candidates", [])
             if c["model"] == MODEL_NAME]
    n_parse = sum(1 for c in cands if c["parse_success"])
    summary = {
        "evaluator_type": EVALUATOR_TYPE,
        "is_official_bird_metric": False,
        "note": (
            "BIRD dev RFT 候选池生成（第 5 源）：{n_q} 题 × rft_bird_v3 × n={n} 采样"
            "候选（T={t} seed={s} top_p={p}，vLLM 单请求 n=16，plain 全参引擎）。"
            "prompt = sqlite_master DDL + question + evidence（有值才加，"
            "ReasoningGeneratorAgent.build_prompt canonical 模板，dialect=sqlite）"
            "+ chat template，截断 {mpt} token，与 gen_bird_pool.py 逐字一致（仅官方"
            "evidence，无 auto evidence / retriever 裁剪）；解析 = "
            "VavSampler.extract_sql。items.json 不去重（裁决阶段做）。gpudebug 50min"
            "切片：生成预算 {budget}s，超预算落 checkpoint 正常退出由链脚本重提续跑；"
            "fully_done=False 表示还需重提。"
        ).format(n_q=len(items), n=args.n, t=args.temperature, s=args.seed,
                 p=args.top_p, mpt=args.max_prompt_tokens, budget=args.max_gen_seconds),
        "total_requested": len(requested_order),
        "fully_done": bool(fully_done),
        "completed_items": n_done,
        "error_items": n_error,
        "per_model": {
            MODEL_NAME: {
                "completed_items": n_done,
                "candidates": len(cands),
                "parse_success": n_parse,
                "parse_rate": round(n_parse / len(cands), 4) if cands else 0.0,
                "serving": "plain",
            },
        },
        "run_config": run_config,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 66)
    print("  BIRD RFT CANDIDATE POOL SUMMARY")
    print("=" * 66)
    s = summary["per_model"][MODEL_NAME]
    print(f"  {MODEL_NAME} [plain]: {s['completed_items']}/{len(requested_order)} items "
          f"| {s['candidates']} candidates | parse {s['parse_success']}/"
          f"{s['candidates']} ({s['parse_rate']:.1%})")
    print(f"  fully_done: {fully_done} ({n_done}/{len(requested_order)} completed, "
          f"error={n_error})")
    print(f"\nItems saved to:   {output_dir / 'items.json'}")
    print(f"Summary saved to: {output_dir / 'summary.json'}")
    print("EXIT fully_done=%s" % fully_done)


if __name__ == "__main__":
    main()
