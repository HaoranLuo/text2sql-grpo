#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BIRD dev 大候选池生成：4 源 checkpoint × 每题 16 采样（vLLM，单引擎多 LoRA）。

把 B1（Spider）大候选池管线移植到 BIRD dev（dev_20240627，1534 题 11 库，
每库单实例 sqlite，无 test-suite 变体）：

  - 源 checkpoint（LoRA adapter，base=models/Qwen2.5-Coder-3B-Instruct）：
    sft_phase1 / sft_v2 / sft_v3 / p2a_500。每题每模型 n=16 采样
    （T=1.0, top_p=1.0, seed=0, max_new_tokens=2048；vLLM 单请求 n=16，
    共享一次 prefill，与 eval_pool_b1 同口径）。
  - 单引擎多 LoRA：enable_lora=True + max_loras=4，逐模型换 LoRARequest，
    不重启引擎（省 4 次引擎加载）；每 adapter 先 1 请求 warmup 强制加载，
    尽早暴露 LoRA 兼容性失败。某 adapter 加载失败 → peft merge 成
    checkpoints/<name>_merged（envs/reasoning3b 子进程，脚本取自
    eval_pool_b1）再以独立 plain 引擎服务该模型（sft_v2_merged /
    sft_v3_merged 已存在则直接复用）。
  - prompt = BIRD schema DDL（sqlite_master 的 CREATE TABLE 语句按表名排序
    拼接）+ question + evidence（有值才加），复用
    ReasoningGeneratorAgent.build_prompt canonical 模板（dialect=sqlite）
    + chat template（add_generation_prompt=True）；截断 3072 token。
    evidence 来源（--evidence-json 可选注入，默认不注入与原池同口径）：
    --evidence-json 传入且文件存在时，auto evidence（SEED 式，question_id→文本）
    优先、dev.json 官方 evidence 兜底；不传该参数时行为与改造前完全一致
    （仅 dev.json 官方 evidence）。
  - 解析 = finer_port/sampler.py VavSampler.extract_sql（返回 dict，
    取 ["sql"]；与 Spider 管线同口径）。
  - 输出 outputs/eval_pool_bird/items.json：dataset_index=question_id、db_id、
    question、gold_sql、difficulty、candidates[{model,sql,parse_success,
    sample_idx}]（按 (model, sample_idx) 排序；不去重——裁决阶段做）。

切片与续跑（gpudebug 墙钟 50min，生成预算 --max-gen-seconds）：
  - 生成时间（不含引擎加载）超过预算即落 checkpoint 后正常退出（exit 0），
    由链脚本重提续跑；checkpoint 每 --checkpoint-every 题写一次，被杀最多
    丢一个 chunk。serving_modes（lora/merged）跨切片持久化，不重复探测。
  - 混配保护：checkpoint 的 run_config 与本次参数逐字段比对，不一致 exit(1)。

用法：
    envs/vllmenv/bin/python src/gen_bird_pool.py --limit 1534
    envs/vllmenv/bin/python src/gen_bird_pool.py --limit 10 \
        --output-dir outputs/eval_pool_bird_smoke
"""
import argparse
import gc
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# 离线模式：避免 vLLM/transformers 尝试访问 HF Hub
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
DEFAULT_BASE_MODEL = str(_PROJECT / "models" / "Qwen2.5-Coder-3B-Instruct")
DEFAULT_CHECKPOINTS_DIR = str(_PROJECT / "checkpoints")
DEFAULT_OUTPUT_DIR = str(_PROJECT / "outputs" / "eval_pool_bird")
DEFAULT_MERGE_PYTHON = str(_PROJECT / "envs" / "reasoning3b" / "bin" / "python")
DEFAULT_LORAS = "sft_phase1,sft_v2,sft_v3,p2a_500"
EVALUATOR_TYPE = "bird_candidate_pool"


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BIRD dev 大候选池生成：4 checkpoint × n=16 采样（单引擎多 LoRA）")
    p.add_argument("--data-json", default=DEFAULT_DATA_JSON)
    p.add_argument("--db-root", default=DEFAULT_DB_ROOT)
    p.add_argument("--evidence-json", default=None,
                   help="SEED 式自动 evidence JSON（question_id→evidence 文本，如 "
                        "data/bird/auto_evidence.json）。不传则仅用 dev.json 官方 "
                        "evidence（与原池同口径）；传入且文件存在时 auto evidence "
                        "优先、官方兜底，文件缺失回退官方 evidence。")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--limit", type=int, required=True,
                   help="评估条数（全量 1534；冒烟 10）——必填，防误触全量")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--loras", default=DEFAULT_LORAS,
                   help="checkpoint 名列表（逗号分隔），顺序 = pass 顺序")
    p.add_argument("--checkpoints-dir", default=DEFAULT_CHECKPOINTS_DIR)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--n", type=int, default=16, help="每题每模型采样候选数")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-prompt-tokens", type=int, default=3072,
                   help="prompt 截断长度（BIRD 实测最长 2386，3072 全量不截断）")
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="每处理 N 题写一次 checkpoint（也即生成 chunk 大小）")
    p.add_argument("--max-gen-seconds", type=int, default=2400,
                   help="单切片生成时间预算（不含引擎加载）；超预算落 checkpoint 正常退出")
    p.add_argument("--max-loras", type=int, default=4,
                   help="单引擎同时挂载的 LoRA adapter 数上限（>= len(--loras)）")
    p.add_argument("--merge-python", default=DEFAULT_MERGE_PYTHON)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-num-seqs", type=int, default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# BIRD 数据加载（dev.json + sqlite_master DDL）
# ---------------------------------------------------------------------------

def load_bird_items(data_json: str, limit: int, start_index: int,
                    evidence_map: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """装载 dev.json 条目。evidence_map（auto evidence，question_id→文本）优先，
    缺失时回退 dev.json 官方 evidence；evidence_map=None 时与原池同口径。"""
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
            "evidence": ((evidence_map or {}).get(str(it["question_id"]))
                         or it.get("evidence") or ""),
            "gold_sql": it.get("SQL") or "",
            "difficulty": it.get("difficulty", ""),
        })
    return items


def read_ddl(db_root: str, db_id: str) -> str:
    """BIRD schema DDL：sqlite_master 的 CREATE TABLE 语句按表名排序拼接。"""
    db_path = Path(db_root) / db_id / f"{db_id}.sqlite"
    if not db_path.is_file():
        raise FileNotFoundError(f"database file not found: {db_path}")
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
# 题级状态 + checkpoint（自包含，不依赖 spider_utils）
# ---------------------------------------------------------------------------

def build_run_config(args: argparse.Namespace, lora_names: List[str]) -> Dict[str, Any]:
    return {
        "evaluator_type": EVALUATOR_TYPE,
        "data_json": str(Path(args.data_json).resolve()),
        "db_root": str(Path(args.db_root).resolve()),
        "evidence_json": (str(Path(args.evidence_json).resolve())
                          if args.evidence_json else None),
        "start_index": args.start_index,
        "limit": args.limit,
        "base_model": str(Path(args.base_model).resolve()),
        "checkpoints_dir": str(Path(args.checkpoints_dir).resolve()),
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "loras": lora_names,
        "checkpoint_every": args.checkpoint_every,
    }


def validate_resume_config(stored: Dict[str, Any], current: Dict[str, Any]) -> None:
    mismatches = []
    for key in sorted(set(list(stored.keys()) + list(current.keys()))):
        if stored.get(key) != current.get(key):
            mismatches.append(f"  {key}: checkpoint={stored.get(key)!r} vs current={current.get(key)!r}")
    if mismatches:
        print("ERROR: resume 参数与 checkpoint 不一致，拒绝混配：")
        for m in mismatches:
            print(m)
        sys.exit(1)


def save_cp(output_dir: Path, items_by_qid: Dict[int, Dict[str, Any]],
            serving_modes: Dict[str, str], run_config: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_config": run_config,
        "serving_modes": serving_modes,
        "items": [items_by_qid[q] for q in sorted(items_by_qid)],
    }
    tmp = output_dir / "checkpoint.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, output_dir / "checkpoint.json")


def load_cp(output_dir: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, str], Optional[Dict[str, Any]]]:
    cp_path = output_dir / "checkpoint.json"
    if not cp_path.exists():
        return {}, {}, None
    with open(cp_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    items_by_qid = {}
    for it in data.get("items", []):
        items_by_qid[int(it["dataset_index"])] = it
    return items_by_qid, dict(data.get("serving_modes", {})), data.get("run_config")


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


# ---------------------------------------------------------------------------
# 引擎（LoRA 多 adapter 单引擎 / merged plain 引擎）与 merge 回退
# ---------------------------------------------------------------------------

_MERGE_SCRIPT = r"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base, lora, out = sys.argv[1], sys.argv[2], sys.argv[3]
print("[merge] loading tokenizer + base model (CPU, bf16)...", flush=True)
tok = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True)
print("[merge] loading LoRA adapter...", flush=True)
model = PeftModel.from_pretrained(model, lora)
print("[merge] merge_and_unload...", flush=True)
model = model.merge_and_unload()
print("[merge] saving merged weights to %s ..." % out, flush=True)
model.save_pretrained(out)
tok.save_pretrained(out)
print("[merge] DONE", flush=True)
"""


def _merged_ok(merged_dir: Path) -> bool:
    return merged_dir.is_dir() and any(merged_dir.glob("*.safetensors"))


def merge_lora_into(base_model: str, lora_path: Path, merged_dir: Path,
                    merge_python: str) -> None:
    if _merged_ok(merged_dir):
        print(f"[merge] {merged_dir} already exists with safetensors, reuse")
        return
    print(f"[merge] peft-merging {lora_path} -> {merged_dir} (python={merge_python})")
    t0 = time.perf_counter()
    merged_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([merge_python, "-c", _MERGE_SCRIPT, base_model,
                    str(lora_path), str(merged_dir)], check=True)
    print(f"[merge] done in {time.perf_counter() - t0:.1f}s")


def _teardown_engine(llm: Any) -> None:
    if llm is None:
        return
    try:
        del llm
    except Exception:  # pragma: no cover
        pass
    gc.collect()
    try:
        free, total = torch.cuda.mem_get_info()
        if free <= 0.5 * total:
            print("[serve] waiting for GPU memory release after engine teardown...")
            deadline = time.time() + 60.0
            while time.time() < deadline:
                time.sleep(2.0)
                free, total = torch.cuda.mem_get_info()
                if free > 0.5 * total:
                    break
        print(f"[serve] GPU memory after teardown: "
              f"{free / 1024**3:.1f}/{total / 1024**3:.1f} GiB free")
    except Exception:  # pragma: no cover
        pass


def _engine_kwargs(args: argparse.Namespace, enable_lora: bool) -> Dict[str, Any]:
    kwargs = dict(dtype="bfloat16", trust_remote_code=True, seed=0,
                  max_model_len=args.max_prompt_tokens + args.max_new_tokens)
    if enable_lora:
        kwargs.update(enable_lora=True, max_loras=args.max_loras, max_lora_rank=32)
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    return kwargs


def _build_lora_engine(args: argparse.Namespace, lora_names: List[str]):
    """返回 (llm, {name: LoRARequest})。引擎参数对齐 eval_pool_b1.py。"""
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    llm = LLM(model=args.base_model, **_engine_kwargs(args, enable_lora=True))
    lrs = {}
    for i, name in enumerate(lora_names):
        lora_path = Path(args.checkpoints_dir) / name
        if not lora_path.is_dir():
            raise RuntimeError(f"LoRA 目录不存在: {lora_path}")
        lrs[name] = LoRARequest(name, i + 1, str(lora_path))
    return llm, lrs


def _build_plain_engine(args: argparse.Namespace, model_dir: str):
    from vllm import LLM
    return LLM(model=model_dir, **_engine_kwargs(args, enable_lora=False))


def _warmup(llm: Any, lr: Any, sample_ids: List[int]) -> None:
    from vllm import SamplingParams
    sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, seed=0, max_tokens=16)
    out = llm.generate([{"prompt_token_ids": sample_ids}], sp, lora_request=lr)
    text = (out[0].outputs[0].text if out and out[0].outputs else "")
    print(f"[serve] warmup ok (first 60 chars): {text[:60]!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    lora_names = [s.strip() for s in args.loras.split(",") if s.strip()]
    if not lora_names:
        raise SystemExit("--loras must not be empty")
    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    if args.max_loras < len(lora_names):
        raise SystemExit(f"--max-loras {args.max_loras} < len(--loras) {len(lora_names)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"BIRD 候选池 | loras={lora_names} | n={args.n} | T={args.temperature} "
          f"seed={args.seed} top_p={args.top_p} | limit={args.limit} "
          f"(start={args.start_index}) | checkpoint-every={args.checkpoint_every} | "
          f"gen budget={args.max_gen_seconds}s | "
          f"evidence-json={args.evidence_json or '(none, 官方 evidence 口径)'}")

    # ---- tokenizer（与 eval_pool_b1 相同设置）----
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- auto evidence（--evidence-json；缺省 None = 与原池同口径）----
    evidence_map: Optional[Dict[str, str]] = None
    if args.evidence_json:
        ev_path = Path(args.evidence_json)
        if ev_path.is_file():
            with open(ev_path, "r", encoding="utf-8") as fh:
                evidence_map = json.load(fh)
            print(f"Auto evidence loaded: {len(evidence_map)} entries from {ev_path}")
        else:
            print(f"[WARN] --evidence-json file not found ({ev_path}), "
                  f"回退 dev.json 官方 evidence")

    items = load_bird_items(args.data_json, args.limit, args.start_index,
                            evidence_map)
    n_auto_ev = sum(1 for it in items
                    if evidence_map and str(it["dataset_index"]) in evidence_map)
    print(f"evidence 命中: auto={n_auto_ev}/{len(items)} items")
    requested_order = [it["dataset_index"] for it in items]
    requested_set = set(requested_order)
    src_by_qid = {it["dataset_index"]: it for it in items}
    print(f"Loaded {len(items)} BIRD items (start={args.start_index}, limit={args.limit})")

    # ---- 预构造 prompt（DDL + question + evidence；与 canonical 模板一致）----
    ddl_cache: Dict[str, str] = {}
    ids_by_qid: Dict[int, Optional[List[int]]] = {}
    prompt_text_by_qid: Dict[int, Optional[str]] = {}
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
            prompt_text_by_qid[qid] = prompt_text
            ids_by_qid[qid] = ids
        except Exception as exc:
            prompt_text_by_qid[qid] = None
            ids_by_qid[qid] = None
            print(f"[WARN] item {qid} db_id={db_id} prompt build failed: {exc}")
    print(f"prompt build done in {time.perf_counter() - t0:.1f}s")

    # ---- checkpoint / resume（自包含协议 + 混配保护）----
    run_config = build_run_config(args, lora_names)
    items_by_qid, serving_modes, stored_cfg = load_cp(output_dir)
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
    # 丢弃 checkpoint 中不属于本次 requested 的条目
    for qid in list(items_by_qid):
        if qid not in requested_set:
            del items_by_qid[qid]

    def pending_for(model: str) -> List[int]:
        out = []
        for qid in requested_order:
            item = items_by_qid[qid]
            if item.get("error"):
                continue
            if _count_model(item, model) < args.n:
                out.append(qid)
        return out

    n_done = sum(1 for qid in requested_order
                 if not items_by_qid[qid].get("error")
                 and all(_count_model(items_by_qid[qid], m) >= args.n for m in lora_names))
    print(f"Resume: {len(items_by_qid)}/{len(items)} items touched ({n_done} fully done) | "
          f"serving_modes={serving_modes}")

    # ---- vLLM 采样参数（与 VavSampler._generate 对齐）----
    from vllm import SamplingParams  # noqa: E402
    sampling_params = SamplingParams(n=args.n, temperature=args.temperature,
                                     top_p=args.top_p, seed=args.seed,
                                     max_tokens=args.max_new_tokens)

    gen_start = time.perf_counter()
    wall_start = gen_start
    deadline = gen_start + args.max_gen_seconds
    MIN_BUDGET_FOR_ENGINE = 120.0  # 预算不足 2 分钟不再起新引擎（避免白加载）

    slice_stats: Dict[str, Dict[str, Any]] = {}

    def budget_left() -> float:
        return deadline - time.perf_counter()

    def run_model(llm: Any, lr: Any, model: str) -> bool:
        """处理 model 的 pending 直至完成或预算耗尽。返回 True=模型完成。"""
        pending = pending_for(model)
        stats = slice_stats.setdefault(model, {
            "serving": "lora" if lr is not None else "merged",
            "generation_seconds": 0.0, "generated_tokens": 0,
            "requests": 0, "applied_items": 0})
        while pending:
            if budget_left() <= 0:
                return False
            chunk = pending[:args.checkpoint_every]
            reqs = [{"prompt_token_ids": ids_by_qid[di]} for di in chunk]
            t0c = time.perf_counter()
            try:
                outputs = llm.generate(reqs, sampling_params, lora_request=lr)
            except Exception as exc:
                print(f"[pass:{model}] generate failed on chunk starting at qid="
                      f"{chunk[0]}: {exc}")
                save_cp(output_dir, items_by_qid, serving_modes, run_config)
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
                        "model": model,
                        "sql": parsed["sql"] if parsed["parse_success"] else "",
                        "parse_success": bool(parsed["parse_success"]),
                        "sample_idx": j,
                    })
                item["candidates"].sort(key=lambda c: (c["model"], c["sample_idx"]))
                n_applied += 1
            for o in outputs:
                for x in o.outputs:
                    n_tokens += len(x.token_ids)
            stats["generation_seconds"] += dt
            stats["generated_tokens"] += n_tokens
            stats["requests"] += len(reqs)
            stats["applied_items"] += n_applied
            pending = pending[len(chunk):]
            save_cp(output_dir, items_by_qid, serving_modes, run_config)
            med = lens[len(lens) // 2] if lens else 0
            print(f"  [pass:{model}] {n_applied}/{len(chunk)} items in {dt:.1f}s "
                  f"({n_tokens / max(dt, 1e-9):.0f} tok/s) | seq lens "
                  f"min/med/max = {lens[0] if lens else 0}/{med}/{lens[-1] if lens else 0} "
                  f"| remaining {len(pending)} | budget {budget_left():.0f}s")
        return True

    # ---- Phase A：LoRA 多 adapter 单引擎 ----
    lora_active = [m for m in lora_names
                   if serving_modes.get(m, "lora") == "lora" and pending_for(m)]
    if lora_active and budget_left() > MIN_BUDGET_FOR_ENGINE:
        print(f"[serve] building single LoRA engine (max_loras={args.max_loras}) "
              f"for {lora_active}")
        t0e = time.perf_counter()
        llm, lrs = _build_lora_engine(args, lora_names)
        print(f"[serve] engine ready in {time.perf_counter() - t0e:.1f}s")
        merged_new: List[str] = []
        for m in list(lora_active):
            if budget_left() <= 0:
                break
            try:
                _warmup(llm, lrs[m], ids_by_qid[pending_for(m)[0]])
            except Exception as exc:
                if "Free memory on device" in str(exc):
                    _teardown_engine(llm)
                    raise RuntimeError(
                        f"[serve] engine init for {m} 显存不足（资源问题，非 LoRA "
                        f"兼容问题，不做 merge 回退）: {exc}") from exc
                print(f"[serve] LoRA serving failed for {m}: {exc}")
                print(f"[serve] falling back to merged weights for {m}")
                serving_modes[m] = "merged"
                merged_new.append(m)
                continue
            done = run_model(llm, lrs[m], m)
            save_cp(output_dir, items_by_qid, serving_modes, run_config)
            if not done:
                break
        _teardown_engine(llm)
        # 本切片内刚转为 merged 的模型：立即 merge（CPU 子进程，不占 GPU 预算太多）
        for m in merged_new:
            if budget_left() > 0:
                merge_lora_into(args.base_model, Path(args.checkpoints_dir) / m,
                                Path(args.checkpoints_dir) / f"{m}_merged",
                                args.merge_python)

    # ---- Phase B：merged 回退模型（每模型一个 plain 引擎）----
    for m in lora_names:
        if serving_modes.get(m) != "merged" or not pending_for(m):
            continue
        if budget_left() <= MIN_BUDGET_FOR_ENGINE:
            break
        merged_dir = Path(args.checkpoints_dir) / f"{m}_merged"
        if not _merged_ok(merged_dir):
            merge_lora_into(args.base_model, Path(args.checkpoints_dir) / m,
                            merged_dir, args.merge_python)
        print(f"[serve] building merged engine for {m} ({merged_dir})")
        llm = _build_plain_engine(args, str(merged_dir))
        _warmup(llm, None, ids_by_qid[pending_for(m)[0]])
        done = run_model(llm, None, m)
        _teardown_engine(llm)
        save_cp(output_dir, items_by_qid, serving_modes, run_config)
        if not done:
            break

    # ---- 收尾：checkpoint + items.json + summary.json ----
    save_cp(output_dir, items_by_qid, serving_modes, run_config)
    fully_done = all(
        (items_by_qid[qid].get("error") or
         all(_count_model(items_by_qid[qid], m) >= args.n for m in lora_names))
        for qid in requested_order)
    n_done = sum(1 for qid in requested_order
                 if not items_by_qid[qid].get("error")
                 and all(_count_model(items_by_qid[qid], m) >= args.n for m in lora_names))
    n_error = sum(1 for qid in requested_order if items_by_qid[qid].get("error"))
    final_items = [items_by_qid[qid] for qid in sorted(items_by_qid) if qid in requested_set]
    with open(output_dir / "items.json", "w", encoding="utf-8") as fh:
        json.dump(final_items, fh, ensure_ascii=False, indent=1)

    per_model: Dict[str, Dict[str, Any]] = {}
    for m in lora_names:
        cands = [c for qid in requested_order
                 for c in items_by_qid[qid].get("candidates", [])
                 if c["model"] == m]
        n_parse = sum(1 for c in cands if c["parse_success"])
        st = slice_stats.get(m, {})
        per_model[m] = {
            "completed_items": sum(
                1 for qid in requested_order
                if not items_by_qid[qid].get("error")
                and _count_model(items_by_qid[qid], m) >= args.n),
            "candidates": len(cands),
            "parse_success": n_parse,
            "parse_rate": round(n_parse / len(cands), 4) if cands else 0.0,
            "serving": serving_modes.get(m, "lora"),
            "this_slice_gen_seconds": round(st.get("generation_seconds", 0.0), 2),
            "this_slice_gen_tokens": st.get("generated_tokens", 0),
            "this_slice_requests": st.get("requests", 0),
        }

    total_wall = time.perf_counter() - wall_start
    summary = {
        "evaluator_type": EVALUATOR_TYPE,
        "is_official_bird_metric": False,
        "note": (
            "BIRD dev 大候选池生成：{n_q} 题 × {loras} × n={n} 采样候选（T={t} "
            "seed={s} top_p={p}，vLLM 单请求 n=16，单引擎 max_loras={ml} 逐模型换 "
            "LoRARequest）。prompt = sqlite_master DDL + question + evidence（有值才加，"
            "ReasoningGeneratorAgent.build_prompt canonical 模板）+ chat template，"
            "evidence 来源 = auto_evidence.json（SEED 式，--evidence-json）优先、"
            "dev.json 官方兜底；不传 --evidence-json 时与原池同口径（仅官方 evidence）。"
            "截断 {mpt} token；解析 = VavSampler.extract_sql（与 Spider 管线同口径）。"
            "items.json 不去重（裁决阶段做）。serving: lora=vLLM LoRARequest；"
            "merged=peft 合并权重回退。gpudebug 50min 切片：生成预算 "
            "{budget}s，超预算落 checkpoint 正常退出由链脚本重提续跑；"
            "fully_done=False 表示还需重提。"
        ).format(n_q=len(items), loras=", ".join(lora_names), n=args.n,
                 t=args.temperature, s=args.seed, p=args.top_p, ml=args.max_loras,
                 mpt=args.max_prompt_tokens, budget=args.max_gen_seconds),
        "total_requested": len(requested_order),
        "fully_done": bool(fully_done),
        "completed_items": n_done,
        "error_items": n_error,
        "serving_modes": serving_modes,
        "per_model": per_model,
        "total_wall_seconds": round(total_wall, 2),
        "run_config": run_config,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 66)
    print("  BIRD CANDIDATE POOL SUMMARY")
    print("=" * 66)
    for m in lora_names:
        s = per_model[m]
        print(f"  {m} [{s['serving']}]: {s['completed_items']}/{len(requested_order)} items "
              f"| {s['candidates']} candidates | parse "
              f"{s['parse_success']}/{s['candidates']} ({s['parse_rate']:.1%}) | "
              f"slice gen {s['this_slice_gen_seconds']:.0f}s")
    print(f"  fully_done: {fully_done} ({n_done}/{len(requested_order)} completed, "
          f"error={n_error})")
    print(f"  total wall: {total_wall:.0f}s")
    print(f"\nItems saved to:   {output_dir / 'items.json'}")
    print(f"Summary saved to: {output_dir / 'summary.json'}")
    print("EXIT fully_done=%s" % fully_done)


if __name__ == "__main__":
    main()
