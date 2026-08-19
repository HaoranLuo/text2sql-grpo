#!/usr/bin/env python3
"""B1 旗舰实验：大候选池生成（Spider dev × {sft_phase1, sft_v2} × n=16，vLLM 采样）。

目的：为裁决阶段实验（多数投票 / 质心过滤 / 大池子一致性分析）生成并入库每个
checkpoint 每题 n=16 个采样候选（T=1.0, seed=0, top_p=1.0, max_new_tokens=2048），
全量 dev 1034 题。生成一次，实验 N 次（中央候选库 src/candidate_store.py）。

与 eval_mem_filter 的一致性（跨实验口径对齐，硬要求）：
  1. Prompt 与 mem_filter 完全一致：VavSampler prompt_style="default" 的 canonical
     prompt。本脚本不实例化 VavSampler（那是 HF 加载路径），而是等价复刻其构建
     路径，来源函数：
       - finer_port/sampler.py VavSampler.build_prompt (L142-150)
         → src/reasoning_generator_agent.py ReasoningGeneratorAgent.build_prompt(
             question=..., ddl_schema=..., schema_links=None, evidence=None,
             dialect="sqlite")
       - finer_port/sampler.py VavSampler.build_chat_text (L152-170) default 分支：
         messages=[{"role":"user","content":prompt_text}] +
         tokenizer.apply_chat_template(..., add_generation_prompt=True)
       - DDL 口径 = SpiderLoader.format_ddl（sqlite_master 优先，tables.json 兜底），
         与 eval_mem_filter.py main() 的预构造 (L855-866) 相同。
       - 截断口径 = tokenizer(..., truncation=True, max_length=1536)，与
         VavSampler.sample_batch 的 max_prompt_tokens=1536 相同。
  2. 解析与 mem_filter 完全一致：直接调用 finer_port/sampler.py VavSampler.
     extract_sql (L175-194)（</think> 尾部 → ```sql 块 → 泛块 → plain SELECT/WITH），
     逐候选记录 parse_success。空 SQL 记为 sql="" + parse_success=False。
  3. 采样参数与 VavSampler._generate 对齐：T=1.0 / top_p=1.0 / n=16。HF 用单次
     forward num_return_sequences=16；vLLM 用单请求 n=16（共享一次 prefill，逐
     序列独立采样）。采样类实验，不要求贪心逐字节一致性（任务口径）。

vLLM 服务（参考 src/eval_5p_vllm.py 的加载写法）：
  - 每 checkpoint 一个 pass，顺序 = --loras 顺序（默认 sft_phase1 先）。每 pass
    起一个引擎（enable_lora=True + LoRARequest，max_loras=1），warmup 1 请求强制
    加载 adapter，尽早暴露 LoRA 兼容性失败。
  - sft_phase1 LoRA 加载失败 → peft merge 出 checkpoints/sft_phase1_merged
    （新目录，不覆盖已有；merge 走子进程 envs/reasoning3b 的 peft 0.14，因
    vllmenv 未装 peft）；sft_v2 失败 → 复用已有 checkpoints/sft_v2_merged。
    合并权重路径下候选库 lora 名仍记 checkpoint 名（口径不变）。
  - 每题 1 个请求 n=16（prompt_token_ids 直送，与 eval_5p_vllm 相同，规避 vLLM
    端 tokenizer/chat-template 差异）；输出按 prompt_token_ids 回配到题。

断点续跑（50min 切片 + --checkpoint-every 25）：
  - 每题需两个模型各 16 候选才算完成。checkpoint.items 允许"半完成"条目（先持久
    化再继续，避免跨 pass 丢已生成候选）。这是对 eval_mem_filter 严格协议
    （completed_indices == items）的**有意放宽**：那里每题一次成型，本脚本双
    pass 才需要半完成态。resume 时按"每模型候选数 < n 即重试该模型"重算 pending；
    混配保护仍复用 spider_utils.validate_resume_config（逐字段比对，不一致
    sys.exit(1)）。
  - 生成失败：chunk 级异常 → 落 checkpoint 后 re-raise（下个切片 resume 重试）；
    题级输出不足 → 不落候选、下个切片重试；DDL 加载失败条目打 error 字段计入
    summary（不再重试）。

输出 outputs/eval_pool_b1/items.json 每题结构（任务口径）：
    {"dataset_index": int, "di": int, "db_id": str, "question": str, "gold_sql": str,
     "candidates": [{"model": "sft_phase1"|"sft_v2", "sql": str,
                     "parse_success": bool, "sample_idx": 0-15}]}
  candidates 按 (model, sample_idx) 排序；去重不做（裁决阶段做）。异常条目另带
  "error" 字段。

候选库 ingest：model="qwen2.5-coder-3b", lora=checkpoint 名, temperature=1.0,
  seed=0, prompt=每题 canonical prompt 文本（candidate_store 内部算 prompt_hash；
  库按 (di, model, lora, T, seed, prompt_hash) 索引，每题 16 候选同 prompt 同
  hash，库内按 norm_sql 去重是 store 的既有设计）。每个 pass 结束时 ingest 一次
  （幂等，切片中断也不丢已完成的 pass）。

用法：
    python src/eval_pool_b1.py --limit 1034 --output-dir outputs/eval_pool_b1 \
        --loras sft_phase1,sft_v2 --n 16 --checkpoint-every 25
"""
import argparse
import gc
import json
import os
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

from spider_utils import (  # noqa: E402
    SpiderLoader,
    build_run_config,
    validate_resume_config,
    save_checkpoint,
    load_checkpoint,
)
# canonical prompt 来源：ReasoningGeneratorAgent.build_prompt（sampler.py 复用同一函数）
from reasoning_generator_agent import ReasoningGeneratorAgent  # noqa: E402
# 解析来源：VavSampler.extract_sql（与 mem_filter 完全同口径）；模块级 import 仅
# 引入 torch/transformers，不加载模型。
from sampler import VavSampler  # noqa: E402

try:
    from candidate_store import ingest as _store_ingest
    from candidate_store import query as _store_query
    from candidate_store import prompt_hash as _store_prompt_hash
    _STORE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _STORE_AVAILABLE = False

DEFAULT_BASE_MODEL = str(_PROJECT / "models" / "Qwen2.5-Coder-3B-Instruct")
DEFAULT_SPIDER_DIR = str(_PROJECT / "data" / "spider_data")
DEFAULT_CHECKPOINTS_DIR = str(_PROJECT / "checkpoints")
DEFAULT_OUTPUT_DIR = str(_PROJECT / "outputs" / "eval_pool_b1")
DEFAULT_MERGE_PYTHON = str(_PROJECT / "envs" / "reasoning3b" / "bin" / "python")
EVALUATOR_TYPE = "b1_candidate_pool"
STORE_MODEL_NAME = "qwen2.5-coder-3b"  # 候选库中的模型名约定（与 eval_5p_vllm 一致）
SPIDER_DEV_SIZE = 1034                  # Spider dev 全量题数（外推基准）


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B1 大候选池生成：双 checkpoint × n=16 采样，解析落盘 + 候选库入库"
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--spider-dir", default=DEFAULT_SPIDER_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, required=True,
                        help="评估条数（全量 1034；冒烟 10）——必填，防误触全量")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--loras", default="sft_phase1,sft_v2",
                        help="checkpoint 名列表（逗号分隔），顺序 = pass 顺序")
    parser.add_argument("--only-lora", default=None,
                        help="本次进程只处理指定 checkpoint（run_config 仍记录完整 "
                             "--loras 列表，保证 resume 混配保护一致；用于每模型独立"
                             "进程的切片调度，规避同进程两引擎的显存回收竞态）")
    parser.add_argument("--checkpoints-dir", default=DEFAULT_CHECKPOINTS_DIR,
                        help="LoRA / 合并权重所在目录")
    parser.add_argument("--n", type=int, default=16, help="每题每模型采样候选数")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="与训练 max_completion / mem_filter 对齐")
    parser.add_argument("--max-prompt-tokens", type=int, default=1536,
                        help="prompt 截断长度（与 VavSampler 一致）")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="每处理 N 题写一次 checkpoint（也即生成 chunk 大小）")
    parser.add_argument("--merge-python", default=DEFAULT_MERGE_PYTHON,
                        help="peft merge 子进程解释器（vllmenv 无 peft，用 reasoning3b）")
    parser.add_argument("--no-cache", action="store_true",
                        help="关闭候选库读写")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="vLLM 关闭 CUDA graph（重试用）")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="vLLM 单批最大序列数（默认 vLLM 自定）")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 题级状态（items 结构 + 完成判定 + pending 计算）
# ---------------------------------------------------------------------------

def _new_item(di: int, src: Dict[str, Any], error: Optional[str] = None) -> Dict[str, Any]:
    item = {
        "dataset_index": di,
        "di": di,
        "db_id": src["db_id"],
        "question": src["question"],
        "gold_sql": src["query"],
        "candidates": [],
    }
    if error:
        item["error"] = error
    return item


def _count_model(item: Dict[str, Any], model: str) -> int:
    return sum(1 for c in item["candidates"] if c["model"] == model)


def _item_done(item: Dict[str, Any], loras: List[str], n: int) -> bool:
    """error 条目视为完成（不再重试）；否则需所有模型各 >= n 候选。"""
    if item.get("error"):
        return True
    return all(_count_model(item, m) >= n for m in loras)


def _pending_for(model: str, requested_order: List[int],
                 items_by_di: Dict[int, Dict[str, Any]], n: int) -> List[int]:
    out = []
    for di in requested_order:
        item = items_by_di.get(di)
        if item is None:
            out.append(di)          # 未处理过：该模型待生成
        elif item.get("error"):
            continue                # 永久错误条目：跳过
        elif _count_model(item, model) < n:
            out.append(di)          # 半完成：该模型待生成
    return out


def _save_checkpoint(output_dir: Path, items_by_di: Dict[int, Dict[str, Any]],
                     loras: List[str], n: int, run_config: Dict[str, Any]) -> None:
    done = sorted(di for di, it in items_by_di.items() if _item_done(it, loras, n))
    items_list = [items_by_di[di] for di in sorted(items_by_di)]
    save_checkpoint(output_dir, {"completed_indices": done, "items": items_list},
                    run_config)


# ---------------------------------------------------------------------------
# vLLM 服务（LoRA 优先，失败回退合并权重）
# ---------------------------------------------------------------------------

# peft merge 子进程脚本（跑在 envs/reasoning3b：peft 0.14；CPU 合并，3B bf16 峰值
# 内存 ~20GB，节点 257GB / 作业 32G 内安全）
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
    if not merged_dir.is_dir():
        return False
    return any(merged_dir.glob("*.safetensors"))


def merge_lora_into(base_model: str, lora_path: Path, merged_dir: Path,
                    merge_python: str) -> None:
    """peft merge → merged_dir（已存在且含 safetensors 则直接复用，不覆盖）。"""
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
    # vLLM 0.11 的 EngineCore 是子进程，终止到显存释放有数秒延迟；轮询等待，避免
    # 下一个引擎 init 时的 "Free memory on device" 检查误判。生产路径（每模型独立
    # 进程）不依赖此等待，但多 lora 单进程手动跑法需要它兜底。
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


def _build_engine(args: argparse.Namespace, model: str, enable_lora: bool,
                  lora: Optional[Tuple[str, str]]) -> Tuple[Any, Any]:
    """返回 (llm, lora_request)。vLLM 引擎参数对齐 eval_5p_vllm.py (L206-220)。"""
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    kwargs = dict(model=model, enable_lora=enable_lora, max_loras=1,
                  max_lora_rank=32, dtype="bfloat16", trust_remote_code=True,
                  seed=0, max_model_len=args.max_prompt_tokens + args.max_new_tokens)
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**kwargs)
    lr = LoRARequest(lora[0], 1, lora[1]) if lora is not None else None
    return llm, lr


def _warmup(llm: Any, lr: Any, sample_ids: List[int]) -> None:
    """1 请求 warmup：强制加载 LoRA（失败在此暴露），并预热 CUDA graph。"""
    from vllm import SamplingParams
    sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, seed=0, max_tokens=16)
    out = llm.generate([{"prompt_token_ids": sample_ids}], sp, lora_request=lr)
    text = (out[0].outputs[0].text if out and out[0].outputs else "")
    print(f"[serve] warmup ok (first 60 chars): {text[:60]!r}")


def _serve_pass(args: argparse.Namespace, lora_name: str,
                warmup_ids: List[int]) -> Tuple[Any, Any, str]:
    """为一个 checkpoint 准备 vLLM 引擎。返回 (llm, lora_request, serving_mode)。

    serving_mode ∈ {"lora", "merged"}。LoRA 优先（eval_5p_vllm 写法）；失败则回退
    合并权重（sft_v2 复用 sft_v2_merged；phase1 现 merge 出 sft_phase1_merged）。
    """
    lora_path = Path(args.checkpoints_dir) / lora_name
    merged_dir = Path(args.checkpoints_dir) / f"{lora_name}_merged"

    if lora_path.is_dir():
        llm = None
        try:
            t0 = time.perf_counter()
            llm, lr = _build_engine(args, args.base_model, enable_lora=True,
                                    lora=(lora_name, str(lora_path)))
            print(f"[serve] engine(LoRA) ready in {time.perf_counter() - t0:.1f}s")
            _warmup(llm, lr, warmup_ids)
            return llm, lr, "lora"
        except Exception as exc:
            if "Free memory on device" in str(exc):
                # 显存不足是资源问题，不是 LoRA 兼容性问题 —— 不得触发 merge 回退
                _teardown_engine(llm)
                raise RuntimeError(
                    f"[serve] engine init for {lora_name} failed due to "
                    f"insufficient free GPU memory (NOT a LoRA issue, no merge "
                    f"fallback): {exc}") from exc
            print(f"[serve] LoRA serving failed for {lora_name}: {exc}")
            print("[serve] falling back to merged weights")
            _teardown_engine(llm)
    else:
        print(f"[serve] {lora_path} not found, go merged directly")

    if not _merged_ok(merged_dir):
        if not lora_path.is_dir():
            raise RuntimeError(
                f"Neither LoRA dir {lora_path} nor merged dir {merged_dir} exists")
        merge_lora_into(args.base_model, lora_path, merged_dir, args.merge_python)
    t0 = time.perf_counter()
    llm, _ = _build_engine(args, str(merged_dir), enable_lora=False, lora=None)
    print(f"[serve] engine(merged) ready in {time.perf_counter() - t0:.1f}s")
    _warmup(llm, None, warmup_ids)
    return llm, None, "merged"


# ---------------------------------------------------------------------------
# 候选库 ingest / 回环
# ---------------------------------------------------------------------------

def _ingest_model(model: str, items_by_di: Dict[int, Dict[str, Any]],
                  prompt_text_by_di: Dict[int, Optional[str]],
                  temperature: float, seed: int) -> Optional[Dict[str, Any]]:
    if not _STORE_AVAILABLE:
        print("[store] candidate_store 不可用，跳过 ingest")
        return None
    payload = []
    for di in sorted(items_by_di):
        item = items_by_di[di]
        if item.get("error"):
            continue
        prompt_text = prompt_text_by_di.get(di)
        if not prompt_text:
            continue
        cands = [
            {"sql": c["sql"], "prompt": prompt_text}
            for c in item["candidates"]
            if c["model"] == model and (c.get("sql") or "").strip()
        ]
        payload.append({"di": item["di"], "question": item["question"],
                        "db_id": item["db_id"], "candidates": cands})
    if not payload:
        return {"added": 0, "skipped": 0, "total": None}
    res = _store_ingest(payload, model=STORE_MODEL_NAME, lora=model,
                        temperature=temperature, seed=seed)
    print(f"[store] ingest {model}: +{res['added']} (skipped {res['skipped']}, "
          f"total {res.get('total')})")
    return res


def _store_roundtrip(di: int, model: str, prompt_text: str,
                     temperature: float, seed: int) -> int:
    """回环验证：按 (di, 配置, prompt_hash) 查询刚 ingest 的候选数。"""
    hits = _store_query(di, model=STORE_MODEL_NAME, lora=model,
                        temperature=temperature, seed=seed,
                        prompt_hash=_store_prompt_hash(prompt_text))
    return len(hits)


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
    if args.only_lora is not None and args.only_lora not in lora_names:
        raise SystemExit(f"--only-lora {args.only_lora!r} not in --loras {lora_names}")
    active_loras = [m for m in lora_names
                    if args.only_lora is None or m == args.only_lora]
    use_store = (not args.no_cache) and _STORE_AVAILABLE
    if not _STORE_AVAILABLE:
        print("[store] WARNING: candidate_store 不可用，本次按无缓存运行")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"B1 大候选池 | loras={lora_names} | n={args.n} | T={args.temperature} "
          f"seed={args.seed} top_p={args.top_p} | limit={args.limit} "
          f"(start={args.start_index}) | checkpoint-every={args.checkpoint_every}")

    # ---- tokenizer（与 mem_filter / eval_5p_vllm 相同的设置） ----
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    loader = SpiderLoader(args.spider_dir)
    items = loader.load_dev(limit=args.limit, start_index=args.start_index)
    requested_order = [it["dataset_index"] for it in items]
    requested_set = set(requested_order)
    src_by_di = {it["dataset_index"]: it for it in items}
    print(f"Loaded {len(items)} items (start={args.start_index}, limit={args.limit})")

    # ---- 预构造 prompt（口径与 mem_filter 完全一致） ----
    prompt_text_by_di: Dict[int, Optional[str]] = {}
    ids_by_di: Dict[int, Optional[List[int]]] = {}
    for it in items:
        di = it["dataset_index"]
        try:
            ddl = loader.format_ddl(it["db_id"])
            # 来源：finer_port/sampler.py VavSampler.build_prompt
            #       (L142-150, prompt_style="default")
            prompt_text = ReasoningGeneratorAgent.build_prompt(
                question=it["question"], ddl_schema=ddl,
                schema_links=None, evidence=None, dialect="sqlite")
            # 来源：finer_port/sampler.py VavSampler.build_chat_text (L164-170)
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}], tokenize=False,
                add_generation_prompt=True)
            # 截断口径：VavSampler.sample_batch max_prompt_tokens=1536
            ids = tokenizer(chat, truncation=True,
                            max_length=args.max_prompt_tokens)["input_ids"]
            prompt_text_by_di[di] = prompt_text
            ids_by_di[di] = ids
        except Exception as exc:
            prompt_text_by_di[di] = None
            ids_by_di[di] = None
            print(f"[WARN] item {di} db_id={it['db_id']} prompt build failed: {exc}")

    # ---- checkpoint / resume ----
    # 复用 mem_filter 的 checkpoint 协议（load/validate_resume/save），但允许
    # items 含半完成条目（双 pass 需要）：resume 时按每模型候选数重算 pending，
    # 不调用 validate_checkpoint_integrity（该函数要求 completed_indices==items，
    # 与半完成态矛盾）。混配保护 = validate_resume_config，逐字段比对。
    run_config = build_run_config(
        spider_dir=args.spider_dir, start_index=args.start_index, limit=args.limit,
        model_path=args.base_model, max_new_tokens=args.max_new_tokens,
        evaluator_type=EVALUATOR_TYPE,
    )
    run_config.update({
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "loras": lora_names,
        "max_prompt_tokens": args.max_prompt_tokens,
        "prompt_style": "default",  # canonical prompt，与 mem_filter 一致
    })
    cp = load_checkpoint(output_dir)
    if cp["run_config"] is not None:
        validate_resume_config(cp["run_config"], run_config)
    items_by_di: Dict[int, Dict[str, Any]] = {}
    for it in cp["items"]:
        di = it.get("dataset_index")
        if di is None or di not in requested_set:
            raise ValueError(f"checkpoint item has invalid dataset_index={di!r}")
        if di in items_by_di:
            raise ValueError(f"checkpoint has duplicate dataset_index={di}")
        items_by_di[di] = it
    n_done = sum(1 for di in requested_order
                 if di in items_by_di and _item_done(items_by_di[di], lora_names, args.n))
    print(f"Resume: {len(items_by_di)}/{len(items)} items touched "
          f"({n_done} fully done)")

    # ---- DDL 失败条目：立即记为 error（不参与生成，不阻塞） ----
    for di in requested_order:
        if ids_by_di[di] is None and di not in items_by_di:
            items_by_di[di] = _new_item(di, src_by_di[di], error="ddl_load_failed")
            print(f"[WARN] item {di} recorded as ddl_load_failed")

    # ---- vLLM 采样参数（与 VavSampler._generate 对齐；采样类实验） ----
    from vllm import SamplingParams  # noqa: E402
    sampling_params = SamplingParams(n=args.n, temperature=args.temperature,
                                     top_p=args.top_p, seed=args.seed,
                                     max_tokens=args.max_new_tokens)

    wall_start = time.perf_counter()
    per_model_stats: Dict[str, Dict[str, Any]] = {}
    ingest_stats: Dict[str, Optional[Dict[str, Any]]] = {}

    for model in active_loras:
        pending = _pending_for(model, requested_order, items_by_di, args.n)
        if not pending:
            print(f"[pass] {model}: nothing pending, skip")
            continue
        print(f"[pass] {model}: {len(pending)} questions pending, serving engine...")
        warmup_ids = ids_by_di[pending[0]]
        llm, lr, serving_mode = _serve_pass(args, model, warmup_ids)

        stats = {"serving": serving_mode, "generation_seconds": 0.0,
                 "generated_tokens": 0, "requests": 0, "applied_items": 0}
        while pending:
            chunk = pending[:args.checkpoint_every]
            reqs = [{"prompt_token_ids": ids_by_di[di]} for di in chunk]
            t0 = time.perf_counter()
            try:
                outputs = llm.generate(reqs, sampling_params, lora_request=lr)
            except Exception as exc:
                print(f"[pass:{model}] generate failed on chunk starting at di="
                      f"{chunk[0]}: {exc}")
                _save_checkpoint(output_dir, items_by_di, lora_names, args.n, run_config)
                raise  # 下个切片 resume 重试
            dt = time.perf_counter() - t0
            by_prompt = {tuple(o.prompt_token_ids): o for o in outputs}
            if len(by_prompt) != len(reqs):
                print(f"[WARN] matched {len(by_prompt)}/{len(reqs)} outputs by prompt ids")
            n_tokens = 0
            n_applied = 0
            lens = sorted(len(x.token_ids) for o in outputs for x in o.outputs)
            for di in chunk:
                o = by_prompt.get(tuple(ids_by_di[di]))
                item = items_by_di.get(di)
                if item is not None and item.get("error"):
                    continue
                if item is None:
                    item = _new_item(di, src_by_di[di])
                    items_by_di[di] = item
                if o is None or len(o.outputs) < args.n:
                    print(f"[WARN] item {di}: {len(o.outputs) if o else 0}/{args.n} "
                          f"outputs, retry next slice")
                    continue
                for j in range(args.n):
                    text = tokenizer.decode(o.outputs[j].token_ids,
                                            skip_special_tokens=True)
                    # 解析口径 = mem_filter：finer_port/sampler.py VavSampler.extract_sql
                    parsed = VavSampler.extract_sql(text)
                    item["candidates"].append({
                        "model": model,
                        "sql": parsed["sql"] if parsed["parse_success"] else "",
                        "parse_success": bool(parsed["parse_success"]),
                        "sample_idx": j,
                    })
                # 按 (model, sample_idx) 排序（任务口径）
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
            _save_checkpoint(output_dir, items_by_di, lora_names, args.n, run_config)
            med = lens[len(lens) // 2] if lens else 0
            print(f"  [pass:{model}] {n_applied}/{len(chunk)} items in {dt:.1f}s "
                  f"({n_tokens / max(dt, 1e-9):.0f} tok/s) | seq lens "
                  f"min/med/max = {lens[0] if lens else 0}/{med}/{lens[-1] if lens else 0} "
                  f"| remaining {len(pending)} | wall "
                  f"{time.perf_counter() - wall_start:.0f}s")

        _teardown_engine(llm)

        # ---- 该 pass 完成 → ingest（幂等，切片中断也不丢已完成 pass） ----
        if use_store:
            ingest_stats[model] = _ingest_model(
                model, items_by_di, prompt_text_by_di,
                temperature=args.temperature, seed=args.seed)
            # 回环验证：任取一个已完成条目，按 (配置, prompt_hash) 查询
            sample_di = next((di for di in requested_order
                              if di in items_by_di
                              and not items_by_di[di].get("error")
                              and _count_model(items_by_di[di], model) >= args.n), None)
            if sample_di is not None and ingest_stats[model]:
                n_hits = _store_roundtrip(sample_di, model,
                                          prompt_text_by_di[sample_di],
                                          args.temperature, args.seed)
                print(f"[store] roundtrip check di={sample_di}: {n_hits} rows "
                      f"(<=16, 库内 norm_sql 去重)")
        else:
            ingest_stats[model] = None

        completed = sum(1 for di in requested_order
                        if di in items_by_di
                        and not items_by_di[di].get("error")
                        and _count_model(items_by_di[di], model) >= args.n)
        cands = [c for di in requested_order
                 for c in items_by_di.get(di, {}).get("candidates", [])
                 if c["model"] == model]
        n_parse = sum(1 for c in cands if c["parse_success"])
        per_model_stats[model] = {
            "serving": stats["serving"],
            "completed_items": completed,
            "candidates": len(cands),
            "parse_success": n_parse,
            "parse_rate": round(n_parse / len(cands), 4) if cands else 0.0,
            "generation_seconds": round(stats["generation_seconds"], 2),
            "generated_tokens": stats["generated_tokens"],
            "requests": stats["requests"],
        }

    # 补全未在本进程运行的模型统计（--only-lora 模式：另一进程负责生成）
    for model in lora_names:
        if model in per_model_stats:
            continue
        cands = [c for di in requested_order
                 for c in items_by_di.get(di, {}).get("candidates", [])
                 if c["model"] == model]
        n_parse = sum(1 for c in cands if c["parse_success"])
        per_model_stats[model] = {
            "serving": "not-run-this-process",
            "completed_items": sum(
                1 for di in requested_order
                if di in items_by_di and not items_by_di[di].get("error")
                and _count_model(items_by_di[di], model) >= args.n),
            "candidates": len(cands),
            "parse_success": n_parse,
            "parse_rate": round(n_parse / len(cands), 4) if cands else 0.0,
            "generation_seconds": None,
            "generated_tokens": None,
            "requests": None,
        }

    total_wall = time.perf_counter() - wall_start

    # ---- 收尾：最终 checkpoint + items.json + summary.json ----
    _save_checkpoint(output_dir, items_by_di, lora_names, args.n, run_config)
    n_done = sum(1 for di in requested_order
                 if di in items_by_di and _item_done(items_by_di[di], lora_names, args.n))
    n_error = sum(1 for di in requested_order
                  if di in items_by_di and items_by_di[di].get("error"))
    final_items = [items_by_di[di] for di in sorted(items_by_di)
                   if di in requested_set]
    with open(output_dir / "items.json", "w", encoding="utf-8") as fh:
        json.dump(final_items, fh, ensure_ascii=False, indent=2)

    summary = {
        "evaluator_type": EVALUATOR_TYPE,
        "is_official_spider_metric": False,
        "note": (
            "B1 大候选池生成：Spider dev × {loras} × n 采样候选（T={t} seed={s} "
            "top_p={p}，vLLM 单请求 n=16，与 mem_filter 的 VavSampler 采样参数对齐）。"
            "prompt = VavSampler prompt_style='default' canonical prompt "
            "（ReasoningGeneratorAgent.build_prompt，DDL=SpiderLoader.format_ddl，"
            "chat template + 1536 截断）；解析 = VavSampler.extract_sql（与 "
            "mem_filter 完全同口径）。items.json 不去重（裁决阶段做）；候选库按 "
            "norm_sql 去重是 store 既有设计。serving: lora=vLLM LoRARequest，"
            "merged=peft 合并权重回退（sft_phase1_merged 现合并，sft_v2_merged 复用）。"
            "半完成条目允许入 checkpoint（双 pass 断点续跑需要）。"
        ).format(loras=", ".join(lora_names), t=args.temperature, s=args.seed,
                 p=args.top_p),
        "total_requested": len(requested_order),
        "total_completed": n_done,
        "error_items": n_error,
        "per_model": per_model_stats,
        "ingest": ingest_stats,
        "store": {"model": STORE_MODEL_NAME, "temperature": args.temperature,
                  "seed": args.seed},
        "total_wall_seconds": round(total_wall, 2),
        "run_config": run_config,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # ---- 控制台结果 ----
    print("\n" + "=" * 66)
    print("  B1 CANDIDATE POOL GENERATION SUMMARY")
    print("=" * 66)
    for m in lora_names:
        s = per_model_stats[m]
        gen_txt = ""
        if s.get("generation_seconds"):
            gen_txt = (f" | gen {s['generation_seconds']:.0f}s "
                       f"({s['generated_tokens'] / max(s['generation_seconds'], 1e-9):.0f} tok/s)")
        print(f"  {m} [{s['serving']}]: {s['completed_items']}/{len(requested_order)} "
              f"items | {s['candidates']} candidates | parse "
              f"{s['parse_success']}/{s['candidates']} ({s['parse_rate']:.1%}){gen_txt}")
    print(f"  fully done: {n_done}/{len(requested_order)} (error={n_error})")
    print(f"  total wall: {total_wall:.0f}s")
    if any(ingest_stats.values()):
        added = sum((v or {}).get("added", 0) for v in ingest_stats.values())
        print(f"  candidate store ingest: +{added}")
    # 全量外推（冒烟用）：按已完成题数 × 每模型 gen 秒数线性外推 1034 题
    # （只对有生成统计的模型求和；--only-lora 模式下每个进程只含自己模型的统计）
    gen_per_item = [
        (s["generation_seconds"] / max(s["completed_items"], 1))
        for s in per_model_stats.values()
        if s.get("generation_seconds") is not None and s.get("completed_items")
    ]
    if gen_per_item:
        est = sum(gen_per_item) * SPIDER_DEV_SIZE / 60.0
        print(f"  EXTRAPOLATE full {SPIDER_DEV_SIZE} gen-only (sum over models with "
              f"stats, linear from {args.limit} items): ~{est:.0f} min")
    print(f"\nItems saved to:   {output_dir / 'items.json'}")
    print(f"Summary saved to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
