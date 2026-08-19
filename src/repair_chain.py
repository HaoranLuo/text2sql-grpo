#!/usr/bin/env python3
"""T1-3 「执行失败修复链」pilot — execution-grounded repair chain（≤1 轮硬上限）。

动机（文献共识）：无执行反馈的 self-correction 已证伪；修复必须携带具体错误信息。
本链对 MI-VAV 胜者 SQL 做全实例执行；任一实例报错即进入修复轮。

数据流（每题）：
  1. 取 MI-VAV 胜者 SQL（outputs/adjudicate_b1/items_arm_vav_multi_all_both.json
     的 predicted_sql），在全部实例上执行。
  2. 任一实例报错 → 修复轮：
     - 修复 prompt = canonical prompt（ReasoningGeneratorAgent.build_prompt，
       与候选生成同源同口径）+ 失败 SQL + 首个报错实例名与错误文本 + 修复指令。
     - sft_v2（vLLM LoRARequest，贪心 T=0，max_new_tokens=2048）生成 →
       VavSampler.extract_sql 解析（与候选生成同解析口径）。
  3. 修复 SQL 再全实例执行；接受条件 = 全部实例执行成功（无报错）。
     执行成功即接受（不用 gold 判定，防泄漏；这是唯一合法的接地信号）。
     失败 → 保留原胜者；最多 --max-repair-rounds 轮（默认 1，设计文档硬上限）。
  4. 输出 items_repair.json（eval_official.sh 兼容；predicted_sql = 修复后或
     原胜者）+ items_baseline.json（同子集原胜者，官方 EX 基线对照）
     + summary.json（触发率 / 接受率 / 生成与执行统计 / 1034 外推）。

执行口径（与 adjudicate_pool 完全一致）：
  - 执行文本 = clean_pred_for_official(predicted_sql)：镜像 eval_official.sh 清洗
    （strip → rstrip(';') → 去 -- 行注释 → 空白折叠）。
  - 实例枚举 = adjudicate_pool.list_instances（sorted + 原始 <db_id>.sqlite 置首）。
  - 引擎 = adjudicate_pool.ExecutionEngine（线程池、每查询独立只读连接、
    30s 墙钟 watchdog + 5M SQLite VM 步数上限 + row_cap 100k）；空 SQL 不触库
    （合成失败）；(sql, db_path) 跨题缓存。
  - 某 db 无实例文件 → 空实例集恒真（镜像官方 eval_exec_match）。

生成口径（与 eval_pool_b1 相同框架）：
  - tokenizer 设置 / chat template / prompt_token_ids 直送 vLLM（规避 vLLM 端
    tokenizer 差异）；LoRA 加载失败回退 checkpoints/sft_v2_merged。
  - 差异点（任务指出的风险，summary 量化）：贪心 T=0 n=1（候选生成为 T=1.0
    n=16 采样），且 prompt 尾部多一段「修复请求」（OOD 于 sft_v2 训练分布）——
    两者都可能影响修复质量，以修复 parse 率 / 接受率量化。

断点续跑：50min 切片 + checkpoint（复用 spider_utils 协议；允许半完成条目）。
  - Phase 1 全量扫描（--scan-file，纯 CPU）→ Phase A 窗口胜者执行 →
    Phase B 修复生成（chunk 落 checkpoint）→ Phase C 修复执行与接受。
  - resume 时按条目状态（winner_ok / repair）重算 pending；混配保护
    validate_resume_config。

用法：
    python src/repair_chain.py --limit 100 --out-dir outputs/repair_pilot \
        --scan-file outputs/adjudicate_b1/items_arm_vav_multi_all_both.json
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
# canonical prompt 来源（与 eval_pool_b1 同源）
from reasoning_generator_agent import ReasoningGeneratorAgent  # noqa: E402
# SQL 解析来源（与候选生成同口径）
from sampler import VavSampler  # noqa: E402
# 执行引擎复用（不修改 adjudicate_pool，纯导入其函数）
import adjudicate_pool as adj  # noqa: E402

DEFAULT_BASE_MODEL = str(_PROJECT / "models" / "Qwen2.5-Coder-3B-Instruct")
DEFAULT_ITEMS = str(_PROJECT / "outputs" / "adjudicate_b1" / "items_arm_vav_multi_all_both.json")
DEFAULT_OUT_DIR = str(_PROJECT / "outputs" / "repair_pilot")
DEFAULT_SPIDER_DIR = str(_PROJECT / "data" / "spider_data")
DEFAULT_CHECKPOINTS_DIR = str(_PROJECT / "checkpoints")
DEFAULT_MERGE_PYTHON = str(_PROJECT / "envs" / "reasoning3b" / "bin" / "python")
EVALUATOR_TYPE = "repair_chain"
SPIDER_DEV_SIZE = 1034  # Spider dev 全量题数（外推基准）

# 修复 prompt 预算（canonical 部分截断 + 修复段字符上限；尾部 truncation 保护）
CANONICAL_TOKEN_CAP = 1200
FAILED_SQL_CHAR_CAP = 1200
ERROR_CHAR_CAP = 500


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T1-3 执行失败修复链：MI-VAV 胜者全实例执行 → 失败则 sft_v2 贪心修复（≤1 轮）"
    )
    parser.add_argument("--items", default=DEFAULT_ITEMS,
                        help="MI-VAV 胜者集（含 predicted_sql 字段）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--spider-dir", default=DEFAULT_SPIDER_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora", default="sft_v2",
                        help="修复生成所用 checkpoint（LoRA 名）")
    parser.add_argument("--checkpoints-dir", default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--merge-python", default=DEFAULT_MERGE_PYTHON)
    parser.add_argument("--limit", type=int, required=True,
                        help="pilot 条数（必填，防误触全量；≤100 为任务口径）")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--scan-file", default=None,
                        help="可选：对该文件做全量 1034 触发率扫描（纯 CPU、只读、"
                             "无修复），写入 summary.population_scan 供外推")
    # 执行参数（与 adjudicate_pool 默认一致）
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--query-timeout", type=float, default=30.0)
    parser.add_argument("--max-vm-steps", type=int, default=5_000_000)
    parser.add_argument("--row-cap", type=int, default=100_000)
    parser.add_argument("--max-instances", type=int, default=None)
    # 生成参数（贪心）
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="修复生成温度（设计口径：贪心 T=0）")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048,
                        help="prompt 截断长度（修复 prompt 含错误信息，比候选生成的 "
                             "1536 放宽；canonical 部分先行截断保护尾部修复段）")
    parser.add_argument("--max-repair-rounds", type=int, default=1,
                        help="修复轮数硬上限（设计文档：1）")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="每处理 N 个修复请求写一次 checkpoint")
    parser.add_argument("--skip-generation", action="store_true",
                        help="纯 CPU 干跑：只做执行/触发检测，不起 vLLM（登录节点冒烟用）")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 题级状态
# ---------------------------------------------------------------------------

def _new_item(src: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_index": src["dataset_index"],
        "di": src.get("di", src.get("dataset_index")),
        "db_id": src["db_id"],
        "question": src["question"],
        "gold_sql": src.get("gold_sql") or "",
        "winner_sql": src.get("predicted_sql") or "",
        "winner_source": src.get("winner_source"),
        "winner_is_correct": src.get("is_correct"),
        "winner_ok": None,       # None=未扫描；True=全实例成功；False=至少一实例报错
        "first_error": None,     # {"instance": str, "error": str, "error_type": str}
        "prompt_error": None,    # DDL/prompt 构建失败原因（有则跳过修复，保留胜者）
        "repair": None,          # None=未尝试；dict=已尝试（见 _repair_record）
    }


def _item_done(item: Dict[str, Any]) -> bool:
    """winner_ok=True 无需修复即完成；winner_ok=False 需 repair 落定；prompt 失败
    条目按「保留胜者」完成，不再重试。"""
    if item.get("prompt_error"):
        return True
    if item.get("winner_ok") is True:
        return True
    return item.get("winner_ok") is False and item.get("repair") is not None


def _save_cp(output_dir: Path, items_by_di: Dict[int, Dict[str, Any]],
             run_config: Dict[str, Any]) -> None:
    done = sorted(di for di, it in items_by_di.items() if _item_done(it))
    save_checkpoint(output_dir, {"completed_indices": done,
                                 "items": [items_by_di[di] for di in sorted(items_by_di)]},
                    run_config)


# ---------------------------------------------------------------------------
# 修复 prompt
# ---------------------------------------------------------------------------

def build_repair_prompt(tokenizer: Any, question: str, ddl: str, failed_sql: str,
                        instance_name: str, error_text: str) -> Tuple[str, str]:
    """修复 prompt = canonical prompt（截断保护尾部）+ 修复段（失败 SQL + 错误）。
    返回 (prompt_text, canonical_text)。canonical 部分截断 CANONICAL_TOKEN_CAP，
    保证最终 chat 尾部 truncation 只可能砍 DDL/指令尾部而非修复段。"""
    canonical = ReasoningGeneratorAgent.build_prompt(
        question=question, ddl_schema=ddl, schema_links=None, evidence=None,
        dialect="sqlite")
    ids = tokenizer(canonical, truncation=True,
                    max_length=CANONICAL_TOKEN_CAP)["input_ids"]
    canonical_text = tokenizer.decode(ids, skip_special_tokens=True).strip()

    fs = (failed_sql or "").strip()
    if len(fs) > FAILED_SQL_CHAR_CAP:
        fs = fs[:FAILED_SQL_CHAR_CAP].rstrip() + "\n... [truncated]"
    er = (error_text or "").strip()
    if len(er) > ERROR_CHAR_CAP:
        er = er[:ERROR_CHAR_CAP] + " ... [truncated]"

    repair_section = (
        "\n\n=== Repair Request (execution feedback) ===\n"
        f'The SQL written for the question above raised an execution error on '
        f'database instance "{instance_name}".\n'
        "The failing SQL:\n```sql\n" + fs + "\n```\n"
        "The error message:\n" + er + "\n\n"
        "Please rewrite the SQL so that it (1) executes successfully on EVERY "
        "database instance of this database and (2) keeps the same meaning with "
        "respect to the original question (do not change what the query asks for; "
        "only fix the execution error). Output the corrected SQL inside a "
        "```sql code block."
    )
    return canonical_text + repair_section, canonical_text


def _tokenize_chat(tokenizer: Any, prompt_text: str, cap: int) -> List[int]:
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}], tokenize=False,
        add_generation_prompt=True)
    return tokenizer(chat, truncation=True, max_length=cap)["input_ids"]


# ---------------------------------------------------------------------------
# vLLM 服务（LoRA 优先，失败回退合并权重；框架照搬 eval_pool_b1._serve_pass）
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
        print(f"[serve] GPU memory after teardown: {free / 1024**3:.1f}/"
              f"{total / 1024**3:.1f} GiB free")
    except Exception:  # pragma: no cover
        pass


def _build_engine(args: argparse.Namespace, model: str, enable_lora: bool,
                  lora: Optional[Tuple[str, str]]) -> Tuple[Any, Any]:
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
    from vllm import SamplingParams
    sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, seed=0, max_tokens=16)
    out = llm.generate([{"prompt_token_ids": sample_ids}], sp, lora_request=lr)
    text = (out[0].outputs[0].text if out and out[0].outputs else "")
    print(f"[serve] warmup ok (first 60 chars): {text[:60]!r}")


def _serve(args: argparse.Namespace, warmup_ids: List[int]) -> Tuple[Any, Any, str]:
    lora_name = args.lora
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
                _teardown_engine(llm)
                raise RuntimeError(
                    f"[serve] engine init for {lora_name} failed due to insufficient "
                    f"free GPU memory (NOT a LoRA issue, no merge fallback): {exc}") from exc
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
# 执行与触发检测
# ---------------------------------------------------------------------------

def _load_items(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        data = data["items"]
    if not isinstance(data, list):
        raise ValueError(f"items 结构异常（期望 list）: {path}")
    return data


def _winner_exec_sql(item: Dict[str, Any]) -> str:
    """执行文本 = 官方清洗后的胜者 SQL（与 eval_official.sh 对 predicted_sql 的
    处理一致）。"""
    return adj.clean_pred_for_official(item.get("winner_sql"))


def _detect_failure(item: Dict[str, Any], engine: Any,
                    insts: List[str]) -> Optional[Dict[str, Any]]:
    """在全部实例上找首个报错；返回 {"instance","error","error_type"} 或 None。
    空实例集合 → 恒真（镜像官方 eval_exec_match）。"""
    sql = _winner_exec_sql(item)
    for p in insts:
        out = engine.get(sql, p)
        if not out["ok"]:
            return {"instance": Path(p).name, "error": out["error"],
                    "error_type": out["error_type"]}
    return None


def scan_population(args: argparse.Namespace, engine: Any,
                    db_instances: Dict[str, List[str]]) -> Dict[str, Any]:
    """全量扫描（纯 CPU）：对 --scan-file 的全部条目做全实例执行，统计触发率。
    返回 summary.population_scan 结构；任务与窗口任务共用引擎缓存。"""
    items = _load_items(Path(args.scan_file))
    database_dir = Path(args.spider_dir) / "database"
    tasks = []
    for it in items:
        db_id = it.get("db_id", "")
        if db_id not in db_instances:
            # 顺带填充窗口共用的实例缓存（Phase A 直接复用）
            db_instances[db_id] = adj.list_instances(
                str(database_dir / db_id), db_id, args.max_instances)
        sql = adj.clean_pred_for_official(it.get("predicted_sql"))
        for p in db_instances[db_id]:
            tasks.append((sql, p))
    n_unique = len(set(tasks))
    print(f"[scan] population {len(items)} 题 × 全实例 = {n_unique} 个唯一执行任务 ...")
    t0 = time.perf_counter()
    engine.run(list(set(tasks)), phase="population_scan")
    wall = time.perf_counter() - t0
    fails = []
    for it in items:
        err = _detect_failure({"winner_sql": it.get("predicted_sql")}, engine,
                              db_instances[it.get("db_id", "")])
        if err:
            fails.append({"di": it.get("dataset_index", it.get("di")),
                          "db_id": it.get("db_id", ""),
                          "winner_source": it.get("winner_source"),
                          "is_correct": it.get("is_correct"),
                          **err})
    print(f"[scan] population triggers: {len(fails)}/{len(items)} "
          f"({len(fails) / max(len(items), 1):.4f}) in {wall:.1f}s")
    for f in fails:
        print(f"[scan]   di={f['di']} {f['db_id']} src={f['winner_source']} "
              f"{f['error_type']}: {f['error'][:80]}")
    return {"file": args.scan_file, "total": len(items), "triggers": len(fails),
            "trigger_rate": round(len(fails) / len(items), 4) if items else 0.0,
            "unique_tasks": n_unique, "wall_seconds": round(wall, 2), "fails": fails}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.temperature != 0.0:
        print("[WARN] 修复生成口径为贪心 T=0；--temperature 非 0 会偏离设计口径")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 输入条目（pilot 窗口） ----
    src_items = _load_items(Path(args.items))[args.start_index:args.start_index + args.limit]
    requested_order = [it["dataset_index"] for it in src_items]
    requested_set = set(requested_order)
    src_by_di = {it["dataset_index"]: it for it in src_items}
    print(f"repair_chain | items={args.items} | window=[{args.start_index}, "
          f"{args.start_index + len(src_items)}) n={len(src_items)} | "
          f"lora={args.lora} T={args.temperature} | out={output_dir}")

    # ---- tokenizer（与 eval_pool_b1 相同设置） ----
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    loader = SpiderLoader(args.spider_dir)
    database_dir = Path(args.spider_dir) / "database"

    # ---- run_config + checkpoint/resume ----
    run_config = build_run_config(
        spider_dir=args.spider_dir, start_index=args.start_index, limit=args.limit,
        model_path=args.base_model, max_new_tokens=args.max_new_tokens,
        evaluator_type=EVALUATOR_TYPE)
    run_config.update({
        "items_path": str(Path(args.items).resolve()),
        "scan_file": str(Path(args.scan_file).resolve()) if args.scan_file else None,
        "lora": args.lora,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_repair_rounds": args.max_repair_rounds,
        "threads": args.threads,
        "query_timeout": args.query_timeout,
        "max_vm_steps": args.max_vm_steps,
        "row_cap": args.row_cap,
        "max_instances": args.max_instances,
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
                 if di in items_by_di and _item_done(items_by_di[di]))
    print(f"Resume: {len(items_by_di)}/{len(src_items)} items touched ({n_done} done)")

    # ---- DDL（每 db 缓存一次；失败条目记录 prompt_error，保留胜者） ----
    ddl_cache: Dict[str, Optional[str]] = {}

    def ddl_for(db_id: str) -> Optional[str]:
        if db_id not in ddl_cache:
            try:
                ddl_cache[db_id] = loader.format_ddl(db_id)
            except Exception as exc:
                print(f"[WARN] DDL load failed for db_id={db_id}: {exc}")
                ddl_cache[db_id] = None
        return ddl_cache[db_id]

    # ---- 实例枚举（按 db_id 缓存；口径 = adjudicate_pool.list_instances） ----
    db_instances: Dict[str, List[str]] = {}

    def instances_for(db_id: str) -> List[str]:
        if db_id not in db_instances:
            db_instances[db_id] = adj.list_instances(
                str(database_dir / db_id), db_id, args.max_instances)
        return db_instances[db_id]

    engine = adj.ExecutionEngine(args.threads, args.query_timeout,
                                 args.max_vm_steps, args.row_cap)

    # ---- Phase 1（可选）：全量触发率扫描（纯 CPU，先跑以共享执行缓存） ----
    population_scan = None
    if args.scan_file:
        population_scan = scan_population(args, engine, db_instances)

    # ---- Phase A：pilot 窗口胜者全实例执行 → 触发检测 ----
    a_pending = [di for di in requested_order
                 if di not in items_by_di or items_by_di[di].get("winner_ok") is None]
    if a_pending:
        tasks = []
        for di in a_pending:
            it = src_by_di[di]
            sql = _winner_exec_sql({"winner_sql": it.get("predicted_sql")})
            for p in instances_for(it["db_id"]):
                tasks.append((sql, p))
        print(f"[phase A] winner scan: {len(a_pending)} 题, "
              f"{len(set(tasks))} 个唯一执行任务")
        t0 = time.perf_counter()
        engine.run(list(set(tasks)), phase="winner_scan")
        for di in a_pending:
            src = src_by_di[di]
            item = items_by_di.get(di)
            if item is None:
                item = _new_item(src)
                items_by_di[di] = item
            err = _detect_failure({"winner_sql": src.get("predicted_sql")}, engine,
                                  instances_for(src["db_id"]))
            if err is None:
                item["winner_ok"] = True
            else:
                item["winner_ok"] = False
                item["first_error"] = err
        print(f"[phase A] done in {time.perf_counter() - t0:.1f}s")
        _save_cp(output_dir, items_by_di, run_config)

    trigger_dis = [di for di in requested_order
                   if di in items_by_di and items_by_di[di].get("winner_ok") is False]
    for di in trigger_dis:
        it = items_by_di[di]
        print(f"[trigger] di={di} db={it['db_id']} src={it.get('winner_source')} "
              f"{it['first_error']['error_type']}: {it['first_error']['error'][:90]}")

    # ---- Phase B + C：修复生成（GPU）+ 修复执行接受（CPU） ----
    gen_stats = {"serving": None, "generation_seconds": 0.0, "generated_tokens": 0,
                 "requests": 0}
    repair_pending = [di for di in requested_order
                      if di in items_by_di and items_by_di[di].get("winner_ok") is False
                      and items_by_di[di].get("repair") is None
                      and not items_by_di[di].get("prompt_error")]

    if repair_pending and args.skip_generation:
        print(f"[phase B] --skip-generation: {len(repair_pending)} 题待修复，跳过生成")

    llm = None
    lr = None
    round_used = 0
    while repair_pending and not args.skip_generation and \
            round_used < args.max_repair_rounds:
        round_used += 1
        print(f"[phase B] round {round_used}/{args.max_repair_rounds}: "
              f"{len(repair_pending)} 题待修复")

        # 本轮的 (di, prompt_text, cur_sql, cur_err) —— 第 1 轮用胜者+首错，
        # 后续轮用上轮修复 SQL+其首错
        round_prompts: Dict[int, Tuple[str, str, Dict[str, Any]]] = {}
        for di in repair_pending:
            item = items_by_di[di]
            if ddl_for(item["db_id"]) is None:
                item["prompt_error"] = "ddl_load_failed"
                continue
            if item.get("repair") is None:
                cur_sql = _winner_exec_sql(item)
                cur_err = item["first_error"]
            else:
                cur_sql = item["repair"].get("sql") or _winner_exec_sql(item)
                cur_err = item["repair"].get("first_error_after_repair") \
                    or item["first_error"]
            prompt_text, _canon = build_repair_prompt(
                tokenizer, item["question"], ddl_for(item["db_id"]),
                cur_sql, cur_err["instance"], cur_err["error"])
            round_prompts[di] = (prompt_text, cur_sql, cur_err)

        gen_pending = [di for di in repair_pending if di in round_prompts]
        repair_pending = [di for di in repair_pending
                          if di not in round_prompts]  # prompt 失败条目出队
        if not gen_pending:
            break

        # 引擎（每进程一次；单 lora 单 pass）
        if llm is None:
            warmup_ids = _tokenize_chat(tokenizer, round_prompts[gen_pending[0]][0],
                                        args.max_prompt_tokens)
            llm, lr, serving_mode = _serve(args, warmup_ids)
            gen_stats["serving"] = serving_mode

        from vllm import SamplingParams  # noqa: E402
        sampling_params = SamplingParams(n=1, temperature=args.temperature,
                                         top_p=args.top_p, seed=0,
                                         max_tokens=args.max_new_tokens)
        ids_by_di: Dict[int, List[int]] = {}
        for di in gen_pending:
            ids_by_di[di] = _tokenize_chat(tokenizer, round_prompts[di][0],
                                           args.max_prompt_tokens)

        queue = list(gen_pending)
        wall_start = time.perf_counter()
        while queue:
            chunk = queue[:args.checkpoint_every]
            reqs = [{"prompt_token_ids": ids_by_di[di]} for di in chunk]
            t0 = time.perf_counter()
            try:
                outputs = llm.generate(reqs, sampling_params, lora_request=lr)
            except Exception as exc:
                print(f"[phase B] generate failed on chunk di={chunk}: {exc}")
                _save_cp(output_dir, items_by_di, run_config)
                raise
            dt = time.perf_counter() - t0
            by_prompt = {tuple(o.prompt_token_ids): o for o in outputs}
            for di in chunk:
                item = items_by_di[di]
                o = by_prompt.get(tuple(ids_by_di[di]))
                if o is None or not o.outputs:
                    print(f"[WARN] di={di} no output, retry next slice")
                    continue
                text = tokenizer.decode(o.outputs[0].token_ids,
                                        skip_special_tokens=True)
                parsed = VavSampler.extract_sql(text)
                rep_sql = parsed["sql"] if parsed["parse_success"] else ""
                clean_rep = adj.clean_pred_for_official(rep_sql)
                insts = instances_for(item["db_id"])
                if clean_rep and insts:
                    engine.run([(clean_rep, p) for p in insts], phase="repair_exec")
                err_after = None
                n_ok = 0
                for p in insts:
                    out = engine.get(clean_rep, p) if clean_rep else \
                        adj._EMPTY_SQL_OUTCOME
                    if not out["ok"]:
                        err_after = {"instance": Path(p).name,
                                     "error": out["error"],
                                     "error_type": out["error_type"]}
                        break
                    n_ok += 1
                accepted = (err_after is None)
                item["repair"] = {
                    "rounds": round_used,
                    "accepted": bool(accepted),
                    "sql": rep_sql,
                    "parse_success": bool(parsed["parse_success"]),
                    "instances_ok": n_ok,
                    "instances_total": len(insts),
                    "first_error_after_repair": err_after,
                    "prompt_tokens": len(ids_by_di[di]),
                    "generated_tokens": len(o.outputs[0].token_ids),
                    "changed_from_winner": adj.normalize_for_dedup(rep_sql) !=
                    adj.normalize_for_dedup(_winner_exec_sql(item)),
                }
                print(f"[repair] di={di} round={round_used} accepted={accepted} "
                      f"parse={parsed['parse_success']} inst_ok="
                      f"{item['repair']['instances_ok']}/{len(insts)} "
                      f"{'' if err_after is None else err_after['error'][:60]}")
            gen_stats["generation_seconds"] += dt
            for o in outputs:
                for x in o.outputs:
                    gen_stats["generated_tokens"] += len(x.token_ids)
            gen_stats["requests"] += len(reqs)
            queue = queue[len(chunk):]
            _save_cp(output_dir, items_by_di, run_config)
        print(f"[phase B] round {round_used} done in "
              f"{gen_stats['generation_seconds']:.0f}s")
        # 未接受的题若还有轮次余量 → 下一轮（首错换成本轮修复 SQL 的错误）
        repair_pending = [di for di in requested_order
                          if di in items_by_di and items_by_di[di].get("winner_ok") is False
                          and items_by_di[di].get("repair") is not None
                          and not items_by_di[di]["repair"].get("accepted")
                          and not items_by_di[di].get("prompt_error")]

    if llm is not None:
        _teardown_engine(llm)

    _save_cp(output_dir, items_by_di, run_config)

    # ---- 输出 items_repair.json / items_baseline.json ----
    final_items = [items_by_di[di] for di in requested_order if di in items_by_di]
    repair_out, baseline_out = [], []
    for item in final_items:
        rep = item.get("repair")
        accepted_sql = (rep.get("sql") or "") if (rep and rep.get("accepted")) else ""
        pred = accepted_sql if accepted_sql else item["winner_sql"]
        base = {
            "dataset_index": item["dataset_index"],
            "di": item["di"],
            "db_id": item["db_id"],
            "question": item["question"],
            "gold_sql": item["gold_sql"],
            "predicted_sql": pred or "SELECT 1",
        }
        repair_out.append({**base, **{
            "winner_sql": item["winner_sql"],
            "winner_source": item.get("winner_source"),
            "winner_is_correct": item.get("winner_is_correct"),
            "repair_attempted": bool(rep),
            "repair_accepted": bool(rep and rep.get("accepted")),
            "error_msg": (item.get("first_error") or {}).get("error"),
            "error_instance": (item.get("first_error") or {}).get("instance"),
            "instances_total": len(db_instances.get(item["db_id"], [])),
            "prompt_error": item.get("prompt_error"),
        }})
        baseline_out.append({**base, **{
            "predicted_sql": item["winner_sql"] or "SELECT 1",
        }})
    with open(output_dir / "items_repair.json", "w", encoding="utf-8") as fh:
        json.dump(repair_out, fh, ensure_ascii=False, indent=2)
    with open(output_dir / "items_baseline.json", "w", encoding="utf-8") as fh:
        json.dump(baseline_out, fh, ensure_ascii=False, indent=2)

    # ---- summary ----
    n_trigger = sum(1 for it in final_items if it.get("winner_ok") is False)
    n_attempted = sum(1 for it in final_items if it.get("repair") is not None)
    n_parse_ok = sum(1 for it in final_items
                     if it.get("repair") and it["repair"].get("parse_success"))
    n_accepted = sum(1 for it in final_items
                     if it.get("repair") and it["repair"].get("accepted"))
    n_prompt_err = sum(1 for it in final_items if it.get("prompt_error"))
    fail_by_type: Dict[str, int] = {}
    fail_by_source: Dict[str, int] = {}
    for it in final_items:
        if it.get("winner_ok") is False:
            t = (it.get("first_error") or {}).get("error_type", "unknown")
            s = it.get("winner_source") or "unknown"
            fail_by_type[t] = fail_by_type.get(t, 0) + 1
            fail_by_source[s] = fail_by_source.get(s, 0) + 1

    # 1034 外推：修复生成成本 = 人口触发数 × 每题修复生成秒数（贪心 n=1）
    extrapolation = {"population_triggers": None,
                     "repair_gen_seconds_per_item": None,
                     "estimated_repair_gen_minutes_full1034": None,
                     "estimated_total_minutes_full1034": None}
    gen_per_item = (gen_stats["generation_seconds"] / max(n_attempted, 1)) \
        if gen_stats["generation_seconds"] else None
    pop_triggers = (population_scan or {}).get("triggers")
    if pop_triggers is not None and gen_per_item is not None:
        est_gen_min = pop_triggers * gen_per_item / 60.0
        # 引擎加载 ~1-2 min + 全量执行（扫描实测 ~0.6 min/1034 题）+ 修复执行 ~0
        extrapolation.update({
            "population_triggers": pop_triggers,
            "repair_gen_seconds_per_item": round(gen_per_item, 2),
            "estimated_repair_gen_minutes_full1034": round(est_gen_min, 2),
            "estimated_total_minutes_full1034": round(est_gen_min + 3.0, 2),
        })

    summary = {
        "evaluator_type": EVALUATOR_TYPE,
        "is_official_spider_metric": False,
        "note": (
            "T1-3 执行失败修复链 pilot。触发 = MI-VAV 胜者 SQL 在全部实例执行时任一 "
            "实例报错（执行口径 = adjudicate_pool 引擎 + clean_pred_for_official 清洗，"
            "与 eval_official.sh 同清洗口径）。修复 = canonical prompt + 失败 SQL + "
            "首个报错实例错误文本，sft_v2 vLLM 贪心 T=0 n=1，max_new_tokens=2048，"
            "VavSampler.extract_sql 解析。接受 = 修复 SQL 全部实例执行成功（无报错），"
            "不用 gold 判定（防泄漏，唯一接地信号 = 可执行性）。失败保留原胜者；"
            "轮数硬上限 --max-repair-rounds（默认 1）。官方 EX 对比需另跑 "
            "scripts/eval_official.sh（items_baseline.json vs items_repair.json）。"
            "风险口径：贪心 T=0 与候选生成 T=1.0 n=16 不同；修复段 prompt OOD 于 "
            "sft_v2 训练分布；二者以 parse/接受率量化。"
        ),
        "meta": {
            "items": str(args.items), "out_dir": str(output_dir),
            "spider_dir": str(args.spider_dir), "lora": args.lora,
            "base_model": args.base_model, "temperature": args.temperature,
            "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_repair_rounds": args.max_repair_rounds,
            "threads": args.threads, "query_timeout_seconds": args.query_timeout,
            "max_vm_steps": args.max_vm_steps, "row_cap": args.row_cap,
            "max_instances": args.max_instances,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "pilot": {"total": len(final_items),
                  "window": [args.start_index, args.start_index + args.limit]},
        "population_scan": population_scan,
        "winner_execution": {
            "total": len(final_items), "exec_ok": len(final_items) - n_trigger,
            "failed": n_trigger,
            "trigger_rate": round(n_trigger / len(final_items), 4) if final_items else 0.0,
            "fail_by_error_type": fail_by_type,
            "fail_by_source": fail_by_source,
            "prompt_errors": n_prompt_err,
        },
        "repair": {
            "attempted": n_attempted,
            "parse_success": n_parse_ok,
            "accepted": n_accepted,
            "accept_rate_of_attempted": round(n_accepted / n_attempted, 4) if n_attempted else 0.0,
            "accepted_rate_of_trigger": round(n_accepted / n_trigger, 4) if n_trigger else 0.0,
            "kept_winner_due_to_failure": n_trigger - n_accepted,
        },
        "generation": gen_stats,
        "execution_stats": {
            "population_scan": engine._stats.get("population_scan", {}),
            "winner_scan": engine._stats.get("winner_scan", {}),
            "repair_exec": engine._stats.get("repair_exec", {}),
        },
        "extrapolation_1034": extrapolation,
        "run_config": run_config,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 66)
    print("  REPAIR CHAIN SUMMARY")
    print("=" * 66)
    print(f"  pilot items: {len(final_items)} | winner exec failures (trigger): "
          f"{n_trigger} ({summary['winner_execution']['trigger_rate']:.1%})")
    print(f"  repair attempted: {n_attempted} | parse ok: {n_parse_ok} | accepted: "
          f"{n_accepted} ({summary['repair']['accept_rate_of_attempted']:.1%} of attempted)")
    print(f"  generation: {gen_stats['generation_seconds']:.0f}s, "
          f"{gen_stats['generated_tokens']} tokens, serving={gen_stats['serving']}")
    if population_scan:
        print(f"  population scan: {population_scan['triggers']}/{population_scan['total']} "
              f"({population_scan['trigger_rate']:.2%})")
    if extrapolation["estimated_total_minutes_full1034"] is not None:
        print(f"  EXTRAPOLATE full {SPIDER_DEV_SIZE} (gen+exec+engine): ~"
              f"{extrapolation['estimated_total_minutes_full1034']:.0f} min")
    print(f"\n  items_repair -> {output_dir / 'items_repair.json'}")
    print(f"  items_baseline -> {output_dir / 'items_baseline.json'}")
    print(f"  summary -> {output_dir / 'summary.json'}")
    print("\n  官方 EX 对比（事后，CPU）：")
    print(f"    bash scripts/eval_official.sh {output_dir / 'items_baseline.json'} "
          f"{output_dir / 'official_baseline'}")
    print(f"    bash scripts/eval_official.sh {output_dir / 'items_repair.json'} "
          f"{output_dir / 'official_repair'}")


if __name__ == "__main__":
    main()
