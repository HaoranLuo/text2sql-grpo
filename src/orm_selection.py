#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/orm_selection.py — T2 收尾实验：ORM 集成选择（4 臂，全部离线/推理侧，不重新生成候选）。

输入
  outputs/eval_pool_b1/items.json（1034 题 × 32 候选，B1 大候选池，复用裁决阶段同一批）
  checkpoints/orm_b1（训练完成后存在：Qwen2.5-Coder-3B-Instruct + LoRA，Yes/No 生成式二分类）
  outputs/adjudicate_b1/items_arm_vav_multi_all_both.json（MI-VAV 基线参考，官方 74.3%）

四个选择臂（同一批候选上离线重算，全部官方语义判定）：
  arm_orm_argmax    纯 ORM：每题去重候选（与 adjudicate_pool._dedupe 完全同口径）逐一打
                    P(Yes)，取最高分（平票 → min_sample_idx 最小，再 key 字符串最大）。
                    打分对象 = 全部唯一候选（20596），成本最高、无裁决器结构先验。
  arm_orm_grouphead 组级 ORM：复用 adjudicate_soft.build_groups 的 MI-VAV 分组（只 SUCCESS
                    候选入组、组大小 = 池内票数加权、组代表 = min_sample_idx 最小），只对
                    各组代表打分，选「组大小 × P(Yes)」综合最高组（平票 → 组大小、key）。
                    打分对象 = 组代表（~5947），成本最低的 ORM 臂。
  arm_orm_tiebreak  MI-VAV 主裁决不变；仅当触发（top1/top2 组大小差 ≤1 或 top1 组 size<2，
                    沿用 adjudicate_soft arm_gated_structural 的触发定义，排序键 = (size,
                    str(key)) 降序）时用 ORM 在两组代表间二选一（P(Yes) 大者胜；ORM 平票
                    保持 MI-VAV top1）。打分对象 = 触发题两组代表（最多 2×触发题数）。
  arm_orm_hybrid    连续打分：score = 组大小归一化(size / 该题最大组 size) × P(Yes)(组代表)，
                    对【全部分组】argmax——无 MI-VAV 的硬分组预筛（空组/全零组同样参与
                    连续打分；与 2 的区别：2 沿用 rankable_groups 的预筛，4 不筛；平票
                    → 组大小、key）。打分对象 = 组代表（~5947）。

打分口径（与 src/train_orm.py 训练/评估完全一致）：
  - 一次前向一个候选；P(Yes) = sigmoid(logit_Yes − logit_No)（Yes/No 均为单 token，
    id 运行时 assert 校验；logprob 差 ≡ logit 差）。
  - prompt = build_orm_prompt（本文件内逐字复制 src/label_orm_data.build_orm_prompt，保证
    train/infer prompt 逐字一致；vllmenv 无 nltk 无法 import 其全模块，故本地复制 +
    运行时自检）：canonical 生成端 prompt + Candidate SQL Query 块 + Yes/No 指令；
    chat template + add_generation_prompt，左截断 max_length=2048（与
    train_orm.tokenize_score 同口径）。
  - 推理后端：vLLM 0.11.2 batch（envs/vllmenv，选快——20596 候选 × ~560 token ≈ 11.5M
    prompt token，A40/3090 上 ~5-15 min；HF+peft batch 约 30-75 min，不采用）。top-k
    logprobs（默认 20, vLLM 上限）取 Yes/No；个别不在 top-k 的请求用全词表 logprobs 重打分兜底。
  - LoRA 服务：vLLM LoRARequest（r=32，与 sft 系同构，写法同 src/eval_pool_b1.py）；
    加载失败自动回退 peft merge（envs/reasoning3b 子进程）→ checkpoints/orm_b1_merged。

冒烟方案（ORM 未训练完之前的全流程验证）：
  --stub-scores random|constant 用随机/恒定分数替代 ORM 输出（不 import torch/vLLM/
  transformers，纯 CPU 可跑，登录节点直接执行），走通 去重 → 打分 → MI-VAV 分组 →
  4 臂选择 → 官方语义判定 → items/summary 输出的完整链路。

输出 outputs/orm_selection/
  items_<arm>.json   4 个，predicted_sql 与 scripts/eval_official.sh 兼容（空胜者按
                     AGENTS.md 铁律 4 写 "SELECT 1" 不跳过）
  summary.json       各臂官方语义判定准确率 + vs MI-VAV 基线 74.3% 的 fixed/broken
                     + ORM 打分成本统计（见下）+ 门控触发统计

成本口径：本实验 4 臂共享一次全候选打分（argmax 臂所需），grouphead/hybrid/tiebreak
只复用其子集；summary.scoring_cost.nominal_cost_per_arm 报告各臂单独部署所需打分次数。

用法
  # 真跑（HPC GPU 节点，envs/vllmenv）：
  envs/vllmenv/bin/python src/orm_selection.py
  # 冒烟（登录节点 CPU，随机分数替代 ORM，~1-2 min）：
  envs/reasoning3b/bin/python src/orm_selection.py --stub-scores random --limit 30 \
      --out-dir outputs/orm_selection_smoke
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(PROJECT_ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adjudicate_pool as AP  # noqa: E402  去重/执行引擎/官方 exec-match 判定（纯 CPU）
import adjudicate_soft as AS  # noqa: E402  build_groups/rankable_groups/_group_rep/_base_record

DEFAULT_ITEMS = PROJECT_ROOT / "outputs" / "eval_pool_b1" / "items.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "orm_selection"
DEFAULT_SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"
DEFAULT_ORM_CKPT = PROJECT_ROOT / "checkpoints" / "orm_b1"
DEFAULT_MERGE_PYTHON = PROJECT_ROOT / "envs" / "reasoning3b" / "bin" / "python"
DEFAULT_BASELINE_ITEMS = (
    PROJECT_ROOT / "outputs" / "adjudicate_b1" / "items_arm_vav_multi_all_both.json")

ARMS = ["arm_orm_argmax", "arm_orm_grouphead", "arm_orm_tiebreak", "arm_orm_hybrid"]
BASELINE_ARM = "arm_vav_multi_all"   # MI-VAV 基线（both 池，官方 74.3%）
ALL_ARMS = [BASELINE_ARM] + ARMS
POOL = "both"
YES_STR, NO_STR = "Yes", "No"


def build_orm_prompt(question: str, ddl_schema: str, candidate_sql: str) -> str:
    """ORM user 侧输入——【逐字复制】src/label_orm_data.build_orm_prompt（train/infer
    prompt 逐字节一致是硬要求；vllmenv 无 nltk，无法 import label_orm_data 全模块
    ——其 import 链经 tools/original_spider_eval 拖入 nltk——故本地复制函数体，
    base prompt 仍来自同一 ReasoningGeneratorAgent.build_prompt）。运行时若环境可
    导入 label_orm_data 则做逐字一致性自检（见 main）。"""
    from reasoning_generator_agent import ReasoningGeneratorAgent  # noqa: E402
    base = ReasoningGeneratorAgent.build_prompt(
        question=question, ddl_schema=ddl_schema,
        schema_links=None, evidence=None, dialect="sqlite")
    cand = (candidate_sql or "").strip().replace("```", "`")
    return (
        f"{base}\n\nCandidate SQL Query:\n```sql\n{cand}\n```\n\n"
        "Task: Judge whether the candidate SQL query above correctly answers the "
        "question (execution-equivalent to the gold query). "
        "Answer with only Yes or No."
    )


# ===================================================================
# 参数
# ===================================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ORM 集成选择实验（4 臂，vLLM 打分 + 官方语义判定）")
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--spider-dir", type=Path, default=DEFAULT_SPIDER_DIR)
    ap.add_argument("--orm-checkpoint", type=Path, default=DEFAULT_ORM_CKPT,
                    help="ORM LoRA 目录（真跑需要 adapter_config.json；亦可指向已合并权重目录）")
    ap.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    ap.add_argument("--merge-python", default=str(DEFAULT_MERGE_PYTHON),
                    help="peft merge 子进程解释器（vllmenv 无 peft，用 reasoning3b）")
    ap.add_argument("--max-length", type=int, default=2048,
                    help="ORM prompt 左截断长度（与 train_orm.tokenize_score 一致）")
    ap.add_argument("--logprobs-topk", type=int, default=20,
                    help="vLLM 每步返回 top-k logprobs（vLLM 0.11.2 上限=20，实测 128 直接杀引擎；"
                         "Yes/No 二选一通常在 top-20 内；缺失的请求自动用全词表 logprobs 重打分兜底）")
    ap.add_argument("--chunk-size", type=int, default=512,
                    help="单次 llm.generate 的请求数（vLLM 内部继续 batch）")
    ap.add_argument("--enforce-eager", action="store_true", help="vLLM 关闭 CUDA graph（重试用）")
    ap.add_argument("--max-num-seqs", type=int, default=None, help="vLLM 单批最大序列数")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--query-timeout", type=float, default=30.0)
    ap.add_argument("--max-vm-steps", type=int, default=5_000_000)
    ap.add_argument("--row-cap", type=int, default=100_000)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--keep-distinct", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 题（冒烟）")
    ap.add_argument("--stub-scores", choices=["off", "random", "constant"], default="off",
                    help="ORM 未训练完时的冒烟方案：随机/恒定分数替代 ORM 输出（不加载模型）")
    ap.add_argument("--stub-const", type=float, default=0.5,
                    help="--stub-scores constant 的分数值（默认 0.5）")
    return ap.parse_args(argv)


# ===================================================================
# ORM 打分器：Stub（冒烟）/ vLLM（真跑）
# ===================================================================


class StubScorer:
    """随机/恒定分数替代 ORM——走通全流程（不 import torch/vLLM/transformers）。"""

    def __init__(self, seed: int, const: Optional[float]) -> None:
        self.seed = seed
        self.const = const

    def score(self, payloads: List[Tuple[int, int, Optional[str]]]) -> List[float]:
        if self.const is not None:
            scores = [float(self.const)] * len(payloads)
        else:
            rng = random.Random(self.seed)
            scores = [rng.random() for _ in payloads]
        return scores

    @property
    def stats(self) -> Dict[str, Any]:
        return {"mode": f"stub-{'constant' if self.const is not None else 'random'}",
                "n_prompts": None, "wall_seconds": None,
                "prompt_tokens": None, "tokens_per_second": None,
                "logprobs_topk": None, "n_topk_covered": None,
                "n_rescored_full_vocab": None, "n_missing": None}


def _p_yes_from_logprobs(lp_yes: float, lp_no: float) -> float:
    """P(Yes) = sigmoid(logit_Yes − logit_No)；logprob 差 ≡ logit 差（train_orm 同式）。"""
    return float(1.0 / (1.0 + math.exp(lp_no - lp_yes)))


def _extract_p_yes(lp_dict: Any, yes_id: int, no_id: int) -> Optional[float]:
    """从 vLLM 单步 logprobs dict[int, Logprob] 提取 P(Yes)；Yes/No 缺失返回 None。"""
    ye = lp_dict.get(yes_id)
    no = lp_dict.get(no_id)
    if ye is None or no is None:
        return None
    return _p_yes_from_logprobs(float(ye.logprob), float(no.logprob))


class VllmScorer:
    """vLLM 0.11 batch 打分（选快的：11.5M prompt token 预填 A40/3090 ≈ 5-15 min）。

    max_tokens=1 只取首 token logprobs（temperature=1.0 保持 logits 未缩放；
    top_p=1.0 无惩罚——vLLM 返回的 logprob 与温度相关，T=1 才等价原始 logits）。
    prompt_token_ids 直送（与 eval_pool_b1 同口径，规避 vLLM 端 chat-template 差异）。
    """

    def __init__(self, args: argparse.Namespace) -> None:
        # 延迟 import：真跑才需要 GPU 依赖（stub 冒烟零 torch/vLLM）
        from transformers import AutoTokenizer  # noqa: E402

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, local_files_only=True, trust_remote_code=True)
        yes_ids = self.tokenizer.encode(YES_STR, add_special_tokens=False)
        no_ids = self.tokenizer.encode(NO_STR, add_special_tokens=False)
        assert len(yes_ids) == 1 and len(no_ids) == 1, \
            f"Yes/No 必须单 token，实际 {yes_ids}/{no_ids}"
        self.yes_id, self.no_id = yes_ids[0], no_ids[0]
        self._stats: Dict[str, Any] = {
            "mode": None, "n_prompts": 0, "wall_seconds": 0.0,
            "prompt_tokens": None, "tokens_per_second": None,
            "logprobs_topk": args.logprobs_topk,
            "n_topk_covered": 0, "n_rescored_full_vocab": 0, "n_missing": 0}

    # ---- vLLM 引擎：LoRA 优先，失败回退 peft merge 权重（同 eval_pool_b1 模式）----

    def _merged_ok(self, merged_dir: Path) -> bool:
        return merged_dir.is_dir() and any(merged_dir.glob("*.safetensors"))

    def _merge_lora(self, lora_path: Path, merged_dir: Path) -> None:
        script = (
            "import sys\n"
            "import torch\n"
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            "from peft import PeftModel\n"
            "base, lora, out = sys.argv[1], sys.argv[2], sys.argv[3]\n"
            "print('[orm-merge] loading base (CPU bf16)...', flush=True)\n"
            "tok = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=True)\n"
            "model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,\n"
            "    local_files_only=True, trust_remote_code=True)\n"
            "model = PeftModel.from_pretrained(model, lora)\n"
            "model = model.merge_and_unload()\n"
            "model.save_pretrained(out)\n"
            "tok.save_pretrained(out)\n"
            "print('[orm-merge] DONE', flush=True)\n")
        merged_dir.mkdir(parents=True, exist_ok=True)
        import subprocess
        print(f"[orm] peft-merging {lora_path} -> {merged_dir} "
              f"(python={self.args.merge_python})")
        subprocess.run([self.args.merge_python, "-c", script, self.args.base_model,
                        str(lora_path), str(merged_dir)], check=True)

    def _warmup(self, llm: Any, lr: Any, warmup_ids: List[int]) -> None:
        from vllm import SamplingParams
        sp = SamplingParams(n=1, temperature=1.0, max_tokens=8, seed=0)
        out = llm.generate([{"prompt_token_ids": warmup_ids}], sp, lora_request=lr)
        text = (out[0].outputs[0].text if out and out[0].outputs else "")
        print(f"[orm] warmup ok (first 40 chars): {text[:40]!r}")

    def _teardown(self, llm: Any) -> None:
        """vLLM 0.11 EngineCore 子进程释放有延迟；轮询等待，避免后续引擎 init 撞上
        'Free memory on device' 检查（同 eval_pool_b1._teardown_engine 模式）。"""
        import gc
        try:
            del llm
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            free, total = torch.cuda.mem_get_info()
            if free <= 0.5 * total:
                print("[orm] waiting for GPU memory release after engine teardown...")
                deadline = time.time() + 60.0
                while time.time() < deadline:
                    time.sleep(2.0)
                    free, total = torch.cuda.mem_get_info()
                    if free > 0.5 * total:
                        break
                print(f"[orm] GPU memory after teardown: "
                      f"{free / 1024**3:.1f}/{total / 1024**3:.1f} GiB free")
        except Exception:
            pass

    def _serve(self, warmup_ids: List[int]) -> Tuple[Any, Any]:
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        lora_path = Path(self.args.orm_checkpoint)
        merged_dir = Path(str(self.args.orm_checkpoint) + "_merged")
        base_kwargs = dict(dtype="bfloat16", trust_remote_code=True, seed=0,
                           max_model_len=self.args.max_length + 16)
        if self.args.enforce_eager:
            base_kwargs["enforce_eager"] = True
        if self.args.max_num_seqs is not None:
            base_kwargs["max_num_seqs"] = self.args.max_num_seqs

        lora_ok = lora_path.is_dir() and (lora_path / "adapter_config.json").exists()
        if not lora_path.exists():
            raise RuntimeError(
                f"ORM checkpoint 不存在: {lora_path}——先完成 scripts/train_orm.slurm 训练，"
                f"或用 --stub-scores 冒烟")
        if lora_ok:
            llm = None
            try:
                t0 = time.perf_counter()
                llm = LLM(model=self.args.base_model, enable_lora=True, max_loras=1,
                          max_lora_rank=32, **base_kwargs)
                lr = LoRARequest("orm_b1", 1, str(lora_path))
                print(f"[orm] engine(LoRA) ready in {time.perf_counter() - t0:.1f}s")
                self._warmup(llm, lr, warmup_ids)
                return llm, lr
            except Exception as exc:
                if "Free memory on device" in str(exc):
                    self._teardown(llm)
                    raise RuntimeError(
                        f"[orm] engine init 显存不足（资源问题，非 LoRA 兼容问题，"
                        f"不做 merge 回退）: {exc}") from exc
                print(f"[orm] LoRA serving failed: {exc}")
                print("[orm] falling back to merged weights")
                self._teardown(llm)

        if not self._merged_ok(merged_dir):
            if not lora_ok:
                raise RuntimeError(
                    f"{lora_path} 既无 adapter_config.json（LoRA 未就绪）也无合并权重——"
                    f"先完成 ORM 训练")
            self._merge_lora(lora_path, merged_dir)
        t0 = time.perf_counter()
        llm = LLM(model=str(merged_dir), enable_lora=False, **base_kwargs)
        print(f"[orm] engine(merged) ready in {time.perf_counter() - t0:.1f}s")
        self._warmup(llm, None, warmup_ids)
        return llm, None

    # ---- 打分主循环 ----

    def score(self, payloads: List[Tuple[int, int, str]]) -> List[float]:
        """payloads: [(qi, ei, prompt_text)]。返回对齐的 P(Yes) 列表。"""
        args = self.args
        tok = self.tokenizer
        ids_list: List[List[int]] = []
        n_tokens = 0
        for (_qi, _ei, prompt) in payloads:
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, return_dict=True)
            ids = enc["input_ids"][-args.max_length:]  # 左截断（保住候选 SQL 与指令）
            ids_list.append(ids)
            n_tokens += len(ids)

        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, seed=0,
                            logprobs=args.logprobs_topk, detokenize=False)
        sp_full = SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, seed=0,
                                 logprobs=200000, detokenize=False)  # > 词表 → 全词表

        llm, lr = self._serve(ids_list[0])
        scores: List[Optional[float]] = [None] * len(payloads)
        t0 = time.perf_counter()
        stragglers: List[int] = []

        def _run(chunk_ids: List[List[int]], sp_i: Any) -> List[Any]:
            reqs = [{"prompt_token_ids": ids} for ids in chunk_ids]
            return llm.generate(reqs, sp_i, lora_request=lr)

        for i in range(0, len(payloads), args.chunk_size):
            chunk = ids_list[i:i + args.chunk_size]
            outputs = _run(chunk, sp)
            by_prompt = {tuple(o.prompt_token_ids): o for o in outputs}
            if len(by_prompt) != len(chunk):
                print(f"[orm] WARN matched {len(by_prompt)}/{len(chunk)} outputs "
                      f"by prompt ids (重复 prompt 合并，分数相同无影响)")
            for j, ids in enumerate(chunk):
                o = by_prompt.get(tuple(ids))
                if o is None or not o.outputs or not o.outputs[0].logprobs:
                    stragglers.append(i + j)
                    continue
                p = _extract_p_yes(o.outputs[0].logprobs[0], self.yes_id, self.no_id)
                if p is None:
                    stragglers.append(i + j)
                else:
                    scores[i + j] = p
                    self._stats["n_topk_covered"] += 1
            if (i // args.chunk_size + 1) % 10 == 0:
                print(f"[orm] scored {min(i + args.chunk_size, len(payloads))}/"
                      f"{len(payloads)} ({time.perf_counter() - t0:.0f}s)")

        # 兜底：top-k 缺 Yes/No 的请求用全词表 logprobs 重打分（应为极少数/零）
        for s in stragglers:
            ids = ids_list[s]
            outputs = _run([ids], sp_full)
            o = outputs[0]
            if not o.outputs or not o.outputs[0].logprobs:
                raise RuntimeError(f"[orm] 全词表重打分无 logprobs: payload {s}")
            p = _extract_p_yes(o.outputs[0].logprobs[0], self.yes_id, self.no_id)
            if p is None:
                raise RuntimeError(
                    f"[orm] 全词表 logprobs 仍缺 Yes/No（id {self.yes_id}/{self.no_id}）: "
                    f"payload {s}")
            scores[s] = p
            self._stats["n_rescored_full_vocab"] += 1

        wall = time.perf_counter() - t0
        self._stats.update({
            "mode": "vllm-lora" if lr is not None else "vllm-merged",
            "n_prompts": len(payloads),
            "wall_seconds": round(wall, 2),
            "prompt_tokens": n_tokens,
            "tokens_per_second": round(n_tokens / max(wall, 1e-9), 1),
            "n_missing": len(payloads) - sum(1 for s in scores if s is not None),
        })
        if any(s is None for s in scores):
            raise RuntimeError("[orm] 存在未打分候选（内部错误）")
        try:
            del llm
        except Exception:
            pass
        return [float(s) for s in scores]

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats


# ===================================================================
# 单题裁决：4 臂 + MI-VAV 基线（全部离线重算，同一批候选）
# ===================================================================


def _rep_and_ei(entries: List[Dict[str, Any]], g: Dict[str, Any],
                ei_by_key: Dict[str, int]) -> Tuple[Dict[str, Any], int]:
    rep = AS._group_rep(entries, g)
    return rep, ei_by_key[rep["key"]]


def _fallback(entries: List[Dict[str, Any]], votes: Dict[int, int],
              n_used: int, grouped: int, excluded: int) -> Dict[str, Any]:
    """NO_RESULTS / 无组 → 回退同池 arm_maj（与 _arm_vav fallback 链一致）。"""
    rec = AS._fallback_record(entries, votes, n_used, grouped, excluded)
    rec["orm_score"] = None
    return rec


def select_arms(entries: List[Dict[str, Any]], scores: List[float],
                sigs_per_entry: List[List[str]], votes: Dict[int, int],
                instances: List[str]) -> Tuple[Dict[str, Dict[str, Any]], Dict]:
    """对一题完成 4 臂 + 基线裁决（只选胜者，不执行 SQL）。
    返回 (results, groups)——groups 供成本统计复用，避免重复构建。"""
    n_used = len(instances)
    ei_by_key = {e["key"]: i for i, e in enumerate(entries)}
    groups, grouped, excluded = AS.build_groups(entries, sigs_per_entry, votes, n_used)
    ranked = AS.rankable_groups(groups)  # 已硬跳过空组/全零组（与 choose_group_vav 一致）
    ordered = sorted(ranked, key=lambda km: (km[1]["size"], str(km[0])), reverse=True)
    joins_cache: Dict[str, Tuple[int, str]] = {}
    results: Dict[str, Dict[str, Any]] = {}

    def finish(rec: Dict[str, Any]) -> Dict[str, Any]:
        rec["empty_winner"] = (rec["text"] == "")
        return rec

    # ---- 基线 arm_vav_multi_all（both 池，= MI-VAV 74.3% 同语义）----
    if not groups:
        base = _fallback(entries, votes, n_used, grouped, excluded)
    else:
        chosen = AP.choose_group_vav(groups)
        if chosen is None:
            base = _fallback(entries, votes, n_used, grouped, excluded)
        else:
            base = AS._base_record(entries, chosen, groups[chosen], "vav",
                                   n_used, grouped, excluded, joins_cache)
            _rep, ei = _rep_and_ei(entries, groups[chosen], ei_by_key)
            base["orm_score"] = scores[ei]
    results[BASELINE_ARM] = finish(base)

    # ---- arm 1：纯 ORM argmax（全部唯一候选，含未成功执行/空 SQL 候选）----
    if not entries:
        rec = {"source": "no_pool", "text": None, "votes": 0, "group_key": None,
               "group_size": 0, "instances_used": n_used, "vav_grouped": 0,
               "vav_excluded": 0, "orm_score": None}
    else:
        best = max(range(len(entries)),
                   key=lambda i: (scores[i], -entries[i]["min_sample_idx"],
                                  entries[i]["key"]))
        e = entries[best]
        rec = {"source": "orm_argmax", "text": e["sql_text"], "votes": e["count"],
               "group_key": None, "group_size": e["count"], "instances_used": 0,
               "vav_grouped": 0, "vav_excluded": 0, "orm_score": scores[best]}
    results["arm_orm_argmax"] = finish(rec)

    # ---- arm 2：组级 ORM（只对各组代表打分，size × P(Yes) 最高组）----
    if not ranked:
        rec = _fallback(entries, votes, n_used, grouped, excluded)
    else:
        def ghead_key(kg: Tuple[Tuple[str, ...], Dict[str, Any]]) -> Tuple[float, int, str]:
            _rep, ei = _rep_and_ei(entries, kg[1], ei_by_key)
            return (kg[1]["size"] * scores[ei], kg[1]["size"], str(kg[0]))

        chosen_key, chosen_g = max(ranked, key=ghead_key)
        rec = AS._base_record(entries, chosen_key, chosen_g, "orm_grouphead",
                              n_used, grouped, excluded, joins_cache)
        _rep, ei = _rep_and_ei(entries, chosen_g, ei_by_key)
        rec["orm_score"] = scores[ei]
    results["arm_orm_grouphead"] = finish(rec)

    # ---- arm 4：混合连续打分（size 归一化 × P(Yes)；无硬分组预筛——空组/全零组也参与）----
    if not groups:
        rec = _fallback(entries, votes, n_used, grouped, excluded)
    else:
        all_groups = list(groups.items())  # 与 arm 2 的区别：不经过 rankable_groups 预筛
        max_size_all = max(g["size"] for _, g in all_groups)

        def hybrid_key(kg: Tuple[Tuple[str, ...], Dict[str, Any]]) -> Tuple[float, int, str]:
            _rep, ei = _rep_and_ei(entries, kg[1], ei_by_key)
            return ((kg[1]["size"] / max_size_all) * scores[ei], kg[1]["size"], str(kg[0]))

        chosen_key, chosen_g = max(all_groups, key=hybrid_key)
        rec = AS._base_record(entries, chosen_key, chosen_g, "orm_hybrid",
                              n_used, grouped, excluded, joins_cache)
        _rep, ei = _rep_and_ei(entries, chosen_g, ei_by_key)
        rec["orm_score"] = scores[ei]
    results["arm_orm_hybrid"] = finish(rec)

    # ---- arm 3：MI-VAV 主裁决 + 触发式 ORM 二选一（触发定义沿用 gated_structural）----
    if not ranked:
        rec = _fallback(entries, votes, n_used, grouped, excluded)
        rec.update({"orm_triggered": False, "top1_size": None, "top2_size": None,
                    "orm_changed_winner": False, "top1_orm_score": None,
                    "top2_orm_score": None})
    else:
        top1_key, top1_g = ordered[0]
        top2 = ordered[1] if len(ordered) > 1 else None
        size1 = top1_g["size"]
        size2 = top2[1]["size"] if top2 is not None else None
        triggered = (size1 < 2) or (top2 is not None and size1 - size2 <= 1)
        _rep1, ei1 = _rep_and_ei(entries, top1_g, ei_by_key)
        s1 = scores[ei1]
        if not triggered or top2 is None:
            rec = AS._base_record(entries, top1_key, top1_g, "vav",
                                  n_used, grouped, excluded, joins_cache)
            rec.update({"orm_triggered": bool(triggered), "top1_size": size1,
                        "top2_size": size2, "orm_changed_winner": False,
                        "top1_orm_score": s1, "top2_orm_score": None})
        else:
            _rep2, ei2 = _rep_and_ei(entries, top2[1], ei_by_key)
            s2 = scores[ei2]
            # ORM 平票 → 保持 MI-VAV top1（确定性）
            chosen_key, chosen_g = (top2[0], top2[1]) if s2 > s1 else (top1_key, top1_g)
            rec = AS._base_record(entries, chosen_key, chosen_g, "orm_tiebreak",
                                  n_used, grouped, excluded, joins_cache)
            rec.update({"orm_triggered": True, "top1_size": size1,
                        "top2_size": size2,
                        "orm_changed_winner": chosen_key != top1_key,
                        "top1_orm_score": s1, "top2_orm_score": s2})
    results["arm_orm_tiebreak"] = finish(rec)

    return results, groups


# ===================================================================
# 主流程
# ===================================================================


def _agg(recs: List[Dict[str, Any]], key: str = "is_correct") -> Dict[str, Any]:
    n = len(recs)
    c = sum(1 for r in recs if r.get(key))
    return {"n": n, "correct": c, "accuracy": round(c / n, 4) if n else None}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    AP.rng = random.Random(args.seed)  # result_eq 列置换剪枝的确定性
    is_stub = args.stub_scores != "off"

    items = AP._load_items(args.items)
    if args.limit:
        items = items[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[orm_select] {len(items)} 题 | stub={args.stub_scores} | seed={args.seed} | "
          f"out={args.out_dir}", file=sys.stderr)

    # ---- 每题去重（与 adjudicate_pool._dedupe 完全同口径）----
    entries_by_q: List[List[Dict[str, Any]]] = []
    for item in items:
        entries_by_q.append(AP._dedupe(item.get("candidates") or []))

    # ---- ORM 打分（stub：随机/恒定；真跑：vLLM batch，prompt = label_orm_data 同一函数）----
    t_score0 = time.perf_counter()
    prompt_check: Optional[str] = None
    if is_stub:
        payloads: List[Tuple[int, int, Optional[str]]] = [
            (qi, ei, None) for qi, entries in enumerate(entries_by_q)
            for ei in range(len(entries))]
        scorer: Any = StubScorer(
            args.seed, args.stub_const if args.stub_scores == "constant" else None)
    else:
        from spider_utils import SpiderLoader  # noqa: E402

        # prompt 一致性自检：环境可导入 label_orm_data 时逐字比对（vllmenv 无 nltk
        # 导入失败则跳过——本地 build_orm_prompt 为逐字复制，见函数 docstring）
        prompt_check = "passed"
        try:
            from label_orm_data import build_orm_prompt as _canonical_prompt
            _sample = _canonical_prompt("__q__", "__ddl__", "SELECT 1;")
            if _sample != build_orm_prompt("__q__", "__ddl__", "SELECT 1;"):
                raise AssertionError("本地 build_orm_prompt 与 label_orm_data 不一致")
            print("[orm_select] build_orm_prompt 与 label_orm_data 逐字一致性自检通过")
        except ImportError:
            prompt_check = "skipped-no-nltk"
            print("[orm_select] WARN 本环境无法 import label_orm_data（缺 nltk），"
                  "跳过 prompt 一致性自检（本地实现为逐字复制）")

        loader = SpiderLoader(str(args.spider_dir))
        ddl_cache: Dict[str, str] = {}

        def ddl_for(db_id: str) -> str:
            if db_id not in ddl_cache:
                ddl_cache[db_id] = loader.format_ddl(db_id)
            return ddl_cache[db_id]

        payloads = []
        for qi, item in enumerate(items):
            entries = entries_by_q[qi]
            if not entries:
                continue
            ddl = ddl_for(item.get("db_id", ""))
            question = item.get("question", "")
            for ei, e in enumerate(entries):
                payloads.append((qi, ei, build_orm_prompt(question, ddl, e["sql_text"])))
        scorer = VllmScorer(args)

    if not payloads:
        raise RuntimeError("无候选可打分（items 为空或全部为 error 条目）")

    score_list = scorer.score(payloads)
    score_map: Dict[Tuple[int, int], float] = {}
    for (qi, ei, _p), s in zip(payloads, score_list):
        score_map[(qi, ei)] = s
    print(f"[orm_select] 打分完成: {len(payloads)} 候选 "
          f"({time.perf_counter() - t_score0:.1f}s, mode={scorer.stats['mode']})",
          file=sys.stderr)

    # ---- 实例枚举 + Phase 1：全部唯一候选原始 SQL × 实例 并行执行（分组签名）----
    database_dir = args.spider_dir / "database"
    engine = AP.ExecutionEngine(args.threads, args.query_timeout,
                                args.max_vm_steps, args.row_cap)
    db_instances_cache: Dict[str, List[str]] = {}

    def instances_for(db_id: str) -> List[str]:
        if db_id not in db_instances_cache:
            db_instances_cache[db_id] = AP.list_instances(
                str(database_dir / db_id), db_id, args.max_instances)
        return db_instances_cache[db_id]

    phase1_tasks: List[Tuple[str, str]] = []
    for qi, item in enumerate(items):
        insts = instances_for(item.get("db_id", ""))
        for e in entries_by_q[qi]:
            text = (e["sql_text"] or "").strip()
            if not text:
                continue
            for inst in insts:
                phase1_tasks.append((text, inst))
    phase1_tasks = list(set(phase1_tasks))
    print(f"[orm_select] phase1: {len(phase1_tasks)} 个唯一 (sql, db_path) 任务",
          file=sys.stderr)
    engine.run(phase1_tasks, phase="grouping")
    print(f"[orm_select] phase1 完成: {engine._stats['grouping']}", file=sys.stderr)

    # ---- 每题裁决（胜者选择，不执行 SQL）----
    per_question: List[Dict[str, Any]] = []
    for qi, item in enumerate(items):
        entries = entries_by_q[qi]
        insts = instances_for(item.get("db_id", ""))
        sigs_per_entry: List[List[str]] = []
        for e in entries:
            if not (e["sql_text"] or "").strip():
                sigs = [AP.ERROR_SIG] * len(insts)
            else:
                sigs = [AP.outcome_signature(engine.get(e["sql_text"], inst))
                        for inst in insts]
            sigs_per_entry.append(sigs)
        # both 池票数（与 adjudicate_soft 的 votes 构建一致）
        votes: Dict[int, int] = defaultdict(int)
        for c in item.get("candidates") or []:
            ck = AP.normalize_for_dedup(c.get("sql"))
            for ei, e in enumerate(entries):
                if ck == e["key"]:
                    votes[ei] += 1
                    break
        q_scores = [score_map[(qi, ei)] for ei in range(len(entries))]
        res, groups = select_arms(entries, q_scores, sigs_per_entry, votes, insts)
        per_question.append({
            "item": item, "entries": entries, "results": res, "groups": groups,
            "num_candidates": len(item.get("candidates") or []),
            "num_unique_candidates": len(entries),
            "num_instances": len(insts),
        })

    # ---- Phase 2：gold 变换后 + 各臂胜者变换后 SQL × 实例 执行 ----
    phase2_tasks: List[Tuple[str, str]] = []
    for qc in per_question:
        item = qc["item"]
        insts = instances_for(item.get("db_id", ""))
        gold_t = AP.official_transform(item.get("gold_sql") or "", is_pred=False,
                                       keep_distinct=args.keep_distinct)
        for inst in insts:
            phase2_tasks.append((gold_t, inst))
        for arm in ALL_ARMS:
            rec = qc["results"][arm]
            if rec["text"] is None:
                continue
            pred_t = AP.official_transform(rec["text"], is_pred=True,
                                           keep_distinct=args.keep_distinct)
            for inst in insts:
                phase2_tasks.append((pred_t, inst))
    phase2_tasks = list(set(phase2_tasks))
    engine.run(phase2_tasks, phase="judgment")
    print(f"[orm_select] phase2 完成: {engine._stats['judgment']}", file=sys.stderr)

    # ---- 判定填充（纯内存计算）----
    for qc in per_question:
        item = qc["item"]
        gold_raw = item.get("gold_sql") or ""
        insts = instances_for(item.get("db_id", ""))
        for arm in ALL_ARMS:
            rec = qc["results"][arm]
            j = AP._judge_winner(rec["text"], gold_raw, insts, engine, args.keep_distinct)
            rec["is_correct"] = j["correct"]
            rec["gold_exec_error"] = j["gold_exec_error"]
            rec["order_matters"] = j["order_matters"]

    # ---- 汇总 ----
    import importlib.util
    sqlparse_ok = importlib.util.find_spec("sqlparse") is not None
    sqlglot_ok = importlib.util.find_spec("sqlglot") is not None

    dataset_stats: Dict[str, Any] = {
        "total_questions": len(items),
        "questions_with_no_instances": 0,
        "questions_with_gold_exec_error": 0,
        "db_instance_count": {
            db: len(insts) for db, insts in db_instances_cache.items()},
    }
    total_cands = unique_cands = 0
    for qc in per_question:
        total_cands += qc["num_candidates"]
        unique_cands += qc["num_unique_candidates"]
        if qc["num_instances"] == 0:
            dataset_stats["questions_with_no_instances"] += 1
        if any(qc["results"][arm].get("gold_exec_error") for arm in ALL_ARMS):
            dataset_stats["questions_with_gold_exec_error"] += 1

    cells: Dict[str, Dict[str, Any]] = {}
    for arm in ALL_ARMS:
        cell: Dict[str, Any] = {
            "total": len(items), "correct": 0, "accuracy": 0.0,
            "winner_sources": Counter(), "empty_winner": 0,
            "gold_exec_error": 0, "candidates_available": 0,
            "orm_triggered": 0, "orm_triggered_correct": 0,
            "orm_changed_winner": 0,
        }
        for qc in per_question:
            rec = qc["results"][arm]
            if rec["source"] != "no_pool":
                cell["candidates_available"] += 1
            cell["winner_sources"][rec["source"]] += 1
            if rec.get("empty_winner"):
                cell["empty_winner"] += 1
            if rec.get("gold_exec_error"):
                cell["gold_exec_error"] += 1
            if rec.get("is_correct"):
                cell["correct"] += 1
            if rec.get("orm_triggered"):
                cell["orm_triggered"] += 1
                if rec.get("is_correct"):
                    cell["orm_triggered_correct"] += 1
            if rec.get("orm_changed_winner"):
                cell["orm_changed_winner"] += 1
        cell["accuracy"] = round(cell["correct"] / cell["total"], 4) if cell["total"] else 0.0
        cell["winner_sources"] = dict(cell["winner_sources"])
        cells[arm] = cell

    # ---- vs MI-VAV 基线 fixed/broken（基线 = 本进程同口径重算；官方参考 74.3%）----
    base_correct = [qc["results"][BASELINE_ARM]["is_correct"] for qc in per_question]
    vs_baseline: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        fixed = broken = same_r = same_w = 0
        f_idx: List[Any] = []
        b_idx: List[Any] = []
        for i, qc in enumerate(per_question):
            a = qc["results"][arm]["is_correct"]
            b = base_correct[i]
            idx = qc["item"].get("dataset_index", qc["item"].get("di"))
            if not b and a:
                fixed += 1
                f_idx.append(idx)
            elif b and not a:
                broken += 1
                b_idx.append(idx)
            elif b:
                same_r += 1
            else:
                same_w += 1
        vs_baseline[arm] = {
            "baseline_accuracy": cells[BASELINE_ARM]["accuracy"],
            "arm_accuracy": cells[arm]["accuracy"],
            "delta": round(cells[arm]["accuracy"] - cells[BASELINE_ARM]["accuracy"], 4),
            "fixed": fixed, "broken": broken, "net": fixed - broken,
            "same_right": same_r, "same_wrong": same_w,
            "fixed_indices": f_idx, "broken_indices": b_idx,
        }

    # ---- 基线参考交叉核对（官方 items 文件的 74.3% vs 本进程重算）----
    baseline_ref: Optional[Dict[str, Any]] = None
    if DEFAULT_BASELINE_ITEMS.exists():
        try:
            ref_data = json.loads(DEFAULT_BASELINE_ITEMS.read_text(encoding="utf-8"))
            if isinstance(ref_data, dict) and isinstance(ref_data.get("items"), list):
                ref_data = ref_data["items"]
            ref_by_q = {int(r.get("dataset_index", r.get("di"))): r for r in ref_data}
            agree = tot = 0
            for qc in per_question:
                idx = qc["item"].get("dataset_index", qc["item"].get("di"))
                r = ref_by_q.get(int(idx))
                if r is None:
                    continue
                tot += 1
                if bool(r.get("is_correct")) == bool(qc["results"][BASELINE_ARM]["is_correct"]):
                    agree += 1
            ref_correct = sum(1 for r in ref_data if r.get("is_correct"))
            baseline_ref = {
                "file": str(DEFAULT_BASELINE_ITEMS),
                "n": len(ref_data),
                "accuracy": round(ref_correct / len(ref_data), 4) if ref_data else None,
                "agreement_with_inprocess_baseline": (
                    round(agree / tot, 4) if tot else None),
                "n_compared": tot,
            }
        except Exception as exc:
            baseline_ref = {"error": str(exc)}

    # ---- tiebreak 门控触发统计 ----
    trig_idx = [i for i, qc in enumerate(per_question)
                if qc["results"]["arm_orm_tiebreak"].get("orm_triggered")]
    changed_idx = [i for i in trig_idx
                   if per_question[i]["results"]["arm_orm_tiebreak"].get("orm_changed_winner")]
    tiebreak_analysis: Dict[str, Any] = {
        "triggered_questions": len(trig_idx),
        "changed_winner": len(changed_idx),
        "changed_improved": sum(
            1 for i in changed_idx
            if not base_correct[i] and per_question[i]["results"]["arm_orm_tiebreak"]["is_correct"]),
        "changed_regressed": sum(
            1 for i in changed_idx
            if base_correct[i] and not per_question[i]["results"]["arm_orm_tiebreak"]["is_correct"]),
        "baseline_on_triggered": _agg([per_question[i]["results"][BASELINE_ARM]
                                       for i in trig_idx]),
        "tiebreak_on_triggered": _agg([per_question[i]["results"]["arm_orm_tiebreak"]
                                       for i in trig_idx]),
        "per_arm_on_triggered": {
            arm: _agg([per_question[i]["results"][arm] for i in trig_idx])
            for arm in ARMS},
    }

    # ---- ORM 打分成本统计 ----
    st = dict(scorer.stats)
    n_groups_total = sum(len(qc["groups"]) for qc in per_question)
    n_trig_two = 0
    for qc in per_question:
        tie_rec = qc["results"]["arm_orm_tiebreak"]
        if tie_rec.get("orm_triggered") and tie_rec.get("top2_size") is not None:
            n_trig_two += 2

    scoring_cost: Dict[str, Any] = {
        "mode": st["mode"],
        "orm_checkpoint": str(args.orm_checkpoint),
        "base_model": args.base_model,
        "stub": is_stub,
        "n_unique_candidates_scored": len(payloads),
        "wall_seconds": st.get("wall_seconds"),
        "prompt_tokens_total": st.get("prompt_tokens"),
        "tokens_per_second": st.get("tokens_per_second"),
        "logprobs_topk": st.get("logprobs_topk"),
        "n_topk_covered": st.get("n_topk_covered"),
        "n_rescored_full_vocab": st.get("n_rescored_full_vocab"),
        "n_missing": st.get("n_missing"),
        "nominal_cost_per_arm": {
            "arm_orm_argmax": {"n_scores": len(payloads),
                               "note": "全部唯一候选逐一打分（成本最高，无结构先验）"},
            "arm_orm_grouphead": {"n_scores": n_groups_total,
                                  "note": "仅各组代表（组级 ORM，成本最低）"},
            "arm_orm_hybrid": {"n_scores": n_groups_total,
                               "note": "仅各组代表（连续打分，与 grouphead 同成本）"},
            "arm_orm_tiebreak": {"n_scores": n_trig_two,
                                 "note": "仅触发题的两组代表（≤2×触发题数）"},
        },
        "note": ("本实验 4 臂共享一次全候选打分（argmax 臂所需），grouphead/hybrid/"
                 "tiebreak 复用其子集；单独部署某臂只需其 nominal 成本。"
                 "时间估算：20596 候选 × ~560 token ≈ 11.5M prompt token，"
                 "vLLM batch 3B LoRA 预填 A40/3090 ≈ 5-15 min。"),
    }

    # ---- 输出 ----
    total_wall = sum(v.get("wall_seconds", 0.0) for v in engine._stats.values())
    summary = {
        "meta": {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "created_by": "src/orm_selection.py",
            "input_items": str(args.items),
            "output_dir": str(args.out_dir),
            "spider_dir": str(args.spider_dir),
            "orm_checkpoint": str(args.orm_checkpoint),
            "threads": args.threads,
            "query_timeout_seconds": args.query_timeout,
            "max_vm_steps": args.max_vm_steps,
            "row_cap": args.row_cap,
            "max_instances_cap": args.max_instances,
            "keep_distinct": args.keep_distinct,
            "seed": args.seed,
            "limit": args.limit,
            "stub_scores": args.stub_scores,
            "scoring_backend": "vLLM 0.11 batch（选快的；HF+peft batch 约 30-75 min 不采用）",
            "prompt_consistency": (
                "ORM prompt = 本文件 build_orm_prompt（逐字复制 label_orm_data.build_orm_prompt："
                "canonical 生成端 prompt + Candidate SQL Query 块 + Yes/No 指令）；"
                "chat template + add_generation_prompt，左截断 2048（train_orm.tokenize_score 同口径）"),
            "prompt_consistency_check": prompt_check,
            "remove_distinct_backend": "sqlparse" if sqlparse_ok else "regex_fallback",
            "join_counter_backend": "sqlglot" if sqlglot_ok else "regex",
            "env_note": (
                "真跑在 envs/vllmenv（无 sqlparse/sqlglot → 上述 fallback；与 adjudicate_soft"
                "（reasoning3b, sqlparse）在『字符串字面量含 distinct 一词』的极端样本上判定"
                "可能不同——官方复评以 scripts/eval_official.sh 的 test_suite_eval 为准）"),
            "semantics": (
                "grouping: bag semantics, row-sorted canonical, column order kept, no "
                "column permutation tolerance (same as adjudicate_pool); ORM score = "
                "P(Yes)=sigmoid(logit_Yes-logit_No); argmax tie -> min_sample_idx then "
                "str(key); grouphead = size*P(Yes) over rankable groups (empty/zero "
                "prescreened), tie -> size then str(key); hybrid = (size/max_group_size)*"
                "P(Yes) over ALL groups (no hard prescreen), tie -> size then str(key); "
                "tiebreak "
                "trigger = |top1-top2|<=1 or top1<2 (adjudicate_soft gated 定义), ORM "
                "二选一 P(Yes) 大者胜, ORM 平票保持 MI-VAV top1; judgment: official "
                "eval_exec_match (postprocess + remove_distinct + replace_cur_year + "
                "result_eq with order_matters and column permutation), all instances "
                "must match; NO_RESULTS falls back to same-pool arm_maj"),
        },
        "dataset_stats": dataset_stats,
        "dedup_stats": {
            "total_candidates": total_cands,
            "unique_after_dedup": unique_cands,
            "merged_duplicates": total_cands - unique_cands,
        },
        "execution_stats": {
            "grouping_phase": engine._stats.get("grouping", {}),
            "judgment_phase": engine._stats.get("judgment", {}),
            "total_wall_seconds": round(total_wall, 2),
        },
        "scoring_cost": scoring_cost,
        "accuracy": cells,
        "vs_baseline": vs_baseline,
        "baseline_reference": baseline_ref,
        "tiebreak_analysis": tiebreak_analysis,
    }

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for arm in ARMS:
        out_items = []
        for qc in per_question:
            item = qc["item"]
            rec = qc["results"][arm]
            predicted = rec["text"]
            if not predicted:
                predicted = "SELECT 1"  # AGENTS.md 铁律 4：空预测不跳过
            out_items.append({
                "dataset_index": item.get("dataset_index", item.get("di")),
                "di": item.get("di", item.get("dataset_index")),
                "db_id": item.get("db_id", ""),
                "question": item.get("question", ""),
                "gold_sql": item.get("gold_sql") or "",
                "predicted_sql": predicted,
                "empty_winner": rec.get("empty_winner", False),
                "winner_source": rec["source"],
                "orm_score": rec.get("orm_score"),
                "winner_votes": rec.get("votes", 0),
                "winner_group_size": rec.get("group_size", 0),
                "winner_group_key": rec.get("group_key"),
                "winner_models": rec.get("winner_models"),
                "winner_dual": rec.get("winner_dual"),
                "n_joins": rec.get("n_joins"),
                "join_counter": rec.get("join_counter"),
                "orm_triggered": rec.get("orm_triggered"),
                "orm_changed_winner": rec.get("orm_changed_winner"),
                "top1_size": rec.get("top1_size"),
                "top2_size": rec.get("top2_size"),
                "top1_orm_score": rec.get("top1_orm_score"),
                "top2_orm_score": rec.get("top2_orm_score"),
                "num_candidates": qc["num_candidates"],
                "num_unique_candidates": qc["num_unique_candidates"],
                "num_instances": qc["num_instances"],
                "instances_used": rec.get("instances_used", 0),
                "order_matters": rec.get("order_matters"),
                "is_correct": rec.get("is_correct", False),
                "gold_exec_error": rec.get("gold_exec_error", False),
                "vav_grouped_candidates": rec.get("vav_grouped", 0),
                "vav_excluded_candidates": rec.get("vav_excluded", 0),
            })
        (args.out_dir / f"items_{arm}.json").write_text(
            json.dumps(out_items, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 终端汇总 ----
    print("\n=== accuracy (correct / total) ===")
    for arm in ALL_ARMS:
        c = cells[arm]
        tag = "*" if arm == BASELINE_ARM else " "
        print(f"  {arm:22s} {c['correct']}/{c['total']} ({c['accuracy']:.4f}){tag}")
    print("\n=== vs MI-VAV baseline (fixed / broken / net) ===")
    for arm in ARMS:
        v = vs_baseline[arm]
        print(f"  {arm:22s} fixed={v['fixed']} broken={v['broken']} net={v['net']:+d} "
              f"delta={v['delta']:+.4f}")
    print(f"\n=== tiebreak: triggered={tiebreak_analysis['triggered_questions']} "
          f"changed={tiebreak_analysis['changed_winner']} "
          f"(improved={tiebreak_analysis['changed_improved']} "
          f"regressed={tiebreak_analysis['changed_regressed']}) ===")
    print(f"=== scoring: mode={scoring_cost['mode']} n={len(payloads)} "
          f"tokens={scoring_cost['prompt_tokens_total']} "
          f"wall={scoring_cost['wall_seconds']}s ===")
    print(f"\nsummary -> {args.out_dir / 'summary.json'}")
    print(f"items   -> {args.out_dir / 'items_<arm>.json'} ({len(ARMS)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
