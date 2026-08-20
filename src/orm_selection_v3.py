#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/orm_selection_v3.py — P1 推理侧两张牌：M 次采样平均（P1-1）+ CAPS 级联判卷（P1-2）。

输入（HPC，复用 v2 全部池与缓存）
  outputs/eval_pool_multi/items.json（1034 题 × 64 候选；官方 ORM grouphead 基线 86.1%）
  outputs/eval_pool_unseen/items.json（1034 题 × 32 候选；官方 ORM grouphead 基线 83.6%）
  outputs/orm_selection_multi|unseen/items_arm_orm_grouphead.json（官方基线对比源）
  outputs/orm_selection_v2(_unseen)/scores/scores_vllm.json（全 SQL 打分缓存，
    T=1.0，Stage-B 与全量对照臂直接复用——与 grouphead 基线同一打分器同一协议）

臂（同一批候选上离线重算；判定 = 官方语义 exec match，全实例等价；空胜者写
SELECT 1；分组 = adjudicate_soft.build_groups 的 MI-VAV 分组，打分对象 = 各组
代表（rankable 域，与 grouphead 完全一致），组级分 = size × 代表分）：

  arm_vav_multi_all        基线 1 = MI-VAV（对照官方 74.3/74.8 线，零 ORM）
  arm_p1_m8                P1-1 主臂：MARS 式 M=8 次采样平均。VllmScorer 以
                           temperature=0.2、n=8 采样组代表的首 token，P̄(Yes) =
                           #{采样为 Yes}/M（蒙特卡洛平均；E[P̄]=sigmoid(Δ/0.2)，
                           二项噪声 std≤√(p(1-p)/8)）。组级分 = size×P̄。
  arm_p1_m8_soft           P1-1 诊断臂：逐样本 logprob 的 P(Yes) 平均。vLLM 返回
                           的 logprob 是 T 缩放后的（同 prompt 同 logits → 逐样本
                           确定性），故 ≡ sigmoid(Δ/0.2) = T=0.2 温度锐化（与
                           P0-4 arm_p04_t05 同构，M 不敏感）。用于把"采样效应"
                           与"温度锐化效应"分开（二者期望相同）。
  arm_caps_k3/k5/k10       P1-2 级联臂：Stage-A 用部分证据（外层 SELECT 投影 +
                           FROM/JOIN + WHERE，去掉 ORDER BY/LIMIT/GROUP BY 等
                           尾部；sqlglot 解析重组，缺失/失败退化正则截尾）对全部
                           组代表打分（T=1.0 同协议），按 (size×P_partial, size,
                           str(key)) 保留 top-K 组；Stage-B 对保留组代表用完整 SQL
                           打分（复用 v2 scores_vllm.json 缓存），组级分 =
                           size×P_full(Yes)，选最高。
  arm_caps_full            对照臂 = 全部组全文打分（= grouphead 复现臂，86.1/83.6
                           同语义，从 v2 缓存重算，用于逐题核对）。

必报统计（summary.analysis）
  ① Stage-A/Stage-B 一致性：P_partial vs P_full 的 Pearson r / Spearman ρ（逐
    代表）、组级 (size×P) 版本、top-1 组一致率、recall@K（全量臂胜者落在 Stage-A
    top-K 内的比例，K=3/5/10；只统计 ranked 组数>K 的题，防平凡覆盖）。
  ② 级联 vs 全量：逐 K 的 EX 差（in-process）+ 判卷 prompt token 预算
    （全量 = Σ tokens_full；级联 = Σ tokens_partial + Σ_topK tokens_full）+
    节省比例 + 打分调用数。
  ③ K 敏感度：K=3/5/10 三档如实报（不选优）。

打分后端
  --dump-payloads（CPU，envs/reasoning3b，sqlglot 30.x）：分组 → 组代表 → 部分
    证据化 → prompt 文本 + token 计数落盘 scores/partial_payloads.json。
  --score-only（GPU，envs/vllmenv，gpudebug 3090）：读 payload 文件，① 部分证据
    prompt 用 orm_selection.VllmScorer 原样打分（T=1.0，logprobs=20）→
    scores/scores_partial_vllm.json；② M8VllmScorer（T=0.2，n=8）打全 SQL prompt
    → scores/scores_m8_vllm.json。vllmenv 无 sqlglot → 部分化文本来自 payload
    文件（sqlglot 产物），GPU 侧不做解析。
  --skip-scoring（CPU）：读三份缓存（partial/m8/本进程 out-dir）+ v2 全量缓存
    （--full-scores 指定）→ 各臂裁决 + 判定 + items/summary。缺分 → CPU HF 回填
    （CpuOrmScorer 复用 v2；m8 缺分用 Bernoulli 模拟，见 cpu_m8_fallback）。
  --stub-scores random|constant：冒烟（登录节点，零 torch/vLLM）。

输出 outputs/orm_selection_v3/（多源池）与 outputs/orm_selection_v3_unseen/：
  items_<arm>.json   与 scripts/eval_official.sh 兼容
  summary.json       各臂 in-process 官方语义准确率 + vs MI-VAV / vs 官方
                     grouphead 基线（86.1/83.6）/ vs arm_caps_full 的 fixed/broken
                     + ①一致性 ②token 预算 ③K 敏感度 + 风险点
  scores/*.json      本牌打分缓存（跨阶段复用）

用法（HPC）
  # 步骤 1（CPU dump，~20 min/池；本文件内嵌在 orm_v3_score.slurm 亦可）：
  envs/reasoning3b/bin/python src/orm_selection_v3.py --items outputs/eval_pool_multi/items.json \
      --out-dir outputs/orm_selection_v3 --dump-payloads --threads 16
  # 步骤 2（GPU 打分，gpudebug 3090，~15-25 min/池）：
  envs/vllmenv/bin/python src/orm_selection_v3.py ... --score-only
  # 步骤 3（CPU 重算 + 输出，~30-40 min/池）：
  envs/reasoning3b/bin/python src/orm_selection_v3.py ... --skip-scoring --threads 16
  # 冒烟（登录节点 CPU，随机分数替代 ORM，~1-2 min）：
  envs/reasoning3b/bin/python src/orm_selection_v3.py --items outputs/eval_pool_multi/items.json \
      --out-dir outputs/orm_selection_v3_smoke --stub-scores random --limit 30
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
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
import orm_selection_v2 as V2  # noqa: E402  CpuOrmScorer（缺分回填）+ 同源常量

DEFAULT_ITEMS_MULTI = PROJECT_ROOT / "outputs" / "eval_pool_multi" / "items.json"
DEFAULT_OUT_MULTI = PROJECT_ROOT / "outputs" / "orm_selection_v3"
DEFAULT_BASELINE_MULTI = (
    PROJECT_ROOT / "outputs" / "orm_selection_multi" / "items_arm_orm_grouphead.json")
DEFAULT_FULL_SCORES_MULTI = (
    PROJECT_ROOT / "outputs" / "orm_selection_v2" / "scores" / "scores_vllm.json")
DEFAULT_SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"
DEFAULT_ORM_CKPT = PROJECT_ROOT / "checkpoints" / "orm_b1"
DEFAULT_MERGE_PYTHON = PROJECT_ROOT / "envs" / "reasoning3b" / "bin" / "python"

# ---- 臂注册表 ----
BASELINE_ARMS = ["arm_vav_multi_all"]
M8_ARMS = ["arm_p1_m8", "arm_p1_m8_soft"]
CAPS_ARMS = ["arm_caps_k3", "arm_caps_k5", "arm_caps_k10", "arm_caps_full"]
GROUP_NAMES = {
    "baseline": BASELINE_ARMS, "m8": M8_ARMS, "caps": CAPS_ARMS,
}
ALL_ARMS = BASELINE_ARMS + M8_ARMS + CAPS_ARMS

DEFAULT_M8_SAMPLES = 8
DEFAULT_M8_TEMP = 0.2
DEFAULT_CAPS_KS = (3, 5, 10)
YES_STR, NO_STR = "Yes", "No"

# ---- 部分证据化（sqlglot 优先，正则退化）----
try:
    import sqlglot  # type: ignore
    from sqlglot import exp  # type: ignore

    _HAS_SQLGLOT = True
except ImportError:  # 退化：正则（记录于 summary）
    sqlglot = None
    exp = None
    _HAS_SQLGLOT = False

_PARTIAL_TAIL_RE = re.compile(
    r"\b(GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET)\b", re.IGNORECASE)


def build_orm_prompt(question: str, ddl_schema: str, candidate_sql: str) -> str:
    """ORM user 侧输入——【逐字复制】orm_selection_v2.build_orm_prompt
    （= orm_selection.build_orm_prompt = label_orm_data.build_orm_prompt；
    train/infer prompt 逐字节一致硬要求）。"""
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
# 部分证据化：关键子句（投影 + FROM/JOIN + WHERE），去尾部
# ===================================================================


def _partialize_sqlglot(text: str) -> Tuple[str, str]:
    """sqlglot 解析后重组：只保留外层 SELECT 的 expressions/from/joins/where
    （+ distinct 标记）；去掉 group/having/order/limit/offset/qualify 等尾部。
    非纯 SELECT（UNION/CTE/子查询外壳等）不部分化 → backend='none'。"""
    tree = sqlglot.parse_one(text, read="sqlite")
    if tree is None:
        raise ValueError("sqlglot parse returned None")
    if not isinstance(tree, exp.Select):
        return text, "none"
    tail_keys = ("group", "having", "order", "limit", "offset", "qualify")
    has_tail = any(k in tree.args for k in tail_keys)
    new = exp.Select(
        expressions=tree.args.get("expressions") or [],
        from_=tree.args.get("from"),
        joins=tree.args.get("joins"),
        where=tree.args.get("where"),
        distinct=tree.args.get("distinct"),
    )
    partial = (new.sql(dialect="sqlite") or "").strip()
    if not partial:
        return text, "none"
    if not has_tail:
        return text, "none"  # 无尾部可去 → 部分化无效果，按 none 计
    return partial, "sqlglot"


def _partialize_regex(text: str) -> Tuple[str, str]:
    """正则截尾兜底（sqlglot 缺失/失败时）：扫描括号深度 0（忽略字符串字面量）
    的最外层首个 GROUP BY/HAVING/ORDER BY/LIMIT/OFFSET 关键字并截断。
    已知局限（summary 记录）：UNION 各分支尾部不逐分支截取；深度 0 括号外仅截
    最外层；关键字出现在反引号标识符内会被误截（Spider 少见）。"""
    i, n, depth = 0, len(text), 0
    in_s: Optional[str] = None
    cut = -1
    while i < n:
        c = text[i]
        if in_s:
            if c == in_s:
                if i + 1 < n and text[i + 1] == in_s:  # '' 转义
                    i += 2
                    continue
                in_s = None
            i += 1
            continue
        if c in "'\"":
            in_s = c
            i += 1
            continue
        if c == "(":
            depth += 1
            i += 1
            continue
        if c == ")":
            depth = max(depth - 1, 0)
            i += 1
            continue
        if depth == 0:
            m = _PARTIAL_TAIL_RE.match(text, i)
            if m:
                cut = i
                break
        i += 1
    if cut < 0:
        return text, "none"
    partial = text[:cut].strip().rstrip(";").strip()
    if not partial:
        return text, "none"
    return partial, "regex"


def partialize_sql(sql: str) -> Tuple[str, str]:
    """部分证据化入口。返回 (partial_sql, backend)，backend ∈ {sqlglot, regex,
    none}；none = 无法部分化（UNION/CTE/非 SELECT/解析失败且正则无尾部），
    partial_sql == 原 SQL（Stage-A 按全文判，统计单列，不参与一致性统计）。"""
    text = (sql or "").strip()
    if not text:
        return text, "none"
    if _HAS_SQLGLOT:
        try:
            return _partialize_sqlglot(text)
        except Exception:
            pass
    return _partialize_regex(text)


# ===================================================================
# M8 打分器（vLLM，T 缩放 + n=M 采样；复用 orm_selection.VllmScorer 的服务层）
# ===================================================================


class M8VllmScorer:
    """MARS 式 M 次采样打分：max_tokens=1、temperature=T、n=M（一次请求 M 条
    独立采样，seed=0）、logprobs=20。
    逐样本输出：
      p_i = sigmoid(logp_yes − logp_no)（logprob 差 ≡ T 缩放 logit 差；同一
      prompt 的 logits 确定 → p_i 逐样本相同，其均值 ≡ sigmoid(Δ/T) 温度锐化）
      token_i = 采样到的首 token（Yes/No/Other）
    汇总：
      vote_rate = #{token_i == Yes}/M —— 蒙特卡洛平均（E = sigmoid(Δ/T)，二项
      噪声 std ≤ √(p(1-p)/M)）；此为 arm_p1_m8 主分数。
      soft_mean = mean(p_i) —— 确定性 T 锐化参考；此为 arm_p1_m8_soft 分数。
    LoRA 服务与兜底逻辑（top-k 缺失 → 全词表重打、engine 复用）与
    orm_selection.VllmScorer 同一实现（组合复用其 _serve/_warmup）。"""

    def __init__(self, args: argparse.Namespace, m: int, temp: float) -> None:
        import orm_selection as OM  # noqa: E402  延迟 import（真跑才需要 GPU 依赖）
        self._OM = OM
        self.args = args
        self.m = int(m)
        self.temp = float(temp)
        self._base = OM.VllmScorer(args)  # tokenizer + yes/no id + LoRA/merge 服务层
        self._stats: Dict[str, Any] = {
            "mode": "vllm-m8", "m": self.m, "temperature": self.temp,
            "n_prompts": 0, "wall_seconds": 0.0, "prompt_tokens": 0,
            "logprobs_topk": args.logprobs_topk,
            "n_rescored_full_vocab": 0, "n_samples_total": 0,
            "vote_rate_mean": None,
        }

    def _extract_sample(self, s: Any, yes_id: int, no_id: int) -> Dict[str, Any]:
        lp0 = s.logprobs[0] if s.logprobs else {}
        p = self._OM._extract_p_yes(lp0, yes_id, no_id)
        tid = getattr(s, "token_ids", None)
        if tid:
            token = "Yes" if tid[0] == yes_id else ("No" if tid[0] == no_id else "Other")
        else:  # token_ids 缺失 → argmax logprob 兜底
            top = max(lp0.items(), key=lambda kv: kv[1].logprob)[0]
            token = "Yes" if top == yes_id else ("No" if top == no_id else "Other")
        return {"token": token, "p": p}

    def score(self, payloads: List[Tuple[int, str, Optional[str]]]) -> List[Dict[str, Any]]:
        """payloads: [(qi, key, prompt)]。返回对齐的逐 payload 汇总 dict。"""
        args = self.args
        tok = self._base.tokenizer
        ids_list: List[List[int]] = []
        n_tokens = 0
        for (_qi, _key, prompt) in payloads:
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, return_dict=True)
            ids = enc["input_ids"][-args.max_length:]
            ids_list.append(ids)
            n_tokens += len(ids)

        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=1, temperature=self.temp, top_p=1.0, seed=0,
                            n=self.m, logprobs=args.logprobs_topk, detokenize=False)
        sp_full = SamplingParams(max_tokens=1, temperature=self.temp, top_p=1.0,
                                 seed=0, n=self.m, logprobs=200000, detokenize=False)

        llm, lr = self._base._serve(ids_list[0])
        out: List[Optional[Dict[str, Any]]] = [None] * len(payloads)
        t0 = time.perf_counter()

        def _run(chunk_ids: List[List[int]], sp_i: Any) -> List[Any]:
            reqs = [{"prompt_token_ids": ids} for ids in chunk_ids]
            return llm.generate(reqs, sp_i, lora_request=lr)

        for i in range(0, len(payloads), args.chunk_size):
            chunk = ids_list[i:i + args.chunk_size]
            outputs = _run(chunk, sp)
            by_prompt = {tuple(o.prompt_token_ids): o for o in outputs}
            for j, ids in enumerate(chunk):
                o = by_prompt.get(tuple(ids))
                if o is None or not o.outputs:
                    raise RuntimeError(f"[orm-v3-m8] 无输出: payload {i + j}")
                samples = o.outputs
                need_full = (len(samples) < self.m or
                             any(not s.logprobs for s in samples) or
                             any(self._OM._extract_p_yes(
                                 s.logprobs[0], self._base.yes_id,
                                 self._base.no_id) is None for s in samples))
                if need_full:  # 兜底：全词表 logprobs 重打（应为极少数/零）
                    o2 = _run([ids], sp_full)[0]
                    samples = o2.outputs
                    self._stats["n_rescored_full_vocab"] += 1
                if len(samples) < self.m:
                    raise RuntimeError(
                        f"[orm-v3-m8] 采样数不足 {len(samples)}/{self.m}: payload {i + j}")
                per = [self._extract_sample(s, self._base.yes_id, self._base.no_id)
                       for s in samples]
                votes = sum(1 for d in per if d["token"] == "Yes")
                ps = [d["p"] for d in per if d["p"] is not None]
                soft = (sum(ps) / len(ps)) if ps else None
                if soft is None:
                    raise RuntimeError(f"[orm-v3-m8] 全词表仍缺 Yes/No: payload {i + j}")
                out[i + j] = {
                    "m": self.m, "votes": votes, "vote_rate": round(votes / self.m, 6),
                    "soft_mean": round(soft, 8), "per_sample": per,
                }
                self._stats["n_samples_total"] += self.m
            if (i // args.chunk_size + 1) % 10 == 0:
                print(f"[orm-v3-m8] scored {min(i + args.chunk_size, len(payloads))}/"
                      f"{len(payloads)} ({time.perf_counter() - t0:.0f}s)",
                      file=sys.stderr)

        wall = time.perf_counter() - t0
        rates = [d["vote_rate"] for d in out if d is not None]
        self._stats.update({
            "n_prompts": len(payloads),
            "wall_seconds": round(wall, 2),
            "prompt_tokens": n_tokens,
            "vote_rate_mean": round(sum(rates) / len(rates), 6) if rates else None,
        })
        try:
            del llm
        except Exception:
            pass
        if any(d is None for d in out):
            raise RuntimeError("[orm-v3-m8] 存在未打分候选（内部错误）")
        return [dict(d) for d in out]  # type: ignore[misc]

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats


# ===================================================================
# 缺分回填（CPU；非冒烟路径，理论只用于 v2 缓存 1 条漂移类场景）
# ===================================================================


def cpu_m8_fallback(cpu_scorer: Any, payloads: List[Tuple[int, str, str]],
                    m: int, temp: float, seed: int) -> List[Dict[str, Any]]:
    """m8 缺分的 CPU 回填：HF 前向取末位 logits，soft_mean = sigmoid(Δ/T)；
    vote_rate = Bernoulli(P) 模拟 M 次采样（与 vLLM token 采样的 Yes 事件同分布；
    T=0.2 下非 Yes/No token 质量可忽略——差异记录于 summary.risks）。"""
    torch = cpu_scorer._torch
    tok = cpu_scorer.tokenizer
    model = cpu_scorer.model
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for _qi, _key, prompt in payloads:
        if not prompt:
            out.append({"m": m, "votes": None, "vote_rate": 0.5, "soft_mean": 0.5,
                        "per_sample": None,
                        "note": "cpu-fallback-no-prompt-default-0.5"})
            continue
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_dict=True)
        ids = enc["input_ids"][-cpu_scorer.max_length:]
        inp = torch.tensor([ids])
        msk = torch.ones(1, len(ids), dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids=inp, attention_mask=msk).logits
        lp_yes = float(logits[0, -1, cpu_scorer.yes_id])
        lp_no = float(logits[0, -1, cpu_scorer.no_id])
        p = 1.0 / (1.0 + math.exp(-(lp_yes - lp_no) / temp))
        votes = sum(1 for _ in range(m) if rng.random() < p)
        out.append({"m": m, "votes": votes, "vote_rate": round(votes / m, 6),
                    "soft_mean": round(p, 8), "per_sample": None,
                    "note": "cpu-fallback-bernoulli"})
    return out


# ===================================================================
# 单题裁决
# ===================================================================


def _rep_ei(entries: List[Dict[str, Any]], g: Dict[str, Any]) -> int:
    rep = AS._group_rep(entries, g)
    return next(i for i, e in enumerate(entries) if e["key"] == rep["key"])


def _fallback(entries: List[Dict[str, Any]], votes: Dict[int, int],
              n_used: int, grouped: int, excluded: int) -> Dict[str, Any]:
    return AS._fallback_record(entries, votes, n_used, grouped, excluded)


def _finish(rec: Dict[str, Any]) -> Dict[str, Any]:
    rec["empty_winner"] = (rec["text"] == "")
    return rec


def _size_x_score_key(kg: Tuple[Tuple[str, ...], Dict[str, Any]],
                      entries: List[Dict[str, Any]],
                      s_of: Any) -> Tuple[float, int, str]:
    """(size×score, size, str(key)) —— grouphead 同款排序键；缺分按 0 计。"""
    g = kg[1]
    ei = _rep_ei(entries, g)
    s = s_of(ei)
    return (g["size"] * (s if s is not None else 0.0), g["size"], str(kg[0]))


def arm_m8(entries: List[Dict[str, Any]],
           ranked: List[Tuple[Tuple[str, ...], Dict[str, Any]]],
           votes: Dict[int, int], n_used: int, grouped: int, excluded: int,
           joins_cache: Dict[str, Tuple[int, str]],
           m8_by_ei: Dict[int, Dict[str, Any]], use_soft: bool) -> Dict[str, Any]:
    """P1-1：size × P̄(Yes)，P̄ = vote_rate（主）或 soft_mean（诊断）。"""
    if not ranked:
        return _fallback(entries, votes, n_used, grouped, excluded)
    field = "soft_mean" if use_soft else "vote_rate"
    chosen_key, chosen_g = max(
        ranked, key=lambda kg: _size_x_score_key(kg, entries, lambda ei: (
            m8_by_ei.get(ei) or {}).get(field)))
    src = "orm_m8_soft" if use_soft else "orm_m8"
    rec = AS._base_record(entries, chosen_key, chosen_g, src,
                          n_used, grouped, excluded, joins_cache)
    ei = _rep_ei(entries, chosen_g)
    d = m8_by_ei.get(ei) or {}
    rec["orm_score"] = d.get(field)
    rec["m8_vote_rate"] = d.get("vote_rate")
    rec["m8_soft_mean"] = d.get("soft_mean")
    rec["m8_n_samples"] = d.get("m")
    return rec


def arm_caps(entries: List[Dict[str, Any]],
             ranked: List[Tuple[Tuple[str, ...], Dict[str, Any]]],
             votes: Dict[int, int], n_used: int, grouped: int, excluded: int,
             joins_cache: Dict[str, Tuple[int, str]],
             partial_by_ei: Dict[int, float], full_by_ei: Dict[int, float],
             k: int) -> Dict[str, Any]:
    """P1-2 级联：Stage-A 部分证据 (size×P_partial) 保 top-K 组 → Stage-B 全文
    (size×P_full) 取最高。返回 (rec, topk_keys)。"""
    if not ranked:
        return _fallback(entries, votes, n_used, grouped, excluded)
    a_keyed = sorted(ranked, key=lambda kg: _size_x_score_key(
        kg, entries, lambda ei: partial_by_ei.get(ei)))
    top_k = a_keyed[-k:]
    chosen_key, chosen_g = max(
        top_k, key=lambda kg: _size_x_score_key(kg, entries, lambda ei: full_by_ei.get(ei)))
    rec = AS._base_record(entries, chosen_key, chosen_g, f"caps_k{k}",
                          n_used, grouped, excluded, joins_cache)
    ei = _rep_ei(entries, chosen_g)
    rec["orm_score"] = full_by_ei.get(ei)
    rec["partial_score"] = partial_by_ei.get(ei)
    rec["caps_stage_b_n"] = len(top_k)
    rec["caps_winner_rank_a"] = len(ranked) - a_keyed.index(
        (chosen_key, chosen_g))  # 1-based 降序位次（第几好）
    return rec


def arm_caps_full(entries: List[Dict[str, Any]],
                  ranked: List[Tuple[Tuple[str, ...], Dict[str, Any]]],
                  votes: Dict[int, int], n_used: int, grouped: int, excluded: int,
                  joins_cache: Dict[str, Tuple[int, str]],
                  full_by_ei: Dict[int, float]) -> Dict[str, Any]:
    """对照臂 = 全部组全文打分 = grouphead 复现（语义同 orm_selection arm2 /
    v2 arm_orm_grouphead，分数来自 v2 缓存）。"""
    if not ranked:
        return _fallback(entries, votes, n_used, grouped, excluded)
    chosen_key, chosen_g = max(
        ranked, key=lambda kg: _size_x_score_key(kg, entries, lambda ei: full_by_ei.get(ei)))
    rec = AS._base_record(entries, chosen_key, chosen_g, "caps_full",
                          n_used, grouped, excluded, joins_cache)
    ei = _rep_ei(entries, chosen_g)
    rec["orm_score"] = full_by_ei.get(ei)
    return rec


# ===================================================================
# 统计辅助
# ===================================================================


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None

    def ranks(v: List[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r

    return _pearson(ranks(xs), ranks(ys))


def _agg(recs: List[Dict[str, Any]], key: str = "is_correct") -> Dict[str, Any]:
    n = len(recs)
    c = sum(1 for r in recs if r.get(key))
    return {"n": n, "correct": c, "accuracy": round(c / n, 4) if n else None}


# ===================================================================
# 参数
# ===================================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="orm_selection_v3：P1-1 M 次采样平均 + P1-2 CAPS 级联判卷"
                    "（离线重算 + 官方语义判定）")
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS_MULTI)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_MULTI)
    ap.add_argument("--spider-dir", type=Path, default=DEFAULT_SPIDER_DIR)
    ap.add_argument("--baseline-items", type=Path, default=DEFAULT_BASELINE_MULTI,
                    help="官方 ORM grouphead 基线 items（86.1/83.6 来源）")
    ap.add_argument("--full-scores", type=Path, default=None,
                    help="v2 全 SQL 打分缓存（T=1.0）；缺省按 --items 池推断："
                         "eval_pool_multi → orm_selection_v2，eval_pool_unseen → "
                         "orm_selection_v2_unseen")
    ap.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    ap.add_argument("--orm-checkpoint", default=str(DEFAULT_ORM_CKPT))
    ap.add_argument("--merge-python", default=str(DEFAULT_MERGE_PYTHON),
                    help="peft merge 子进程解释器（vllmenv 无 peft，用 reasoning3b）")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--logprobs-topk", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="m8 每请求 n=M 条采样，chunk 减半防显存/时延尖峰")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--query-timeout", type=float, default=30.0)
    ap.add_argument("--max-vm-steps", type=int, default=5_000_000)
    ap.add_argument("--row-cap", type=int, default=100_000)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--keep-distinct", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 题（冒烟）")
    ap.add_argument("--arms", default="all",
                    help="逗号分隔：baseline,m8,caps 或 all")
    ap.add_argument("--m8-samples", type=int, default=DEFAULT_M8_SAMPLES)
    ap.add_argument("--m8-temp", type=float, default=DEFAULT_M8_TEMP)
    ap.add_argument("--caps-k", default=",".join(map(str, DEFAULT_CAPS_KS)),
                    help="级联 top-K 三档（如 3,5,10）")
    ap.add_argument("--stub-scores", choices=["off", "random", "constant"],
                    default="off")
    ap.add_argument("--stub-const", type=float, default=0.5)
    ap.add_argument("--dump-payloads", action="store_true",
                    help="只做分组 + 部分证据化 + prompt/token 落盘后退出（CPU）")
    ap.add_argument("--score-only", action="store_true",
                    help="只读 payload 文件做 GPU 打分（partial T=1 + m8 T×M）落盘后退出")
    ap.add_argument("--skip-scoring", action="store_true",
                    help="不打分，只用已有缓存；缺分走 --no-cpu-fallback 关闭回填")
    ap.add_argument("--cpu-fallback", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="缺分候选用 CPU HF 回填（partial/full 用 v2 CpuOrmScorer；"
                         "m8 用 Bernoulli 模拟）；--no-cpu-fallback 关闭（缺分记 0 并警告）")
    ap.add_argument("--cpu-threads", type=int, default=32)
    ap.add_argument("--score-batch", type=int, default=8)
    return ap.parse_args(argv)


def _resolve_arms(spec: str) -> List[str]:
    if spec.strip().lower() == "all":
        return list(ALL_ARMS)
    out: List[str] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in GROUP_NAMES:
            out.extend(GROUP_NAMES[tok])
        elif tok in ALL_ARMS:
            out.append(tok)
        else:
            raise ValueError(f"未知臂/组: {tok}")
    seen: List[str] = []
    for a in out:
        if a not in seen:
            seen.append(a)
    return seen


# ===================================================================
# 主流程
# ===================================================================


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    AP.rng = random.Random(args.seed)
    arms = _resolve_arms(args.arms)
    caps_ks = [int(x) for x in args.caps_k.split(",") if x.strip()]
    is_stub = args.stub_scores != "off"

    if args.full_scores is None:
        if "eval_pool_unseen" in str(args.items):
            args.full_scores = Path(
                str(PROJECT_ROOT / "outputs" / "orm_selection_v2_unseen" /
                    "scores" / "scores_vllm.json"))
        else:
            args.full_scores = Path(str(DEFAULT_FULL_SCORES_MULTI))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = args.out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    payload_file = scores_dir / "partial_payloads.json"

    items = AP._load_items(args.items)
    if args.limit:
        items = items[: args.limit]
    print(f"[orm-v3] {len(items)} 题 | arms={arms} | stub={args.stub_scores} | "
          f"out={args.out_dir} | full-scores={args.full_scores}", file=sys.stderr)

    # ---- 每题去重 ----
    entries_by_q: List[List[Dict[str, Any]]] = []
    for item in items:
        entries_by_q.append(AP._dedupe(item.get("candidates") or []))

    # ---- 实例枚举 + Phase 1：全部唯一候选原始 SQL × 实例（分组签名）----
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
    print(f"[orm-v3] phase1: {len(phase1_tasks)} 个唯一 (sql, db_path) 任务",
          file=sys.stderr)
    engine.run(phase1_tasks, phase="grouping")
    print(f"[orm-v3] phase1 完成: {engine._stats['grouping']}", file=sys.stderr)

    # ---- 每题：签名向量 + 分组 + 基线 + 打分对象（rankable 组代表）----
    per_question: List[Dict[str, Any]] = []
    rep_set: Dict[Tuple[int, str], int] = {}  # (qi,key) -> ei（rankable 组代表）
    for qi, item in enumerate(items):
        entries = entries_by_q[qi]
        insts = instances_for(item.get("db_id", ""))
        sigs: List[List[str]] = []
        for e in entries:
            if not (e["sql_text"] or "").strip():
                sigs.append([V2.ERROR_SIG] * len(insts))
            else:
                sigs.append([AP.outcome_signature(engine.get(e["sql_text"], inst))
                             for inst in insts])
        votes: Dict[int, int] = defaultdict(int)
        for c in item.get("candidates") or []:
            ck = AP.normalize_for_dedup(c.get("sql"))
            for ei, e in enumerate(entries):
                if ck == e["key"]:
                    votes[ei] += 1
                    break
        joins_cache: Dict[str, Tuple[int, str]] = {}
        groups, grouped, excluded = AS.build_groups(entries, sigs, votes, len(insts))
        results: Dict[str, Dict[str, Any]] = {}
        results["arm_vav_multi_all"] = AS.arm_baseline(
            entries, sigs, votes, insts, joins_cache)
        ranked = AS.rankable_groups(groups)
        for _k, g in ranked:
            rep = AS._group_rep(entries, g)
            rep_set[(qi, rep["key"])] = next(
                i for i, e in enumerate(entries) if e["key"] == rep["key"])
        per_question.append({
            "item": item, "entries": entries, "sigs": sigs, "votes": votes,
            "groups": groups, "ranked": ranked, "grouped": grouped,
            "excluded": excluded, "results": results,
            "num_candidates": len(item.get("candidates") or []),
            "num_unique_candidates": len(entries),
            "num_instances": len(insts),
        })
    n_reps = len(rep_set)
    print(f"[orm-v3] 打分对象（rankable 组代表）: {n_reps}", file=sys.stderr)

    # =================================================================
    # 路径 A：--dump-payloads（CPU，sqlglot 部分化 + prompt/token 落盘）
    # =================================================================
    if args.dump_payloads:
        from spider_utils import SpiderLoader  # noqa: E402
        from transformers import AutoTokenizer  # noqa: E402

        loader = SpiderLoader(str(args.spider_dir))
        ddl_cache: Dict[str, str] = {}
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, local_files_only=True, trust_remote_code=True)
        payloads: List[Dict[str, Any]] = []
        backend_cnt: Dict[str, int] = Counter()
        for (qi, key), ei in sorted(rep_set.items()):
            qc = per_question[qi]
            item = qc["item"]
            db_id = item.get("db_id", "")
            if db_id not in ddl_cache:
                ddl_cache[db_id] = loader.format_ddl(db_id)
            ddl = ddl_cache[db_id]
            question = item.get("question", "")
            sql = qc["entries"][ei]["sql_text"]
            partial_sql, backend = partialize_sql(sql)
            backend_cnt[backend] += 1
            prompt_full = build_orm_prompt(question, ddl, sql)
            prompt_partial = build_orm_prompt(question, ddl, partial_sql)

            def tok_len(prompt: str) -> Tuple[int, bool]:
                enc = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True, return_dict=True)
                raw = len(enc["input_ids"])
                return (min(raw, args.max_length), raw > args.max_length)

            tokens_full, trunc_full = tok_len(prompt_full)
            tokens_partial, trunc_partial = tok_len(prompt_partial)
            payloads.append({
                "qi": qi, "key": key, "sql": sql,
                "partial_sql": partial_sql, "partial_backend": backend,
                "prompt_full": prompt_full, "prompt_partial": prompt_partial,
                "tokens_full": tokens_full, "tokens_partial": tokens_partial,
                "truncated_full": trunc_full, "truncated_partial": trunc_partial,
            })
        meta = {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "items": str(args.items), "n_questions": len(items),
            "n_reps": n_reps, "partial_backend_counts": dict(backend_cnt),
            "sqlglot_available": _HAS_SQLGLOT,
            "m": args.m8_samples, "temp": args.m8_temp,
            "max_length": args.max_length,
        }
        payload_file.write_text(
            json.dumps({"meta": meta, "payloads": payloads},
                       ensure_ascii=False), encoding="utf-8")
        print(f"[orm-v3] payload 落盘: {payload_file} "
              f"(backend={dict(backend_cnt)})", file=sys.stderr)
        return 0

    # =================================================================
    # 路径 B：--score-only（GPU：partial T=1.0 + m8 T×M）
    # =================================================================
    if args.score_only:
        if not payload_file.exists():
            raise SystemExit(
                f"[orm-v3] 缺 payload 文件 {payload_file}——先跑 --dump-payloads")
        import orm_selection as OM  # noqa: E402
        from types import SimpleNamespace  # noqa: E402

        data = json.loads(payload_file.read_text(encoding="utf-8"))
        pl = data["payloads"]
        print(f"[orm-v3] score-only: {len(pl)} payloads from {payload_file}",
              file=sys.stderr)
        om_args = SimpleNamespace(
            base_model=args.base_model,
            orm_checkpoint=args.orm_checkpoint,
            max_length=args.max_length,
            logprobs_topk=args.logprobs_topk,
            chunk_size=args.chunk_size,
            enforce_eager=False,
            max_num_seqs=None,
            merge_python=args.merge_python,
        )

        # ---- Stage-A：部分证据 T=1.0（VllmScorer 原样）----
        partial_payloads = [(p["qi"], p["key"], p["prompt_partial"]) for p in pl]
        scorer = OM.VllmScorer(om_args)
        partial_scores = scorer.score(partial_payloads)
        partial_cache: Dict[str, float] = {}
        for (qi, key, _pr), s in zip(partial_payloads, partial_scores):
            partial_cache[f"{qi}\t{key}"] = float(s)
        (scores_dir / "scores_partial_vllm.json").write_text(
            json.dumps(partial_cache, ensure_ascii=False), encoding="utf-8")
        partial_stats = dict(scorer.stats)

        # ---- M8：全 SQL T=0.2 × M 采样 ----
        m8_payloads = [(p["qi"], p["key"], p["prompt_full"]) for p in pl]
        m8_scorer = M8VllmScorer(om_args, args.m8_samples, args.m8_temp)
        m8_scores = m8_scorer.score(m8_payloads)
        m8_cache: Dict[str, Dict[str, Any]] = {}
        for (qi, key, _pr), d in zip(m8_payloads, m8_scores):
            m8_cache[f"{qi}\t{key}"] = d
        (scores_dir / "scores_m8_vllm.json").write_text(
            json.dumps(m8_cache, ensure_ascii=False), encoding="utf-8")
        m8_stats = dict(m8_scorer.stats)

        meta_out = {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "payload_file": str(payload_file),
            "payload_meta": data["meta"],
            "partial_scorer": partial_stats,
            "m8_scorer": m8_stats,
            "n_scored": len(pl),
        }
        (scores_dir / "payload_meta.json").write_text(
            json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[orm-v3] score-only 完成: partial n={len(partial_scores)} "
              f"({partial_stats['wall_seconds']}s) | m8 n={len(m8_scores)} "
              f"({m8_stats['wall_seconds']}s)", file=sys.stderr)
        return 0

    # =================================================================
    # 路径 C：重算（--skip-scoring 或 stub）——读缓存 → 各臂 → 判定 → 输出
    # =================================================================
    if not is_stub and not args.skip_scoring:
        if not (scores_dir / "scores_partial_vllm.json").exists() or \
                not (scores_dir / "scores_m8_vllm.json").exists():
            raise SystemExit(
                f"[orm-v3] 缺本牌分数缓存 {scores_dir}——先跑 "
                f"orm_v3_score.slurm（GPU 打分），再以 --skip-scoring 重算；"
                f"或 --stub-scores random 冒烟")
        print("[orm-v3] WARN 未加 --skip-scoring：按已有缓存读取重算",
              file=sys.stderr)
    # ---- payload 信息（token 预算统计用；文件缺失则现场重建）----
    payload_info: Dict[Tuple[int, str], Dict[str, Any]] = {}
    payload_meta: Optional[Dict[str, Any]] = None
    if payload_file.exists():
        try:
            data = json.loads(payload_file.read_text(encoding="utf-8"))
            payload_meta = data.get("meta")
            for p in data.get("payloads", []):
                payload_info[(int(p["qi"]), p["key"])] = p
        except Exception as exc:
            print(f"[orm-v3] WARN payload 文件不可读: {exc}", file=sys.stderr)

    if not payload_info:
        # 现场重建（SpiderLoader + tokenizer + partialize；reasoning3b 环境）
        # stub 冒烟同样重建——覆盖 payload 构建路径
        from spider_utils import SpiderLoader  # noqa: E402
        from transformers import AutoTokenizer  # noqa: E402

        loader = SpiderLoader(str(args.spider_dir))
        ddl_cache: Dict[str, str] = {}
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, local_files_only=True, trust_remote_code=True)
        print("[orm-v3] payload 文件缺失，现场重建 prompt/token 信息", file=sys.stderr)
        for (qi, key), ei in sorted(rep_set.items()):
            qc = per_question[qi]
            item = qc["item"]
            db_id = item.get("db_id", "")
            if db_id not in ddl_cache:
                ddl_cache[db_id] = loader.format_ddl(db_id)
            ddl = ddl_cache[db_id]
            sql = qc["entries"][ei]["sql_text"]
            partial_sql, backend = partialize_sql(sql)
            prompt_full = build_orm_prompt(item.get("question", ""), ddl, sql)
            prompt_partial = build_orm_prompt(item.get("question", ""), ddl, partial_sql)

            def tok_len(prompt: str) -> Tuple[int, bool]:
                enc = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True, return_dict=True)
                raw = len(enc["input_ids"])
                return (min(raw, args.max_length), raw > args.max_length)

            tf, trf = tok_len(prompt_full)
            tp, trp = tok_len(prompt_partial)
            payload_info[(qi, key)] = {
                "qi": qi, "key": key, "sql": sql, "partial_sql": partial_sql,
                "partial_backend": backend, "prompt_full": prompt_full,
                "prompt_partial": prompt_partial, "tokens_full": tf,
                "tokens_partial": tp, "truncated_full": trf,
                "truncated_partial": trp,
            }

    # ---- 分数加载 ----
    partial_by_q: Dict[int, Dict[int, float]] = defaultdict(dict)
    m8_by_q: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    full_by_q: Dict[int, Dict[int, float]] = defaultdict(dict)
    n_partial_missing = n_m8_missing = n_full_missing = 0

    if is_stub:
        rng = random.Random(args.seed)
        for (qi, key), ei in sorted(rep_set.items()):
            if args.stub_scores == "constant":
                p = args.stub_const
            else:
                p = rng.random()
            partial_by_q[qi][ei] = p
            full_by_q[qi][ei] = rng.random()
            votes = int(round(args.stub_const * args.m8_samples)) \
                if args.stub_scores == "constant" else rng.randrange(args.m8_samples + 1)
            m8_by_q[qi][ei] = {
                "m": args.m8_samples, "votes": votes,
                "vote_rate": round(votes / args.m8_samples, 6),
                "soft_mean": round(args.stub_const, 6)
                if args.stub_scores == "constant" else round(rng.random(), 6),
                "per_sample": None, "note": f"stub-{args.stub_scores}",
            }
    else:
        # partial 缓存（本牌 GPU 产物）
        if (scores_dir / "scores_partial_vllm.json").exists():
            pc = json.loads((scores_dir / "scores_partial_vllm.json")
                            .read_text(encoding="utf-8"))
        else:
            pc = {}
        # m8 缓存
        if (scores_dir / "scores_m8_vllm.json").exists():
            mc = json.loads((scores_dir / "scores_m8_vllm.json")
                            .read_text(encoding="utf-8"))
        else:
            mc = {}
        # full 缓存（v2 产物，--full-scores）
        fc: Dict[str, float] = {}
        if args.full_scores.exists():
            fc = json.loads(args.full_scores.read_text(encoding="utf-8"))
        else:
            print(f"[orm-v3] WARN full-scores 不存在: {args.full_scores}",
                  file=sys.stderr)

        for (qi, key), ei in sorted(rep_set.items()):
            ck = f"{qi}\t{key}"
            if ck in pc:
                partial_by_q[qi][ei] = float(pc[ck])
            else:
                n_partial_missing += 1
            if ck in mc:
                m8_by_q[qi][ei] = mc[ck]
            else:
                n_m8_missing += 1
            if ck in fc:
                full_by_q[qi][ei] = float(fc[ck])
            else:
                n_full_missing += 1

        # ---- CPU 回填（缺分）----
        if (n_partial_missing or n_full_missing or n_m8_missing) and args.cpu_fallback:
            print(f"[orm-v3] 缺分 partial={n_partial_missing} full={n_full_missing} "
                  f"m8={n_m8_missing} → CPU 回填", file=sys.stderr)
            cpu_scorer = V2.CpuOrmScorer(
                args.base_model, args.orm_checkpoint, args.cpu_threads,
                args.max_length, args.score_batch, scores_dir, None, [])
            # partial 回填
            todo: List[Tuple[int, str, str]] = []
            for (qi, key), ei in sorted(rep_set.items()):
                if ei not in partial_by_q[qi]:
                    info = payload_info.get((qi, key))
                    todo.append((qi, key, info["prompt_partial"] if info else None))
            if todo:
                print(f"[orm-v3] CPU 回填 partial: {len(todo)} 条", file=sys.stderr)
                for (qi, key, _pr), s in zip(todo, cpu_scorer.score(todo)):
                    if isinstance(s, float) and not math.isnan(s):
                        partial_by_q[qi][rep_set[(qi, key)]] = float(s)
            # full 回填
            todo = []
            for (qi, key), ei in sorted(rep_set.items()):
                if ei not in full_by_q[qi]:
                    info = payload_info.get((qi, key))
                    todo.append((qi, key, info["prompt_full"] if info else None))
            if todo:
                print(f"[orm-v3] CPU 回填 full: {len(todo)} 条", file=sys.stderr)
                for (qi, key, _pr), s in zip(todo, cpu_scorer.score(todo)):
                    if isinstance(s, float) and not math.isnan(s):
                        full_by_q[qi][rep_set[(qi, key)]] = float(s)
            # m8 回填（Bernoulli 模拟）
            todo = []
            for (qi, key), ei in sorted(rep_set.items()):
                if ei not in m8_by_q[qi]:
                    info = payload_info.get((qi, key))
                    todo.append((qi, key, info["prompt_full"] if info else None))
            if todo:
                print(f"[orm-v3] CPU 回填 m8: {len(todo)} 条（Bernoulli 模拟）",
                      file=sys.stderr)
                for (qi, key, _pr), d in zip(
                        todo, cpu_m8_fallback(cpu_scorer, todo,
                                              args.m8_samples, args.m8_temp,
                                              args.seed)):
                    m8_by_q[qi][rep_set[(qi, key)]] = d
            del cpu_scorer
            n_partial_missing = sum(1 for (qi, key), ei in rep_set.items()
                                    if ei not in partial_by_q[qi])
            n_full_missing = sum(1 for (qi, key), ei in rep_set.items()
                                 if ei not in full_by_q[qi])
            n_m8_missing = sum(1 for (qi, key), ei in rep_set.items()
                               if ei not in m8_by_q[qi])
        elif n_partial_missing or n_full_missing or n_m8_missing:
            print(f"[orm-v3] WARN 缺分未回填 partial={n_partial_missing} "
                  f"full={n_full_missing} m8={n_m8_missing}——对应臂按 0 分处理，"
                  f"结果不可信！", file=sys.stderr)

    # ---- 各臂裁决 + 级联 top-K 集合（token 预算用）----
    for qi, qc in enumerate(per_question):
        entries = qc["entries"]
        votes = qc["votes"]
        n_used = qc["num_instances"]
        joins_cache: Dict[str, Tuple[int, str]] = {}
        grouped, excluded = qc["grouped"], qc["excluded"]
        ranked = qc["ranked"]
        if "arm_p1_m8" in arms:
            qc["results"]["arm_p1_m8"] = _finish(arm_m8(
                entries, ranked, votes, n_used, grouped, excluded, joins_cache,
                m8_by_q[qi], use_soft=False))
        if "arm_p1_m8_soft" in arms:
            qc["results"]["arm_p1_m8_soft"] = _finish(arm_m8(
                entries, ranked, votes, n_used, grouped, excluded, joins_cache,
                m8_by_q[qi], use_soft=True))
        need_caps = any(a in CAPS_ARMS for a in arms)
        if need_caps:
            for k in caps_ks:
                if f"arm_caps_k{k}" in arms:
                    qc["results"][f"arm_caps_k{k}"] = _finish(arm_caps(
                        entries, ranked, votes, n_used, grouped, excluded,
                        joins_cache, partial_by_q[qi], full_by_q[qi], k))
            if "arm_caps_full" in arms:
                qc["results"]["arm_caps_full"] = _finish(arm_caps_full(
                    entries, ranked, votes, n_used, grouped, excluded,
                    joins_cache, full_by_q[qi]))

    # ---- Phase 2：gold 变换后 + 各臂胜者变换后 SQL × 实例 执行 ----
    active_arms = [a for a in arms if a in ALL_ARMS]
    if "arm_vav_multi_all" not in active_arms:
        active_arms.insert(0, "arm_vav_multi_all")
    phase2_tasks: List[Tuple[str, str]] = []
    for qc in per_question:
        item = qc["item"]
        insts = instances_for(item.get("db_id", ""))
        gold_t = AP.official_transform(item.get("gold_sql") or "", is_pred=False,
                                       keep_distinct=args.keep_distinct)
        for inst in insts:
            phase2_tasks.append((gold_t, inst))
        for arm in active_arms:
            rec = qc["results"].get(arm)
            if rec is None or rec.get("text") is None:
                continue
            pred_t = AP.official_transform(rec["text"], is_pred=True,
                                           keep_distinct=args.keep_distinct)
            for inst in insts:
                phase2_tasks.append((pred_t, inst))
    phase2_tasks = list(set(phase2_tasks))
    engine.run(phase2_tasks, phase="judgment")
    print(f"[orm-v3] phase2 完成: {engine._stats['judgment']}", file=sys.stderr)

    for qc in per_question:
        item = qc["item"]
        gold_raw = item.get("gold_sql") or ""
        insts = instances_for(item.get("db_id", ""))
        for arm in active_arms:
            rec = qc["results"].get(arm)
            if rec is None:
                continue
            j = AP._judge_winner(rec["text"], gold_raw, insts, engine,
                                 args.keep_distinct)
            rec["is_correct"] = j["correct"]
            rec["gold_exec_error"] = j["gold_exec_error"]
            rec["order_matters"] = j["order_matters"]

    # ---- 汇总 cells ----
    dataset_stats: Dict[str, Any] = {
        "total_questions": len(items),
        "questions_with_no_instances": 0,
        "questions_with_gold_exec_error": 0,
        "db_instance_count": {db: len(insts)
                              for db, insts in db_instances_cache.items()},
    }
    total_cands = unique_cands = 0
    for qc in per_question:
        total_cands += qc["num_candidates"]
        unique_cands += qc["num_unique_candidates"]
        if qc["num_instances"] == 0:
            dataset_stats["questions_with_no_instances"] += 1
        if any(r.get("gold_exec_error") for r in qc["results"].values()):
            dataset_stats["questions_with_gold_exec_error"] += 1

    cells: Dict[str, Dict[str, Any]] = {}
    for arm in active_arms:
        cell: Dict[str, Any] = {
            "total": len(items), "correct": 0, "accuracy": 0.0,
            "winner_sources": Counter(), "empty_winner": 0,
            "gold_exec_error": 0, "candidates_available": 0,
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
        cell["accuracy"] = round(cell["correct"] / cell["total"], 4) \
            if cell["total"] else 0.0
        cell["winner_sources"] = dict(cell["winner_sources"])
        cells[arm] = cell

    # ---- vs 基线（generic fixed/broken）----
    def _vs(base_correct: List[bool], arm_list: List[str],
            baseline_name: str, baseline_accuracy: float) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for arm in arm_list:
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
            out[arm] = {
                "baseline": baseline_name,
                "baseline_accuracy": baseline_accuracy,
                "arm_accuracy": cells[arm]["accuracy"],
                "delta": round(cells[arm]["accuracy"] - baseline_accuracy, 4),
                "fixed": fixed, "broken": broken, "net": fixed - broken,
                "same_right": same_r, "same_wrong": same_w,
                "fixed_indices": f_idx, "broken_indices": b_idx,
            }
        return out

    base_correct = [qc["results"]["arm_vav_multi_all"]["is_correct"]
                    for qc in per_question]
    vs_mivav = _vs(base_correct, [a for a in active_arms
                                  if a != "arm_vav_multi_all"],
                   "arm_vav_multi_all", cells["arm_vav_multi_all"]["accuracy"])

    # ---- vs 基线 2：官方 ORM grouphead（persisted items，86.1/83.6）----
    vs_orm: Optional[Dict[str, Any]] = None
    if args.baseline_items.exists():
        try:
            data = json.loads(args.baseline_items.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                data = data["items"]
            ref_by_q = {int(r.get("dataset_index", r.get("di"))): r
                        for r in data if isinstance(r, dict)}
            ref_idx = [int(qc["item"].get("dataset_index", qc["item"].get("di")))
                       for qc in per_question]
            base_official = [bool(ref_by_q[i]["is_correct"]) if i in ref_by_q else False
                             for i in ref_idx]
            n_ref = sum(1 for i in ref_idx if i in ref_by_q)
            ref_correct = sum(1 for r in ref_by_q.values() if r.get("is_correct"))
            official_acc = round(ref_correct / len(ref_by_q), 4) if ref_by_q else 0.0
            vs_orm = {
                "file": str(args.baseline_items),
                "n_ref": len(ref_by_q),
                "official_accuracy": official_acc,
                "n_compared": n_ref,
                "vs": _vs(base_official, [a for a in active_arms
                                          if a != "arm_vav_multi_all"],
                          "orm_grouphead_official", official_acc),
            }
            if "arm_caps_full" in active_arms:
                agree = tot = 0
                for qc, i in zip(per_question, ref_idx):
                    r = ref_by_q.get(i)
                    if r is None:
                        continue
                    tot += 1
                    if bool(r.get("is_correct")) == \
                            bool(qc["results"]["arm_caps_full"]["is_correct"]):
                        agree += 1
                vs_orm["caps_full_recompute_agreement"] = (
                    round(agree / tot, 4) if tot else None)
        except Exception as exc:
            vs_orm = {"error": str(exc)}

    # ---- vs 基线 3：arm_caps_full（= grouphead 本进程复现）----
    vs_caps_full: Optional[Dict[str, Dict[str, Any]]] = None
    if "arm_caps_full" in active_arms:
        full_correct = [qc["results"]["arm_caps_full"]["is_correct"]
                        for qc in per_question]
        others = [a for a in active_arms
                  if a not in ("arm_caps_full", "arm_vav_multi_all")]
        vs_caps_full = {
            "arm_accuracy": cells["arm_caps_full"]["accuracy"],
            "vs": _vs(full_correct, others, "arm_caps_full",
                      cells["arm_caps_full"]["accuracy"]),
        }

    # =================================================================
    # 必报统计 ①：Stage-A vs Stage-B 一致性
    # =================================================================
    caps_stats: Dict[str, Any] = {}
    if "arm_caps_full" in active_arms or any(
            a.startswith("arm_caps_k") for a in active_arms):
        p_pairs: List[float] = []
        p_full_pairs: List[float] = []
        g_pairs: List[float] = []
        g_full_pairs: List[float] = []
        n_excl_none = 0
        top1_agree = top1_tot = 0
        recall_cnt = {k: 0 for k in caps_ks}
        recall_tot = {k: 0 for k in caps_ks}
        full_rank_missing = 0
        for qi, qc in enumerate(per_question):
            ranked = qc["ranked"]
            if not ranked:
                continue
            infos = []  # (kg, a_score, b_score)
            for kg in ranked:
                ei = _rep_ei(qc["entries"], kg[1])
                key = qc["entries"][ei]["key"]
                pinfo = payload_info.get((qi, key), {})
                backend = pinfo.get("partial_backend", "none")
                pp = partial_by_q[qi].get(ei)
                pf = full_by_q[qi].get(ei)
                if pp is None or pf is None:
                    full_rank_missing += 1
                    continue
                if backend == "none":
                    n_excl_none += 1
                    continue
                p_pairs.append(pp)
                p_full_pairs.append(pf)
                g_pairs.append(kg[1]["size"] * pp)
                g_full_pairs.append(kg[1]["size"] * pf)
                infos.append((kg, pp, pf))
            if len(infos) < 2:
                continue
            # top-1 一致（arm 语义：size×p 排序，缺分计 0 —— 此处只统计双分齐全代表）
            top1_tot += 1
            a_keyed = sorted(infos, key=lambda t: (t[0][1]["size"] * t[1],
                                                   t[0][1]["size"],
                                                   str(t[0][0])), reverse=True)
            b_keyed = sorted(infos, key=lambda t: (t[0][1]["size"] * t[2],
                                                   t[0][1]["size"],
                                                   str(t[0][0])), reverse=True)
            if a_keyed[0][0][0] == b_keyed[0][0][0]:
                top1_agree += 1
            for k in caps_ks:
                if len(infos) > k:
                    recall_tot[k] += 1
                    topk_keys = {t[0][0] for t in a_keyed[:k]}
                    if b_keyed[0][0][0] in topk_keys:
                        recall_cnt[k] += 1
        caps_stats["stageA_vs_stageB"] = {
            "n_pairs": len(p_pairs),
            "n_excluded_backend_none": n_excl_none,
            "n_ranked_reps_with_missing_scores": full_rank_missing,
            "pearson_r_pointwise": (round(_pearson(p_pairs, p_full_pairs), 4)
                                    if len(p_pairs) >= 2 else None),
            "spearman_rho_pointwise": (round(_spearman(p_pairs, p_full_pairs), 4)
                                       if len(p_pairs) >= 2 else None),
            "pearson_r_group_level": (round(_pearson(g_pairs, g_full_pairs), 4)
                                      if len(g_pairs) >= 2 else None),
            "spearman_rho_group_level": (round(_spearman(g_pairs, g_full_pairs), 4)
                                         if len(g_pairs) >= 2 else None),
            "top1_group_agreement": {
                "n": top1_tot,
                "agree": top1_agree,
                "rate": round(top1_agree / top1_tot, 4) if top1_tot else None,
            },
            "recall_at_k": {
                str(k): {
                    "n": recall_tot[k],
                    "hit": recall_cnt[k],
                    "rate": round(recall_cnt[k] / recall_tot[k], 4)
                    if recall_tot[k] else None,
                } for k in caps_ks
            },
            "note": "recall@K 只统计『双分齐全且 partial_backend != none 的代表数 > K』"
                    "的题（防平凡覆盖）；相关系数只统计双分齐全且 backend != none 的代表",
        }

        # 必报统计 ②：级联 vs 全量（成绩差 + token 预算）
        full_tokens = 0
        partial_tokens = 0
        per_q_topk: Dict[int, Dict[int, int]] = defaultdict(dict)
        for qi, qc in enumerate(per_question):
            ranked = qc["ranked"]
            for kg in ranked:
                ei = _rep_ei(qc["entries"], kg[1])
                key = qc["entries"][ei]["key"]
                pinfo = payload_info.get((qi, key), {})
                full_tokens += pinfo.get("tokens_full", 0)
                partial_tokens += pinfo.get("tokens_partial", 0)
            if not ranked:
                continue
            a_keyed = sorted(ranked, key=lambda kg: _size_x_score_key(
                kg, qc["entries"], lambda ei: partial_by_q[qi].get(ei)))
            for k in caps_ks:
                per_q_topk[k][qi] = sum(
                    payload_info.get((qi, qc["entries"][_rep_ei(qc["entries"], kg[1])]
                                      ["key"]), {}).get("tokens_full", 0)
                    for kg in a_keyed[-k:])
        token_budget: Dict[str, Any] = {
            "n_reps": n_reps,
            "full_arm_prompt_tokens": full_tokens,
            "cascade_stageA_prompt_tokens": partial_tokens,
            "stageA_vs_full_token_ratio": round(partial_tokens / full_tokens, 4)
            if full_tokens else None,
            "per_k": {},
            "score_calls": {
                "full_arm": n_reps,
                "m8_arm": n_reps * args.m8_samples,
            },
        }
        for k in caps_ks:
            stage_b_tokens = sum(per_q_topk[k].values())
            cascade_tokens = partial_tokens + stage_b_tokens
            stage_b_calls = sum(min(k, len(qc["ranked"])) for qc in per_question
                                if qc["ranked"])
            token_budget["per_k"][str(k)] = {
                "stageB_prompt_tokens": stage_b_tokens,
                "cascade_total_prompt_tokens": cascade_tokens,
                "savings_pct_vs_full": round(
                    100.0 * (1.0 - cascade_tokens / full_tokens), 2)
                if full_tokens else None,
                "score_calls": n_reps + stage_b_calls,
            }
        token_budget["note"] = (
            "prompt token 口径 = chat template 后左截断 2048 的长度（与打分同口径）；"
            "full 臂 = 全部组代表全文打分；级联 = 全部组代表部分证据 + top-K 全文。"
            "ORM 判卷节省只在判卷阶段成立（生成阶段不变）；m8 臂成本 = prefill "
            "tokens_full + M×1 token 解码。")

        # 必报统计 ③：K 敏感度
        k_sens: Dict[str, Any] = {}
        for k in caps_ks:
            arm = f"arm_caps_k{k}"
            k_sens[str(k)] = {
                "accuracy": cells[arm]["accuracy"] if arm in cells else None,
                "delta_vs_full": round(
                    cells[arm]["accuracy"] - cells["arm_caps_full"]["accuracy"], 4)
                if arm in cells and "arm_caps_full" in cells else None,
            }
            if vs_caps_full and arm in vs_caps_full["vs"]:
                v = vs_caps_full["vs"][arm]
                k_sens[str(k)].update({
                    "fixed_vs_full": v["fixed"], "broken_vs_full": v["broken"],
                    "net_vs_full": v["net"],
                })
        caps_stats["token_budget"] = token_budget
        caps_stats["k_sensitivity"] = k_sens
        caps_stats["k_note"] = "K=3/5/10 三档如实报，不选优。"

    # ---- m8 诊断 ----
    m8_stats: Dict[str, Any] = {}
    if any(a in M8_ARMS for a in active_arms):
        vrs: List[float] = []
        sms: List[float] = []
        for qmap in m8_by_q.values():
            for d in qmap.values():
                if d.get("vote_rate") is not None:
                    vrs.append(d["vote_rate"])
                if d.get("soft_mean") is not None:
                    sms.append(d["soft_mean"])
        # m8 vs 全分（同代表的 vote_rate 与 P_full 相关性）
        x, y = [], []
        for qi, qmap in full_by_q.items():
            for ei, pf in qmap.items():
                d = m8_by_q[qi].get(ei)
                if d and d.get("vote_rate") is not None:
                    x.append(d["vote_rate"])
                    y.append(pf)
        m8_stats = {
            "m": args.m8_samples,
            "temperature": args.m8_temp,
            "n_reps": len(vrs),
            "vote_rate_mean": round(sum(vrs) / len(vrs), 4) if vrs else None,
            "soft_mean_mean": round(sum(sms) / len(sms), 4) if sms else None,
            "pearson_vote_rate_vs_p_full": round(_pearson(x, y), 4)
            if len(x) >= 2 else None,
            "note": ("arm_p1_m8 = MARS 式蒙特卡洛平均（#{Yes}/M，二项噪声 "
                     "std≤√(p(1-p)/8)≈0.18）；arm_p1_m8_soft = 逐样本 logprob "
                     "平均 ≡ sigmoid(Δ/0.2)（T=0.2 锐化，与 P0-4 t05 同构；两者"
                     "期望相同，差异即采样噪声）"),
        }

    # ---- 输出 ----
    total_wall = sum(v.get("wall_seconds", 0.0) for v in engine._stats.values())
    gpu_meta: Optional[Dict[str, Any]] = None
    if (scores_dir / "payload_meta.json").exists():
        try:
            gpu_meta = json.loads((scores_dir / "payload_meta.json")
                                  .read_text(encoding="utf-8"))
        except Exception:
            gpu_meta = None
    summary = {
        "meta": {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "created_by": "src/orm_selection_v3.py",
            "input_items": str(args.items),
            "output_dir": str(args.out_dir),
            "spider_dir": str(args.spider_dir),
            "baseline_items": str(args.baseline_items),
            "full_scores": str(args.full_scores),
            "threads": args.threads,
            "max_instances_cap": args.max_instances,
            "keep_distinct": args.keep_distinct,
            "seed": args.seed,
            "limit": args.limit,
            "arms_requested": arms,
            "arms_computed": active_arms,
            "stub_scores": args.stub_scores,
            "m8_samples": args.m8_samples,
            "m8_temperature": args.m8_temp,
            "caps_k": caps_ks,
            "sqlglot_available": _HAS_SQLGLOT,
            "payload_file": str(payload_file),
            "gpu_scoring_meta": gpu_meta,
            "semantics": (
                "分组 = adjudicate_soft.build_groups（MI-VAV，只 SUCCESS 入组、"
                "size=票数加权），打分对象 = rankable 组代表（min_sample_idx 最小，"
                "与 grouphead 完全一致），组级分 = size×代表分，平票 → (size, "
                "str(key))，域 = rankable_groups（空组/全零组预筛）。"
                "arm_p1_m8: P̄ = #{Yes}/M（M=8, T=0.2, seed=0, n=M 一次采样）。"
                "arm_p1_m8_soft: P̄ = mean_i sigmoid(logp_yes_i−logp_no_i) ≡ "
                "sigmoid(Δ/0.2)。"
                "arm_caps_k*: Stage-A 部分证据（外层投影+FROM/JOIN+WHERE，去尾部）"
                "T=1.0 打全部组代表，按 (size×P_partial, size, str(key)) 保 top-K；"
                "Stage-B 全文 T=1.0 打分（复用 v2 scores_vllm.json 缓存，与 86.1/"
                "83.6 基线同一打分器）取 (size×P_full) 最高。"
                "arm_caps_full: 全部组全文打分（= grouphead 复现臂）。"
                "判定: 官方 eval_exec_match 全实例等价；NO_RESULTS 回退 arm_maj；"
                "空胜者写 SELECT 1"),
            "risks": [
                "部分证据去掉了 ORDER BY/LIMIT/GROUP BY/HAVING——GROUP BY 去掉后"
                "聚合语义在 Stage-A 缺失（设计意图：粗筛只需关键子句；但 ORM 可能"
                "因此改变判断，这正是 ① 一致性统计要量化的）。",
                "partialize 对 UNION/CTE/非纯 SELECT 不部分化（backend=none，"
                "Stage-A 按全文判、不参与一致性统计）；sqlglot 缺失时正则截尾"
                "（UNION 分支不逐分支截、深度 0 之外不截，见 _partialize_regex "
                "docstring）。",
                "v2 全量缓存（multi 池）有 1 条 vs 本进程重算的漂移（25794/25795），"
                "本进程缺分走 CPU HF 回填（bf16，与 vLLM 值差 ~1e-3）。",
                "m8 vote_rate 为二项估计（M=8，std≤0.177），组级排名受采样噪声"
                "影响；m8_soft 为同期望的确定性锐化参考。",
                "Stage-A/Stage-B 是同一 ORM 在不同输入上打分，一致性上界受 ORM"
                "自身可靠性限制；级联的收益上限 = 全量臂（任何筛选只省 token 不"
                "可能增益）。",
                "判卷 token 节省仅对 ORM 判卷阶段成立（候选生成阶段不变）；"
                "部分证据对 prompt 的缩短有限（base prompt 的 question+DDL 占"
                "大头），节省主要来自 Stage-B 只打 top-K。",
                "m8 与 partial 打分用本牌新采的 vLLM 分数；caps 全文分数复用 v2 "
                "缓存（同日同池同协议）。",
            ],
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
        "scoring_stats": {
            "n_reps_scored": n_reps,
            "payload_meta": payload_meta,
            "n_partial_missing_after_fallback": n_partial_missing,
            "n_m8_missing_after_fallback": n_m8_missing,
            "n_full_missing_after_fallback": n_full_missing,
            "cpu_fallback_used": args.cpu_fallback and not is_stub,
            "stub": is_stub,
        },
        "accuracy": cells,
        "vs_baseline_mivav": vs_mivav,
        "vs_baseline_orm_grouphead_official": vs_orm,
        "vs_arm_caps_full": vs_caps_full,
        "analysis": {
            "caps": caps_stats,
            "m8": m8_stats,
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for arm in active_arms:
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
                "winner_votes": rec.get("votes", 0),
                "winner_group_size": rec.get("group_size", 0),
                "winner_group_key": rec.get("group_key"),
                "winner_models": rec.get("winner_models"),
                "winner_dual": rec.get("winner_dual"),
                "orm_score": rec.get("orm_score"),
                "m8_vote_rate": rec.get("m8_vote_rate"),
                "m8_soft_mean": rec.get("m8_soft_mean"),
                "m8_n_samples": rec.get("m8_n_samples"),
                "partial_score": rec.get("partial_score"),
                "caps_stage_b_n": rec.get("caps_stage_b_n"),
                "caps_winner_rank_a": rec.get("caps_winner_rank_a"),
                "n_joins": rec.get("n_joins"),
                "join_counter": rec.get("join_counter"),
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
    for arm in active_arms:
        c = cells[arm]
        tag = "*" if arm == "arm_vav_multi_all" else " "
        print(f"  {arm:22s} {c['correct']}/{c['total']} ({c['accuracy']:.4f}){tag}")
    print("\n=== vs MI-VAV baseline (fixed / broken / net) ===")
    for arm in active_arms:
        if arm == "arm_vav_multi_all":
            continue
        v = vs_mivav[arm]
        print(f"  {arm:22s} fixed={v['fixed']} broken={v['broken']} "
              f"net={v['net']:+d} delta={v['delta']:+.4f}")
    if vs_orm and "vs" in vs_orm:
        print("\n=== vs official ORM grouphead baseline (fixed / broken / net) ===")
        for arm, v in vs_orm["vs"].items():
            print(f"  {arm:22s} fixed={v['fixed']} broken={v['broken']} "
                  f"net={v['net']:+d} delta={v['delta']:+.4f}")
    if caps_stats.get("stageA_vs_stageB"):
        s = caps_stats["stageA_vs_stageB"]
        print("\n=== CAPS: Stage-A vs Stage-B 一致性 ===")
        print(f"  pointwise  r={s['pearson_r_pointwise']} "
              f"rho={s['spearman_rho_pointwise']} (n={s['n_pairs']})")
        print(f"  group-level r={s['pearson_r_group_level']} "
              f"rho={s['spearman_rho_group_level']}")
        print(f"  top1 agree={s['top1_group_agreement']['rate']} | "
              f"recall@K={ {k: v['rate'] for k, v in s['recall_at_k'].items()} }")
    if caps_stats.get("token_budget"):
        tb = caps_stats["token_budget"]
        print("\n=== CAPS: token 预算 ===")
        print(f"  full={tb['full_arm_prompt_tokens']} | stageA={tb['cascade_stageA_prompt_tokens']} "
              f"(ratio {tb['stageA_vs_full_token_ratio']})")
        for k, v in tb["per_k"].items():
            print(f"  K={k}: cascade={v['cascade_total_prompt_tokens']} "
                  f"savings={v['savings_pct_vs_full']}% "
                  f"calls={v['score_calls']}")
    print(f"\nsummary -> {args.out_dir / 'summary.json'}")
    print(f"items   -> {args.out_dir / 'items_<arm>.json'} ({len(active_arms)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
