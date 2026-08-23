#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoSQL co-evolve 测试期 harness 移植（BIRD dev，零训练）。

背景（EvoSQL arXiv 2607.20489 + tmp_idea_research/p2_research/repos/EvoSQL 代码精读）：
  - Coder-3B BIRD-dev：Maj@16 51.24 → co-evolve 测试期 harness 60.43（+9.19）
    → +SDPO 训练 60.89（+0.46）。大头是测试期 harness，本文件只复刻测试期部分。
  - harness 本质（零训练）：每轮"生成 K=16 候选 → 单实例执行 → LLM critic 打分
    （0-10 rubric + 问题诊断）→ 效用聚合（Eq.6-8：conf + γ 时间衰减 +
    λ_cons·log(1+N) 执行一致性 bonus）→ softmax 采样 payload → 聚合突变
    prompt 再生成"迭代 T=3 轮 + 一致性早停 + 最终效用 top-K 贪心选择。
    默认配方 K=16/T=3/M=1/γ=0.9/λ_cons=0.3/τ=8.0，生成温度 0.8。
  - 我们的移植映射（详见 COEVOLVE_PLAN.md）：
      EvoSQL LLM critic 打分          → 已有 ORM 判卷 P(Yes)（orm_bird_bird_bal2）
      EvoSQL 执行结果簇（result_hash）→ 已有 outcome_signature 签名组
      EvoSQL 每轮 16 个 elicit 候选   → 现有多源池（4 checkpoint × 16，round=0）
      EvoSQL 突变轮（round 1..T-1）   → --phase gen（GPU，主控排期；本文件
                                        --dry-run 只落聚合突变 prompt JSON）
  - 离线可跑部分 = prep/score/final：用现有池的执行签名 + ORM 分做 EvoSQL
    Eq.6-8 效用聚合与最终选择（arm_evosql 等），并产出官方 EX。这度量的是
    "选择侧"收益；"生成侧"（突变轮）收益留给主控排 GPU。

三阶段（与 src/bird_select.py 同构，不修改现有 pipeline）：
  --phase prep   （CPU）复用 bird_select.phase_prep：去重 → 单实例执行 →
                 签名分组 → arm_vav + ORM payloads → work/prep.json。
  --phase score  （GPU, vllmenv）复用 bird_select.phase_score，默认判卷
                 checkpoints/orm_bird_bird_bal2；冒烟用 --stub-scores。
  --phase final  （CPU）EvoSQL Eq.6-8 效用 → 早停诊断 → 多臂选择（arm_evosql
                 主臂 + 网格臂 + 参考臂）→ 官方评估器 → 预注册判定。
  --phase gen    （GPU，主控排期；默认 --dry-run 只落 prompt JSON 不调 LLM）
                 对未早停题构造 EvoSQL 聚合突变 prompt（round 1..T-1）。
  --selftest     （CPU）纯数学自检（无 DB、无 GPU）。

用法（与 bird_select 同口径）：
  envs/reasoning3b/bin/python src/bird_coevolve.py --phase prep \
      --items outputs/eval_pool_bird/items.json --out-dir outputs/bird_coevolve
  envs/vllmenv/bin/python src/bird_coevolve.py --phase score \
      --out-dir outputs/bird_coevolve \
      --orm-checkpoint checkpoints/orm_bird_bird_bal2
  envs/reasoning3b/bin/python src/bird_coevolve.py --phase final \
      --out-dir outputs/bird_coevolve --num-cpus 12
  复用 bird_select 已有产物（HPC 上若已跑过，跳过 prep/score）：
  envs/reasoning3b/bin/python src/bird_coevolve.py --phase final \
      --out-dir outputs/bird_coevolve \
      --reuse-work outputs/bird_select/work --num-cpus 12
  冒烟：各阶段加 --limit 10（final 自动用前 N 行 dev.sql / 前 N 条 dev.json）。
"""
import argparse
import json
import math
import random
import sys
import time
from argparse import Namespace
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(PROJECT_ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adjudicate_pool as AP  # noqa: E402  去重/执行引擎/签名（纯 CPU）
import adjudicate_soft as AS  # noqa: E402  build_groups/rankable_groups/_group_rep

DATA_DIR = PROJECT_ROOT / "data" / "bird" / "bird_dev" / "dev_20240627"
DEFAULT_ITEMS = PROJECT_ROOT / "outputs" / "eval_pool_bird" / "items.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "bird_coevolve"
DEFAULT_DB_ROOT = DATA_DIR / "dev_databases"
DEFAULT_DATA_JSON = DATA_DIR / "dev.json"
DEFAULT_GROUND_TRUTH = DATA_DIR / "dev.sql"
DEFAULT_EVALUATOR = (PROJECT_ROOT / "tmp_idea_research" / "finer-sql" / "evaluation"
                     / "official_bird_evaluation" / "evaluation_bird_ex.py")
DEFAULT_ORM_CKPT = PROJECT_ROOT / "checkpoints" / "orm_bird_bird_bal2"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"
DEFAULT_MERGE_PYTHON = PROJECT_ROOT / "envs" / "reasoning3b" / "bin" / "python"
DEFAULT_REFERENCE_OUT = PROJECT_ROOT / "outputs" / "bird_select"

# ---------------------------------------------------------------------------
# 预注册（2026-08-23 封盘；本文件为唯一执行载体，改此处 = 新实验）
# ---------------------------------------------------------------------------
PRE_REGISTERED = {
    "sealed_at": "2026-08-23",
    "baseline_adapted_EX": 60.37,   # 适配判卷终裁 EX（bird_select arm_orm_grouphead 口径）
    "baseline_official_EX": "实施日以 outputs/bird_select/summary.json 的官方 EX 封盘为准",
    "selection_arm": {
        "name": "arm_evosql",
        "config": {"lambda_cons": 0.3, "tau_raw": 8.0, "gamma": 1.0,
                   "consistency_count": "sql", "K": 16},
        "rationale": (
            "多源池 4 checkpoint×16 全部是 round-0 elicit 候选（无突变轮），"
            "EvoSQL 的 γ 时间衰减无意义 → γ=1.0（γ=0.9 进网格作敏感性）。"
            "τ=8.0 是 EvoSQL raw-score(0-10) 阈值，映射到 P(Yes)=0.8。"
            "consistency_count=sql 与 EvoSQL Eq.8 逐 SQL 去重计数一致。"),
        "success_criteria": (
            "官方 EX ≥ 官方基线（bird_select arm_orm_grouphead）+0.5pp → "
            "选择侧贡献成立；≥ +1.0pp → 选择侧强贡献。"
            "若 ≤ 基线，选择侧无法独立扛收益 → 转主控排 GPU 跑突变轮（gen）。"),
        "official_arms": ["arm_evosql"],
        "promotion_rule": (
            "网格臂内部 EX ≥ arm_evosql 内部 EX +0.3pp 且 ≥ arm_orm_grouphead "
            "内部 EX +0.5pp → 自动追加官方评估（上限 3 个官方臂）。"),
    },
    "full_harness": {
        "note": "pool-seeded 突变轮（round 1..2）为 GPU 阶段，主控排期；",
        "success_criteria": (
            "先跑 200 题校准子集（EvoSQL 同款 leaderboard tuning）：内部 EX "
            "≥ 选择侧臂 +1.0pp 才放行全量 dev。全量官方 EX ≥ 官方基线 +1.5pp "
            "（目标 62+）判成功；≥ 基线 +3.0pp 判强成功（对照 EvoSQL +9.19 "
            "的一半量级）。"),
        "no_go": "200 题子集内部增益 < +1.0pp → NO-GO，不扩全量。",
    },
    "grid": {
        "lambda_cons": [0.0, 0.1, 0.3, 0.5],
        "tau_raw": [8.0],
        "gamma": [1.0, 0.9],
        "consistency_count": ["sql", "votes"],
    },
}

# EvoSQL 默认超参（repos/EvoSQL src/config/evolve_config.yaml 公开配方）
EVO_K = 16          # 每轮候选数
EVO_T = 3           # 最大轮数
EVO_M = 1           # 每 payload 候选数
EVO_GAMMA = 0.9     # 时间衰减（离线池全 round-0 → 主臂用 1.0）
EVO_LAMBDA_CONS = 0.3
EVO_TAU_RAW = 8.0   # raw-score(0-10) 置信阈值 → P(Yes)=0.8
EVO_TEMP_DEFAULT = 1.0    # softmax 选择温度
EVO_SCORE_OFFSET = 1.0    # softmax 数值稳定偏移
EVO_VALID_RATIO = 0.9     # 早停：top-K 有效占比
EVO_CONSISTENCY = 0.9     # 早停：top-K 高置信多数簇占比
EVO_MIN_ACTIVE = 0.2      # 全局早停：活动题占比下限

ARMS = ["arm_evosql", "arm_evosql_maj"]          # 本文件新臂
REFERENCE_ARMS = ["arm_vav", "arm_orm_grouphead"]  # bird_select 既有臂（参考）


# ===========================================================================
# EvoSQL 打分核心（逐语义移植 repos/EvoSQL/src/memory/score.py 与 pool.py）
# ===========================================================================

def normalize_sql(sql: str) -> str:
    if not sql:
        return ""
    return " ".join(sql.lower().split())


def evo_cluster_key(sig: str) -> Tuple[str, str]:
    """执行结果簇 key（对应 EvoSQL execution_cluster_key）。

    - 执行错误 → ('error', 'error')
    - 成功但空结果 → ('empty', 'empty')（EvoSQL 中空结果 conf=0 但属合法候选）
    - 成功非空   → ('result', sig)（sig = outcome_signature，等价 result_hash）
    """
    if sig == AP.ERROR_SIG:
        return ("error", "error")
    if sig.startswith(AP.SUCCESS_PREFIX):
        body = sig[len(AP.SUCCESS_PREFIX):]
        if not body.strip("|"):
            return ("empty", "empty")
        return ("result", sig)
    return ("other", sig)


def compute_confidence(cluster_tag: str, orm_score: Optional[float]) -> float:
    """EvoSQL Eq.6：Conf(c)。

    - error → -1.0（EvoSQL：语法/模式错误）
    - empty → 0.0（EvoSQL：空结果/TLE）
    - result → raw_score/10（EvoSQL raw 为 0-10；我们的 ORM P(Yes)∈[0,1]
      直接就是归一化后的 conf，数值区间完全一致）
    """
    if cluster_tag == "error":
        return -1.0
    if cluster_tag == "empty":
        return 0.0
    if orm_score is None:
        return 0.0
    return float(min(1.0, max(0.0, orm_score)))


def compute_consistency_bonuses(
        entries_clusters: Dict[int, str],      # entry_idx -> cluster_key
        conf: Dict[int, float],                # entry_idx -> confidence
        tau: float,
        counts: Dict[int, int],                # entry_idx -> 票数（votes 模式权重）
        mode: str = "sql",
) -> Dict[str, float]:
    """EvoSQL Eq.8：Const(cluster) = log(1 + N_τ(cluster))。

    N_τ = 该执行结果簇内 conf ≥ τ 的去重候选数（mode=sql，EvoSQL 口径：
    逐 SQL 去重后计数）或票数（mode=votes，与 MI-VAV 组 size 口径一致）。
    返回 cluster_key -> bonus；没有任何 conf≥τ 成员的簇拿 0。
    注意：EvoSQL 原实现中候选自身计入 N（单成员簇 bonus=log2≈0.69）。
    """
    cluster_members: Dict[str, List[int]] = defaultdict(list)
    for ei, ck in entries_clusters.items():
        cluster_members[ck].append(ei)
    bonuses: Dict[str, float] = {}
    for ck, members in cluster_members.items():
        if mode == "votes":
            n = sum(counts.get(ei, 1) for ei in members if conf.get(ei, -1.0) >= tau)
        else:
            n = sum(1 for ei in members if conf.get(ei, -1.0) >= tau)
        if n >= 1:
            bonuses[ck] = math.log(1.0 + n)
        else:
            bonuses[ck] = 0.0
    return bonuses


def compute_utility(conf: float, bonus: float, lambda_cons: float,
                    created_iter: int, current_iter: int, gamma: float) -> float:
    """EvoSQL Eq.7：u_t(c) = γ^(t−t')·Conf(c) + λ_cons·Const(c)。"""
    time_diff = max(current_iter - created_iter, 0)
    return (gamma ** time_diff) * conf + lambda_cons * bonus


def build_pool_state(qc: Dict[str, Any], score_map: Dict[Tuple[int, int], float],
                     tau_raw: float, lambda_cons: float, gamma: float,
                     consistency_count: str = "sql",
                     ) -> Dict[str, Any]:
    """由 bird_select 的 prep 条目 + ORM 分构造 EvoSQL 池状态（单题）。

    输入 qc 为 work/prep.json 的 items 元素。ORM 分只存在于 rankable 组代表
    （bird_select 口径），按 EvoSQL group-cluster-score 模式传播到同签名组成员；
    错误/空结果簇按 EvoSQL 硬上限处理（-1/0），无需 ORM。
    返回：
      entries, sigs, clusters, conf, bonuses, utility, ranks(有效候选按效用排序的
      entry_idx 列表), early_stop(诊断), stats
    """
    tau = tau_raw / 10.0  # raw 0-10 → [0,1]
    entries = qc["entries"]
    sigs_per_entry = qc["sigs_per_entry"]
    votes = {int(k): v for k, v in (qc.get("votes") or {}).items()}
    n_inst = int(qc.get("num_instances", 1))

    # 1) 单实例签名 → 簇
    sigs: Dict[int, str] = {}
    for ei, sig_list in enumerate(sigs_per_entry):
        sigs[ei] = sig_list[0] if (sig_list and n_inst >= 1) else AP.ERROR_SIG
    clusters: Dict[int, str] = {ei: evo_cluster_key(s)[0] for ei, s in sigs.items()}

    # 2) ORM 分传播：ranked 组代表 → 同簇成员（group-cluster-score 模式）
    ranked = qc.get("groups_meta", {}).get("ranked", [])
    rep_scores: Dict[int, float] = {}
    for kg in ranked:
        rep_ei = int(kg["rep_ei"])
        s = score_map.get((int(qc["qi"]), rep_ei))
        if s is not None:
            rep_scores[rep_ei] = float(s)
    rep_cluster_of: Dict[str, float] = {}
    for ei, s in rep_scores.items():
        ck = sigs.get(ei, AP.ERROR_SIG)
        rep_cluster_of[ck] = max(rep_cluster_of.get(ck, -1.0), s)
    orm_of: Dict[int, Optional[float]] = {}
    for ei in range(len(entries)):
        ck = sigs.get(ei, AP.ERROR_SIG)
        orm_of[ei] = rep_cluster_of.get(ck)

    # 3) Conf / Const / Utility（Eq.6→8→7）
    conf: Dict[int, float] = {}
    for ei in range(len(entries)):
        conf[ei] = compute_confidence(clusters[ei], orm_of.get(ei))
    bonuses = compute_consistency_bonuses(
        {ei: sigs[ei] for ei in range(len(entries))}, conf, tau,
        {ei: votes.get(ei, entries[ei].get("count", 1)) for ei in range(len(entries))},
        mode=consistency_count)
    utility: Dict[int, float] = {}
    for ei in range(len(entries)):
        tag = clusters[ei]
        if tag != "result":
            utility[ei] = conf[ei]          # 无效候选：utility=conf（EvoSQL 同）
        else:
            utility[ei] = compute_utility(
                conf[ei], bonuses.get(sigs[ei], 0.0), lambda_cons, 0, 0, gamma)

    # 4) 有效非空候选按效用排序（EvoSQL get_top_k：utility desc → created_iter
    #    desc（新者优先，此处全 0）→ key asc 保证确定性）
    valid_entries = [
        ei for ei in range(len(entries))
        if clusters[ei] == "result"
    ]
    ranks = sorted(valid_entries,
                   key=lambda ei: (-utility[ei],
                                   entries[ei].get("key", "")))
    top_k_rank = ranks[:EVO_K]

    # 5) 早停诊断（EvoSQL EarlyStopPolicy.should_stop_problem）
    valid_in_topk = sum(1 for ei in top_k_rank if clusters[ei] == "result")
    valid_ratio = valid_in_topk / EVO_K if top_k_rank else 0.0
    hi_conf = [ei for ei in top_k_rank if clusters[ei] == "result"
               and conf[ei] >= tau]
    consistency_ratio = 0.0
    top_cluster = None
    if hi_conf:
        cnt = Counter(sigs[ei] for ei in hi_conf)
        top_cluster, top_n = cnt.most_common(1)[0]
        consistency_ratio = top_n / len(hi_conf)
    early_stop = (valid_ratio >= EVO_VALID_RATIO
                  and bool(hi_conf)
                  and consistency_ratio >= EVO_CONSISTENCY)

    return {
        "entries": entries, "sigs": sigs, "clusters": clusters,
        "conf": conf, "bonuses": bonuses, "utility": utility,
        "ranks": ranks, "top_k": top_k_rank,
        "early_stop": early_stop, "valid_ratio": valid_ratio,
        "consistency_ratio": consistency_ratio, "top_cluster": top_cluster,
        "n_hi_conf": len(hi_conf),
        "orm_rep_scores": rep_scores, "orm_of": orm_of,
    }


def _arm_evosql_rec(qc: Dict[str, Any], st: Dict[str, Any]) -> Dict[str, Any]:
    """arm_evosql：效用贪心（EvoSQL 最终 greedy = top-K 有效候选效用最大者）。"""
    entries = st["entries"]
    if not st["ranks"]:
        return {"source": "evosql_fallback", "text": None,
                "winner_ei": None, "utility": None, "empty_winner": True}
    ei = st["ranks"][0]
    return {
        "source": "evosql", "text": entries[ei]["sql_text"],
        "winner_ei": ei, "utility": st["utility"][ei],
        "conf": st["conf"][ei], "bonus": st["bonuses"].get(st["sigs"][ei], 0.0),
        "group_key": st["sigs"][ei],
        "group_size": sum(1 for x in st["clusters"] if st["sigs"].get(x) == st["sigs"][ei]),
        "votes": int(entries[ei].get("count", 1)),
        "empty_winner": False,
    }


def _arm_evosql_maj_rec(qc: Dict[str, Any], st: Dict[str, Any],
                        tau_raw: float) -> Dict[str, Any]:
    """arm_evosql_maj：论文 §4.2.2 终选口径——Top-K 效用候选里"多数执行簇"
    （限 conf≥τ 有效候选），簇内取效用最大代表。"""
    tau = tau_raw / 10.0
    entries = st["entries"]
    hi_conf = [ei for ei in st["top_k"] if st["clusters"][ei] == "result"
               and st["conf"][ei] >= tau]
    if not hi_conf:
        return _arm_evosql_rec(qc, st)
    cnt = Counter(st["sigs"][ei] for ei in hi_conf)
    top_sig, _ = cnt.most_common(1)[0]
    winner = max((ei for ei in hi_conf if st["sigs"][ei] == top_sig),
                 key=lambda ei: st["utility"][ei])
    return {
        "source": "evosql_maj", "text": entries[winner]["sql_text"],
        "winner_ei": winner, "utility": st["utility"][winner],
        "conf": st["conf"][winner], "group_key": top_sig,
        "group_size": sum(1 for x in st["clusters"] if st["sigs"].get(x) == top_sig),
        "votes": int(entries[winner].get("count", 1)),
        "empty_winner": False,
    }


def _arm_orm_grouphead_rec(qc: Dict[str, Any],
                           score_map: Dict[Tuple[int, int], float]
                           ) -> Dict[str, Any]:
    """参考臂 arm_orm_grouphead（与 bird_select.phase_final 同逻辑复刻）。"""
    entries = qc["entries"]
    ranked = qc.get("groups_meta", {}).get("ranked", [])
    rec_vav = qc.get("arm_vav") or {}
    if not ranked:
        return {"source": rec_vav.get("source", "fallback"),
                "text": rec_vav.get("text"), "empty_winner": not bool(rec_vav.get("text"))}
    best = None
    best_key = None
    for kg in ranked:
        s = score_map.get((int(qc["qi"]), int(kg["rep_ei"])), 0.0)
        k = (float(kg["size"]) * s, int(kg["size"]), kg["key"])
        if best_key is None or k > best_key:
            best, best_key = kg, k
    rep = entries[int(best["rep_ei"])]
    return {"source": "orm_grouphead", "text": rep["sql_text"],
            "orm_score": score_map.get((int(qc["qi"]), int(best["rep_ei"]))),
            "group_key": best["key"], "group_size": best["size"],
            "empty_winner": False}


def _finalize_winner(rec: Dict[str, Any], qc: Dict[str, Any]) -> str:
    """胜者文本回退链（与 bird_select 铁律一致：空胜者写 SELECT 1，不跳过）。"""
    text = rec.get("text")
    if not text:
        fb = qc.get("arm_vav") or {}
        text = fb.get("text")
    if not text:
        text = "SELECT 1"
    return text


# ===========================================================================
# EvoSQL 聚合突变 prompt（移植 AGGREGATED_MUTATION_TEMPLATE；gen 阶段用）
# ===========================================================================

# 模板语义逐条对齐 repos/EvoSQL/src/prompt/generator.py AGGREGATED_MUTATION_TEMPLATE
# （MIT；候选卡片 = SQL + 执行摘要 + 判卷分 + 问题诊断；Schema/Question/Hint 权威，
# 候选只是证据）。
MUTATION_TEMPLATE = """Schema:
{schema}

Question:
{question}

Hint:
{hint}

{summary_section}
## Candidate Cards
Each card may contain both useful evidence and serious mistakes.
The execution summary shows behavior, not semantic correctness.
The judge score and issue summary are useful but may be incomplete or wrong.
Do not trust a candidate merely because it executed successfully.
Do not merge SQL fragments unless each fragment is supported by the Schema, Question, and Hint.

{candidates_section}
## Task
Produce exactly ONE corrected SQL query.

Think step by step before writing the final SQL.

Reasoning protocol:
1. Re-read the original Question and Hint first.
2. Identify the requested output columns and row granularity.
3. Identify the required tables, joins, filters, aggregation, ordering, and limits.
4. Compare candidates as possible evidence, not as truth.
5. Reject candidate logic that uses unsupported filters, wrong joins, wrong aggregation, wrong ordering, wrong limit, or wrong output columns.
6. Prefer the simplest SQL that directly answers the Question.
7. Prefer a small correction to a well-supported candidate, but rewrite from scratch if all candidates share the same semantic error.

SQL construction constraints:
- Treat the Question and Hint as separate fields.
- Use the Hint only as evidence for this specific question.
- Use schema-defined keys for joins when available.
- Return only the requested columns, in the requested order.
- Match the requested row cardinality and granularity.
- Use ordering, aggregation, grouping, or limits only when required by the question.
- Preserve the numeric scale implied by the schema and question.
- If the question explicitly defines a formula, compute that formula directly.
- Do not add DISTINCT, GROUP BY, HAVING, filters, joins, or extra output columns unless required.

Output format:
Reasoning:
<brief step-by-step reasoning>

Final SQL:
```sql
SELECT column FROM table WHERE condition;
```
"""

MAX_PREVIEW_ROWS = 10
MAX_PREVIEW_CHARS = 500


def format_exec_preview(outcome: Optional[Dict[str, Any]]) -> str:
    """执行摘要（镜像 EvoSQL format_exec_status：错误文本 / 行数 + 结果预览）。"""
    if not outcome:
        return "Not executed"
    if outcome.get("error"):
        return f"Error: {str(outcome.get('error'))[:200]}"
    rows = outcome.get("rows") or outcome.get("result")
    if rows is None:
        return "Executed; no rows" if outcome.get("ok") else "Executed with unknown result"
    n = len(rows) if isinstance(rows, list) else 0
    if n == 0:
        return "Executed successfully; returned 0 rows"
    preview = str(rows[:MAX_PREVIEW_ROWS])
    if len(preview) > MAX_PREVIEW_CHARS:
        preview = preview[:MAX_PREVIEW_CHARS] + "... (truncated)"
    return f"Executed successfully; returned {n} rows. Preview: {preview}"


def build_candidate_card(idx: int, sql: str, exec_status: str,
                         score_line: str, issue_line: str) -> str:
    parts = [f"### Candidate {idx}", "**SQL:**", f"```sql\n{sql}\n```",
             f"**Execution:** {exec_status}"]
    if score_line:
        parts.append(f"**Score:** {score_line}")
    if issue_line:
        parts.append(f"**Issue:** {issue_line}")
    return "\n".join(parts) + "\n"


def build_mutation_prompt(schema: str, question: str, hint: str,
                          cards_section: str, summary_section: str = "") -> str:
    return MUTATION_TEMPLATE.format(
        schema=schema, question=question, hint=hint,
        summary_section=summary_section.strip(),
        candidates_section=cards_section.strip())


# ===========================================================================
# 参数
# ===========================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="EvoSQL co-evolve 测试期 harness 移植（BIRD dev，零训练）")
    ap.add_argument("--phase", choices=["prep", "score", "final", "gen"],
                    default=None, help="None + --selftest 时跑纯数学自检")
    ap.add_argument("--selftest", action="store_true",
                    help="纯数学自检（无 DB、无 GPU）")
    # 通用
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--work-dir", type=Path, default=None,
                    help="中间产物目录（默认 <out-dir>/work）")
    ap.add_argument("--reuse-work", type=Path, default=None,
                    help="复用 bird_select 已有 work 目录（跳过 prep/score 的产物读取）")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 题（冒烟）")
    ap.add_argument("--seed", type=int, default=0)
    # prep/score（转发给 bird_select）
    ap.add_argument("--db-root", type=Path, default=DEFAULT_DB_ROOT)
    ap.add_argument("--data-json", type=Path, default=DEFAULT_DATA_JSON)
    ap.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    ap.add_argument("--evaluator-py", type=Path, default=DEFAULT_EVALUATOR)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--query-timeout", type=float, default=30.0)
    ap.add_argument("--max-vm-steps", type=int, default=5_000_000)
    ap.add_argument("--row-cap", type=int, default=100_000)
    ap.add_argument("--orm-checkpoint", type=Path, default=DEFAULT_ORM_CKPT)
    ap.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    ap.add_argument("--merge-python", default=str(DEFAULT_MERGE_PYTHON))
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--logprobs-topk", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--stub-scores", choices=["off", "random", "constant"],
                    default="off", help="冒烟：跳过 vLLM 直接写 orm_scores.json")
    # final（co-evolve 超参）
    ap.add_argument("--lambda-cons", type=float, default=EVO_LAMBDA_CONS)
    ap.add_argument("--tau-raw", type=float, default=EVO_TAU_RAW)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--consistency-count", choices=["sql", "votes"], default="sql")
    ap.add_argument("--k", type=int, default=EVO_K)
    ap.add_argument("--grid", action="store_true", help="final：跑全网格内部 EX 诊断")
    ap.add_argument("--official-arms", default="arm_evosql",
                    help="官方评估臂列表（逗号分隔，默认 arm_evosql）")
    ap.add_argument("--skip-official", action="store_true",
                    help="final：不调官方评估器（只出内部 EX 与诊断）")
    ap.add_argument("--num-cpus", type=int, default=12)
    ap.add_argument("--meta-time-out", type=float, default=30.0)
    # gen（GPU 阶段，主控排期；dry-run 只落 prompt JSON）
    ap.add_argument("--round", type=int, default=1, help="gen：突变轮号（1..T-1）")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="gen：只构造并落 mutation prompt JSON（默认，不调 LLM）")
    ap.add_argument("--no-dry-run", action="store_true", dest="no_dry_run",
                    help="gen：实际调用 LLM（仅主控授权 GPU 后使用；需 vllmenv）")
    return ap.parse_args(argv)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _resolve_work(args: argparse.Namespace) -> Path:
    if args.reuse_work:
        return Path(args.reuse_work)
    if args.work_dir:
        return Path(args.work_dir)
    return Path(args.out_dir) / "work"


def _bird_select_ns(args: argparse.Namespace, phase: str) -> Namespace:
    return Namespace(
        phase=phase, items=Path(args.items), out_dir=Path(args.out_dir),
        work_dir=Path(args.work_dir) if args.work_dir else None,
        db_root=Path(args.db_root), data_json=Path(args.data_json),
        ground_truth=Path(args.ground_truth), evaluator_py=Path(args.evaluator_py),
        threads=args.threads, query_timeout=args.query_timeout,
        max_vm_steps=args.max_vm_steps, row_cap=args.row_cap, limit=args.limit,
        seed=args.seed, orm_checkpoint=Path(args.orm_checkpoint),
        base_model=args.base_model, merge_python=args.merge_python,
        max_length=args.max_length, logprobs_topk=args.logprobs_topk,
        chunk_size=args.chunk_size, enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs, num_cpus=args.num_cpus,
        meta_time_out=args.meta_time_out,
    )


# ===========================================================================
# Phase prep / score（复用 bird_select，不改动既有产物口径）
# ===========================================================================

def phase_prep(args: argparse.Namespace) -> None:
    if args.reuse_work:
        work = _resolve_work(args)
        need = [work / "prep.json", work / "orm_payloads.json"]
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            raise RuntimeError(f"--reuse-work 缺产物: {missing}；先跑 bird_select --phase prep")
        print(f"[coevolve_prep] reuse: {work}", file=sys.stderr)
        return
    from bird_select import phase_prep as _bird_prep  # noqa: E402
    _bird_prep(_bird_select_ns(args, "prep"))


def phase_score(args: argparse.Namespace) -> None:
    work = _resolve_work(args)
    if args.stub_scores != "off":
        payloads = json.loads((work / "orm_payloads.json").read_text(encoding="utf-8"))
        rng = random.Random(args.seed)
        const = 0.9 if args.stub_scores == "constant" else None
        scores = [(const if const is not None else rng.random()) for _ in payloads]
        _write_json(work / "orm_scores.json", {
            "entries": [{"qi": p["qi"], "ei": p["ei"], "score": float(s)}
                        for p, s in zip(payloads, scores)],
            "stats": {"mode": f"stub_{args.stub_scores}", "seed": args.seed},
        })
        print(f"[coevolve_score] stub={args.stub_scores} -> {work / 'orm_scores.json'}",
              file=sys.stderr)
        return
    from bird_select import phase_score as _bird_score  # noqa: E402
    _bird_score(_bird_select_ns(args, "score"))


# ===========================================================================
# Phase final：EvoSQL 效用选择 + 官方 EX + 预注册判定
# ===========================================================================

def _load_gold_sigs(args: argparse.Namespace, work: Path, n_questions: int,
                    engine: AP.ExecutionEngine) -> List[Optional[str]]:
    """gold SQL 单实例执行签名缓存（诊断用内部 EX；官方 EX 以官方评估器为准）。"""
    dev = json.loads(Path(args.data_json).read_text(encoding="utf-8"))
    if n_questions < len(dev):
        dev = dev[:n_questions]
    tasks = []
    for i, d in enumerate(dev):
        sql = (d.get("SQL") or "").strip()
        if not sql:
            tasks.append((None, ""))
            continue
        db_id = d["db_id"]
        inst = Path(args.db_root) / db_id / f"{db_id}.sqlite"
        tasks.append((sql, str(inst)))
    engine.run([t for t in tasks if t[0]], phase="gold")
    sigs = []
    for sql, inst in tasks:
        if not sql:
            sigs.append(None)
            continue
        sigs.append(AP.outcome_signature(engine.get(sql, inst)))
    _write_json(work / "gold_sigs.json", {"sigs": sigs,
                                          "note": "内部诊断口径（签名相等），官方 EX 以官方评估器为准"})
    return sigs


def _is_correct(sig: Optional[str], gold_sig: Optional[str]) -> bool:
    if sig is None or gold_sig is None:
        return False
    return sig == gold_sig and sig != AP.ERROR_SIG


def _load_reference_official() -> Optional[Dict[str, Any]]:
    p = DEFAULT_REFERENCE_OUT / "summary.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("official_exec_accuracy")
    except Exception:
        return None


def run_official_arm(arm: str, rows: List[Dict[str, Any]], args: argparse.Namespace,
                     work: Path) -> Dict[str, Any]:
    from bird_select import run_official_evaluator  # noqa: E402
    arm_dir = Path(args.out_dir) / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    pred_list = [
        [r["question"], f"{r['predicted_sql']}\t----- bird -----\t{r['db_id']}"]
        for r in rows
    ]
    _write_json(arm_dir / "predict_dev.json", pred_list)
    ns = _bird_select_ns(args, "final")
    return run_official_evaluator(arm_dir / "predict_dev.json", ns, len(rows), work, arm)


def _grid_configs() -> List[Tuple[float, float, float, str]]:
    g = PRE_REGISTERED["grid"]
    return [(lc, tr, gm, cc)
            for lc in g["lambda_cons"] for tr in g["tau_raw"]
            for gm in g["gamma"] for cc in g["consistency_count"]]


def phase_final(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    work = _resolve_work(args)
    prep_file = work / "prep.json"
    scores_file = work / "orm_scores.json"
    if not prep_file.exists():
        raise RuntimeError(f"缺少 {prep_file}——先跑 --phase prep（或 --reuse-work 指向 bird_select work）")
    if not scores_file.exists():
        raise RuntimeError(f"缺少 {scores_file}——先跑 --phase score（冒烟可 --stub-scores）")

    prep = json.loads(prep_file.read_text(encoding="utf-8"))
    qcs = prep["items"]
    n_questions = len(qcs)
    score_entries = json.loads(scores_file.read_text(encoding="utf-8"))["entries"]
    score_map: Dict[Tuple[int, int], float] = {}
    for e in score_entries:
        score_map[(int(e["qi"]), int(e["ei"]))] = float(e["score"])

    qids = [qc["dataset_index"] for qc in qcs]
    if qids != sorted(qids) or len(set(qids)) != len(qids):
        raise RuntimeError("prep 条目顺序/ID 异常（必须与 dev.json 顺序一致）")

    # ---- gold 执行签名（内部 EX 诊断口径）----
    engine = AP.ExecutionEngine(args.threads, args.query_timeout,
                                args.max_vm_steps, args.row_cap)
    gold_sigs = _load_gold_sigs(args, work, n_questions, engine)

    # ---- 主臂 + 网格臂状态 ----
    primary = PRE_REGISTERED["selection_arm"]["config"]
    states: Dict[str, Dict[str, Any]] = {}
    arm_configs: Dict[str, Tuple[float, float, float, str]] = {}
    states["primary"] = {i: build_pool_state(
        qc, score_map, primary["tau_raw"], primary["lambda_cons"],
        primary["gamma"], primary["consistency_count"]) for i, qc in enumerate(qcs)}
    arm_configs["arm_evosql"] = (primary["lambda_cons"], primary["tau_raw"],
                                 primary["gamma"], primary["consistency_count"])
    for (lc, tr, gm, cc) in _grid_configs():
        tag = f"l{lc}_t{tr}_g{gm}_{cc}"
        states[tag] = {i: build_pool_state(qc, score_map, tr, lc, gm, cc)
                       for i, qc in enumerate(qcs)}
        arm_configs[tag] = (lc, tr, gm, cc)
    if args.grid is False:
        # 默认只跑主臂 + 4 个固定敏感性变体（成本控制）
        keep = {k for k in states if k == "primary"}
        for (lc, tr, gm, cc) in [(0.0, 8.0, 1.0, "sql"), (0.3, 8.0, 0.9, "sql"),
                                 (0.3, 8.0, 1.0, "votes"), (0.5, 8.0, 1.0, "sql")]:
            tag = f"l{lc}_t{tr}_g{gm}_{cc}"
            keep.add(tag)
        states = {k: v for k, v in states.items() if k in keep}

    # ---- 每臂胜者 + 内部 EX ----
    arm_rows: Dict[str, List[Dict[str, Any]]] = {}
    arm_internal_ex: Dict[str, Dict[str, float]] = {}
    arm_tags = sorted(states.keys())
    for tag in arm_tags:
        st_map = states[tag]
        lc, tr, gm, cc = arm_configs["arm_evosql" if tag == "primary" else tag]
        rows = []
        correct = 0
        for i, qc in enumerate(qcs):
            st = st_map[i]
            if tag == "primary":
                rec = _arm_evosql_rec(qc, st)
            else:
                rec = _arm_evosql_rec(qc, st)
            winner_ei = rec.get("winner_ei")
            gold_sig = gold_sigs[i]
            is_correct = (winner_ei is not None
                          and _is_correct(st["sigs"].get(winner_ei), gold_sig))
            correct += int(is_correct)
            rows.append({
                "dataset_index": qc["dataset_index"], "db_id": qc["db_id"],
                "question": qc["question"], "gold_sql": qc["gold_sql"],
                "difficulty": qc["difficulty"],
                "predicted_sql": _finalize_winner(rec, qc),
                "correct_internal": bool(is_correct),
                "empty_winner": bool(rec.get("empty_winner")),
                "winner_source": rec.get("source"),
                "winner_utility": rec.get("utility"),
                "winner_conf": rec.get("conf"),
                "winner_bonus": rec.get("bonus"),
                "winner_group_key": rec.get("group_key"),
                "winner_group_size": rec.get("group_size"),
                "winner_votes": rec.get("votes", 0),
                "n_candidates": qc["num_candidates"],
                "n_unique": qc["num_unique_candidates"],
            })
        arm_rows[tag] = rows
        arm_internal_ex[tag] = {
            "internal_EX": round(correct / n_questions * 100, 2),
            "config": {"lambda_cons": lc, "tau_raw": tr, "gamma": gm,
                       "consistency_count": cc},
        }
        _write_json(out_dir / f"items_{tag}.json", rows)
    _write_json(out_dir / "items_arm_evosql.json", arm_rows["primary"])  # 主臂别名

    # ---- arm_evosql_maj（主配方口径）----
    maj_rows = []
    maj_correct = 0
    st_map = states["primary"]
    for i, qc in enumerate(qcs):
        rec = _arm_evosql_maj_rec(qc, st_map[i], primary["tau_raw"])
        winner_ei = rec.get("winner_ei")
        gold_sig = gold_sigs[i]
        is_correct = (winner_ei is not None
                      and _is_correct(st_map[i]["sigs"].get(winner_ei), gold_sig))
        maj_correct += int(is_correct)
        maj_rows.append({
            "dataset_index": qc["dataset_index"], "db_id": qc["db_id"],
            "question": qc["question"], "gold_sql": qc["gold_sql"],
            "difficulty": qc["difficulty"],
            "predicted_sql": _finalize_winner(rec, qc),
            "correct_internal": bool(is_correct),
            "empty_winner": bool(rec.get("empty_winner")),
            "winner_source": rec.get("source"),
            "winner_group_key": rec.get("group_key"),
            "winner_group_size": rec.get("group_size"),
        })
    arm_rows["arm_evosql_maj"] = maj_rows
    arm_internal_ex["arm_evosql_maj"] = {
        "internal_EX": round(maj_correct / n_questions * 100, 2),
        "config": {"mode": "majority cluster among top-K (conf>=tau)",
                   "tau_raw": primary["tau_raw"], "K": EVO_K},
    }
    _write_json(out_dir / "items_arm_evosql_maj.json", maj_rows)

    # ---- 参考臂内部 EX（与 bird_select 同逻辑复刻，便于对照）----
    # 先预执行参考臂胜者 SQL（engine.get 只查缓存，不懒执行）
    ref_tasks: List[Tuple[str, str]] = []
    ref_text_of: Dict[Tuple[str, int], Optional[str]] = {}
    for i, qc in enumerate(qcs):
        inst = str(Path(args.db_root) / qc["db_id"] / f"{qc['db_id']}.sqlite")
        vav_rec = qc.get("arm_vav") or {}
        ref_text_of[("vav", i)] = vav_rec.get("text")
        orgh = _arm_orm_grouphead_rec(qc, score_map)
        ref_text_of[("orgh", i)] = orgh.get("text")
        for text in (ref_text_of[("vav", i)], ref_text_of[("orgh", i)]):
            if text:
                ref_tasks.append((text, inst))
    engine.run(list(set(ref_tasks)), phase="ref")
    ref_internal: Dict[str, Dict[str, float]] = {}
    for ref in REFERENCE_ARMS:
        correct = 0
        for i, qc in enumerate(qcs):
            text = ref_text_of[("vav" if ref == "arm_vav" else "orgh", i)]
            gold_sig = gold_sigs[i]
            sig = None
            if text:
                outcome = engine.get(text, str(Path(args.db_root) / qc["db_id"]
                                                / f"{qc['db_id']}.sqlite"))
                sig = AP.outcome_signature(outcome) if outcome else None
            correct += int(_is_correct(sig, gold_sig))
        ref_internal[ref] = {"internal_EX": round(correct / n_questions * 100, 2)}
    ref_official = _load_reference_official()

    # ---- 官方 EX（预注册：主臂必跑；网格臂按 promotion_rule 追加）----
    official_results: Dict[str, Any] = {}
    official_arms = [a.strip() for a in args.official_arms.split(",") if a.strip()]
    if not args.skip_official and official_arms:
        if "arm_evosql" not in official_arms:
            raise RuntimeError("主臂 arm_evosql 必须官方评估（预注册主判定）")
        promoted = []
        base_internal = arm_internal_ex["primary"]["internal_EX"]
        ref_internal_ex = ref_internal.get("arm_orm_grouphead", {}).get("internal_EX", 0.0)
        for tag in arm_tags:
            if tag == "primary":
                continue
            ex = arm_internal_ex[tag]["internal_EX"]
            if ex >= base_internal + 0.3 and ex >= ref_internal_ex + 0.5:
                promoted.append(tag)
        promoted = promoted[:2]  # 上限：官方臂总数 ≤ 3
        arm_name_of = {"primary": "arm_evosql"}
        for tag in promoted:
            arm_name_of[tag] = f"arm_evosql_{tag}"
        eval_plan = ["arm_evosql"] + [arm_name_of[t] for t in promoted]
        official_results["arm_evosql"] = run_official_arm(
            "arm_evosql", arm_rows["primary"], args, work)
        for tag in promoted:
            name = arm_name_of[tag]
            official_results[name] = run_official_arm(name, arm_rows[tag], args, work)
        print(f"[coevolve_final] official arms: {eval_plan}", file=sys.stderr)
    elif not args.skip_official:
        print("[coevolve_final] --official-arms 为空 → 跳过官方评估", file=sys.stderr)

    # ---- 早停诊断 + pass@K oracle ----
    n_early_stop = 0
    n_pass = 0                      # 池内存在与 gold 签名一致的候选（pass@池 oracle）
    n_maj_wrong = 0                 # arm_evosql 错但 arm_vav 对（迁移风险诊断）
    by_diff: Dict[str, List[int]] = defaultdict(list)
    per_problem_diag = []
    st_map = states["primary"]
    for i, qc in enumerate(qcs):
        st = st_map[i]
        gold_sig = gold_sigs[i]
        hit = any(_is_correct(st["sigs"].get(ei), gold_sig)
                  for ei in range(len(st["entries"])))
        n_pass += int(hit)
        n_early_stop += int(st["early_stop"])
        vav_rec = qc.get("arm_vav") or {}
        vav_text = vav_rec.get("text")
        vav_sig = None
        if vav_text:
            outcome = engine.get(vav_text, str(Path(args.db_root) / qc["db_id"]
                                                / f"{qc['db_id']}.sqlite"))
            vav_sig = AP.outcome_signature(outcome) if outcome else None
        vav_correct = _is_correct(vav_sig, gold_sig)
        evo_rec = _arm_evosql_rec(qc, st)
        evo_correct = (evo_rec.get("winner_ei") is not None
                       and _is_correct(st["sigs"].get(evo_rec["winner_ei"]), gold_sig))
        if evo_correct is False and vav_correct:
            n_maj_wrong += 1
        by_diff[qc["difficulty"]].append(int(evo_correct))
        per_problem_diag.append({
            "qi": i, "db_id": qc["db_id"], "difficulty": qc["difficulty"],
            "n_candidates": qc["num_candidates"],
            "n_unique": qc["num_unique_candidates"],
            "n_valid_result": len(st["ranks"]),
            "n_clusters": len(set(st["sigs"].values())),
            "pass_pool_oracle": bool(hit),
            "early_stop_met": bool(st["early_stop"]),
            "valid_ratio": round(st["valid_ratio"], 3),
            "consistency_ratio": round(st["consistency_ratio"], 3),
            "n_hi_conf": st["n_hi_conf"],
            "evosql_correct": bool(evo_correct),
            "vav_correct": bool(vav_correct),
        })
    _write_json(out_dir / "diagnostics_per_problem.json", per_problem_diag)
    diff_ex = {d: round(sum(v) / len(v) * 100, 2) for d, v in by_diff.items()}

    summary = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": "src/bird_coevolve.py（EvoSQL co-evolve 测试期 harness 移植）",
            "prep_file": str(prep_file), "scores_file": str(scores_file),
            "out_dir": str(out_dir), "n_questions": n_questions,
            "limit": args.limit, "reference_out": str(DEFAULT_REFERENCE_OUT),
            "note": (
                "critic = ORM 判卷 P(Yes)（orm_bird_bird_bal2），只打 rankable 组代表"
                "并向同签名组成员传播（EvoSQL group-cluster-score 模式）；执行簇 = "
                "outcome_signature 签名；离线池全为 round-0 elicit 候选 → γ 衰减项恒为 1；"
                "内部 EX 用签名相等口径做诊断，唯一有效数字 = 官方评估器 EX。"),
        },
        "pre_registered": PRE_REGISTERED,
        "arm_internal_EX": arm_internal_ex,
        "reference_internal_EX": ref_internal,
        "reference_official_EX": ref_official,
        "official_exec_accuracy": official_results,
        "early_stop_diagnostics": {
            "n_early_stop_met": n_early_stop,
            "early_stop_rate": round(n_early_stop / n_questions * 100, 2),
            "pool_pass_oracle": round(n_pass / n_questions * 100, 2),
            "note": ("pass@池 oracle = 池内存在与 gold 执行签名一致的候选（选择侧天花板）；"
                     "early_stop 判定用 EvoSQL 默认 valid_ratio>=0.9 & 高置信多数簇>=0.9"),
        },
        "difficulty_internal_EX": diff_ex,
        "risk_diag": {
            "n_evosql_wrong_but_vav_right": n_maj_wrong,
            "note": "arm_evosql 错但 arm_vav 对的题数（>0 提示效用选择替换了正确的多数簇）",
        },
    }
    _write_json(out_dir / "summary.json", summary)

    print("\n=== BIRD 官方执行准确率（FINER evaluation_bird_ex.py）===")
    for arm, r in official_results.items():
        print(f"  {arm:22s} simple={r['simple']:.2f} moderate={r['moderate']:.2f} "
              f"challenging={r['challenging']:.2f} total={r['total']:.2f}")
    if ref_official:
        print("--- 参考臂（bird_select 官方）---")
        for arm in REFERENCE_ARMS:
            r = ref_official.get(arm)
            if r:
                print(f"  {arm:22s} total={r['total']:.2f}")
    print(f"\nsummary -> {out_dir / 'summary.json'}")


# ===========================================================================
# Phase gen：聚合突变 prompt（GPU 阶段，主控排期；dry-run 只落 JSON）
# ===========================================================================

def phase_gen(args: argparse.Namespace) -> None:
    work = _resolve_work(args)
    out_dir = Path(args.out_dir)
    prep_file = work / "prep.json"
    scores_file = work / "orm_scores.json"
    for p in (prep_file, scores_file):
        if not p.exists():
            raise RuntimeError(f"缺少 {p}——gen 阶段依赖 prep/score 产物")
    prep = json.loads(prep_file.read_text(encoding="utf-8"))
    qcs = prep["items"]
    score_entries = json.loads(scores_file.read_text(encoding="utf-8"))["entries"]
    score_map: Dict[Tuple[int, int], float] = {
        (int(e["qi"]), int(e["ei"])): float(e["score"]) for e in score_entries}
    cfg = PRE_REGISTERED["selection_arm"]["config"]

    from bird_select import read_ddl  # noqa: E402
    engine = AP.ExecutionEngine(args.threads, args.query_timeout,
                                args.max_vm_steps, args.row_cap)

    # 1) 池状态 + 早停 → 决定哪些题进入本轮突变
    states = {}
    active = []
    for i, qc in enumerate(qcs):
        st = build_pool_state(qc, score_map, cfg["tau_raw"], cfg["lambda_cons"],
                              cfg["gamma"], cfg["consistency_count"])
        states[i] = st
        if not st["early_stop"] and st["ranks"]:
            active.append(i)

    # 2) 预执行本轮 payload 候选（组代表级预览，最多 n_active × 1 次执行）
    payload_sqls = []
    for i in active:
        ei = states[i]["ranks"][0]
        payload_sqls.append((states[i]["entries"][ei]["sql_text"],
                             str(Path(args.db_root) / qcs[i]["db_id"]
                                 / f"{qcs[i]['db_id']}.sqlite")))
    engine.run(payload_sqls, phase="gen_preview")

    # 3) 构造 EvoSQL 聚合突变 prompt（M=1：payload = 效用最高候选）
    ddl_cache: Dict[str, str] = {}
    prompts = []
    for i in active:
        qc = qcs[i]
        st = states[i]
        db_id = qc["db_id"]
        if db_id not in ddl_cache:
            ddl_cache[db_id] = read_ddl(Path(args.db_root), db_id)
        ei = st["ranks"][0]
        entry = st["entries"][ei]
        outcome = engine.get(entry["sql_text"],
                             str(Path(args.db_root) / db_id / f"{db_id}.sqlite"))
        exec_status = format_exec_preview(outcome)
        score_line = (f"judge P(Yes)={st['conf'][ei]:.3f} "
                      f"utility={st['utility'][ei]:.4f}")
        issue_line = _issue_for(st, ei)
        card = build_candidate_card(1, entry["sql_text"], exec_status,
                                    score_line, issue_line)
        prompt_text = build_mutation_prompt(
            schema=ddl_cache[db_id], question=qc["question"],
            hint="", cards_section=card, summary_section="")
        prompts.append({
            "qi": i, "dataset_index": qc["dataset_index"], "db_id": db_id,
            "round": args.round, "payload_ei": ei,
            "payload_utility": st["utility"][ei],
            "payload_sql": entry["sql_text"],
            "prompt": prompt_text,
        })

    out_file = work / f"mutation_prompts_r{args.round}.json"
    _write_json(out_file, {
        "meta": {
            "round": args.round, "n_active": len(active),
            "n_total": len(qcs),
            "template": "EvoSQL AGGREGATED_MUTATION_TEMPLATE 移植（M=1 payload）",
            "config": cfg,
            "note": ("dry-run 产物：主控 GPU runner 按此 JSON 用生成模型（K=16 采样/"
                     "题）执行 → 新候选走 ExecutionEngine + ORM 判卷 → 下一轮。"
                     "执行预览来自本轮 CPU 重执行（组代表）。"),
        },
        "prompts": prompts,
    })
    print(f"[coevolve_gen] round={args.round} active={len(active)}/{len(qcs)} "
          f"-> {out_file}", file=sys.stderr)
    if not args.no_dry_run:
        print("[coevolve_gen] dry-run 完成；实际 LLM 生成由主控排 GPU 执行。",
              file=sys.stderr)


def _issue_for(st: Dict[str, Any], ei: int) -> str:
    """低置信/异常候选的修复引导（EvoSQL critic issues 的 ORM 近似版）。"""
    conf = st["conf"][ei]
    if conf < 0.5:
        return ("problem: low judge confidence; fix: re-check filters, joins and "
                "output columns against the question and schema")
    return ""


# ===========================================================================
# Selftest（纯数学；无 DB/GPU）
# ===========================================================================

def _selftest() -> None:
    def approx(a, b, eps=1e-6):
        assert abs(a - b) < eps, f"{a} != {b}"

    # Eq.6
    approx(compute_confidence("error", 0.9), -1.0)
    approx(compute_confidence("empty", None), 0.0)
    approx(compute_confidence("result", 0.9), 0.9)
    approx(compute_confidence("result", None), 0.0)

    # Eq.8（EvoSQL 口径：成员含自身；单成员簇 bonus=log(2)）
    entries_clusters = {0: "sA", 1: "sA", 2: "sA", 3: "sB"}
    conf = {0: 0.9, 1: 0.7, 2: 0.85, 3: 0.9}
    b = compute_consistency_bonuses(entries_clusters, conf, tau=0.8,
                                    counts={e: 1 for e in entries_clusters},
                                    mode="sql")
    approx(b["sA"], math.log(1 + 2))   # 0.9,0.85 ≥ 0.8 → N=2
    approx(b["sB"], math.log(1 + 1))   # 自身
    bv = compute_consistency_bonuses(entries_clusters, conf, tau=0.8,
                                     counts={0: 3, 1: 1, 2: 1, 3: 2}, mode="votes")
    approx(bv["sA"], math.log(1 + 4))  # 3+1 votes
    approx(bv["sB"], math.log(1 + 2))

    # Eq.7
    u = compute_utility(0.9, math.log(3.0), lambda_cons=0.3,
                        created_iter=1, current_iter=2, gamma=0.9)
    approx(u, 0.9 * 0.9 + 0.3 * math.log(3.0))

    # 合成池：两簇，簇 A 高置信大、簇 B 高置信小 → 效用排序 A 胜
    qc = {
        "qi": 0, "num_instances": 1,
        "entries": [
            {"key": "a", "sql_text": "SELECT 1", "count": 4, "min_sample_idx": 0,
             "models": ["m1"]},
            {"key": "b", "sql_text": "SELECT 2", "count": 1, "min_sample_idx": 1,
             "models": ["m1"]},
            {"key": "c", "sql_text": "SELECT bad", "count": 1, "min_sample_idx": 2,
             "models": ["m2"]},
        ],
        "sigs_per_entry": [["SUCCESS_VALUES:|1|"], ["SUCCESS_VALUES:|2|"], ["ERROR"]],
        "votes": {"0": 4, "1": 1, "2": 1},
        "groups_meta": {"ranked": [
            {"key": "('SUCCESS_VALUES:|1|',)", "size": 4, "rep_ei": 0,
             "models": ["m1"], "rep_text": "SELECT 1"},
            {"key": "('SUCCESS_VALUES:|2|',)", "size": 1, "rep_ei": 1,
             "models": ["m1"], "rep_text": "SELECT 2"},
        ]},
        "arm_vav": {"source": "vav", "text": "SELECT 1"},
        "num_candidates": 6, "num_unique_candidates": 3,
        "dataset_index": 0, "db_id": "x", "question": "q", "gold_sql": "",
        "difficulty": "simple",
    }
    score_map = {(0, 0): 0.9, (0, 1): 0.85}
    st = build_pool_state(qc, score_map, tau_raw=8.0, lambda_cons=0.3,
                          gamma=1.0, consistency_count="sql")
    # 传播：簇 B 只有一个 rep 0.85 → conf 0.85
    approx(st["conf"][0], 0.9)
    approx(st["conf"][1], 0.85)
    approx(st["conf"][2], -1.0)
    # bonus：簇 A N=1（只有 entry0 ≥0.8）→ log2；簇 B N=1 → log2
    approx(st["bonuses"]["SUCCESS_VALUES:|1|"], math.log(2.0))
    approx(st["bonuses"]["SUCCESS_VALUES:|2|"], math.log(2.0))
    u0 = 0.9 + 0.3 * math.log(2.0)
    u1 = 0.85 + 0.3 * math.log(2.0)
    approx(st["utility"][0], u0)
    approx(st["utility"][1], u1)
    assert st["ranks"][0] == 0, "效用最高者应为 entry 0"
    rec = _arm_evosql_rec(qc, st)
    assert rec["text"] == "SELECT 1" and rec["winner_ei"] == 0
    rec_maj = _arm_evosql_maj_rec(qc, st, tau_raw=8.0)
    assert rec_maj["text"] == "SELECT 1"
    # 早停诊断：top_k 只有 2 个有效 → valid_ratio 0.125 < 0.9 → 不早停
    assert st["early_stop"] is False

    # votes 模式：簇 A 票数 4、簇 B 票数 1 → 簇 A bonus log(1+4) 更高
    st_v = build_pool_state(qc, score_map, tau_raw=8.0, lambda_cons=0.3,
                            gamma=1.0, consistency_count="votes")
    approx(st_v["bonuses"]["SUCCESS_VALUES:|1|"], math.log(1 + 4))
    assert st_v["ranks"][0] == 0

    print("[selftest] 6 组断言全部通过（Eq.6/7/8、传播、排序、maj、早停、votes）")


# ===========================================================================
# main
# ===========================================================================

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    if args.phase == "prep":
        phase_prep(args)
    elif args.phase == "score":
        phase_score(args)
    elif args.phase == "final":
        phase_final(args)
    elif args.phase == "gen":
        phase_gen(args)
    else:
        raise SystemExit("必须指定 --phase 或 --selftest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
