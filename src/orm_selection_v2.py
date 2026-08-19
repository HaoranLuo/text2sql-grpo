#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/orm_selection_v2.py — P0 融合层升级：5 个零训练臂（全部离线重算，CPU only）。

输入（HPC）
  outputs/eval_pool_multi/items.json（1034 题 × 64 候选，4 源池：sft_phase1/sft_v2/
    sft_v3/p2a_500，官方 ORM grouphead 基线 86.1%）
  outputs/eval_pool_unseen/items.json（1034 题 × 32 候选，2 源池：sft_v3/p2a_500，
    官方 ORM grouphead 基线 83.4%）
  checkpoints/orm_b1（Qwen2.5-Coder-3B-Instruct + LoRA，Yes/No 生成式 ORM）
  outputs/orm_selection_multi|unseen/items_arm_orm_grouphead.json（官方基线对比源）

臂（同一批候选上离线重算；判定一律官方语义 exec match：postprocess →
remove_distinct → replace_cur_year → result_eq，全实例等价；空胜者写 SELECT 1）：

  arm_vav_multi_all        基线 1 = MI-VAV（行排序保列序签名，choose_group_vav）
                           （对照官方 74.3% 线）
  arm_p02_rci              P0-2：分组签名换成"行列序双不敏感"
                           （SISelection result_to_normal_form 语义：行排序 +
                           行内值排序 + 剔全 None 行；重复行保留——与管线包语义一致，
                           偏离点见 summary）。裁决语义 = MI-VAV（最大组胜出）。
  arm_p03_gate             P0-3：SIRIUS 式门控 + 平票兜底链。触发条件
                           |top1-top2|<=1 或 top1<2（沿 gated_structural 定义）；
                           触发后二次裁决链 = ① 跨源支持度（组内不同来源模型数）
                           ② 组大小 ③ 保守保留原 MI-VAV 胜者（只有 (n_models,size)
                           严格更优才覆盖）。零 ORM。
  arm_p04_t05/t10/t20      P0-4：执行分层先置（clean=所有实例非空 > empty=存在空
                           实例；error 因分组前置条件恒为空）+ 层内 argmax
                           size×P_T(Yes)，P_T=σ(σ⁻¹(P)/T)，T∈{0.5,1.0,2.0} 三档
                           如实报（T=1.0 = 恒等缩放，只剩分层效应）。打分对象 =
                           各组代表（含 all-empty 组——GradeSQL 语义"empty 仅在无
                           clean 组时参与"；全零组仍剔除）。
  arm_p05_bsf1             P0-5：BS-F1 软比较器（列名感知版）替换硬等值。
                           score(c)=mean over 实例 BS-F1(result(c), result(consensus))，
                           consensus = MI-VAV 最大组代表（arm_vav_multi_all 胜者）。
                           BS-F1 = DPC metrics.py 移植（值归一化 + 匈牙利行匹配）
                           + 列名对齐（按列名映射两表后逐行比较，修 DPC 列位置敏感
                           洞）；行数超 cap 退化多重集硬等值。零 ORM。
  arm_p05_bsf1_group       P0-5 组级变体：score(g)=size×BS-F1(rep(g), consensus)
                           （与现行 size×P(Yes) 同构，只是把 P(Yes) 换成 BS-F1）。
  arm_p06_r3               P0-6：R³ 组级打分 (r_list, r_point) + τ=0.05 决策化。
                           u(g)=max pointwise P(Yes)（组内最高分）；r_list(g)=
                           #{j: u(g)>u(j)+τ}（τ=0.05 决策化边际——零训练伪 pairwise
                           对 R³ 概率阈值 τ 的对应物，见 summary 说明）；r_point(g)=
                           |g|×u(g)；按 (r_list, r_point) 字典序取胜组，组内取
                           pointwise 最高 SQL。打分对象 = 全部入组候选。
  arm_orm_grouphead        基线 2 = 现行 ORM 终裁 size×P(Yes)（本进程 CPU 重打分
                           重算，用于与官方 86.1/83.4 基线核对与 fixed/broken）。

ORM 打分（零 GPU 纪律下的方案）
  CPU HF 前向（transformers + peft merge_and_unload，bf16）：只算末位 logits 的
  P(Yes)=sigmoid(logit_Yes−logit_No)，不做生成；prompt 与 vLLM 基线完全同一协议
  （chat template + 左截断 2048，build_orm_prompt 逐字复制自 orm_selection.py）。
  实测 cpu8358 32 线程 batch=8 ≈ 3.6 s/候选。分数按 (题,候选 key) 落盘 JSON 缓存，
  --score-shard K/N 支持跨节点分片并行；完成后用 --crosscheck-items 对基线胜者
  orm_score 做数值一致性核对（vLLM vs CPU bf16 预期 ~1e-3 量级差）。

输出 outputs/orm_selection_v2/（多源池）与 outputs/orm_selection_v2_unseen/：
  items_<arm>.json    与 scripts/eval_official.sh 兼容
  summary.json        各臂 in-process 官方语义准确率 + vs MI-VAV 基线与 vs 官方
                      ORM grouphead 基线（86.1/83.4）的 fixed/broken + 各臂诊断
  scores/scores_shard{K}of{N}.json    ORM 分数缓存（跨阶段复用）

用法（HPC CPU）
  # 阶段 1（零 ORM 臂 + 基线；~1-1.5h）：
  envs/reasoning3b/bin/python src/orm_selection_v2.py --items outputs/eval_pool_multi/items.json \
      --out-dir outputs/orm_selection_v2 --baseline-items outputs/orm_selection_multi/items_arm_orm_grouphead.json \
      --orm-backend none --arms baseline,p02,p03,p05 --threads 16
  # 阶段 2（分片打分；4 个节点并行，各 ~8h）：
  envs/reasoning3b/bin/python src/orm_selection_v2.py ... --orm-backend cpu --score-only --score-shard K 4 \
      --cpu-threads 32 --score-batch 8 --crosscheck-items outputs/orm_selection_multi/items_arm_orm_grouphead.json
  # 阶段 3（ORM 臂；复用分数缓存；~1h）：
  envs/reasoning3b/bin/python src/orm_selection_v2.py ... --orm-backend cpu --arms p04,p06,ormgrouphead \
      --skip-scoring
  # 冒烟（零 ORM，纯逻辑）：
  envs/reasoning3b/bin/python src/orm_selection_v2.py ... --orm-backend stub-random --limit 30 \
      --out-dir outputs/orm_selection_v2_smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(PROJECT_ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adjudicate_pool as AP  # noqa: E402  去重/执行引擎/官方 exec-match 判定（纯 CPU）
import adjudicate_soft as AS  # noqa: E402  build_groups/rankable_groups/_group_rep/_base_record

from adjudicate_pool import (  # noqa: E402
    ERROR_SIG, SUCCESS_PREFIX, _SEP_ROW, _SEP_VAL, _TRUNC_MARK, _sig_value)

DEFAULT_ITEMS_MULTI = PROJECT_ROOT / "outputs" / "eval_pool_multi" / "items.json"
DEFAULT_OUT_MULTI = PROJECT_ROOT / "outputs" / "orm_selection_v2"
DEFAULT_BASELINE_MULTI = (
    PROJECT_ROOT / "outputs" / "orm_selection_multi" / "items_arm_orm_grouphead.json")
DEFAULT_SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"
DEFAULT_ORM_CKPT = PROJECT_ROOT / "checkpoints" / "orm_b1"

# ---- 臂注册表 ----
BASELINE_ARMS = ["arm_vav_multi_all", "arm_orm_grouphead"]
P02_ARMS = ["arm_p02_rci"]
P03_ARMS = ["arm_p03_gate"]
P04_ARMS = ["arm_p04_t05", "arm_p04_t10", "arm_p04_t20"]
P05_ARMS = ["arm_p05_bsf1", "arm_p05_bsf1_group"]
P06_ARMS = ["arm_p06_r3"]
GROUP_NAMES = {
    "baseline": BASELINE_ARMS, "p02": P02_ARMS, "p03": P03_ARMS,
    "p04": P04_ARMS, "p05": P05_ARMS, "p06": P06_ARMS,
    "ormgrouphead": ["arm_orm_grouphead"],
}
ALL_ARMS = (["arm_vav_multi_all"] + P02_ARMS + P03_ARMS + P04_ARMS + P05_ARMS +
            P06_ARMS + ["arm_orm_grouphead"])
P04_TEMPS = {"arm_p04_t05": 0.5, "arm_p04_t10": 1.0, "arm_p04_t20": 2.0}
R3_TAU = 0.05
YES_STR, NO_STR = "Yes", "No"

# BS-F1 参数
BSF1_MAX_HUNGARIAN_ROWS = 200        # 超出 → 多重集硬等值退化（成本上限）
BSF1_MAX_MATCHED_CELLS = 40_000      # n_pred*n_gt 上限
MAX_NAME_ENGINE_THREADS = 16

_SKIP_MARKER = "__SKIPPED_NO_SCORES__"


def build_orm_prompt(question: str, ddl_schema: str, candidate_sql: str) -> str:
    """ORM user 侧输入——【逐字复制】src/orm_selection.build_orm_prompt
    （= src/label_orm_data.build_orm_prompt；train/infer prompt 逐字节一致硬要求）。"""
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
# P0-2：行列序双不敏感签名（SISelection result_to_normal_form 语义）
# ===================================================================


def rows_to_group_signature_rci(rows: List[List[Any]], truncated: bool = False) -> str:
    """SISelection result_to_normal_form 的签名版：
    ① 剔全 None 行（`t != tuple([None]*len(t))`）；② 行排序；③ 行内值排序
    （frozenset 语义 → 列序不敏感）；重复行保留（与管线包语义一致——SISelection
    的 frozenset 会去重，此为本实现的显式偏离，summary 记录）。
    值编码仍用 AP._sig_value 带类型标签 json（空组/全零组判定函数可直接复用）。"""
    kept = [row for row in rows if not all(v is None for v in row)]
    row_strings = [
        _SEP_VAL.join(sorted(
            json.dumps(_sig_value(v), ensure_ascii=False, separators=(",", ":"))
            for v in row
        ))
        for row in kept
    ]
    sig = SUCCESS_PREFIX + _SEP_ROW.join(sorted(row_strings))
    if truncated:
        sig += _SEP_VAL + _TRUNC_MARK
    return sig


def outcome_signature_rci(outcome: Dict[str, Any]) -> str:
    if not outcome.get("ok"):
        return ERROR_SIG
    return rows_to_group_signature_rci(
        outcome.get("rows") or [], outcome.get("truncated", False))


def build_groups_rci(entries: List[Dict[str, Any]],
                     sigs_per_entry: List[List[str]],
                     votes: Dict[int, int],
                     n_instances: int) -> Tuple[Dict, int, int]:
    """镜像 AS.build_groups，但用 RCI 签名向量分组（其余语义完全一致）。"""
    groups: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    grouped = 0
    excluded = 0
    for ei, cnt in sorted(votes.items()):
        sigs = sigs_per_entry[ei][:n_instances]
        if any(s == ERROR_SIG for s in sigs):
            excluded += cnt
            continue
        grouped += cnt
        key = tuple(sigs)
        g = groups.setdefault(key, {"size": 0, "members": [], "models": set()})
        g["size"] += cnt
        g["members"].append(ei)
        for m in entries[ei]["models"]:
            if m:
                g["models"].add(m)
    return groups, grouped, excluded


# ===================================================================
# P0-5：BS-F1 软比较器（DPC metrics.py 移植 + 列名感知修正）
# ===================================================================


def normalize_value_bsf1(v: Any) -> Any:
    """DPC normalize_value 移植：None/NaN/none/null → None；float round 4；
    int 保 int；bytes 解 utf-8；字符串 strip。已知激进点（DPC 同款）：字符串
    "None"/"null" 也归 None——只影响软分，summary 记录。"""
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return round(float(v), 4)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, (bytes, bytearray)):
        try:
            s = bytes(v).decode("utf-8")
        except UnicodeDecodeError:
            s = bytes(v).hex()
        return s.strip()
    s = str(v).strip()
    if s.lower() in ("nan", "none", "null"):
        return None
    return s


def _multiset_eq(a: List[Tuple], b: List[Tuple]) -> bool:
    if len(a) != len(b):
        return False
    d: Dict[Any, int] = defaultdict(int)
    for e in a:
        d[e] += 1
    for e in b:
        d[e] -= 1
        if d[e] < 0:
            return False
    return True


def align_results_by_name(rows_a: List[List[Any]], names_a: List[Optional[str]],
                          rows_b: List[List[Any]], names_b: List[Optional[str]]
                          ) -> Tuple[List[Tuple], List[Tuple], int]:
    """列名感知对齐（修 DPC 列位置敏感的洞）：
    列名并集（A 序优先、B-only 追加）；无名列（SQLite description None，如表达式）
    落合成名 __c{i} → 自动退化位置对齐；同名重复列取首次出现（已知局限，summary
    记录）。两侧行投影到并集列序（缺失列 → None），返回 (A', B', n_cols)。"""
    na = [n or f"__c{i}" for i, n in enumerate(names_a or [])]
    nb = [n or f"__c{i}" for i, n in enumerate(names_b or [])]
    seen: Dict[str, int] = {}
    union: List[str] = []
    for n in na + nb:
        if n not in seen:
            seen[n] = len(union)
            union.append(n)

    def project(rows: List[List[Any]], names: List[str]) -> List[Tuple]:
        idx: Dict[str, int] = {}
        for i, n in enumerate(names):
            if n not in idx:
                idx[n] = i
        out: List[Tuple] = []
        for r in rows:
            out.append(tuple(
                normalize_value_bsf1(r[idx[n]]) if n in idx else None
                for n in union))
        return out

    return project(rows_a, na), project(rows_b, nb), len(union)


def _hungarian_pure(cost: List[List[float]]) -> Tuple[List[int], List[int]]:
    """纯 Python Kuhn-Munkres（矩形，min-cost assignment；小矩阵兜底用）。
    返回 (row_ind, col_ind)。"""
    n, m = len(cost), len(cost[0]) if cost else 0
    if n == 0 or m == 0:
        return [], []
    # 补方阵：虚拟行/列 cost 置大，避免 0 成本虚拟匹配偷走真实列
    big = 1e9
    size = max(n, m)
    c = [[big] * size for _ in range(size)]
    for i in range(n):
        for j in range(m):
            c[i][j] = cost[i][j]
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [big] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = big
            j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = c[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    row_ind, col_ind = [], []
    for j in range(1, size + 1):
        if p[j] and p[j] - 1 < n and j - 1 < m:
            row_ind.append(p[j] - 1)
            col_ind.append(j - 1)
    return row_ind, col_ind


def _hungarian(cost: List[List[float]]) -> Tuple[List[int], List[int]]:
    try:
        import numpy as np  # noqa: E402
        from scipy.optimize import linear_sum_assignment  # noqa: E402
        a = np.asarray(cost, dtype=np.float64)
        r, c = linear_sum_assignment(a)
        return list(map(int, r)), list(map(int, c))
    except ImportError:
        return _hungarian_pure(cost)


def calculate_soft_f1_name_aware(rows_a: List[List[Any]], names_a: List[Optional[str]],
                                 rows_b: List[List[Any]], names_b: List[Optional[str]]
                                 ) -> Tuple[float, str]:
    """列名感知 BS-F1。返回 (score, method)，method ∈ {bsf1, hard_eq}（超 cap 退化）。"""
    if not rows_a and not rows_b:
        return 1.0, "bsf1"
    if not rows_a or not rows_b:
        return 0.0, "bsf1"
    if len(rows_a) * len(rows_b) > BSF1_MAX_MATCHED_CELLS or \
            max(len(rows_a), len(rows_b)) > BSF1_MAX_HUNGARIAN_ROWS:
        # 成本上限：退化多重集硬等值（对齐投影后比较）
        pa, pb, _ = align_results_by_name(rows_a, names_a, rows_b, names_b)
        return (1.0 if _multiset_eq(pa, pb) else 0.0), "hard_eq"
    pa, pb, n_cols = align_results_by_name(rows_a, names_a, rows_b, names_b)
    if n_cols == 0:
        return 1.0, "bsf1"
    n, m = len(pa), len(pb)
    cost = [[0.0] * m for _ in range(n)]
    for i in range(n):
        ri = pa[i]
        for j in range(m):
            rj = pb[j]
            matches = sum(1 for k in range(n_cols) if ri[k] == rj[k])
            cost[i][j] = 1.0 - matches / n_cols
    row_i, col_j = _hungarian(cost)
    tp = sum(1.0 - cost[r][c] for r, c in zip(row_i, col_j))
    fp = float(n - len(row_i))
    fn = float(m - len(col_j))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0, "bsf1"


# ===================================================================
# 带列名的执行引擎（P0-5 用；AP 引擎只存行值不含列名）
# ===================================================================


def _query_thread_names(sql: str, db_path: str, holder: Dict[str, Any],
                        row_cap: int, max_vm_steps: int) -> None:
    """AP._query_thread 同款（进度回调 + watchdog），额外返回 cursor.description
    列名。连接在线程内开/闭，无跨线程共享。"""
    conn = None
    start = time.perf_counter()
    try:
        if not os.path.exists(db_path):
            holder["result"] = AP._failure(
                f"Database file not found: {db_path}", "db_missing")
            return
        uri = Path(db_path).resolve().as_uri()
        conn = __import__("sqlite3").connect(f"{uri}?mode=ro", uri=True)
        holder["conn"] = conn
        conn.text_factory = lambda b: b.decode(errors="ignore")
        conn.row_factory = lambda _c, row: list(row)
        ticks = [0]

        def _handler() -> int:
            ticks[0] += 1000
            return 1 if ticks[0] >= max_vm_steps else 0

        conn.set_progress_handler(_handler, 1000)
        try:
            cur = conn.execute(sql)
        except __import__("sqlite3").Warning:
            holder["result"] = AP._failure(
                "SQL rejected: multi-statement input", "multi_statement")
            return
        names = [d[0] for d in (cur.description or [])]
        rows: List[List[Any]] = []
        total = 0
        while True:
            batch = cur.fetchmany(1000)
            if not batch:
                break
            for row in batch:
                total += 1
                if len(rows) < row_cap:
                    rows.append(list(row))
        holder["result"] = {
            "ok": True, "rows": rows, "names": names, "row_count": total,
            "truncated": total > row_cap, "error": None, "error_type": None,
            "seconds": round(time.perf_counter() - start, 4),
        }
    except __import__("sqlite3").OperationalError as exc:
        if "interrupt" in str(exc).lower():
            holder["result"] = AP._failure(
                f"Query interrupted after {max_vm_steps} SQLite VM steps: {exc}",
                "interrupted")
        else:
            holder["result"] = AP._failure(str(exc), "sqlite_error")
    except Exception as exc:
        holder["result"] = AP._failure(f"{type(exc).__name__}: {exc}", "sqlite_error")
    finally:
        holder["conn"] = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


class NameEngine:
    """按 (sql, db_path) 缓存 rows+names 的执行器（P0-5 专用；与 AP 引擎同超时
    语义；结果独立缓存，不打进 AP 引擎）。"""

    def __init__(self, threads: int, query_timeout: float,
                 max_vm_steps: int, row_cap: int) -> None:
        self.threads = max(1, int(threads))
        self.query_timeout = float(query_timeout)
        self.max_vm_steps = int(max_vm_steps)
        self.row_cap = int(row_cap)
        self._results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._stats: Dict[str, Dict[str, Any]] = {}

    def get(self, sql: str, db_path: str) -> Dict[str, Any]:
        if (sql or "").strip() == "":
            return AP._EMPTY_SQL_OUTCOME
        return self._results.get((sql, db_path))  # type: ignore

    def run(self, tasks: List[Tuple[str, str]], phase: str) -> None:
        stats = {"tasks": len(tasks), "hits": 0, "executed": 0, "failures": 0,
                 "wall_seconds": 0.0}
        start = time.perf_counter()
        todo: List[Tuple[str, str]] = []
        for sql, db_path in tasks:
            if (sql or "").strip() == "":
                continue
            if (sql, db_path) in self._results:
                stats["hits"] += 1
                continue
            todo.append((sql, db_path))
        done = 0
        if todo:
            with ThreadPoolExecutor(max_workers=self.threads) as ex:
                futs = {ex.submit(self._execute_one, sql, db_path): (sql, db_path)
                        for sql, db_path in todo}
                for fut in as_completed(futs):
                    sql, db_path = futs[fut]
                    try:
                        outcome = fut.result()
                    except Exception as exc:
                        outcome = AP._failure(f"Worker crash: {exc}", "worker_error")
                    self._results[(sql, db_path)] = outcome
                    stats["executed"] += 1
                    if not outcome["ok"]:
                        stats["failures"] += 1
                    done += 1
                    if done % 2000 == 0:
                        print(f"  [name:{phase}] {done}/{len(todo)} executions ...",
                              file=sys.stderr)
        stats["wall_seconds"] = round(time.perf_counter() - start, 2)
        self._stats[phase] = stats

    def _execute_one(self, sql: str, db_path: str) -> Dict[str, Any]:
        holder: Dict[str, Any] = {"conn": None, "result": None}
        t = threading.Thread(
            target=_query_thread_names,
            args=(sql, db_path, holder, self.row_cap, self.max_vm_steps))
        t.start()
        t.join(self.query_timeout)
        if not t.is_alive():
            return holder["result"] if holder["result"] is not None else \
                AP._failure("Worker finished without result", "worker_error")
        if holder["conn"] is not None:
            try:
                holder["conn"].interrupt()
            except Exception:
                pass
        t.join(5.0)
        if t.is_alive():
            return AP._failure(
                f"Wall-clock timeout after {self.query_timeout}s and worker "
                f"unresponsive after interrupt", "worker_hang")
        return AP._failure(f"Wall-clock timeout after {self.query_timeout}s", "timeout")


# ===================================================================
# ORM 打分器：Stub / CPU-HF（零 GPU；分数按 (题,候选) 落盘缓存，支持分片）
# ===================================================================


def temp_scale_p(p: float, t: float) -> float:
    """P_T = σ(σ⁻¹(P)/T)。T=1 恒等；P 夹到 [1e-9,1-1e-9] 防 logit 溢出。"""
    if t == 1.0:
        return p
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    logit = math.log(p / (1.0 - p))
    x = -logit / t
    x = min(max(x, -745.0), 745.0)
    return 1.0 / (1.0 + math.exp(x))


class StubScorer:
    def __init__(self, seed: int, const: Optional[float]) -> None:
        self.seed = seed
        self.const = const

    def score(self, payloads: List[Tuple[int, str, Optional[str]]]) -> List[float]:
        if self.const is not None:
            return [float(self.const)] * len(payloads)
        rng = random.Random(self.seed)
        return [rng.random() for _ in payloads]

    @property
    def stats(self) -> Dict[str, Any]:
        return {"mode": f"stub-{'constant' if self.const is not None else 'random'}"}


class CpuOrmScorer:
    """CPU HF 打分：peft merge_and_unload（bf16 基座）→ 批量前向 → 末位 logits
    提取 P(Yes)。协议与 orm_selection.VllmScorer 完全一致（同 prompt、同 chat
    template、同左截断 2048、同 P(Yes) 公式），数值差异仅来自 vLLM vs HF-CPU 的
    bf16 实现差（~1e-3 量级，crosscheck 核对）。"""

    def __init__(self, base_model: str, orm_checkpoint: str, threads: int,
                 max_length: int, batch_size: int, scores_dir: Path,
                 shard: Optional[Tuple[int, int]],
                 crosscheck_items: Optional[List[Dict[str, Any]]]) -> None:
        import torch  # noqa: E402  真打分才需要 torch/transformers（延迟 import）
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
        from peft import PeftModel  # noqa: E402

        torch.set_num_threads(int(threads))
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length)
        self.scores_dir = scores_dir
        self.shard = shard
        self._torch = torch

        print(f"[orm-v2-cpu] loading tokenizer {base_model} ...", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, local_files_only=True, trust_remote_code=True)
        yes_ids = self.tokenizer.encode(YES_STR, add_special_tokens=False)
        no_ids = self.tokenizer.encode(NO_STR, add_special_tokens=False)
        assert len(yes_ids) == 1 and len(no_ids) == 1, \
            f"Yes/No 必须单 token，实际 {yes_ids}/{no_ids}"
        self.yes_id, self.no_id = yes_ids[0], no_ids[0]

        print("[orm-v2-cpu] loading base model (bf16) ...", file=sys.stderr)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            local_files_only=True, trust_remote_code=True)
        print("[orm-v2-cpu] loading LoRA + merge_and_unload ...", file=sys.stderr)
        model = PeftModel.from_pretrained(base, orm_checkpoint)
        self.model = model.merge_and_unload()
        self.model.eval()
        import gc  # noqa: E402
        del base, model
        gc.collect()

        # 已打分缓存（分片文件 + 历史全量文件）
        self._cache: Dict[str, float] = {}
        self._load_cache()
        # crosscheck 参考表：{(di, 归一化预测 SQL): orm_score}
        self._cross_ref: Dict[Tuple[Any, str], float] = {}
        self._cross_diffs: List[float] = []
        for it in (crosscheck_items or []):
            sql = AP.normalize_for_dedup(it.get("predicted_sql"))
            if sql:
                self._cross_ref[(it.get("dataset_index", it.get("di")), sql)] = \
                    float(it.get("orm_score") or 0.0)
        self._stats: Dict[str, Any] = {
            "mode": "cpu-hf", "n_scored": 0, "n_cache_hits": 0,
            "wall_seconds": 0.0, "prompt_tokens": 0, "crosscheck": None,
        }

    # ---- 缓存 ----

    def _cache_file(self) -> Path:
        if self.shard is not None:
            k, n = self.shard
            return self.scores_dir / f"scores_shard{k}of{n}.json"
        return self.scores_dir / "scores.json"

    def _all_cache_files(self) -> List[Path]:
        if not self.scores_dir.is_dir():
            return []
        return sorted(self.scores_dir.glob("scores*.json"))

    def _load_cache(self) -> None:
        for f in self._all_cache_files():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                self._cache.update(data)
            except Exception as exc:
                print(f"[orm-v2-cpu] WARN cache {f} unreadable: {exc}", file=sys.stderr)

    def _save_cache(self) -> None:
        f = self._cache_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(f)

    # ---- 打分 ----

    def score(self, payloads: List[Tuple[int, str, Optional[str]]],
              di_by_qi: Optional[Dict[int, Any]] = None) -> List[float]:
        """payloads: [(qi, key, prompt)]。返回对齐的 P(Yes)。分片：只打
        index % N == K 的；其余置 NaN（本进程不负责）。缓存命中跳过。"""
        torch = self._torch
        n = len(payloads)
        out = [float("nan")] * n
        todo: List[int] = []
        for i, (qi, key, prompt) in enumerate(payloads):
            if prompt is None:
                continue
            if self.shard is not None:
                k, nn = self.shard
                if i % nn != k:
                    continue
            ck = f"{qi}\t{key}"
            if ck in self._cache:
                out[i] = float(self._cache[ck])
                self._stats["n_cache_hits"] += 1
                continue
            todo.append(i)
        print(f"[orm-v2-cpu] 需要打分 {len(todo)}/{n}（分片={self.shard}，"
              f"缓存命中 {self._stats['n_cache_hits']}）", file=sys.stderr)
        t0 = time.perf_counter()
        n_tokens = 0
        for b0 in range(0, len(todo), self.batch_size):
            idxs = todo[b0:b0 + self.batch_size]
            encs = []
            for i in idxs:
                prompt = payloads[i][2]
                enc = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True, return_dict=True)
                ids = enc["input_ids"][-self.max_length:]
                encs.append((i, ids))
            mx = max(len(ids) for _, ids in encs)
            pad = [ids + [0] * (mx - len(ids)) for _, ids in encs]
            mask = [[1] * len(ids) + [0] * (mx - len(ids)) for _, ids in encs]
            inp = torch.tensor(pad)
            msk = torch.tensor(mask)
            with torch.no_grad():
                logits = self.model(input_ids=inp, attention_mask=msk).logits
            # 每行末位（最后非 pad token）
            last_pos = msk.sum(dim=1) - 1
            rows = logits[torch.arange(len(idxs)), last_pos]
            lp_yes = rows[:, self.yes_id]
            lp_no = rows[:, self.no_id]
            p = torch.sigmoid(lp_yes - lp_no).tolist()
            for (i, ids), pv in zip(encs, p):
                qi, key, _prompt = payloads[i]
                self._cache[f"{qi}\t{key}"] = float(pv)
                out[i] = float(pv)
                n_tokens += len(ids)
                # crosscheck
                if di_by_qi is not None:
                    di = di_by_qi.get(qi)
                    ref = self._cross_ref.get((di, key))
                    if ref is not None:
                        self._cross_diffs.append(abs(float(pv) - ref))
            self._stats["n_scored"] += len(idxs)
            if (b0 // self.batch_size + 1) % 20 == 0 or \
                    b0 + self.batch_size >= len(todo):
                wall = time.perf_counter() - t0
                done = b0 + len(idxs)
                print(f"[orm-v2-cpu] {done}/{len(todo)} ({wall:.0f}s, "
                      f"{done * 560.0 / max(wall, 1e-9):.0f} tok/s 估算)",
                      file=sys.stderr)
                self._save_cache()
        self._stats["wall_seconds"] = round(time.perf_counter() - t0, 2)
        self._stats["prompt_tokens"] = n_tokens
        if self._cross_diffs:
            self._stats["crosscheck"] = {
                "n": len(self._cross_diffs),
                "max_abs_diff": round(max(self._cross_diffs), 6),
                "mean_abs_diff": round(sum(self._cross_diffs) / len(self._cross_diffs), 6),
            }
        return out

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats


# ===================================================================
# 单题裁决：基线 + 6 组臂（全部离线重算，同一批候选）
# ===================================================================


def _rankable_order(groups: Dict, rci: bool = False) -> List[Tuple[Tuple[str, ...], Dict]]:
    return AS.rankable_groups(groups)


def _fallback(entries: List[Dict[str, Any]], votes: Dict[int, int],
              n_used: int, grouped: int, excluded: int) -> Dict[str, Any]:
    rec = AS._fallback_record(entries, votes, n_used, grouped, excluded)
    return rec


def _finish(rec: Dict[str, Any]) -> Dict[str, Any]:
    rec["empty_winner"] = (rec["text"] == "")
    return rec


def arm_p02(entries: List[Dict[str, Any]], groups_rci: Dict, votes: Dict[int, int],
            n_used: int, grouped: int, excluded: int,
            joins_cache: Dict[str, Tuple[int, str]]) -> Dict[str, Any]:
    """P0-2：RCI 签名分组 + MI-VAV 最大组裁决（choose_group_vav 语义照抄）。"""
    if not groups_rci:
        return _fallback(entries, votes, n_used, grouped, excluded)
    ranked = AS.rankable_groups(groups_rci)
    if not ranked:
        return _fallback(entries, votes, n_used, grouped, excluded)
    chosen_key, chosen_g = max(
        ranked, key=lambda km: (int(km[1].get("size", 0)), str(km[0])))
    rec = AS._base_record(entries, chosen_key, chosen_g, "p02_rci",
                          n_used, grouped, excluded, joins_cache)
    rec["rci_group_count"] = len(groups_rci)
    rec["rci_rankable_count"] = len(ranked)
    return rec


def arm_p03(entries: List[Dict[str, Any]], groups: Dict, votes: Dict[int, int],
            n_used: int, grouped: int, excluded: int,
            joins_cache: Dict[str, Tuple[int, str]]) -> Dict[str, Any]:
    """P0-3：SIRIUS 式门控 + 平票兜底链。
    触发 = |top1-top2|<=1 或 top1<2（gated_structural 定义，排序键 (size,str(key))）。
    触发后兜底链：① 跨源支持度（n_models）② 组大小 ③ 保守保留 MI-VAV 胜者——
    仅当竞争组的 (n_models, size) 严格大于 top1 时覆盖，否则保持原胜者。"""
    ranked = AS.rankable_groups(groups)
    if not ranked:
        rec = _fallback(entries, votes, n_used, grouped, excluded)
        rec.update({"gated_triggered": False, "top1_size": None, "top2_size": None,
                    "gated_changed_winner": False,
                    "top1_models": None, "winner_models_gain": None})
        return rec
    ordered = sorted(ranked, key=lambda km: (km[1]["size"], str(km[0])), reverse=True)
    top1_key, top1_g = ordered[0]
    top2 = ordered[1] if len(ordered) > 1 else None
    size1 = top1_g["size"]
    size2 = top2[1]["size"] if top2 is not None else None
    triggered = (size1 < 2) or (top2 is not None and size1 - size2 <= 1)
    if not triggered:
        rec = AS._base_record(entries, top1_key, top1_g, "vav",
                              n_used, grouped, excluded, joins_cache)
        rec.update({"gated_triggered": False, "top1_size": size1, "top2_size": size2,
                    "gated_changed_winner": False,
                    "top1_models": len(top1_g["models"]), "winner_models_gain": 0})
        return rec
    # 兜底链 ①②：max (n_models, size)；③ 保守：严格更优才覆盖
    cand_key, cand_g = max(
        ranked, key=lambda km: (len(km[1]["models"]), km[1]["size"], str(km[0])))
    better = (len(cand_g["models"]), cand_g["size"]) > (len(top1_g["models"]), size1)
    if not better:
        chosen_key, chosen_g = top1_key, top1_g
        source = "vav"
    else:
        chosen_key, chosen_g = cand_key, cand_g
        source = "p03_gated"
    rec = AS._base_record(entries, chosen_key, chosen_g, source,
                          n_used, grouped, excluded, joins_cache)
    rec.update({"gated_triggered": True, "top1_size": size1, "top2_size": size2,
                "gated_changed_winner": str(chosen_key) != str(top1_key),
                "top1_models": len(top1_g["models"]),
                "winner_models_gain": len(chosen_g["models"]) - len(top1_g["models"])})
    return rec


def _instance_empty(sig: str) -> bool:
    """单实例签名是否空结果（0 值 token）。"""
    return len(AP.parse_signature_values(sig)) == 0


def arm_p04(entries: List[Dict[str, Any]], groups: Dict, votes: Dict[int, int],
            n_used: int, grouped: int, excluded: int, temp: float,
            score_map: Dict[int, float],
            joins_cache: Dict[str, Tuple[int, str]]) -> Dict[str, Any]:
    """P0-4：执行分层先置（clean=所有实例非空 > empty=存在空实例；error 层因分组
    前置条件恒空）+ 层内 argmax size×P_T(Yes)。
    参与域 = 全组剔除全零组（all-empty 组保留在 empty 层——GradeSQL"empty 仅在无
    clean 时参与"语义；与 rankable_groups 的差异见 summary）。"""
    if not groups:
        return _fallback(entries, votes, n_used, grouped, excluded)
    kept = {k: g for k, g in groups.items() if not AP.vector_is_all_zero(k)}
    if not kept:
        return _fallback(entries, votes, n_used, grouped, excluded)
    clean = {k: g for k, g in kept.items()
             if not any(_instance_empty(s) for s in k)}
    tier_groups = clean if clean else kept
    tier = "clean" if clean else "empty"

    def p04_key(km: Tuple[Tuple[str, ...], Dict[str, Any]]) -> Tuple:
        rep = AS._group_rep(entries, km[1])
        ei = next(i for i, e in enumerate(entries) if e["key"] == rep["key"])
        p = score_map.get(ei)
        if p is None or (isinstance(p, float) and math.isnan(p)):
            p = 0.0
        return (km[1]["size"] * temp_scale_p(p, temp), km[1]["size"], str(km[0]))

    chosen_key, chosen_g = max(tier_groups.items(), key=p04_key)
    rep = AS._group_rep(entries, chosen_g)
    ei = next(i for i, e in enumerate(entries) if e["key"] == rep["key"])
    p_rep = score_map.get(ei)
    rec = AS._base_record(entries, chosen_key, chosen_g, f"p04_t{temp}",
                          n_used, grouped, excluded, joins_cache)
    rec.update({"p04_tier": tier, "p04_clean_groups": len(clean),
                "p04_empty_groups": len(kept) - len(clean),
                "p04_rep_score": (None if p_rep is None or
                                  (isinstance(p_rep, float) and math.isnan(p_rep))
                                  else p_rep),
                "p04_rep_score_scaled": temp_scale_p(float(p_rep), temp)
                if p_rep is not None and not (isinstance(p_rep, float) and math.isnan(p_rep))
                else None})
    return rec


def arm_p05(entries: List[Dict[str, Any]], votes: Dict[int, int],
            n_used: int, grouped: int, excluded: int, group_level: bool,
            consensus_sql: Optional[str],
            bsf1_scores: Dict[int, float], group_membership: Dict[int, Tuple],
            joins_cache: Dict[str, Tuple[int, str]]) -> Dict[str, Any]:
    """P0-5：BS-F1 软比较器（列名感知）。
    候选级：argmax mean-BS-F1(c, consensus)（tie → 组大小 → -min_sample_idx → key）；
    组级：argmax size×BS-F1(rep(g), consensus)（tie → size → str(key)）。
    consensus = MI-VAV 基线胜者 SQL（基线 fallback 时沿用同一 fallback SQL）。"""
    if consensus_sql is None:
        rec = {"source": "no_pool", "text": None, "votes": 0, "group_key": None,
               "group_size": 0, "instances_used": n_used, "vav_grouped": grouped,
               "vav_excluded": excluded}
        return rec
    if not group_level:
        eligible = [ei for ei, s in bsf1_scores.items()]
        if not eligible:
            return _fallback(entries, votes, n_used, grouped, excluded)
        best = max(eligible, key=lambda ei: (
            bsf1_scores[ei],
            group_membership.get(ei, (0,))[0] if ei in group_membership else 0,
            -entries[ei]["min_sample_idx"], entries[ei]["key"]))
        gsz, gkey = group_membership.get(best, (0, None))
        return {"source": "p05_bsf1", "text": entries[best]["sql_text"],
                "votes": gsz, "group_key": str(gkey), "group_size": gsz,
                "instances_used": n_used, "vav_grouped": grouped,
                "vav_excluded": excluded, "bsf1_score": bsf1_scores[best],
                "winner_models": len(entries[best]["models"]),
                "winner_dual": len(entries[best]["models"]) >= 2}
    # 组级：bsf1_scores 键 = 组 key，值 = (代表 BS-F1, 组大小, 代表 ei)
    if not bsf1_scores:
        return _fallback(entries, votes, n_used, grouped, excluded)
    best_gkey = max(bsf1_scores, key=lambda gk: (
        bsf1_scores[gk][0] * bsf1_scores[gk][1], bsf1_scores[gk][1], str(gk)))
    rep_score = bsf1_scores[best_gkey][0]
    g = group_membership.get(best_gkey)
    if g is None:
        return _fallback(entries, votes, n_used, grouped, excluded)
    rec = AS._base_record(entries, best_gkey, g, "p05_bsf1_group",
                          n_used, grouped, excluded, joins_cache)
    rec["bsf1_score"] = rep_score
    rec["bsf1_group_score"] = g["size"] * rep_score
    return rec


def arm_p06(entries: List[Dict[str, Any]], ranked: List[Tuple[Tuple[str, ...], Dict]],
            votes: Dict[int, int], n_used: int, grouped: int, excluded: int,
            score_map: Dict[int, float],
            joins_cache: Dict[str, Tuple[int, str]]) -> Dict[str, Any]:
    """P0-6：R³ 组级打分 (r_list, r_point) + τ=0.05 决策化。
    u(g)=max pointwise；r_list(g)=#{j: u(g)>u(j)+τ}；r_point(g)=|g|×u(g)；
    (r_list, r_point, size, str(key)) 降序取胜组；组内 pointwise 最高者胜
    （tie → min_sample_idx → key）。"""
    if not ranked:
        return _fallback(entries, votes, n_used, grouped, excluded)

    def u_of(g: Dict[str, Any]) -> float:
        vals = []
        for ei in g["members"]:
            p = score_map.get(ei)
            if p is not None and not (isinstance(p, float) and math.isnan(p)):
                vals.append(float(p))
        return max(vals) if vals else 0.0

    us = {k: u_of(g) for k, g in ranked}
    keys = list(us)

    def r_list(gk: Tuple[str, ...]) -> int:
        return sum(1 for h in keys if h != gk and us[gk] > us[h] + R3_TAU)

    def p06_key(km: Tuple[Tuple[str, ...], Dict[str, Any]]) -> Tuple:
        k, g = km
        return (r_list(k), g["size"] * us[k], g["size"], str(k))

    chosen_key, chosen_g = max(ranked, key=p06_key)
    best_ei = max(chosen_g["members"], key=lambda ei: (
        score_map.get(ei) if score_map.get(ei) is not None and
        not (isinstance(score_map.get(ei), float) and math.isnan(score_map.get(ei)))
        else -1.0,
        -entries[ei]["min_sample_idx"], entries[ei]["key"]))
    rec = AS._base_record(entries, chosen_key, chosen_g, "p06_r3",
                          n_used, grouped, excluded, joins_cache)
    rec.update({"r3_u": us[chosen_key], "r3_r_list": r_list(chosen_key),
                "r3_r_point": chosen_g["size"] * us[chosen_key],
                "r3_group_count": len(ranked),
                "r3_best_member": best_ei})
    return rec


# ===================================================================
# 参数与主流程
# ===================================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="orm_selection_v2：P0-2~P0-6 零训练融合臂离线重算（CPU only）")
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS_MULTI)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_MULTI)
    ap.add_argument("--spider-dir", type=Path, default=DEFAULT_SPIDER_DIR)
    ap.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    ap.add_argument("--orm-checkpoint", default=str(DEFAULT_ORM_CKPT))
    ap.add_argument("--baseline-items", type=Path, default=DEFAULT_BASELINE_MULTI,
                    help="官方 ORM grouphead 基线 items（86.1/83.4 来源），供 "
                         "fixed/broken 对比与分数 crosscheck")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--query-timeout", type=float, default=30.0)
    ap.add_argument("--max-vm-steps", type=int, default=5_000_000)
    ap.add_argument("--row-cap", type=int, default=100_000)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--keep-distinct", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 题（冒烟）")
    ap.add_argument("--arms", default="all",
                    help="逗号分隔：baseline,p02,p03,p04,p05,p06,ormgrouphead 或 all")
    ap.add_argument("--orm-backend",
                    choices=["none", "stub-random", "stub-const", "cpu", "vllm"],
                    default="none",
                    help="vllm = 复用 orm_selection.VllmScorer（86.1/83.4 基线同一打分器，"
                         "GPU；需 envs/vllmenv）")
    ap.add_argument("--stub-const", type=float, default=0.5)
    ap.add_argument("--cpu-threads", type=int, default=32)
    ap.add_argument("--score-batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--score-shard", default=None, help="K N：只打 index%N==K 的候选")
    ap.add_argument("--score-only", action="store_true",
                    help="只打分并落盘缓存后退出（打分对象=所有入组候选+组代表）")
    ap.add_argument("--skip-scoring", action="store_true",
                    help="不打分，只用已有缓存；缺分的 ORM 臂跳过")
    return ap.parse_args(argv)


def _resolve_arms(spec: str) -> List[str]:
    if spec.strip().lower() == "all":
        return ALL_ARMS
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
    seen = []
    for a in out:
        if a not in seen:
            seen.append(a)
    return seen


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    AP.rng = random.Random(args.seed)
    arms = _resolve_arms(args.arms)
    # 零 ORM 阶段（backend=none 且非缓存复用）：显式跳过 ORM 消费臂，防退化为纯
    # size 选择（分数全缺时 size×0 平票 → 最大组 = MI-VAV，结果被误标为
    # orm_grouphead 的历史教训）
    SCORE_ARMS = P04_ARMS + P06_ARMS + ["arm_orm_grouphead"]
    skipped_arms: List[str] = []
    if args.orm_backend == "none" and not args.skip_scoring:
        for a in list(arms):
            if a in SCORE_ARMS:
                arms.remove(a)
                skipped_arms.append(a)
        if skipped_arms:
            print(f"[orm-v2] WARN backend=none：跳过 ORM 消费臂 {skipped_arms} "
                  f"（需先 --orm-backend cpu|vllm 打分）", file=sys.stderr)
    shard = None
    if args.score_shard:
        k, n = args.score_shard.split()
        shard = (int(k), int(n))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = args.out_dir / "scores"
    if shard is not None:
        scores_dir.mkdir(parents=True, exist_ok=True)

    items = AP._load_items(args.items)
    if args.limit:
        items = items[: args.limit]
    print(f"[orm-v2] {len(items)} 题 | arms={arms} | backend={args.orm_backend} | "
          f"shard={shard} | out={args.out_dir}", file=sys.stderr)

    # ---- 每题去重 ----
    entries_by_q: List[List[Dict[str, Any]]] = []
    for item in items:
        entries_by_q.append(AP._dedupe(item.get("candidates") or []))

    # ---- 实例枚举 + Phase 1：全部唯一候选原始 SQL × 实例（签名用）----
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
    print(f"[orm-v2] phase1: {len(phase1_tasks)} 个唯一 (sql, db_path) 任务",
          file=sys.stderr)
    engine.run(phase1_tasks, phase="grouping")
    print(f"[orm-v2] phase1 完成: {engine._stats['grouping']}", file=sys.stderr)

    # ---- 每题：签名向量 + 分组 + 基线 ----
    per_question: List[Dict[str, Any]] = []
    for qi, item in enumerate(items):
        entries = entries_by_q[qi]
        insts = instances_for(item.get("db_id", ""))
        sigs: List[List[str]] = []
        sigs_rci: List[List[str]] = []
        for e in entries:
            if not (e["sql_text"] or "").strip():
                sigs.append([ERROR_SIG] * len(insts))
                sigs_rci.append([ERROR_SIG] * len(insts))
            else:
                sigs.append([AP.outcome_signature(engine.get(e["sql_text"], inst))
                             for inst in insts])
                sigs_rci.append([outcome_signature_rci(engine.get(e["sql_text"], inst))
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
        groups_rci, grouped_rci, excluded_rci = build_groups_rci(
            entries, sigs_rci, votes, len(insts))
        results: Dict[str, Dict[str, Any]] = {}
        # 基线恒算：P0-5 共识、vs 基线、官方基线对比都依赖它（成本为零）
        if "arm_vav_multi_all" in arms or any(a in P05_ARMS for a in arms) or True:
            results["arm_vav_multi_all"] = AS.arm_baseline(
                entries, sigs, votes, insts, joins_cache)
        per_question.append({
            "item": item, "entries": entries, "sigs": sigs, "sigs_rci": sigs_rci,
            "votes": votes, "groups": groups, "groups_rci": groups_rci,
            "grouped": grouped, "excluded": excluded,
            "grouped_rci": grouped_rci, "excluded_rci": excluded_rci,
            "results": results,
            "num_candidates": len(item.get("candidates") or []),
            "num_unique_candidates": len(entries),
            "num_instances": len(insts),
        })

    # ---- 打分对象集合（供 P0-4/P0-6/ormgrouphead）----
    need_p04 = any(a in P04_ARMS for a in arms) or "arm_orm_grouphead" in arms
    need_p06 = any(a in P06_ARMS for a in arms)
    need_scores = need_p04 or need_p06
    scoring_set: Dict[Tuple[int, str], Tuple[int, int]] = {}  # (qi,key)->(ei,type)
    if need_scores:
        for qi, qc in enumerate(per_question):
            ranked = AS.rankable_groups(qc["groups"])
            if need_p04:
                kept_p04 = {k: g for k, g in qc["groups"].items()
                            if not AP.vector_is_all_zero(k)}
                for g in kept_p04.values():
                    rep = AS._group_rep(qc["entries"], g)
                    scoring_set.setdefault((qi, rep["key"]),
                                           (next(i for i, e in enumerate(qc["entries"])
                                                 if e["key"] == rep["key"]), 0))
            if need_p06:
                for _k, g in ranked:
                    for ei in g["members"]:
                        scoring_set.setdefault((qi, qc["entries"][ei]["key"]), (ei, 1))
            for _k, g in ranked:  # grouphead 基线的组代表（rankable 域）
                rep = AS._group_rep(qc["entries"], g)
                scoring_set.setdefault((qi, rep["key"]),
                                       (next(i for i, e in enumerate(qc["entries"])
                                             if e["key"] == rep["key"]), 0))

    # ---- ORM 打分（stub / CPU / 跳过）----
    score_map_by_q: Dict[int, Dict[int, float]] = defaultdict(dict)
    scoring_stats: Dict[str, Any] = {"mode": args.orm_backend,
                                     "n_needed": len(scoring_set),
                                     "n_missing_in_map": None}
    payloads: List[Tuple[int, str, Optional[str]]] = []
    di_by_qi: Dict[int, Any] = {}
    crosscheck_items: List[Dict[str, Any]] = []
    if args.baseline_items.exists():
        try:
            data = json.loads(args.baseline_items.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                data = data["items"]
            crosscheck_items = data if isinstance(data, list) else []
        except Exception as exc:
            print(f"[orm-v2] WARN baseline-items 读取失败: {exc}", file=sys.stderr)

    def _load_cached_scores() -> Dict[str, float]:
        cache: Dict[str, float] = {}
        if scores_dir.is_dir():
            for f in sorted(scores_dir.glob("scores*.json")):
                try:
                    cache.update(json.loads(f.read_text(encoding="utf-8")))
                except Exception as exc:
                    print(f"[orm-v2] WARN cache {f} unreadable: {exc}", file=sys.stderr)
        return cache

    if (need_scores or args.score_only) and not args.skip_scoring:
        ddl_cache: Dict[str, str] = {}
        _loader = None
        if args.orm_backend in ("cpu", "vllm"):
            try:
                from spider_utils import SpiderLoader  # noqa: E402
                _loader = SpiderLoader(str(args.spider_dir))
            except Exception as exc:
                print(f"[orm-v2] WARN SpiderLoader 不可用（ddl 置空）: {exc}",
                      file=sys.stderr)
                _loader = None

        def ddl_for(db_id: str) -> str:
            if db_id not in ddl_cache:
                try:
                    ddl_cache[db_id] = _loader.format_ddl(db_id) if _loader else ""
                except Exception:
                    ddl_cache[db_id] = ""
            return ddl_cache[db_id]

        for (qi, key), (_ei, _typ) in sorted(scoring_set.items()):
            if qi >= len(items):
                continue
            qc = per_question[qi]
            item = qc["item"]
            di_by_qi[qi] = item.get("dataset_index", item.get("di"))
            entry = qc["entries"][_ei]
            if not (entry["sql_text"] or "").strip():
                payloads.append((qi, key, None))
                continue
            if args.orm_backend in ("cpu", "vllm"):
                prompt = build_orm_prompt(item.get("question", ""),
                                          ddl_for(item.get("db_id", "")),
                                          entry["sql_text"])
            else:
                prompt = None
            payloads.append((qi, key, prompt))

    if args.orm_backend in ("stub-random", "stub-const"):
        scorer: Any = StubScorer(
            args.seed, args.stub_const if args.orm_backend == "stub-const" else None)
        scores = scorer.score(payloads)
        for (qi, key, _p), s in zip(payloads, scores):
            ei = scoring_set[(qi, key)][0]
            score_map_by_q[qi][ei] = s
        scoring_stats.update(scorer.stats)
    elif args.orm_backend in ("cpu", "vllm") and args.skip_scoring:
        cache = _load_cached_scores()
        n_hit = n_miss = 0
        for (qi, key), (ei, _typ) in sorted(scoring_set.items()):
            v = cache.get(f"{qi}\t{key}")
            if v is None:
                n_miss += 1
                continue
            score_map_by_q[qi][ei] = float(v)
            n_hit += 1
        scoring_stats.update({"mode": "cache-only", "n_cache_hits": n_hit,
                              "n_missing_in_map": n_miss})
    elif args.orm_backend == "cpu":
        scorer = CpuOrmScorer(args.base_model, args.orm_checkpoint, args.cpu_threads,
                              args.max_length, args.score_batch, scores_dir, shard,
                              crosscheck_items)
        scores = scorer.score(payloads, di_by_qi)
        n_miss = 0
        for (qi, key, _p), s in zip(payloads, scores):
            if isinstance(s, float) and math.isnan(s):
                n_miss += 1
                continue  # 非本分片
            ei = scoring_set[(qi, key)][0]
            score_map_by_q[qi][ei] = s
        scoring_stats.update(scorer.stats)
        scoring_stats["n_scored_in_shard"] = sum(
            1 for s in scores if not (isinstance(s, float) and math.isnan(s)))
        if n_miss > 0:
            print(f"[orm-v2] WARN {n_miss} 个候选不在本分片（shard={shard}）——"
                  f"属正常分片行为", file=sys.stderr)
    elif args.orm_backend == "vllm":
        # 复用 orm_selection.VllmScorer（86.1/83.4 基线同一打分器；vLLM 0.11 batch，
        # LoRA 优先、失败回退 peft merge；单进程全量打分，不支持分片）
        if shard is not None:
            raise SystemExit("[orm-v2] vllm 后端不支持分片（一次 GPU 打分全量即可）")
        import orm_selection as OM  # noqa: E402
        from types import SimpleNamespace  # noqa: E402
        om_args = SimpleNamespace(
            base_model=args.base_model,
            orm_checkpoint=args.orm_checkpoint,
            max_length=args.max_length,
            logprobs_topk=20,
            chunk_size=512,
            enforce_eager=False,
            max_num_seqs=None,
            merge_python=str(PROJECT_ROOT / "envs" / "reasoning3b" / "bin" / "python"),
        )
        scorer = OM.VllmScorer(om_args)
        scores = scorer.score(payloads)
        for (qi, key, _p), s in zip(payloads, scores):
            ei = scoring_set[(qi, key)][0]
            score_map_by_q[qi][ei] = float(s)
        scoring_stats.update(scorer.stats)
        # 分数落盘（与 CPU 路径同一缓存格式，供 stage3 复用）
        cache_out: Dict[str, float] = {}
        for (qi, key, _p), s in zip(payloads, scores):
            cache_out[f"{qi}\t{key}"] = float(s)
        scores_dir.mkdir(parents=True, exist_ok=True)
        (scores_dir / "scores_vllm.json").write_text(
            json.dumps(cache_out, ensure_ascii=False), encoding="utf-8")
        # 与官方基线胜者 orm_score 逐点核对（同打分器 → 期望 ~1e-6 级一致）
        if crosscheck_items:
            di_to_qi = {v: k for k, v in di_by_qi.items()}
            diffs: List[float] = []
            for it in crosscheck_items:
                sql = AP.normalize_for_dedup(it.get("predicted_sql"))
                ref = it.get("orm_score")
                if not sql or ref is None:
                    continue
                qi = di_to_qi.get(it.get("dataset_index", it.get("di")))
                if qi is None:
                    continue
                tup = scoring_set.get((qi, sql))
                if tup is None:
                    continue
                p = score_map_by_q[qi].get(tup[0])
                if p is None:
                    continue
                diffs.append(abs(float(p) - float(ref)))
            if diffs:
                scoring_stats["crosscheck"] = {
                    "n": len(diffs),
                    "max_abs_diff": round(max(diffs), 6),
                    "mean_abs_diff": round(sum(diffs) / len(diffs), 6),
                }
    else:
        scoring_stats["note"] = "backend=none：不打分（零 ORM 臂或缓存复用阶段）"
        if need_scores and not args.skip_scoring:
            print("[orm-v2] WARN 请求了 ORM 臂但 backend=none——这些臂会按缺分"
                  "（0 分）运行，结果不可信！请用 --orm-backend cpu 或先跑分片打分",
                  file=sys.stderr)
    if args.score_only:
        print("[orm-v2] score-only 完成", file=sys.stderr)
        return 0

    # 缺分检查（诚实报告：ORM 臂在分数不全时结果不可信）
    if need_scores and args.orm_backend in ("cpu", "vllm") and not args.skip_scoring:
        missing = [k for k in scoring_set
                   if scoring_set[k][0] not in score_map_by_q[k[0]]]
        scoring_stats["n_missing_in_map"] = len(missing)
        if missing:
            print(f"[orm-v2] WARN 本进程后仍有 {len(missing)} 个候选缺分"
                  f"（分片未跑完？ORM 臂会按 0 分处理——结果不可信，勿用！）",
                  file=sys.stderr)

    # ---- P0-5 前置：共识结果 + 带列名执行 ----
    need_p05 = any(a in P05_ARMS for a in arms)
    name_engine: Optional[NameEngine] = None
    consensus_by_q: Dict[int, Optional[str]] = {}
    if need_p05:
        name_engine = NameEngine(min(args.threads, MAX_NAME_ENGINE_THREADS),
                                 args.query_timeout, args.max_vm_steps, args.row_cap)
        # 共识 SQL = MI-VAV 基线胜者（无组 → fallback_maj 同一 SQL）
        name_tasks: List[Tuple[str, str]] = []
        for qi, qc in enumerate(per_question):
            base_rec = qc["results"].get("arm_vav_multi_all")
            if base_rec is None:
                consensus_by_q[qi] = None
                continue
            if base_rec["source"] in ("no_pool",):
                consensus_by_q[qi] = None
                continue
            ctext = (base_rec["text"] or "").strip()
            consensus_by_q[qi] = ctext
            for inst in instances_for(qc["item"].get("db_id", "")):
                name_tasks.append((ctext, inst))
            # 所有全成功候选的结果（列名）
            for ei, e in enumerate(qc["entries"]):
                if any(s == ERROR_SIG for s in qc["sigs"][ei]):
                    continue
                if not (e["sql_text"] or "").strip():
                    continue
                for inst in instances_for(qc["item"].get("db_id", "")):
                    name_tasks.append((e["sql_text"], inst))
        name_tasks = list(set(name_tasks))
        print(f"[orm-v2] name-engine: {len(name_tasks)} 个 (sql, db_path) 任务",
              file=sys.stderr)
        name_engine.run(name_tasks, phase="names")
        print(f"[orm-v2] name-engine 完成: {name_engine._stats['names']}",
              file=sys.stderr)

    # ---- P0-5：逐题 BS-F1 分数 ----
    p05_cand_scores_by_q: Dict[int, Dict[int, float]] = defaultdict(dict)
    p05_group_scores_by_q: Dict[int, Dict[Tuple, Tuple[float, int, int]]] = defaultdict(dict)
    p05_stats: Dict[str, Any] = {"n_hard_eq_fallback": 0, "n_name_missing": 0,
                                 "n_comparisons": 0}
    if need_p05:
        for qi, qc in enumerate(per_question):
            consensus = consensus_by_q.get(qi)
            insts = instances_for(qc["item"].get("db_id", ""))
            if consensus is None:
                continue
            group_of_ei: Dict[int, Tuple[int, Optional[Tuple]]] = {}
            for gk, g in qc["groups"].items():
                for ei in g["members"]:
                    group_of_ei[ei] = (g["size"], gk)
            for ei, e in enumerate(qc["entries"]):
                if any(s == ERROR_SIG for s in qc["sigs"][ei]):
                    continue
                if not (e["sql_text"] or "").strip():
                    continue
                acc = 0.0
                for inst in insts:
                    c_out = name_engine.get(consensus, inst)  # type: ignore[union-attr]
                    o_out = name_engine.get(e["sql_text"], inst)  # type: ignore[union-attr]
                    if not (c_out and c_out.get("ok")) or not (o_out and o_out.get("ok")):
                        acc += 0.0
                        continue
                    names_c = c_out.get("names")
                    names_o = o_out.get("names")
                    if not names_c and not names_o:
                        p05_stats["n_name_missing"] += 1
                    s, method = calculate_soft_f1_name_aware(
                        c_out["rows"], names_c, o_out["rows"], names_o)
                    if method == "hard_eq":
                        p05_stats["n_hard_eq_fallback"] += 1
                    p05_stats["n_comparisons"] += 1
                    acc += s
                if insts:
                    p05_cand_scores_by_q[qi][ei] = acc / len(insts)
            # 组级分数（rankable 组的代表）
            for gk, g in AS.rankable_groups(qc["groups"]):
                rep = AS._group_rep(qc["entries"], g)
                rep_ei = next(i for i, e2 in enumerate(qc["entries"])
                              if e2["key"] == rep["key"])
                if rep_ei in p05_cand_scores_by_q[qi]:
                    p05_group_scores_by_q[qi][gk] = (
                        p05_cand_scores_by_q[qi][rep_ei], g["size"], rep_ei)

    # ---- 各臂裁决 ----
    for qi, qc in enumerate(per_question):
        entries = qc["entries"]
        votes = qc["votes"]
        n_used = qc["num_instances"]
        joins_cache: Dict[str, Tuple[int, str]] = {}
        grouped, excluded = qc["grouped"], qc["excluded"]
        if any(a in P02_ARMS for a in arms):
            qc["results"]["arm_p02_rci"] = _finish(arm_p02(
                entries, qc["groups_rci"], votes, n_used,
                qc["grouped_rci"], qc["excluded_rci"], joins_cache))
        if any(a in P03_ARMS for a in arms):
            qc["results"]["arm_p03_gate"] = _finish(arm_p03(
                entries, qc["groups"], votes, n_used, grouped, excluded, joins_cache))
        if any(a in P04_ARMS for a in arms) or "arm_orm_grouphead" in arms:
            ranked = AS.rankable_groups(qc["groups"])
            for arm in P04_ARMS:
                if arm not in arms:
                    continue
                t = P04_TEMPS[arm]
                qc["results"][arm] = _finish(arm_p04(
                    entries, qc["groups"], votes, n_used, grouped, excluded, t,
                    score_map_by_q[qi], joins_cache))
            if "arm_orm_grouphead" in arms:
                if ranked:
                    def gh_key(kg: Tuple[Tuple[str, ...], Dict[str, Any]]) -> Tuple:
                        rep = AS._group_rep(entries, kg[1])
                        ei = next(i for i, e in enumerate(entries)
                                  if e["key"] == rep["key"])
                        p = score_map_by_q[qi].get(ei)
                        if p is None:
                            p = 0.0
                        return (kg[1]["size"] * p, kg[1]["size"], str(kg[0]))

                    chosen_key, chosen_g = max(ranked, key=gh_key)
                    rep = AS._group_rep(entries, chosen_g)
                    ei = next(i for i, e in enumerate(entries)
                              if e["key"] == rep["key"])
                    rec = AS._base_record(entries, chosen_key, chosen_g,
                                          "orm_grouphead", n_used, grouped, excluded,
                                          joins_cache)
                    rec["orm_score"] = score_map_by_q[qi].get(ei)
                    qc["results"]["arm_orm_grouphead"] = _finish(rec)
                else:
                    qc["results"]["arm_orm_grouphead"] = _finish(_fallback(
                        entries, votes, n_used, grouped, excluded))
        if any(a in P05_ARMS for a in arms):
            consensus = consensus_by_q.get(qi)
            group_membership: Dict[int, Tuple[int, Optional[Tuple]]] = {}
            for gk, g in qc["groups"].items():
                for ei in g["members"]:
                    group_membership[ei] = (g["size"], gk)
            if "arm_p05_bsf1" in arms:
                qc["results"]["arm_p05_bsf1"] = _finish(arm_p05(
                    entries, votes, n_used, grouped, excluded, False, consensus,
                    p05_cand_scores_by_q[qi], group_membership, joins_cache))
            if "arm_p05_bsf1_group" in arms:
                qc["results"]["arm_p05_bsf1_group"] = _finish(arm_p05(
                    entries, votes, n_used, grouped, excluded, True, consensus,
                    p05_group_scores_by_q[qi], qc["groups"], joins_cache))
        if any(a in P06_ARMS for a in arms):
            ranked = AS.rankable_groups(qc["groups"])
            qc["results"]["arm_p06_r3"] = _finish(arm_p06(
                entries, ranked, votes, n_used, grouped, excluded,
                score_map_by_q[qi], joins_cache))

    # ---- Phase 2：gold 变换后 + 各臂胜者变换后 SQL × 实例 执行 ----
    active_arms = [a for a in arms if a in ALL_ARMS]
    if "arm_vav_multi_all" not in active_arms:
        active_arms.insert(0, "arm_vav_multi_all")  # 基线恒算恒输出（供对比）
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
    print(f"[orm-v2] phase2 完成: {engine._stats['judgment']}", file=sys.stderr)

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

    # ---- 汇总 ----
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
        cell["accuracy"] = round(cell["correct"] / cell["total"], 4) if cell["total"] else 0.0
        cell["winner_sources"] = dict(cell["winner_sources"])
        cells[arm] = cell

    # ---- vs 基线 1：MI-VAV（本进程同口径重算）----
    def _vs(base_correct: List[bool], arm_list: List[str]) -> Dict[str, Dict[str, Any]]:
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
                "baseline_accuracy": cells["arm_vav_multi_all"]["accuracy"],
                "arm_accuracy": cells[arm]["accuracy"],
                "delta": round(cells[arm]["accuracy"] -
                               cells["arm_vav_multi_all"]["accuracy"], 4),
                "fixed": fixed, "broken": broken, "net": fixed - broken,
                "same_right": same_r, "same_wrong": same_w,
                "fixed_indices": f_idx, "broken_indices": b_idx,
            }
        return out

    base_correct = ([qc["results"]["arm_vav_multi_all"]["is_correct"]
                     for qc in per_question]
                    if "arm_vav_multi_all" in active_arms else None)
    vs_mivav: Dict[str, Dict[str, Any]] = {}
    if base_correct is not None:
        vs_mivav = _vs(base_correct, [a for a in active_arms
                                      if a != "arm_vav_multi_all"])

    # ---- vs 基线 2：官方 ORM grouphead（persisted items，86.1/83.4）----
    vs_orm: Optional[Dict[str, Any]] = None
    if args.baseline_items.exists():
        try:
            data = json.loads(args.baseline_items.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                data = data["items"]
            ref_by_q = {int(r.get("dataset_index", r.get("di"))): r
                        for r in data if isinstance(r, dict)}
            base_official = [bool(ref_by_q[int(
                qc["item"].get("dataset_index", qc["item"].get("di")))]["is_correct"])
                if int(qc["item"].get("dataset_index", qc["item"].get("di")))
                in ref_by_q else False
                for qc in per_question]
            n_ref = sum(1 for qc in per_question
                        if int(qc["item"].get("dataset_index", qc["item"].get("di")))
                        in ref_by_q)
            ref_correct = sum(1 for r in ref_by_q.values() if r.get("is_correct"))
            vs_orm = {
                "file": str(args.baseline_items),
                "n_ref": len(ref_by_q),
                "official_accuracy": round(ref_correct / len(ref_by_q), 4)
                if ref_by_q else None,
                "n_compared": n_ref,
                "vs": _vs(base_official, [a for a in active_arms
                                          if a != "arm_orm_grouphead"]),
            }
            # arm_orm_grouphead 本进程 vs 官方文件逐题一致性
            if "arm_orm_grouphead" in active_arms:
                agree = tot = 0
                for qc in per_question:
                    idx = int(qc["item"].get("dataset_index", qc["item"].get("di")))
                    r = ref_by_q.get(idx)
                    if r is None:
                        continue
                    tot += 1
                    if bool(r.get("is_correct")) == \
                            bool(qc["results"]["arm_orm_grouphead"]["is_correct"]):
                        agree += 1
                vs_orm["ormgrouphead_recompute_agreement"] = (
                    round(agree / tot, 4) if tot else None)
        except Exception as exc:
            vs_orm = {"error": str(exc)}

    # ---- 各臂诊断 ----
    analysis: Dict[str, Any] = {}
    if "arm_p03_gate" in active_arms:
        trig = [i for i, qc in enumerate(per_question)
                if qc["results"]["arm_p03_gate"].get("gated_triggered")]
        changed = [i for i in trig
                   if per_question[i]["results"]["arm_p03_gate"].get("gated_changed_winner")]
        analysis["p03_gate"] = {
            "triggered_questions": len(trig),
            "changed_winner": len(changed),
            "changed_improved": sum(
                1 for i in changed
                if base_correct is not None and not base_correct[i] and
                per_question[i]["results"]["arm_p03_gate"]["is_correct"]),
            "changed_regressed": sum(
                1 for i in changed
                if base_correct is not None and base_correct[i] and
                not per_question[i]["results"]["arm_p03_gate"]["is_correct"]),
        }
    if any(a in P04_ARMS for a in active_arms):
        tiers = Counter()
        for qc in per_question:
            for arm in P04_ARMS:
                if arm in active_arms:
                    tiers[qc["results"][arm].get("p04_tier")] += 1
        analysis["p04_tier"] = {"per_arm_tier_counts": dict(tiers),
                                "note": "clean=所有实例非空；empty=存在空实例；"
                                        "error 层恒空（分组前置条件排除 ERROR 候选）"}
    if any(a in P05_ARMS for a in active_arms):
        analysis["p05_bsf1"] = p05_stats
    if "arm_p06_r3" in active_arms:
        r3_u = [qc["results"]["arm_p06_r3"].get("r3_u") for qc in per_question]
        r3_u = [u for u in r3_u if u is not None]
        analysis["p06_r3"] = {
            "tau": R3_TAU,
            "n_questions": len(r3_u),
            "u_mean": round(sum(r3_u) / len(r3_u), 4) if r3_u else None,
        }

    # ---- 输出 ----
    total_wall = sum(v.get("wall_seconds", 0.0) for v in engine._stats.values())
    if name_engine is not None:
        total_wall += sum(v.get("wall_seconds", 0.0)
                          for v in name_engine._stats.values())
    summary = {
        "meta": {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "created_by": "src/orm_selection_v2.py",
            "input_items": str(args.items),
            "output_dir": str(args.out_dir),
            "spider_dir": str(args.spider_dir),
            "baseline_items": str(args.baseline_items),
            "threads": args.threads,
            "max_instances_cap": args.max_instances,
            "keep_distinct": args.keep_distinct,
            "seed": args.seed,
            "limit": args.limit,
            "arms_requested": arms,
            "arms_computed": active_arms,
            "arms_skipped_no_scores": skipped_arms,
            "orm_backend": args.orm_backend,
            "score_shard": shard,
            "scoring_note": (
                "ORM 打分 = CPU HF 前向（bf16，peft merge_and_unload），只取末位 "
                "logits 的 P(Yes)=sigmoid(logit_Yes-logit_No)，prompt/chat template/"
                "左截断 2048 与 orm_selection VllmScorer 完全一致；与 vLLM 基线分"
                "数值差 ~1e-3 量级（crosscheck 核对）。温度缩放 P_T=σ(σ⁻¹(P)/T)。"),
            "semantics": (
                "P0-2: RCI 签名 = 剔全 None 行 + 行排序 + 行内值排序（SISelection "
                "result_to_normal_form 语义；重复行保留——与管线包语义一致，"
                "frozenset 去重为显式偏离），裁决 = MI-VAV 最大组；"
                "P0-3: 触发 |top1-top2|<=1 或 top1<2（gated 定义），兜底链 = "
                "(跨源支持度 n_models, 组大小) 严格更优才覆盖，否则保守保留 "
                "MI-VAV 胜者；"
                "P0-4: 分层 clean(全实例非空)>empty(存在空实例)>error(恒空)，"
                "层内 argmax size×P_T(Yes)，T∈{0.5,1.0,2.0}，参与域 = 全组剔全零"
                "（all-empty 组保留在 empty 层，GradeSQL 语义；与 rankable_groups "
                "差异见 meta.semantics 本段）；"
                "P0-5: score(c)=mean over 实例 BS-F1(result(c), result(consensus))，"
                "consensus=MI-VAV 基线胜者；BS-F1 = DPC 移植（值归一化+匈牙利）"
                "+ 列名对齐（无名列 __c{i} 位置退化；同名重复列取首次出现；"
                "字符串 'none'/'null' 归 None 为 DPC 同款激进点；行数超 cap 退化 "
                "多重集硬等值）；组级变体 = size×BS-F1(rep)；"
                "P0-6: u(g)=组内最高 pointwise P(Yes)；r_list=#{j:u(g)>u(j)+τ}，"
                "τ=0.05 决策化（零训练伪 pairwise：把 R³ 的概率阈值 τ 平移为 "
                "pointwise 边际，见报告）；r_point=|g|×u(g)；"
                "(r_list,r_point,size,str(key)) 字典序取胜组，组内 pointwise 最高"
                "者胜；"
                "判定: 官方 eval_exec_match（postprocess+remove_distinct+"
                "replace_cur_year+result_eq），全实例等价；NO_RESULTS 回退 "
                "arm_maj；空胜者写 SELECT 1"),
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
            "name_exec_phase": name_engine._stats.get("names") if name_engine else None,
            "total_wall_seconds": round(total_wall, 2),
        },
        "scoring_stats": scoring_stats,
        "accuracy": cells,
        "vs_baseline_mivav": vs_mivav,
        "vs_baseline_orm_grouphead_official": vs_orm,
        "analysis": analysis,
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
                "bsf1_score": rec.get("bsf1_score"),
                "bsf1_group_score": rec.get("bsf1_group_score"),
                "gated_triggered": rec.get("gated_triggered"),
                "gated_changed_winner": rec.get("gated_changed_winner"),
                "top1_size": rec.get("top1_size"),
                "top2_size": rec.get("top2_size"),
                "p04_tier": rec.get("p04_tier"),
                "p04_rep_score_scaled": rec.get("p04_rep_score_scaled"),
                "r3_r_list": rec.get("r3_r_list"),
                "r3_r_point": rec.get("r3_r_point"),
                "r3_u": rec.get("r3_u"),
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
        print(f"  {arm:24s} {c['correct']}/{c['total']} ({c['accuracy']:.4f}){tag}")
    if base_correct is not None:
        print("\n=== vs MI-VAV baseline (fixed / broken / net) ===")
        for arm in active_arms:
            if arm == "arm_vav_multi_all":
                continue
            v = vs_mivav[arm]
            print(f"  {arm:24s} fixed={v['fixed']} broken={v['broken']} "
                  f"net={v['net']:+d} delta={v['delta']:+.4f}")
    if vs_orm and "vs" in vs_orm:
        print("\n=== vs official ORM grouphead baseline (fixed / broken / net) ===")
        for arm, v in vs_orm["vs"].items():
            print(f"  {arm:24s} fixed={v['fixed']} broken={v['broken']} "
                  f"net={v['net']:+d} delta={v['delta']:+.4f}")
    print(f"\nsummary -> {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
