#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/gen_hard_trajectories.py — 难题定向补课轨迹合成（路线 B，RevDecomp-SFT 式）。

目标清单 = 无解题 ∪ hard/extra（去重）：
  - 无解题：outputs/adjudicate_soft/group_level_correctness.json 中，某题所有组
    correct 均为 false 的题（候选池无正确答案，103 题）。
  - hard/extra：canonical Spider hardness（eval_hardness 及其 helper，
    逐函数照抄 src/gating_calibrate.py ← tools/original_spider_eval/evaluation.py，
    作用于 data/spider_data/dev.json 的 `sql` 解析树）。

轨迹（RevDecomp 式多步轨迹，SFT 训练数据格式，与 prep_sft_data 一致）：
  user      = ReasoningGeneratorAgent.build_prompt(question, format_ddl(db_id))
              （canonical prompt 模板在本文件内逐字复刻，训练/推理一致）
  assistant = <think>
              1. Schema 链接 / 2. 逻辑分解 / 3. 草稿 SQL / 4. 修订 / 5. 最终 SQL
              </think>
  最终 SQL = 回复中最后一个 ```sql 块；经 spider_utils.DatabaseExecutor 在原始库
  执行并与 gold 对比（compare_execution_results）。验证失败（parse/执行/不匹配）
  的轨迹丢弃，只进 records（审计），不进 trajectories.jsonl。

API 调用模式照 scripts/gen_distill_v3.slurm / src/gen_reasoning_data.py：
  AsyncOpenAI + deepseek-v4-flash（--model 可换 v4-pro）、thinking disabled
  （extra_body）、指数退避重试、prompt cache hit/miss 计费、DEEPSEEK_API_KEY
  从环境变量读取（绝不硬编码）。

分片：--shard k --n-shards N 按 dataset_index 升序 stride 取片（4 进程并行互不
重叠），各自写 records_shard<k>.jsonl；最后 --merge-shards 汇总去重并产出
trajectories.jsonl（仅验证通过）+ summary.json（通过率/成本）。

用法
  # 1) 生成目标清单（纯 CPU，可交互跑）
  python src/gen_hard_trajectories.py --build-targets
  # 2) pilot 20 题（单进程；难题集推荐 pro + 思考 + 修复轮）
  python src/gen_hard_trajectories.py --limit 20 --shard 1 --n-shards 1 \
      --concurrency 4 --model deepseek-v4-pro --thinking-enabled \
      --repair-rounds 2 --max-tokens 8192
  # 3) 全量：4 进程 stride 分片（scripts/gen_hard_traj.slurm），完成后合并
  python src/gen_hard_trajectories.py --shard $SHARD --n-shards 4 --concurrency 8
  python src/gen_hard_trajectories.py --merge-shards
  # 4) 预检（无 API 调用）
  python src/gen_hard_trajectories.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spider_utils import DatabaseExecutor, SpiderLoader, compare_execution_results  # noqa: E402

try:
    # 官方 exec_eval 语义（与候选池裁决同口径，纯标准库 CPU 可用）
    from adjudicate_pool import (  # noqa: E402
        ExecutionEngine,
        _judge_winner,
        list_instances,
        official_transform,
    )
    _AP_AVAILABLE = True
except Exception:
    _AP_AVAILABLE = False

try:
    from openai import AsyncOpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    AsyncOpenAI = None
    _OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# 常量 / 定价
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPIDER_DIR = (
    "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data"
)
DEFAULT_GROUP_CORRECTNESS = (
    "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/"
    "outputs/adjudicate_soft/group_level_correctness.json"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "hard_traj"
DEFAULT_BASE_URL = "https://api.deepseek.com"
MODEL_CHOICES = ("deepseek-v4-flash", "deepseek-v4-pro")

# DeepSeek 定价，USD / 1M tokens（照 src/gen_reasoning_data.py 2026-08 现值，
# 每次跑前以官方定价页 api-docs.deepseek.com/quick_start/pricing 为准）
PRICING_USD_PER_M: Dict[str, Dict[str, float]] = {
    "deepseek-v4-flash": {"input_miss": 0.14, "input_hit": 0.0028, "output": 0.28},
    "deepseek-v4-pro": {"input_miss": 0.435, "input_hit": 0.003625, "output": 0.87},
}
CNY_PER_USD = 7.2  # 成本报告换算口径（说明性，非官方汇率）

# ===================================================================
# canonical Spider hardness 分类器
# （逐函数照抄 src/gating_calibrate.py ←
#   tools/original_spider_eval/evaluation.py，作用于 process_sql 解析树）
# ===================================================================

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')


def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    for k in ('intersect', 'except', 'union'):
        if sql[k] is not None:
            nested.append(sql[k])
    return nested


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])
    return count


def count_component2(sql):
    return len(get_nestedSQL(sql))


def count_others(sql):
    count = 0
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                               [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1
    if len(sql['select'][1]) > 1:
        count += 1
    if len(sql['where']) > 1:
        count += 1
    if len(sql['groupBy']) > 1:
        count += 1
    return count


def eval_hardness(sql):
    count_comp1_ = count_component1(sql)
    count_comp2_ = count_component2(sql)
    count_others_ = count_others(sql)
    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
            (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
        return "medium"
    elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
            (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
            (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
        return "hard"
    else:
        return "extra"


# ===================================================================
# canonical user prompt（逐字复刻 ReasoningGeneratorAgent.build_prompt，
# schema_links/evidence 走默认 None，与评估端/prep_sft_data 完全一致）
# ===================================================================

def build_canonical_prompt(question: str, ddl_schema: str,
                           dialect: str = "sqlite") -> str:
    schema_links_text = "Not provided"
    evidence_text = "Not provided"
    return f"""
Task Overview:
You are a Text-to-SQL Reasoning Generator Agent.
Given a database schema and a natural-language question,
reason about the required tables, columns, joins, filters,
grouping, aggregation, and ordering. Then generate exactly
one valid SQL query.

Database Dialect:
{dialect}

Database Schema (DDL):
{ddl_schema}

Question:
{question}

Optional Schema Links:
{schema_links_text}

Optional External Evidence:
{evidence_text}

Instructions:
1. Use only tables and columns that appear in the schema.
2. Do not invent database objects.
3. Use the minimum number of tables required to answer the question.
4. Do not join a table merely because a foreign-key relationship exists.
5. If all required columns are available in one table, use only that table.
6. Treat Optional Schema Links as hints about relevant columns and tables.
7. Explain the reasoning before producing the final SQL.
8. Generate exactly one final SQL query.
9. Put the final SQL inside a ```sql code block.
10. Do not use Gold SQL or expected answers.
11. The final query must use the {dialect} dialect.
""".strip()


# ===================================================================
# 教师 system prompt（RevDecomp 五段式，段标题按任务规格逐字固定）
# ===================================================================

TEACHER_SYSTEM = """You are a meticulous Text-to-SQL teacher. Given a database schema and a natural-language question, produce a complete RevDecomp-style reasoning trajectory that a student model can learn from. Your response must follow EXACTLY this structure, with the five numbered section headers verbatim:

<think>
1. Schema 链接: [which tables and columns are relevant, and why]
2. 逻辑分解: [decompose the question into sub-steps, e.g. CTEs or derived-table pieces]
3. 草稿 SQL: ```sql ... ```
4. 修订: [re-check the JOIN paths, filter semantics, aggregation, ordering and edge cases; point out at least one real issue found in the draft and how it is fixed; if the draft is already correct, state which edge cases you verified instead]
5. 最终 SQL: ```sql ... ```
</think>

Rules:
- Keep the five section headers (Schema 链接 / 逻辑分解 / 草稿 SQL / 修订 / 最终 SQL) exactly as written. Write the reasoning prose in English; SQL identifiers must come from the schema.
- The draft SQL must be a plausible first attempt, and the revision step must genuinely improve it.
- The final SQL must be valid SQLite, use only tables and columns present in the schema, and exactly answer the question.
- The final SQL must be the LAST ```sql code block of the whole response.
- Output nothing outside the <think>...</think> section.
"""

# 讲解式（混合路线 Part A）：给定 schema + 题目 + 已知正确的目标 SQL，
# 倒推五段式讲解轨迹。修订段只写真实自查要点，不编造错误。
EXPLAIN_SYSTEM = """You are a meticulous Text-to-SQL teacher. You are given a database schema, a natural-language question, and a known-correct SQL query (the target answer). Produce a complete RevDecomp-style reasoning trajectory that explains, step by step, how the target SQL answers the question. Your response must follow EXACTLY this structure, with the five numbered section headers verbatim:

<think>
1. Schema 链接: [which tables and columns the target SQL relies on, and why]
2. 逻辑分解: [decompose the question into sub-steps, e.g. CTEs or derived-table pieces, mirroring the target SQL]
3. 草稿 SQL: ```sql ... ```
4. 修订: [real self-checks: JOIN paths, filter semantics, boundary/edge values, aggregation, ordering, determinism. Do NOT invent errors the draft does not have — if the draft is already correct, state which checks passed and describe any legitimate refinements (e.g. adding ORDER BY for determinism, narrowing SELECT *)]
5. 最终 SQL: ```sql ... ```
</think>

Rules:
- Keep the five section headers (Schema 链接 / 逻辑分解 / 草稿 SQL / 修订 / 最终 SQL) exactly as written. Write the reasoning prose in English; SQL identifiers must come from the schema.
- The 最终 SQL must be EXACTLY the provided target SQL (identical SQL text).
- The trajectory must be self-contained: derive the reasoning from the schema and the question only; do not mention that a target answer was provided, any checker, or this prompt.
- The final SQL must be the LAST ```sql code block of the whole response.
- Output nothing outside the <think>...</think> section.
"""


# ===================================================================
# 目标清单
# ===================================================================

# L7 数据点：无解题的"来源"粗分类（零成本启发式，仅用 gold SQL 与其执行结果）
_GOLD_QUIRK_OPS_RE = re.compile(r"\b(EXCEPT|INTERSECT|UNION)\b", flags=re.IGNORECASE)
_GOLD_DQ_LITERAL_RE = re.compile(r"""(=|!=|>|<|>=|<=|in\s*\(|like)\s*"[^"]+"\s*[)]?""",
                                 flags=re.IGNORECASE)


def classify_unsolved_subtype(gold_sql: str, gold_exec_ok: bool,
                              gold_rows: Optional[int]) -> str:
    """无解题来源标注（启发式，口径写入报告）：
      - gold_exec_fail: gold 本身无法执行（恒定不可验证）
      - gold_quirk_empty_result: gold 可执行但 0 行（题目/库不匹配类怪题，如
        flight_2 大小写不匹配导致 gold 空结果）
      - gold_quirk_syntax: gold 含 EXCEPT/INTERSECT/UNION 或双引号字符串字面量
        （SQLite 引号怪癖，池子系统性不擅长）
      - pool_blindspot: gold 干净且非空但池子 32 候选全错（纯模型盲区）
    """
    if not gold_exec_ok:
        return "gold_exec_fail"
    if gold_rows == 0:
        return "gold_quirk_empty_result"
    if _GOLD_QUIRK_OPS_RE.search(gold_sql) or _GOLD_DQ_LITERAL_RE.search(gold_sql):
        return "gold_quirk_syntax"
    return "pool_blindspot"


def load_unsolved_set(group_path: Path) -> Set[int]:
    """无解题 = group_level_correctness.json 中所有组 correct 均为 false 的题。"""
    data = json.loads(group_path.read_text(encoding="utf-8"))
    per_di: Dict[int, List[bool]] = defaultdict(list)
    for entry in data:
        if not isinstance(entry, dict):
            continue
        di = entry.get("dataset_index")
        if di is None:
            di = entry.get("qi")
        if not isinstance(di, int):
            continue
        per_di[di].append(bool(entry.get("correct")))
    unsolved = {di for di, cs in per_di.items() if cs and not any(cs)}
    return unsolved


def build_targets(spider_dir: str, group_path: Path) -> List[Dict[str, Any]]:
    """无解题 ∪ hard/extra（canonical eval_hardness）→ 去重、验 gold 可执行性。"""
    unsolved = load_unsolved_set(group_path)
    dev_path = Path(spider_dir) / "dev.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8"))

    n_sql_tree_missing = 0
    targets: List[Dict[str, Any]] = []
    for di, item in enumerate(dev):
        tree = item.get("sql")
        if isinstance(tree, dict):
            difficulty = eval_hardness(tree)
        else:
            difficulty = None
            n_sql_tree_missing += 1
        source: List[str] = []
        if di in unsolved:
            source.append("unsolved")
        if difficulty in ("hard", "extra"):
            source.append("hardness")
        if not source:
            continue
        targets.append({
            "di": di,
            "db_id": item.get("db_id", ""),
            "question": item.get("question", ""),
            "gold_sql": item.get("query", ""),
            "difficulty": difficulty or "",
            "source": source,
        })

    # gold 可执行性检查（DatabaseExecutor 在原始库执行 gold；失败则无法验证）
    # + 无解题来源标注（L7 数据点）
    executor = DatabaseExecutor(spider_dir)
    n_unverifiable = 0
    for t in targets:
        res = executor.execute(t["db_id"], t["gold_sql"])
        t["gold_exec_ok"] = bool(res["success"])
        t["gold_exec_error"] = res.get("error") if not res["success"] else None
        t["verifiable"] = bool(res["success"])
        if not res["success"]:
            n_unverifiable += 1
        if "unsolved" in t["source"]:
            t["unsolved_subtype"] = classify_unsolved_subtype(
                t["gold_sql"], bool(res["success"]),
                res.get("row_count") if res.get("success") else None)

    targets.sort(key=lambda t: t["di"])
    print(f"[targets] dev={len(dev)}  unsolved(全组false)={len(unsolved)}  "
          f"hard/extra={sum(1 for t in targets if 'hardness' in t['source'])}  "
          f"union={len(targets)}  gold执行失败={n_unverifiable}  "
          f"sql树缺失={n_sql_tree_missing}", flush=True)
    both = sum(1 for t in targets if len(t["source"]) == 2)
    print(f"[targets] 来源: 仅unsolved={sum(1 for t in targets if t['source'] == ['unsolved'])}  "
          f"仅hardness={sum(1 for t in targets if t['source'] == ['hardness'])}  "
          f"两者={both}", flush=True)
    diff_of_unsolved = Counter(t["difficulty"] for t in targets if "unsolved" in t["source"])
    print(f"[targets] 无解题的难度标注分布: {dict(diff_of_unsolved)}", flush=True)
    sub_all = Counter(t.get("unsolved_subtype") for t in targets if "unsolved" in t["source"])
    sub_em = Counter(t.get("unsolved_subtype") for t in targets
                     if "unsolved" in t["source"] and t["difficulty"] in ("easy", "medium"))
    print(f"[targets] 无解题来源标注(全部103): {dict(sub_all)}", flush=True)
    print(f"[targets] 无解题来源标注(easy/medium 37, L7 数据点): {dict(sub_em)}", flush=True)
    return targets


# ===================================================================
# 最终 SQL 提取（最后一个 ```sql 块）
# ===================================================================

_SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_GENERIC_BLOCK_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)


def extract_final_sql(text: str) -> str:
    blocks = _SQL_BLOCK_RE.findall(text)
    if not blocks:
        blocks = _GENERIC_BLOCK_RE.findall(text)
    if not blocks:
        return ""
    sql = blocks[-1].strip()
    sql = sql.rstrip(";").strip()
    return sql


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # 预检口径启发式（真实 token 以 API usage 为准）


# ===================================================================
# 讲解式目标（混合路线 Part A）：池内已验证正确 SQL → 倒推五段式讲解轨迹
# ===================================================================

class OfficialVerifier:
    """官方 exec_eval 语义验证（与候选池裁决同一口径：postprocess +
    remove_distinct(keep=False) + replace_cur_year + result_eq 含 order_matters
    与列置换容忍，全实例一致才算对）。仅用于讲解式路由——其目标 SQL 的来源
    （group_level_correctness correct 组）本身就是该语义判定的产物。"""

    def __init__(self, spider_dir: str, threads: int = 8,
                 query_timeout: float = 30.0):
        self.engine = ExecutionEngine(threads, query_timeout, 5_000_000, 100_000)
        self._db_dir = Path(spider_dir) / "database"
        self._inst_cache: Dict[str, List[str]] = {}
        self._j_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def instances(self, db_id: str) -> List[str]:
        if db_id not in self._inst_cache:
            self._inst_cache[db_id] = list_instances(
                str(self._db_dir / db_id), db_id, None)
        return self._inst_cache[db_id]

    def verify(self, pred_text: str, gold_text: str, db_id: str) -> Dict[str, Any]:
        key = (pred_text or "", gold_text or "", db_id)
        if key in self._j_cache:
            return self._j_cache[key]
        insts = self.instances(db_id)
        gold_t = official_transform(gold_text, is_pred=False, keep_distinct=False)
        pred_t = official_transform(pred_text, is_pred=True, keep_distinct=False)
        tasks = [(gold_t, inst) for inst in insts] + [(pred_t, inst) for inst in insts]
        self.engine.run(tasks, "hard_traj")
        j = _judge_winner(pred_text, gold_text, insts, self.engine,
                          keep_distinct=False)
        self._j_cache[key] = j
        return j


def verify_final_sql_official(target: Dict[str, Any], final_sql: str,
                              ov: "OfficialVerifier") -> Dict[str, Any]:
    """讲解式路由验证：官方 exec_eval 全实例语义（与池内 SQL 的判定同口径）。"""
    if not final_sql:
        return {"exec_success": None, "exec_error": "no_sql_block_parsed",
                "match": False, "match_reason": None, "failure_stage": None,
                "predicted_row_count": None, "gold_row_count": None,
                "normalized_sql_string_match": None, "official": None}
    j = ov.verify(final_sql, target["gold_sql"], target["db_id"])
    return {
        "exec_success": True, "exec_error": None,
        "match": bool(j["correct"]),
        "match_reason": None if j["correct"] else "official_exec_mismatch",
        "failure_stage": None if j["correct"] else "official_exec",
        "predicted_row_count": None, "gold_row_count": None,
        "normalized_sql_string_match": (
            final_sql.strip().lower()
            == target["gold_sql"].strip().rstrip(";").lower()),
        "official": {
            "gold_exec_error": bool(j["gold_exec_error"]),
            "order_matters": bool(j["order_matters"]),
            "n_instances": len(ov.instances(target["db_id"])),
        },
    }


def build_explain_targets(spider_dir: str, group_path: Path,
                          targets: List[Dict[str, Any]],
                          out_dir: Path) -> List[Dict[str, Any]]:
    """hard/extra 且非无解题（候选池有正确答案，274 题）→ 取池内正确组的
    rep_text（原始 SQL）作为讲解目标。正确性来源 = group_level_correctness 的
    correct 标记（官方 exec_eval 全实例语义，keep_distinct=False，与官方口径
    一致）；轨迹生成后再以同一语义对最终 SQL 复验。"""
    hardness_only = [t for t in targets if t["source"] == ["hardness"]]
    groups = json.loads(group_path.read_text(encoding="utf-8"))
    correct_by_di: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for e in groups:
        if not isinstance(e, dict) or not bool(e.get("correct")):
            continue
        di = e.get("dataset_index")
        if di is None:
            di = e.get("qi")
        if not isinstance(di, int):
            continue
        text = (e.get("rep_text") or "").strip()
        if not text:
            continue
        correct_by_di[di].append({
            "sql": text,
            "models": e.get("models") or [],
            "size": e.get("size"),
        })

    explain: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for t in hardness_only:
        cands = correct_by_di.get(t["di"], [])
        if not cands:
            skipped.append({"di": t["di"], "db_id": t["db_id"],
                            "reason": "no_correct_group"})
            continue
        # 去重保序，取第一条（组序即文件序；全部已官方验证正确）
        seen: Set[str] = set()
        chosen: Optional[Dict[str, Any]] = None
        for c in cands:
            if c["sql"].lower() in seen:
                continue
            seen.add(c["sql"].lower())
            chosen = c
            break
        explain.append({
            "di": t["di"], "db_id": t["db_id"], "question": t["question"],
            "gold_sql": t["gold_sql"], "difficulty": t["difficulty"],
            "source": t["source"],
            "target_sql": chosen["sql"],
            "target_models": chosen["models"],
            "target_group_size": chosen["size"],
            "n_pool_correct_groups": len(cands),
            "correctness_source": "official_exec_eval(group_level_correctness, "
                                  "keep_distinct=False)",
        })
    explain.sort(key=lambda x: x["di"])
    print(f"[explain-targets] hardness-only={len(hardness_only)}  "
          f"usable={len(explain)}  skipped={len(skipped)}", flush=True)
    for s in skipped[:10]:
        print(f"  skip di={s['di']} {s['db_id']}: {s['reason']}", flush=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "definition": "hard/extra 非无解题，目标 SQL = 池内已验证正确组的 rep_text"
                      "（官方 exec_eval 全实例语义来源；轨迹最终 SQL 以同一语义复验）",
        "count": len(explain),
        "skipped": skipped,
        "targets": explain,
    }
    (out_dir / "explain_targets.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[explain-targets] wrote {len(explain)} -> {out_dir / 'explain_targets.json'}",
          flush=True)
    return explain


def _norm_eq(a: str, b: str) -> bool:
    """对齐检查：空白/关键词大小写不敏感（标识符大小写也忽略，仅用于
    '最终 SQL == 目标 SQL' 的契约校验，非答案判定）。"""
    na = re.sub(r"\s+", " ", a.strip().rstrip(";")).lower().strip()
    nb = re.sub(r"\s+", " ", b.strip().rstrip(";")).lower().strip()
    return na == nb


async def process_one_explain(target: Dict[str, Any], ddl: str, client: Any,
                              sem: asyncio.Semaphore, args: argparse.Namespace,
                              ov: "OfficialVerifier",
                              ) -> Dict[str, Any]:
    """讲解式单题：目标 SQL 只进生成侧 prompt（存盘 messages 用干净 canonical
    prompt），最终 SQL 必须 (1) 官方 exec_eval 全实例验证通过 (2) 与目标 SQL
    对齐 (3) 自洽。"""
    rec = make_base_record(target, args.model, args.temperature)
    rec["route"] = "explain"
    rec["target_sql"] = target.get("target_sql")
    user_prompt = build_canonical_prompt(target["question"], ddl)
    gen_user = (user_prompt + "\n\nTarget SQL (the correct answer to explain):\n"
                "```sql\n" + target["target_sql"] + "\n```")
    convo: List[Dict[str, str]] = [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user", "content": gen_user},
    ]
    async with sem:
        if args.sleep > 0:
            await asyncio.sleep(args.sleep)
        last_text = None
        final_ver: Optional[Dict[str, Any]] = None
        for rnd in range(1 + args.repair_rounds):
            # 讲解是格式/对齐任务，恒关思考（省预算、格式稳）
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
            final_ver = verify_final_sql_official(target, final_sql, ov)
            sql_ok = bool(final_ver.get("match") and final_ver.get("exec_success"))
            aligned = sql_ok and _norm_eq(final_sql, target["target_sql"])
            if sql_ok and aligned and trajectory_is_self_contained(text or ""):
                rec["final_matches_target"] = True
                rec["conversation"] = convo
                break
            if rnd < args.repair_rounds:
                if not sql_ok:
                    fb = ("Your final SQL did not pass execution verification on "
                          "the real database (it must produce exactly the same "
                          "result as the target SQL on every database instance). "
                          "Rewrite the complete trajectory with the 最终 SQL "
                          "exactly equal to:\n```sql\n"
                          + target["target_sql"] + "\n```")
                else:
                    fb = ("Your final SQL passed execution verification but "
                          "differs from the target SQL. Rewrite the complete "
                          "trajectory with the 最终 SQL exactly equal to:\n"
                          "```sql\n" + target["target_sql"] + "\n```")
                convo.append({"role": "user", "content": fb})

    rec["response"] = last_text
    rec["final_sql"] = extract_final_sql(last_text or "") or None
    rec["verification"] = final_ver
    rec["final_matches_target"] = bool(
        final_ver and final_ver.get("match") and final_ver.get("exec_success")
        and _norm_eq(rec["final_sql"] or "", target["target_sql"]))
    rec["messages"] = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": (last_text or "").strip()},
    ]
    rec["generation_user_prompt"] = gen_user  # 审计（含目标 SQL）
    if (final_ver and final_ver.get("match") and final_ver.get("exec_success")
            and rec["final_matches_target"]
            and trajectory_is_self_contained(last_text or "")):
        rec["success"] = True
    else:
        if rec.get("conversation") is None:
            rec["conversation"] = convo
        stage = (final_ver or {}).get("failure_stage") or (final_ver or {}).get("exec_error")
        if stage is None and not rec["final_matches_target"]:
            stage = "target_sql_misaligned"
        rec["error"] = f"verify_failed: stage={stage}"
    return rec


# ===================================================================
# 执行验证
# ===================================================================

def verify_final_sql(target: Dict[str, Any], final_sql: str,
                     executor: DatabaseExecutor,
                     gold_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """DatabaseExecutor 执行最终 SQL 并与 gold 对比（项目自定义执行口径）。"""
    ver: Dict[str, Any] = {
        "exec_success": False, "exec_error": None,
        "match": False, "match_reason": None, "failure_stage": None,
        "predicted_row_count": None, "gold_row_count": None,
        "normalized_sql_string_match": None,
    }
    db_id = target["db_id"]
    if not final_sql:
        ver["exec_error"] = "no_sql_block_parsed"
        return ver
    pred = executor.execute(db_id, final_sql)
    if not pred["success"]:
        ver["exec_error"] = f"{pred.get('error_type')}: {pred.get('error')}"[:500]
        return ver
    if db_id not in gold_cache:
        gold_cache[db_id] = executor.execute(db_id, target["gold_sql"])
    gold = gold_cache[db_id]
    if not gold["success"]:
        ver["exec_error"] = f"gold_unexecutable: {gold.get('error')}"[:300]
        return ver
    cmp = compare_execution_results(pred["full_rows"], gold["full_rows"],
                                    target["gold_sql"])
    ver["exec_success"] = True
    ver["match"] = bool(cmp["match"])
    ver["match_reason"] = cmp.get("match_reason")
    ver["failure_stage"] = cmp.get("failure_stage")
    ver["predicted_row_count"] = cmp.get("predicted_row_count")
    ver["gold_row_count"] = cmp.get("gold_row_count")
    ver["normalized_sql_string_match"] = (
        final_sql.strip().lower() == target["gold_sql"].strip().rstrip(";").lower()
    )
    return ver


# ===================================================================
# DeepSeek 调用（AsyncOpenAI + 指数退避，照 gen_reasoning_data.py）
# ===================================================================

async def call_deepseek(client: Any, model: str,
                        messages: List[Dict[str, str]],
                        temperature: float, max_tokens: int,
                        max_retries: int, timeout: int,
                        thinking_enabled: bool,
                        ) -> Tuple[Optional[str], Optional[str], Any, Optional[Exception]]:
    """Returns (content, reasoning_content, usage, error).

    thinking_enabled=False 时用 extra_body 关思考（模型直接输出可见五段式轨迹）；
    thinking_enabled=True 时保留 V4 原生思考（hidden CoT），reasoning_content 单独
    记录，可见 content 仍要求是五段式轨迹（SFT 用的 assistant 文本）。
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "temperature": temperature,
    }
    if not thinking_enabled:
        # V4 关闭思考: OpenAI 兼容格式用 extra_body（官方文档，照项目既有模式）
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            reasoning = getattr(msg, "reasoning_content", None)
            return content, reasoning, resp.usage, None
        except Exception as exc:  # network / 429 / 5xx
            last_error = exc
            if attempt < max_retries:
                wait = min(30.0, 2.0 ** attempt) * (1 + random.random() * 0.25)
                await asyncio.sleep(wait)
    return None, None, None, last_error


def _usage_int(usage: Any, field: str) -> Optional[int]:
    if usage is None:
        return None
    val = getattr(usage, field, None)
    return val if isinstance(val, int) else None


# ===================================================================
# 单题处理
# ===================================================================

def make_base_record(target: Dict[str, Any], model: str,
                     temperature: float) -> Dict[str, Any]:
    return {
        "dataset_index": target["di"],
        "db_id": target["db_id"],
        "question": target["question"],
        "gold_sql": target["gold_sql"],
        "difficulty": target["difficulty"],
        "source": target["source"],
        "model": model,
        "temperature": temperature,
        "success": False,          # 最终验证通过（写 trajectories 的唯一判据）
        "error": None,
        "response": None,
        "final_sql": None,
        "verification": None,
        "messages": None,          # SFT 用 [user(canonical), assistant(最终轨迹)]
        "conversation": None,      # 审计用（含修复轮完整多轮）
        "reasoning_contents": [],  # 思考模式下的 hidden CoT（审计用）
        "n_calls": 0,
        "attempts_used": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "prompt_cache_hit_tokens": None,
        "prompt_cache_miss_tokens": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_repair_feedback(ver: Dict[str, Any]) -> str:
    """验证失败 → 修复轮反馈。给出具体失败点（含行差示例，实测对修复成功率
    关键——v2 带行差反馈 8/20 vs 只给行数 5/20），但要求产出不得引用检查器/
    反馈（自洽轨迹，防止 SFT 数据带答案窥探痕迹）。"""
    if not ver.get("exec_success"):
        problem = f"execution failed: {ver.get('exec_error')}"[:500]
    else:
        problem = (str(ver.get("match_reason") or "") or
                   f"result mismatch (stage {ver.get('failure_stage')})")[:600]
    return (
        "Verification of your final SQL against the real database failed:\n"
        f"{problem}\n\n"
        "Please produce the complete trajectory again (the same five-section "
        "format inside a fresh <think>...</think> block) with a corrected final "
        "SQL. Requirements:\n"
        "- The trajectory must be self-contained: do NOT mention this feedback, "
        "any checker/verifier, or previous attempts.\n"
        "- The 修订 section must identify the schema/query-level issue yourself "
        "(JOIN path, filter semantics, aggregation, ordering, edge cases) and "
        "explain the fix.\n"
        "- The corrected final SQL must be the LAST ```sql code block."
    )


_SELF_CONTAINMENT_VIOLATION_RE = re.compile(
    r"\b(checker|verifier|feedback|previous attempt|earlier (response|answer|attempt)|"
    r"expected result|verification (failed|result|of))\b",
    flags=re.IGNORECASE)
def trajectory_is_self_contained(text: str) -> bool:
    """轨迹文本不得引用检查器/反馈/先前尝试（答案窥探痕迹 QC）。"""
    return not bool(_SELF_CONTAINMENT_VIOLATION_RE.search(text or ""))


def build_polish_feedback() -> str:
    return (
        "Your final SQL is now correct, but the trajectory text references the "
        "checker, feedback, or a previous attempt. Rewrite the COMPLETE trajectory "
        "as a self-contained artifact: same five-section format, same final SQL, "
        "but describe the reasoning and the revision as if derived from the "
        "schema and question alone, with no mention of any external verification."
    )


def _accumulate_usage(rec: Dict[str, Any], usage: Any) -> None:
    rec["prompt_tokens"] = (rec["prompt_tokens"] or 0) + (_usage_int(usage, "prompt_tokens") or 0)
    rec["completion_tokens"] = (rec["completion_tokens"] or 0) + (_usage_int(usage, "completion_tokens") or 0)
    rec["prompt_cache_hit_tokens"] = (rec["prompt_cache_hit_tokens"] or 0) + (_usage_int(usage, "prompt_cache_hit_tokens") or 0)
    rec["prompt_cache_miss_tokens"] = (rec["prompt_cache_miss_tokens"] or 0) + (_usage_int(usage, "prompt_cache_miss_tokens") or 0)


async def process_one(target: Dict[str, Any], ddl: str, client: Any,
                      sem: asyncio.Semaphore, args: argparse.Namespace,
                      executor: DatabaseExecutor,
                      gold_cache: Dict[str, Dict[str, Any]],
                      ) -> Dict[str, Any]:
    rec = make_base_record(target, args.model, args.temperature)
    rec["route"] = "free"
    user_prompt = build_canonical_prompt(target["question"], ddl)
    convo: List[Dict[str, str]] = [
        {"role": "system", "content": TEACHER_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    async with sem:
        if args.sleep > 0:
            await asyncio.sleep(args.sleep)
        # 多尝试 × 修复轮：attempts 条独立轨迹链，验证通过即停；
        # 单链 = 首轮（可选思考）+ repair_rounds 轮修复（关思考、低温度）。
        # 最终只存验证通过的那条自洽轨迹（messages），其余链进 conversation 审计。
        last_text = None
        final_ver: Optional[Dict[str, Any]] = None
        chain_ok = False
        for attempt in range(1, args.attempts + 1):
            convo = [
                {"role": "system", "content": TEACHER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
            for rnd in range(1 + args.repair_rounds):
                think_on = args.thinking_enabled and (
                    args.thinking_all_rounds or rnd == 0)
                temp = args.temperature if rnd == 0 else args.repair_temperature
                text, reasoning, usage, err = await call_deepseek(
                    client, args.model, convo, temp, args.max_tokens,
                    args.max_retries, args.timeout, think_on)
                if err is not None:
                    rec["error"] = f"api_error: {type(err).__name__}: {err}"[:500]
                    return rec
                rec["n_calls"] += 1
                _accumulate_usage(rec, usage)
                if reasoning:
                    rec["reasoning_contents"].append(reasoning)
                convo.append({"role": "assistant", "content": text or ""})
                last_text = text
                final_sql = extract_final_sql(text or "")
                final_ver = verify_final_sql(target, final_sql, executor, gold_cache)
                sql_ok = bool(final_ver.get("match") and final_ver.get("exec_success"))
                if sql_ok and trajectory_is_self_contained(text or ""):
                    chain_ok = True
                    break
                if rnd < args.repair_rounds:
                    fb = (build_polish_feedback() if sql_ok
                          else build_repair_feedback(final_ver))
                    convo.append({"role": "user", "content": fb})
            if chain_ok:
                rec["attempts_used"] = attempt
                rec["conversation"] = convo
                break

    rec["response"] = last_text
    rec["final_sql"] = extract_final_sql(last_text or "") or None
    rec["verification"] = final_ver
    if rec.get("conversation") is None:
        rec["conversation"] = convo
    rec["messages"] = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": (last_text or "").strip()},
    ]
    if chain_ok:
        rec["success"] = True
    else:
        if final_ver and (final_ver.get("match") and final_ver.get("exec_success")):
            rec["error"] = "verify_failed: stage=not_self_contained"
        else:
            stage = (final_ver or {}).get("failure_stage") or (final_ver or {}).get("exec_error")
            rec["error"] = f"verify_failed: stage={stage}"
    return rec


# ===================================================================
# 记录 IO / 续跑
# ===================================================================

def load_done(records_path: Path) -> Set[int]:
    done: Set[int] = set()
    if not records_path.exists():
        return done
    with open(records_path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            di = rec.get("dataset_index")
            if isinstance(di, int):
                done.add(di)
    return done


def compute_cost_usd(records: List[Dict[str, Any]], model: str) -> Tuple[float, int, int, int]:
    p = PRICING_USD_PER_M.get(model, PRICING_USD_PER_M["deepseek-v4-flash"])
    in_hit = in_miss = out = 0
    for r in records:
        if r.get("prompt_tokens") is None:  # API 调用未发生或失败
            continue
        hit = r.get("prompt_cache_hit_tokens")
        miss = r.get("prompt_cache_miss_tokens")
        if isinstance(hit, int) and isinstance(miss, int):
            in_hit += hit
            in_miss += miss
        else:
            in_miss += r.get("prompt_tokens") or 0
        out += r.get("completion_tokens") or 0
    usd = (in_miss / 1e6) * p["input_miss"] + (in_hit / 1e6) * p["input_hit"] \
        + (out / 1e6) * p["output"]
    return usd, in_hit, in_miss, out


def print_run_summary(records: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    n = len(records)
    verified = sum(1 for r in records if r.get("success"))
    api_err = sum(1 for r in records if (r.get("error") or "").startswith("api_error"))
    parse_fail = sum(1 for r in records if r.get("verification", {}).get("exec_error") == "no_sql_block_parsed")
    exec_fail = sum(1 for r in records if r.get("verification", {}).get("exec_error") not in (None, "no_sql_block_parsed"))
    mismatch = sum(1 for r in records if (r.get("verification", {}).get("exec_success")
                                         and not r.get("verification", {}).get("match")))
    usd, in_hit, in_miss, out = compute_cost_usd(records, args.model)
    p = PRICING_USD_PER_M[args.model]
    avg_in = (in_miss + in_hit) / max(1, n)
    avg_out = out / max(1, n)
    avg_calls = sum(r.get("n_calls") or 0 for r in records) / max(1, n)
    print("=" * 70, flush=True)
    print(f"RUN SUMMARY ({args.model}, temperature={args.temperature}, "
          f"thinking={'on(r0)' if args.thinking_enabled else 'off'}, "
          f"repair_rounds={args.repair_rounds}, attempts={args.attempts}, "
          f"shard={args.shard}/{args.n_shards}, limit={args.limit})", flush=True)
    print(f"  attempted        : {n}", flush=True)
    print(f"  verified pass    : {verified}  ({verified / max(1, n):.1%})", flush=True)
    print(f"  fail breakdown   : api_error={api_err} parse_fail={parse_fail} "
          f"exec_fail={exec_fail} mismatch={mismatch}", flush=True)
    print(f"  avg calls/item   : {avg_calls:.2f}", flush=True)
    if args.attempts > 1:
        used = [r.get("attempts_used") or args.attempts for r in records]
        print(f"  attempts used    : avg={sum(used) / max(1, n):.2f} "
              f"(first-attempt pass: {sum(1 for u in used if u == 1)}/{n})", flush=True)
    print(f"  tokens           : prompt={in_miss + in_hit:,} "
          f"(hit {in_hit:,} / miss {in_miss:,}) completion={out:,}", flush=True)
    print(f"  avg tokens/item  : prompt={avg_in:.0f} completion={avg_out:.0f}", flush=True)
    print(f"  unit price       : input ${p['input_miss']}/M (miss), ${p['input_hit']}/M (hit), "
          f"output ${p['output']}/M", flush=True)
    print(f"  cost             : ${usd:.4f} USD ≈ ¥{usd * CNY_PER_USD:.4f} "
          f"(@{CNY_PER_USD} CNY/USD)", flush=True)
    print(f"  cost/verified    : ${usd / max(1, verified):.4f} USD "
          f"({verified} 条验证通过)", flush=True)
    print("=" * 70, flush=True)


# ===================================================================
# 合并分片 / 汇总
# ===================================================================

def merge_shards(out_dir: Path, args: argparse.Namespace) -> int:
    rec_paths = (sorted(out_dir.glob("records_shard*.jsonl"))
                 + sorted(out_dir.glob("explain_records_shard*.jsonl")))
    if not rec_paths:
        print(f"ERROR: no records_shard*.jsonl / explain_records_shard*.jsonl "
              f"found in {out_dir}")
        return 1
    by_di: Dict[int, Dict[str, Any]] = {}
    all_lines: List[Dict[str, Any]] = []  # 含被去重覆盖的行（成本按真实 API 调用计）
    for p in rec_paths:
        with open(p, "r", encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                all_lines.append(rec)
                if isinstance(rec.get("dataset_index"), int):
                    by_di[rec["dataset_index"]] = rec  # 同 di 后者覆盖（--retry-failed 语义）
    all_recs = [by_di[di] for di in sorted(by_di)]
    verified = [r for r in all_recs if r.get("success")]

    records_path = out_dir / "records.jsonl"
    traj_path = out_dir / "trajectories.jsonl"
    with open(records_path, "w", encoding="utf-8") as fh:
        for r in all_recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(traj_path, "w", encoding="utf-8") as fh:
        for r in verified:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    model = all_recs[0]["model"] if all_recs else args.model
    usd, in_hit, in_miss, out = compute_cost_usd(all_lines, model)
    n_targets = 0
    targets_path = out_dir / "targets.json"
    if targets_path.exists():
        payload = json.loads(targets_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("targets", [])
        n_targets = len(payload) if isinstance(payload, list) else 0
    route_stats: Dict[str, Dict[str, int]] = {}
    for route in ("explain", "free"):
        rs = [r for r in all_recs if r.get("route") == route]
        route_stats[route] = {
            "attempted": len(rs),
            "verified": sum(1 for r in rs if r.get("success")),
        }
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "hard trajectory synthesis (RevDecomp-SFT route B)",
        "model": model,
        "temperature": (all_recs[0]["temperature"] if all_recs else None),
        "targets_total": n_targets,
        "attempted": len(all_recs),
        "verified": len(verified),
        "pass_rate": round(len(verified) / max(1, len(all_recs)), 4),
        "route_breakdown": route_stats,
        "fail_breakdown": {
            "api_error": sum(1 for r in all_recs if (r.get("error") or "").startswith("api_error")),
            "parse_fail": sum(1 for r in all_recs if r.get("verification", {}).get("exec_error") == "no_sql_block_parsed"),
            "exec_fail": sum(1 for r in all_recs if r.get("verification", {}).get("exec_error") not in (None, "no_sql_block_parsed")),
            "mismatch": sum(1 for r in all_recs if (r.get("verification", {}).get("exec_success")
                                                    and not r.get("verification", {}).get("match"))),
        },
        "tokens": {"prompt_total": in_miss + in_hit, "prompt_hit": in_hit,
                   "prompt_miss": in_miss, "completion": out},
        "avg_tokens_per_item": {
            "prompt": round((in_miss + in_hit) / max(1, len(all_recs)), 1),
            "completion": round(out / max(1, len(all_recs)), 1),
        },
        "cost": {
            "usd": round(usd, 4),
            "cny": round(usd * CNY_PER_USD, 4),
            "cny_per_usd": CNY_PER_USD,
            "pricing_usd_per_m": PRICING_USD_PER_M[model],
            "usd_per_verified": round(usd / max(1, len(verified)), 4),
        },
        "files": {
            "records": str(records_path),
            "trajectories": str(traj_path),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge] {len(all_recs)} records ({len(verified)} verified) -> "
          f"{traj_path}; summary -> {out_dir / 'summary.json'}", flush=True)
    print(f"[merge] cost ${usd:.4f} ≈ ¥{usd * CNY_PER_USD:.4f} "
          f"(pass rate {len(verified) / max(1, len(all_recs)):.1%})", flush=True)
    return 0


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="难题定向补课轨迹合成（RevDecomp-SFT 路线 B）")
    ap.add_argument("--spider-dir", default=DEFAULT_SPIDER_DIR)
    ap.add_argument("--group-correctness", default=DEFAULT_GROUP_CORRECTNESS,
                    help="outputs/adjudicate_soft/group_level_correctness.json")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--build-targets", action="store_true",
                    help="只生成目标清单 targets.json，不调 API")
    ap.add_argument("--merge-shards", action="store_true",
                    help="合并 records_shard*.jsonl -> records.jsonl + "
                         "trajectories.jsonl + summary.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理 N 题（pilot；0=全部。N>0 时先按 --seed 洗牌取前 N）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--model", choices=MODEL_CHOICES, default="deepseek-v4-flash")
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="采样温度（照 gen_distill_v3 的 0.5）")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking-enabled", action="store_true",
                    help="开启 V4 原生思考（hidden CoT 提难题准确率；可见内容仍须五段式）")
    ap.add_argument("--thinking-all-rounds", action="store_true",
                    help="思考不止首轮：修复轮也开（v2 实测修复轮思考转换率更高，"
                         "但吃输出预算，需配大 --max-tokens）")
    ap.add_argument("--repair-rounds", type=int, default=0,
                    help="验证失败后的修复轮数：把执行/比对失败点反馈给教师重做整条轨迹 "
                         "（每轮多一次 API 调用）")
    ap.add_argument("--repair-temperature", type=float, default=0.3,
                    help="修复轮采样温度（更确定，利于按反馈修正）")
    ap.add_argument("--attempts", type=int, default=1,
                    help="每题最多 N 条独立轨迹链（验证通过即停）；难题集可用 2-3 "
                         "提高每题产出率（yield），单链失败才起新链")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retry-failed", action="store_true",
                    help="重跑本分片已失败（非 verified）的题")
    ap.add_argument("--explain", action="store_true",
                    help="混合路线 Part A：讲解式轨迹（池内正确 SQL 倒推五段式；"
                         "只处理 explain_targets.json）")
    ap.add_argument("--build-explain-targets", action="store_true",
                    help="只构建讲解式目标清单 explain_targets.json，不调 API")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_shards:
        return merge_shards(out_dir, args)

    targets_path = out_dir / "targets.json"
    group_path = Path(args.group_correctness)
    if not group_path.exists():
        print(f"ERROR: group correctness file not found: {group_path}")
        return 1
    if args.build_targets or not targets_path.exists():
        targets = build_targets(args.spider_dir, group_path)
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "definition": "无解题（group_level_correctness 全组 correct=false）"
                          " ∪ hard/extra（canonical eval_hardness on dev.json sql 树）",
            "count": len(targets),
            "counts": {
                "total": len(targets),
                "unsolved": sum(1 for t in targets if "unsolved" in t["source"]),
                "hardness": sum(1 for t in targets if "hardness" in t["source"]),
                "both": sum(1 for t in targets if len(t["source"]) == 2),
                "unverifiable_gold_exec_fail": sum(1 for t in targets if not t["verifiable"]),
            },
            "targets": targets,
        }
        targets_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        print(f"[targets] wrote {len(targets)} targets -> {targets_path}", flush=True)
    else:
        payload = json.loads(targets_path.read_text(encoding="utf-8"))
        targets = payload.get("targets", [])
        print(f"[targets] loaded {len(targets)} targets from {targets_path}", flush=True)

    if args.build_targets:
        print("[targets] done (--build-targets, no API calls).")
        return 0

    # ---- 路由：explain（Part A 讲解式，池内正确 SQL 倒推） vs free（Part B 自由生成，仅无解题）----
    if args.explain:
        explain_path = out_dir / "explain_targets.json"
        if args.build_explain_targets or not explain_path.exists():
            explain = build_explain_targets(args.spider_dir, group_path, targets, out_dir)
        else:
            payload = json.loads(explain_path.read_text(encoding="utf-8"))
            explain = payload.get("targets", [])
            print(f"[explain-targets] loaded {len(explain)} from {explain_path}", flush=True)
        if args.build_explain_targets:
            print("[explain-targets] done (--build-explain-targets, no API calls).")
            return 0
        pending = [t for t in explain if t.get("verifiable", True)]
        records_name = f"explain_records_shard{args.shard}.jsonl"
        route = "explain"
        base = explain
    else:
        # 自由生成（混合路线 Part B）：仅无解题（103，easy/medium 已带 unsolved_subtype 标注）
        base = [t for t in targets if "unsolved" in t["source"]]
        pending = [t for t in base if t.get("verifiable", True)]
        records_name = f"records_shard{args.shard}.jsonl"
        route = "free"
    skipped_unverifiable = len(base) - len(pending)
    if skipped_unverifiable:
        print(f"[gen] route={route}: skip {skipped_unverifiable} unverifiable "
              f"(gold exec fail)", flush=True)

    if args.limit > 0:
        rng = random.Random(args.seed)
        shuffled = pending[:]
        rng.shuffle(shuffled)
        pending = shuffled[: args.limit]
        comp = Counter("/".join(t["source"]) for t in pending)
        print(f"[gen] pilot: seed={args.seed} limit={args.limit} -> "
              f"{len(pending)} targets, 来源构成={dict(comp)}", flush=True)

    if args.n_shards > 1:
        pending = [t for i, t in enumerate(pending) if i % args.n_shards == args.shard - 1]
        print(f"[gen] shard {args.shard}/{args.n_shards}: {len(pending)} targets", flush=True)

    # --- DDL 预取（fail fast，避免 API 白烧） ---
    loader = SpiderLoader(args.spider_dir)
    ddl_cache: Dict[int, str] = {}
    ddl_missing: List[int] = []
    for t in pending:
        if t["di"] in ddl_cache:
            continue
        try:
            ddl_cache[t["di"]] = loader.format_ddl(t["db_id"])
        except Exception as exc:
            ddl_missing.append(t["di"])
            print(f"WARNING: no DDL for di={t['di']} db={t['db_id']!r} ({exc})", flush=True)
    pending = [t for t in pending if t["di"] in ddl_cache]

    records_path = out_dir / records_name
    done = load_done(records_path)
    if args.retry_failed:
        # 只重跑本分片失败的题：失败 di 集合直接作为待处理集合
        failed: Set[int] = set()
        with open(records_path, "r", encoding="utf-8-sig") as fh:
            for line in fh:
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
    print(f"[gen] to process: {len(pending)} (already recorded: "
          f"{sum(1 for t in (base if args.retry_failed else targets) if t['di'] in done)})",
          flush=True)

    est_in = sum(estimate_tokens(build_canonical_prompt(t["question"], ddl_cache[t["di"]]))
                 + estimate_tokens(TEACHER_SYSTEM) for t in pending)
    # RevDecomp 轨迹输出量级（预检口径）；修复轮按满轮估
    est_out = 2500 * len(pending) * (1 + args.repair_rounds)
    p = PRICING_USD_PER_M[args.model]
    est_usd = est_in / 1e6 * p["input_miss"] + est_out / 1e6 * p["output"]
    print(f"[gen] cost estimate (pre-run): ~${est_usd:.2f} USD ≈ ¥{est_usd * CNY_PER_USD:.2f} "
          f"(est prompt {est_in:,} tok + output {est_out:,} tok)", flush=True)

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
    gold_cache: Dict[str, Dict[str, Any]] = {}
    if args.explain and not _AP_AVAILABLE:
        print("ERROR: adjudicate_pool import failed; explain route requires "
              "official exec_eval verification.")
        return 1
    ov = OfficialVerifier(args.spider_dir) if args.explain else None
    client = AsyncOpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)
    sem = asyncio.Semaphore(args.concurrency)

    async def run_all() -> List[Dict[str, Any]]:
        if args.explain:
            tasks = {
                asyncio.create_task(process_one_explain(
                    t, ddl_cache[t["di"]], client, sem, args, ov)): t["di"]
                for t in pending
            }
        else:
            tasks = {
                asyncio.create_task(process_one(
                    t, ddl_cache[t["di"]], client, sem, args,
                    executor, gold_cache)): t["di"]
                for t in pending
            }
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

    print(f"[gen] route={route}: generating {len(pending)} trajectories via "
          f"{args.model} (concurrency={args.concurrency})...", flush=True)
    records = asyncio.run(run_all())
    print_run_summary(records, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
