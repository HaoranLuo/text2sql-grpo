#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/expo_localize.py — EXPO-SQL 错误子句定位子程序（零训练可复用部分）。

来源：EXPO-SQL（arXiv:2606.23693，ACL Findings 2026）附录 Algorithm 1
（clause-wise incremental execution）与 Appendix A.3 SQLite 错误分类表（Table 14）
+ A.4 复杂结构归因策略（Table 15）。论文代码未发布（github.com/jhn25/EXPO-SQL 404），
本模块按论文附录独立复现，并针对 BIRD/SQLite 实测错误信息做了扩展。

职责（纯 CPU、零 GPU、仅标准库，无 sqlglot/torch 依赖）：
  1. localize(sql, db_path=None, error=None) —— 执行失败分支：
     输入错误 SQL + 数据库路径 + 错误信息（缺 error 时自动执行获取），
     输出定位诊断 dict：C_err 根因子句列表 + 锚点子句 + 自然语言诊断 +
     英文修复提示（供修复 prompt 注入）。
  2. incremental_analyze(sql, db_path, gold_cols, gold_rows) —— 结果错误分支：
     Algorithm 1 子句级增量执行 + 10 类 diff_type 分类，输出 C_err。
  3. CLI scan —— 离线诊断：对候选池（outputs/eval_pool_bird/items.json）执行
     失败样本批量定位 + 统计 + 50 条人工抽查样本（outputs/expo_diag/）。

设计要点（与论文对齐）：
  - 逻辑执行顺序：FROM/JOIN → WHERE → GROUP BY → HAVING → SELECT → ORDER BY
    → LIMIT；错误信息回引用沿该顺序向后回溯找根因（论文 Fig.2/Fig.8 例：
    ORDER BY 报 no such column: us.engagement_count → 根因是定义别名 us 的 JOIN）。
  - 复杂结构（Table 15）：嵌套子查询归到外层子句（按文本 span 归属）；UNION/
    INTERSECT/EXCEPT 在边界切开分别分析；CTE 内部不拆（报错锚点落在 WITH 段
    时标注 WITH(CTE)）；窗口函数归 SELECT。
  - SQLite 报错为 prepare 期整句报错（不指明子句），故"锚点子句"用启发式：
    标识符/错误 token 的文本 span 归属 + 逻辑执行顺序取最后一个出现处；
    根因子句再沿执行顺序回溯。
  - 回退纪律：未识别的错误类走 method=fallback（err_clauses 为空），修复链
    注入通用错误文本（与现状等价，无损降级）。

用法：
  # 单条定位（修复链内嵌或人工排查）
  python src/expo_localize.py one --sql "SELECT ..." \
      --db data/bird/bird_dev/dev_20240627/dev_databases/california_schools/california_schools.sqlite
  # 候选池离线诊断（HPC, CPU）
  python src/expo_localize.py scan \
      --items outputs/eval_pool_bird/items.json \
      --db-root data/bird/bird_dev/dev_20240627/dev_databases \
      --out-dir outputs/expo_diag
"""
import argparse
import difflib
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(PROJECT_ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 纯 CPU 导入（adjudicate_pool 无 torch 依赖；执行引擎口径与裁决/修复链一致）
import adjudicate_pool as AP  # noqa: E402

# ---------------------------------------------------------------------------
# 常量：错误分类 / diff 类型 / 子句
# ---------------------------------------------------------------------------

# Table 14 分类（中文标签用于统计与诊断）
CAT_SCHEMA_REF = "schema_reference"    # no such column/table、no such function …
CAT_LOGICAL = "logical_misuse"         # misuse of aggregate、ambiguous column …
CAT_DATA = "data_constraint"           # datatype mismatch、constraint failed …
CAT_SYNTAX = "syntax_error"            # near "…"、unrecognized token、incomplete input
CAT_SETOP = "setop_mismatch"           # UNION 列数不一致、sub-select 列数
CAT_ENGINE = "engine"                  # 执行引擎层（超时/中断/空 SQL/多语句/库缺失）
CAT_OTHER = "other"                    # 未覆盖

LOGICAL_ORDER = ["WITH", "FROM", "JOIN", "WHERE", "GROUP BY", "HAVING",
                 "SELECT", "ORDER BY", "LIMIT"]
_CLAUSE_RANK = {c: i for i, c in enumerate(LOGICAL_ORDER)}

# Table 12：diff_type → 可能错误子句（诊断用语）
DIFF_CLAUSE_HINT = {
    "col_count": "SELECT / FROM / JOIN",
    "col_name": "SELECT / FROM / JOIN",
    "row_order": "ORDER BY",
    "row_dedup": "SELECT (DISTINCT) / GROUP BY",
    "row_subset": "WHERE / JOIN / HAVING / LIMIT（过紧，漏行）",
    "row_superset": "JOIN / WHERE(OR/IN)（过松，多行/笛卡尔积）",
    "row_emptied": "WHERE / JOIN（不匹配）/ HAVING",
    "row_created": "FROM / JOIN",
    "row_disjoint": "FROM / JOIN / WHERE",
    "row_partial": "SELECT（聚合/表达式值错）",
}

_SETOP_WORDS = {"UNION", "INTERSECT", "EXCEPT"}
_JOIN_MODIFIERS = {"INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "NATURAL"}


# ---------------------------------------------------------------------------
# SQL 顶层子句切分（手写扫描器：括号深度 + 引号/注释；容忍语法错 SQL）
# ---------------------------------------------------------------------------

class Segment:
    __slots__ = ("label", "start", "end", "text", "branch")

    def __init__(self, label: str, start: int, end: int, text: str, branch: int):
        self.label = label
        self.start = start
        self.end = end
        self.text = text
        self.branch = branch

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Seg {self.label}@{self.branch} [{self.start}:{self.end}]>"


def _scan_words(sql: str) -> List[Tuple[str, int, int]]:
    """词法扫描：返回 [(word_upper, start, end)]，跳过空白/注释/字符串；
    同时返回每个词所在括号深度在旁路函数中处理。"""
    words: List[Tuple[str, int, int]] = []
    i, n = 0, len(sql)
    depth = 0
    while i < n:
        c = sql[i]
        if c in " \t\r\n\f\v":
            i += 1
            continue
        if c == "-" and i + 1 < n and sql[i + 1] == "-":  # 行注释
            j = sql.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if c == "/" and i + 1 < n and sql[i + 1] == "*":  # 块注释
            j = sql.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == "'":
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if c == '"' or c == "`":
            q = c
            i += 1
            while i < n:
                if sql[i] == q:
                    if i + 1 < n and sql[i + 1] == q:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if c == "[":
            j = sql.find("]", i + 1)
            i = n if j < 0 else j + 1
            continue
        if c == "(":
            depth += 1
            i += 1
            continue
        if c == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (sql[j].isalnum() or sql[j] in "_$"):
                j += 1
            words.append((sql[i:j].upper(), i, j))
            i = j
            continue
        i += 1  # 数字/运算符/其他
    return words


def split_sql_clauses(sql: str) -> Optional[List[List[Segment]]]:
    """把 SQL 切成顶层子句段，按 UNION/INTERSECT/EXCEPT 边界分组（每分支一组）。

    返回 [[Segment, ...], ...]（分支列表；每个分支内按文本顺序排列的段）。
    无法切分（空/异常）返回 None。容忍语法错 SQL（词法层面即可切分）。
    """
    if not sql or not sql.strip():
        return None
    words = _scan_words(sql)
    # 事件：(pos, kind, label, branch)。kind: 'clause' | 'setop'
    events: List[Tuple[int, str, str, int]] = []
    branch = 0
    # 首词为 WITH → CTE 段；主 SELECT 出现前的内容归 WITH
    i = 0
    # 重建每个词对应的括号深度（重扫一遍，O(n) 但简单可靠）
    wd = _word_depths(sql, words)
    while i < len(words):
        w, s, e = words[i]
        d = wd[i]
        if d != 0:
            i += 1
            continue
        if w == "WITH" and not events:
            events.append((s, "clause", "WITH", branch))
            i += 1
            continue
        if w in _SETOP_WORDS:
            events.append((s, "setop", w, branch))
            branch += 1
            i += 1
            continue
        if w == "SELECT":
            events.append((s, "clause", "SELECT", branch))
            i += 1
            continue
        if w == "FROM":
            events.append((s, "clause", "FROM", branch))
            i += 1
            continue
        if w == "WHERE":
            events.append((s, "clause", "WHERE", branch))
            i += 1
            continue
        if w == "GROUP":
            nxt = words[i + 1] if i + 1 < len(words) else None
            if nxt is not None and nxt[0] == "BY" and wd[i + 1] == d:
                events.append((s, "clause", "GROUP BY", branch))
                i += 2
                continue
            i += 1
            continue
        if w == "HAVING":
            events.append((s, "clause", "HAVING", branch))
            i += 1
            continue
        if w == "ORDER":
            nxt = words[i + 1] if i + 1 < len(words) else None
            if nxt is not None and nxt[0] == "BY" and wd[i + 1] == d:
                events.append((s, "clause", "ORDER BY", branch))
                i += 2
                continue
            i += 1
            continue
        if w == "LIMIT":
            events.append((s, "clause", "LIMIT", branch))
            i += 1
            continue
        if w == "JOIN":
            events.append((s, "clause", "JOIN", branch))
            i += 1
            continue
        if w in _JOIN_MODIFIERS:
            # LEFT/RIGHT/FULL/INNER/OUTER/CROSS/NATURAL [OUTER] JOIN：
            # 向后看 1~3 个词，若遇 JOIN 则此处开 JOIN 段并跳过整个短语
            j = i + 1
            saw_join = False
            while j < len(words) and j <= i + 3:
                wj = words[j][0]
                if wj == "JOIN":
                    saw_join = True
                    break
                if wj in _JOIN_MODIFIERS:
                    j += 1
                    continue
                break
            if saw_join:
                events.append((s, "clause", "JOIN", branch))
                i = j + 1
                continue
            i += 1
            continue
        i += 1

    if not events:
        return None
    # 若 SQL 以 SELECT 子查询开头（无任何事件的首词是 ( 等），仍按无子句处理
    branches: Dict[int, List[Segment]] = defaultdict(list)
    positions = events
    for k, (pos, kind, label, br) in enumerate(positions):
        end = positions[k + 1][0] if k + 1 < len(positions) else len(sql)
        if kind == "clause":
            branches[br].append(Segment(label, pos, end, sql[pos:end], br))
    out = [branches[b] for b in sorted(branches)]
    return out if out else None


def _word_depths(sql: str, words: List[Tuple[str, int, int]]) -> List[int]:
    """为 words 中每个词计算其起始位置的括号深度。"""
    depth = 0
    idx = 0
    depths = []
    cursor = 0
    n = len(sql)
    for (_, s, _) in words:
        while cursor < s:
            c = sql[cursor]
            if c == "-" and cursor + 1 < n and sql[cursor + 1] == "-":
                j = sql.find("\n", cursor)
                cursor = n if j < 0 else j + 1
                continue
            if c == "/" and cursor + 1 < n and sql[cursor + 1] == "*":
                j = sql.find("*/", cursor + 2)
                cursor = n if j < 0 else j + 2
                continue
            if c == "'":
                cursor += 1
                while cursor < n:
                    if sql[cursor] == "'":
                        if cursor + 1 < n and sql[cursor + 1] == "'":
                            cursor += 2
                            continue
                        cursor += 1
                        break
                    cursor += 1
                continue
            if c in '"[`':
                q = c
                cursor += 1
                while cursor < n:
                    if sql[cursor] == q:
                        if cursor + 1 < n and sql[cursor + 1] == q:
                            cursor += 2
                            continue
                        cursor += 1
                        break
                    cursor += 1
                continue
            if c == "[":
                j = sql.find("]", cursor + 1)
                cursor = n if j < 0 else j + 1
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth = max(0, depth - 1)
            cursor += 1
        depths.append(depth)
        idx += 1
    return depths


def _segments_flat(branches: List[List[Segment]]) -> List[Segment]:
    return [s for b in branches for s in b]


# ---------------------------------------------------------------------------
# 文本锚定工具
# ---------------------------------------------------------------------------

def _identifier_regex(identifier: str) -> re.Pattern:
    """把 'us.engagement_count' 形式标识符转成可匹配 SQL 文本的正则
    （容忍引用符 ` " ' [ ] 与空格）。"""
    parts = [p for p in identifier.split(".") if p]
    if not parts:
        return re.compile(r"\b\b")  # 永不匹配
    q = r'[\`"\'\[\]]*'
    body = (q + r"\s*\.\s*" + q).join(re.escape(p) for p in parts)
    return re.compile(r"(?<![A-Za-z0-9_$])" + q + body + q +
                      r"(?![A-Za-z0-9_$])", re.IGNORECASE)


def _bare_identifier_regex(name: str) -> re.Pattern:
    q = r'[\`"\'\[\]]*'
    return re.compile(r"(?<![A-Za-z0-9_$])" + q + re.escape(name) + q +
                      r"(?![A-Za-z0-9_$])", re.IGNORECASE)


def _word_regex(word: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9_$])" + re.escape(word) +
                      r"(?![A-Za-z0-9_$])", re.IGNORECASE)


def _segment_at(branches: List[List[Segment]], pos: int) -> Optional[Segment]:
    for seg in _segments_flat(branches):
        if seg.start <= pos < seg.end:
            return seg
    return None


def _last_in_logical_order(segs: List[Segment]) -> Optional[Segment]:
    """按逻辑执行顺序取最后一个子句段（FROM/JOIN 内部按文本序）。"""
    if not segs:
        return None
    return max(segs, key=lambda s: (_CLAUSE_RANK.get(s.label, 99), s.branch, s.start))


def _first_defining_segment(branches: List[List[Segment]], alias: str) -> Optional[Segment]:
    """找第一个定义别名 alias 的 FROM/JOIN 段（文本序；排除 '.' 前缀引用）。"""
    rx = _word_regex(alias)
    for seg in _segments_flat(branches):
        if seg.label not in ("FROM", "JOIN"):
            continue
        for m in rx.finditer(seg.text):
            # 排除以 '.' 结尾的前缀引用（如 `x.alias`）
            if m.start() > 0 and seg.text[m.start() - 1] == ".":
                continue
            return seg
    return None


def _resolve_alias_table(def_seg: Segment, alias: str) -> Optional[str]:
    """把 FROM/JOIN 段里定义的别名解析回表名（如 'FROM superhero h' → 'superhero'；
    'JOIN major AS m ON ...' → 'major'）。失败返回 None。"""
    words = _scan_words(def_seg.text)
    skip = {"AS", "JOIN", "FROM", "INNER", "LEFT", "RIGHT", "FULL", "OUTER",
            "CROSS", "NATURAL", "ON", "USING", "WHERE", "SELECT", "GROUP",
            "ORDER", "HAVING", "LIMIT"}
    alias_up = alias.upper()
    for i, (w, _s, _e) in enumerate(words):
        if w != alias_up:
            continue
        j = i - 1
        while j >= 0 and words[j][0] in skip:
            j -= 1
        if j >= 0:
            return words[j][0]
    return None


def _clauses_containing(branches: List[List[Segment]], pattern: re.Pattern) -> List[Segment]:
    """所有包含 pattern 匹配的子句段（含嵌套子查询 → 按 span 归到外层子句）。"""
    out: List[Segment] = []
    for seg in _segments_flat(branches):
        if pattern.search(seg.text):
            out.append(seg)
    return out


def _dedupe_labels(segs: List[Segment]) -> List[str]:
    """段列表 → 去重子句标签（按逻辑执行顺序 + 分支序）。"""
    seen = set()
    out = []
    for s in sorted(segs, key=lambda x: (_CLAUSE_RANK.get(x.label, 99), x.branch, x.start)):
        if s.label not in seen:
            seen.add(s.label)
            out.append(s.label)
    return out


# ---------------------------------------------------------------------------
# Schema 探测（sqlite_master；按 db 路径缓存）
# ---------------------------------------------------------------------------

_schema_cache: Dict[str, Dict[str, Any]] = {}


def _schema(db_path: str) -> Dict[str, Any]:
    """返回 {tables: {table_name: [col, ...]}, table_list: [...]}；失败返回空。
    额外提供 case-insensitive 索引：tables_lower: {table_lower: {col_lower: col}}。"""
    if db_path in _schema_cache:
        return _schema_cache[db_path]
    info: Dict[str, Any] = {"tables": {}, "table_list": []}
    try:
        uri = Path(db_path).resolve().as_uri()
        con = sqlite3.connect(f"{uri}?mode=ro", uri=True, timeout=5)
        try:
            for (name,) in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%'").fetchall():
                cols = [r[1] for r in con.execute(f"PRAGMA table_info({name})").fetchall()]
                info["tables"][name] = cols
                info["table_list"].append(name)
        finally:
            con.close()
    except Exception:
        pass
    info["tables_lower"] = {
        t.lower(): {c.lower(): c for c in cols}
        for t, cols in info["tables"].items()
    }
    info["table_lower_to_name"] = {t.lower(): t for t in info["tables"]}
    _schema_cache[db_path] = info
    return info


def _tables_with_col(schema: Dict[str, Any], name: str) -> List[str]:
    """case-insensitive：包含列 name 的表名列表（原始大小写）。"""
    key = (name or "").lower()
    out = []
    for t_lower, cols in schema.get("tables_lower", {}).items():
        if key in cols:
            out.append(schema.get("table_lower_to_name", {}).get(t_lower, t_lower))
    return out


def _col_suggestions(schema: Dict[str, Any], table: str, name: str,
                     limit: int = 3) -> List[str]:
    cols = schema.get("tables", {}).get(table)
    if not cols:
        cols = list((schema.get("tables_lower", {}) or {}).get(table.lower(), {}).values())
    return _similar(cols, name, limit)


def _similar(names: Sequence[str], target: str, limit: int = 3) -> List[str]:
    target = (target or "").lower()
    scored = sorted(((difflib.SequenceMatcher(None, target, n.lower()).ratio(), n)
                     for n in names), reverse=True)
    return [n for r, n in scored[:limit] if r >= 0.5]


def _referenced_tables(branches: List[List[Segment]], schema: Dict[str, Any]) -> List[str]:
    """FROM/JOIN 段中出现的 schema 表名（词边界匹配，容忍引号）。"""
    tables = []
    join_text = " ".join(s.text for s in _segments_flat(branches)
                         if s.label in ("FROM", "JOIN"))
    for t in schema.get("table_list", []):
        if _word_regex(t).search(join_text):
            tables.append(t)
    return tables


# ---------------------------------------------------------------------------
# 错误信息分类（Table 14 + BIRD 实测扩展）
# ---------------------------------------------------------------------------

def _extract_error_identifier(msg: str, pattern: str, strip_quotes: bool = True) -> Optional[str]:
    m = re.search(pattern, msg, re.IGNORECASE)
    if not m:
        return None
    v = m.group(1)
    if strip_quotes:
        v = v.strip('"\'`[]')
    return v


def classify_error(error: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """把错误信息归类为 (category, payload)。payload 含 anchor 关键词等。
    未识别 → (CAT_OTHER, None)。"""
    msg = (error or "").strip()
    low = msg.lower()
    # ---- 语法错误 ----
    m = re.search(r'near\s+"(.*?)":\s*syntax error', low, re.DOTALL)
    if m:
        return CAT_SYNTAX, {"kind": "near_token", "token": m.group(1)}
    m = re.search(r'unrecognized token:\s*"(.*?)"', low, re.DOTALL)
    if m:
        return CAT_SYNTAX, {"kind": "near_token", "token": m.group(1)}
    if "incomplete input" in low:
        return CAT_SYNTAX, {"kind": "incomplete"}
    if "parser stack overflow" in low:
        return CAT_SYNTAX, {"kind": "generic"}
    if re.search(r'keyword ".*" reserved|syntax error', low):
        return CAT_SYNTAX, {"kind": "generic"}
    # ---- Schema Reference ----
    m = re.search(r"no such column:\s*([^\n]+)", low)
    if m:
        ident = m.group(1).strip()
        ident = re.sub(r"\s+$", "", ident)
        return CAT_SCHEMA_REF, {"kind": "no_col",
                                "ident": ident.strip('"\'`[] ')}
    m = re.search(r"no such table:\s*([^\n]+)", low)
    if m:
        ident = m.group(1).strip()
        ident = re.sub(r"\s+$", "", ident)
        return CAT_SCHEMA_REF, {"kind": "no_table",
                                "ident": ident.strip('"\'`[] ')}
    m = re.search(r"no such function:\s*([^\s,;]+)", low)
    if m:
        return CAT_SCHEMA_REF, {"kind": "no_func", "func": m.group(1).strip('"\'`')}
    m = re.search(r"no such collation sequence:\s*([^\s,;]+)", low)
    if m:
        return CAT_SCHEMA_REF, {"kind": "collation", "ident": m.group(1).strip('"\'`')}
    m = re.search(r"(\d+(?:st|nd|rd|th))\s+order by term out of range", low)
    if m:
        return CAT_SCHEMA_REF, {"kind": "order_range", "nth": m.group(1)}
    if "order by term does not match any column" in low:
        return CAT_SCHEMA_REF, {"kind": "order_no_match"}
    m = re.search(r"no query solution", low)
    if m:
        return CAT_SCHEMA_REF, {"kind": "no_query_solution"}
    # ---- Logical Misuse ----
    m = re.search(r"misuse of aggregate(?: function)?:?\s*([A-Za-z_]\w*)\s*\(?", low)
    if m:
        return CAT_LOGICAL, {"kind": "misuse_agg", "func": m.group(1)}
    m = re.search(r"misuse of window function\s+([A-Za-z_]\w*)\s*\(?", low)
    if m:
        return CAT_LOGICAL, {"kind": "agg_window", "func": m.group(1)}
    m = re.search(r"wrong number of arguments to function\s+([A-Za-z_]\w*)\s*\(",
                  low)
    if m:
        return CAT_LOGICAL, {"kind": "wrong_args", "func": m.group(1)}
    if "having clause on a non-aggregate query" in low:
        return CAT_LOGICAL, {"kind": "having_nonagg"}
    m = re.search(r"cannot join using column (\S+?)\s*-\s*column not present in both tables",
                  low)
    if m:
        return CAT_SCHEMA_REF, {"kind": "using_col", "col": m.group(1).strip('`"\'')}
    if "escape expression must be a single character" in low:
        return CAT_SYNTAX, {"kind": "escape_char"}
    m = re.search(r"unknown join type:\s*(\w+)", low)
    if m:
        return CAT_DATA, {"kind": "unknown_join", "op": m.group(1)}
    if "distinct is not supported for window functions" in low:
        return CAT_LOGICAL, {"kind": "distinct_window"}
    if "incorrect number of bindings" in low:
        return CAT_SYNTAX, {"kind": "bindings"}
    if re.search(r"\btable .* already exists\b", low):
        return CAT_OTHER, {"kind": "non_select_sql"}
    if "recursion limit" in low:
        return CAT_SYNTAX, {"kind": "generic"}
    m = re.search(r"order by clause should come after (union(?: all)?|intersect|except) not before",
                  low)
    if m:
        return CAT_SYNTAX, {"kind": "order_before_union", "op": m.group(1)}
    if "a join clause is required before on" in low:
        return CAT_SYNTAX, {"kind": "on_without_join"}
    m = re.search(r"circular reference:\s*([^\s,;]+)", low)
    if m:
        return CAT_SCHEMA_REF, {"kind": "circular_ref",
                                "ident": m.group(1).strip('"\'`')}
    m = re.search(r"aggregate functions are not allowed in ([^\n]+)", low)
    if m:
        return CAT_LOGICAL, {"kind": "agg_forbidden", "where": m.group(1).strip()}
    m = re.search(r"ambiguous column name:\s*([^\s,;]+)", low)
    if m:
        return CAT_LOGICAL, {"kind": "ambiguous",
                             "ident": _extract_error_identifier(
                                 msg, r"ambiguous column name:\s*([^\s,;]+)", True)}
    m = re.search(r"window functions are not allowed in ([^\n]+)", low)
    if m:
        return CAT_LOGICAL, {"kind": "window_forbidden", "where": m.group(1).strip()}
    m = re.search(r"misuse of aliased identifier", low)
    if m:
        return CAT_LOGICAL, {"kind": "alias_misuse"}
    m = re.search(r"(\w+\(\)) may not be used with the window frame", low)
    if m:
        return CAT_LOGICAL, {"kind": "agg_window", "func": m.group(1)}
    # ---- Data & Constraint ----
    if "datatype mismatch" in low:
        return CAT_DATA, {"kind": "datatype"}
    m = re.search(r"(NOT NULL|UNIQUE|CHECK|FOREIGN KEY) constraint failed", low)
    if m:
        return CAT_DATA, {"kind": "constraint", "type": m.group(1)}
    if "right and full outer joins are not currently supported" in low:
        return CAT_DATA, {"kind": "outer_join"}
    # ---- Set-Op 结构错 ----
    if re.search(r"selects to the left and right of (\w+)", low):
        m = re.search(r"selects to the left and right of (\w+)", low)
        return CAT_SETOP, {"kind": "union_cols", "op": m.group(1)}
    m = re.search(r"sub-select returns (\d+) columns - expected (\d+)", low)
    if m:
        return CAT_SETOP, {"kind": "subselect_cols", "n": int(m.group(1)),
                           "m": int(m.group(2))}
    if "distinct aggregates must have exactly one argument" in low:
        return CAT_LOGICAL, {"kind": "distinct_agg_args"}
    if "too many terms in compound select" in low:
        return CAT_SETOP, {"kind": "compound_terms"}
    return CAT_OTHER, None


# ---------------------------------------------------------------------------
# 定位主函数
# ---------------------------------------------------------------------------

def _default_loc(error: str, category: str, method: str, sql: str,
                 note: str = "") -> Dict[str, Any]:
    return {
        "ok": False,
        "category": category,
        "method": method,
        "err_clauses": [],
        "anchor_clause": None,
        "anchors": [],
        "root_cause": note or "（未定位到具体子句）",
        "diagnosis": "",
        "hint_en": "",
        "details": {},
    }


def _make_loc(ok: bool, category: str, method: str, err_clauses: List[str],
              anchor_clause: Optional[str], anchors: List[str], root_cause: str,
              diagnosis: str, hint_en: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": ok,
        "category": category,
        "method": method,
        "err_clauses": err_clauses,
        "anchor_clause": anchor_clause,
        "anchors": anchors,
        "root_cause": root_cause,
        "diagnosis": diagnosis,
        "hint_en": hint_en,
        "details": details or {},
    }


def _seg_excerpt(seg: Segment, width: int = 60) -> str:
    t = re.sub(r"\s+", " ", seg.text).strip()
    return (t[:width] + "…") if len(t) > width else t


def _clause_names(segs: List[Segment]) -> str:
    labels = []
    for s in segs:
        nm = s.label
        cnt = 1
        while nm in labels or any(l.startswith(nm + "#") for l in labels):
            nm = f"{s.label}#{cnt}"
            cnt += 1
        labels.append(nm)
    return ", ".join(labels) if labels else "?"


def _find_token_segment(branches: List[List[Segment]], token: str) -> Optional[Segment]:
    """在 SQL 中定位 near-token：多候选变体（原样/去引号/去方括号），
    取【第一次】出现（SQLite 在首个解析失败点报 near-token）。"""
    if not token:
        return None
    variants = []
    for t in [token, token.replace('""', '"'),
              token.replace("[", "").replace("]", ""),
              token.strip('"\'`[]')]:
        if t and t not in variants:
            variants.append(t)
    segs = _segments_flat(branches)
    for v in variants:
        pos = -1
        for seg in segs:
            m = re.search(re.escape(v), seg.text, re.IGNORECASE)
            if m:
                pos = seg.start + m.start()
                return _segment_at(branches, pos)
    # 拆词匹配（token 含空格时）
    for seg in segs:
        for m in re.finditer(r"\S+", seg.text):
            if m.group().lower() == token.lower():
                return seg
    return None


def _err_clauses_with_label(err_clauses: List[str]) -> str:
    return "+".join(err_clauses) if err_clauses else "(未定位)"


def localize(sql: str, db_path: Optional[str] = None, error: Optional[str] = None,
             error_type: Optional[str] = None) -> Dict[str, Any]:
    """EXPO-SQL 执行失败分支定位。返回诊断 dict。

    参数：
      sql       失败 SQL 原文（用于子句切分/锚定）
      db_path   数据库文件路径（schema 探测；error 缺失时用于执行取错误）
      error     错误信息文本（已有则直接使用，不再执行）
      error_type 可选（adjudicate_pool 引擎的错误类型）
    """
    sql = (sql or "").strip()
    if not sql:
        return _default_loc(error or "", CAT_ENGINE, "engine", sql,
                            "空 SQL（不可执行）")

    # 引擎层错误直接透传（不进行子句定位）
    if error_type in ("empty_sql", "multi_statement", "db_missing", "interrupted",
                      "timeout", "worker_hang", "worker_error"):
        return _default_loc(error or "", CAT_ENGINE, "engine", sql,
                            f"执行引擎层错误（{error_type}），非子句级 SQL 错误")

    if not error and db_path:
        error = _execute_get_error(sql, db_path)
        if error is None:
            # 执行成功 → 无错误可定位
            return _make_loc(False, CAT_OTHER, "no_error", [], None, [],
                             "SQL 执行成功，无错误可定位", "", "")
    if not error:
        return _default_loc("", CAT_OTHER, "fallback", sql, "无错误信息")

    category, payload = classify_error(error)
    branches = split_sql_clauses(sql)

    handler = {
        ("schema_reference", "no_col"): _loc_no_column,
        ("schema_reference", "no_table"): _loc_no_table,
        ("schema_reference", "no_func"): _loc_no_func,
        ("schema_reference", "collation"): _loc_collation,
        ("schema_reference", "order_range"): _loc_order_range,
        ("schema_reference", "order_no_match"): _loc_order_no_match,
        ("schema_reference", "no_query_solution"): _loc_no_solution,
        ("logical_misuse", "misuse_agg"): _loc_misuse_agg,
        ("logical_misuse", "agg_forbidden"): _loc_agg_forbidden,
        ("logical_misuse", "ambiguous"): _loc_ambiguous,
        ("logical_misuse", "window_forbidden"): _loc_window_forbidden,
        ("logical_misuse", "alias_misuse"): _loc_alias_misuse,
        ("logical_misuse", "agg_window"): _loc_agg_window,
        ("logical_misuse", "distinct_agg_args"): _loc_distinct_agg,
        ("logical_misuse", "wrong_args"): _loc_wrong_args,
        ("logical_misuse", "having_nonagg"): _loc_having_nonagg,
        ("syntax_error", "order_before_union"): _loc_order_before_union,
        ("syntax_error", "on_without_join"): _loc_on_without_join,
        ("syntax_error", "escape_char"): _loc_escape_char,
        ("syntax_error", "bindings"): _loc_bindings,
        ("schema_reference", "circular_ref"): _loc_circular_ref,
        ("schema_reference", "using_col"): _loc_using_col,
        ("data_constraint", "unknown_join"): _loc_unknown_join,
        ("logical_misuse", "distinct_window"): _loc_distinct_window,
        ("other", "non_select_sql"): _loc_non_select_sql,
        ("data_constraint", "datatype"): _loc_datatype,
        ("data_constraint", "constraint"): _loc_constraint,
        ("data_constraint", "outer_join"): _loc_outer_join,
        ("syntax_error", "near_token"): _loc_near_token,
        ("syntax_error", "incomplete"): _loc_incomplete,
        ("syntax_error", "generic"): _loc_syntax_generic,
        ("setop_mismatch", "union_cols"): _loc_union_cols,
        ("setop_mismatch", "subselect_cols"): _loc_subselect_cols,
        ("setop_mismatch", "compound_terms"): _loc_compound_terms,
    }.get((category, (payload or {}).get("kind")))

    if handler is not None:
        try:
            return handler(sql, db_path, error, payload, branches)
        except Exception as exc:  # 定位器自身异常 → 无损降级
            return _default_loc(error, category, "fallback", sql,
                                f"定位器内部异常：{exc}")
    return _default_loc(error, category, "fallback", sql,
                        f"未覆盖的错误模式：{error[:120]}")


def _execute_get_error(sql: str, db_path: str, timeout: float = 20.0) -> Optional[str]:
    """执行一次 SQL 取错误文本（只读；进度处理器限制 VM 步数）。成功返回 None。"""
    import threading
    holder: Dict[str, Any] = {"err": None, "done": False}

    def _run() -> None:
        con = None
        try:
            uri = Path(db_path).resolve().as_uri()
            con = sqlite3.connect(f"{uri}?mode=ro", uri=True, timeout=5)
            ticks = [0]

            def _handler() -> int:
                ticks[0] += 1000
                return 1 if ticks[0] >= 5_000_000 else 0

            con.set_progress_handler(_handler, 1000)
            cur = con.execute(sql)
            cur.fetchall()
        except Exception as exc:
            holder["err"] = str(exc)
        finally:
            holder["done"] = True
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

    try:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout)
        if not holder["done"]:
            return f"Wall-clock timeout after {timeout}s (localization probe)"
        return holder["err"]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


# ---- 各错误类定位实现 ----

def _loc_no_column(sql, db_path, error, payload, branches):
    ident = payload.get("ident") or ""
    if not ident:
        return _default_loc(error, CAT_SCHEMA_REF, "fallback", sql,
                            f"无法提取列名：{error[:120]}")
    if "." in ident:
        prefix, name = ident.rsplit(".", 1)
    else:
        prefix, name = "", ident
    rx = _identifier_regex(ident)
    containing = _clauses_containing(branches, rx)
    if not containing:
        rx_bare = _bare_identifier_regex(name)
        containing = _clauses_containing(branches, rx_bare)
    anchor = _last_in_logical_order(containing) if containing else None
    anchor_label = anchor.label if anchor else None
    schema = _schema(db_path) if db_path else {"tables": {}, "table_list": []}

    if prefix:
        def_seg = _first_defining_segment(branches, prefix)
        tables = schema["tables"]
        tbl_name = schema.get("table_lower_to_name", {}).get(prefix.lower())

        def _both(cl: str) -> List[str]:
            # 根因 + 报错点：论文 C_err 只含根因（定义/引用别名处），
            # 修复 prompt 使用超集（根因在前、锚点在后，按逻辑执行序去重）。
            out = []
            for s in sorted([def_seg, anchor], key=lambda x: (
                    _CLAUSE_RANK.get(x.label, 99) if x else 99, 0)):
                if s is not None and s.label not in out:
                    out.append(s.label)
            if cl and cl not in out:
                out.insert(0, cl)
            return out[:3]

        if tbl_name:
            # 前缀是真实表名（大小写不敏感）：列是否存在决定根因
            cols = tables[tbl_name]
            cols_lower = {c.lower() for c in cols}
            if name.lower() in cols_lower:
                # 表有该列但报错（引号/大小写/遮蔽等边缘）→ 锚点即引用处
                err_cl = [anchor_label] if anchor else ["SELECT"]
                diag = (f"引用 {ident} 报 no such column，但表 {tbl_name} 确有列 "
                        f"{name}（可能大小写/引号/别名遮蔽）。检查引用写法。")
                hint = (f"Root cause clause: {anchor_label or '?'}. Qualified "
                        f"column reference {ident} is not resolved although table "
                        f"{tbl_name} has column {name}. Check quoting/case/alias "
                        f"shadowing at the reference.")
                return _make_loc(True, CAT_SCHEMA_REF, "schema_probe", err_cl,
                                 anchor_label, [ident], hint.split(".")[0] + ".",
                                 diag, hint,
                                 {"table": tbl_name, "column": name})
            sugg = _col_suggestions(schema, tbl_name, name)
            err_cl = _both(def_seg.label if def_seg else "")
            root = (f"表 {tbl_name} 无列 {name}（限定 {prefix} 来自 "
                    f"{def_seg.label if def_seg else 'FROM/JOIN'}）")
            if sugg:
                root += f"；相近列：{', '.join(sugg)}"
            diag = (f"no such column: {ident}。表 {tbl_name} 没有列 {name}。"
                    f"根因子句 = {def_seg.label if def_seg else 'FROM/JOIN'}，"
                    f"报错点 = {anchor_label or '?'}。"
                    + (f"相近列名：{', '.join(sugg)}。" if sugg else ""))
            hint = (f"Root cause clause: {def_seg.label if def_seg else 'FROM/JOIN'}"
                    f". Column {name} does not exist in table {tbl_name}; error "
                    f"surfaces at {anchor_label or '?'}. Rewrite the "
                    f"{def_seg.label if def_seg else 'FROM/JOIN'} clause first."
                    + (f" Similar columns: {', '.join(sugg)}." if sugg else ""))
            return _make_loc(True, CAT_SCHEMA_REF, "error_trace", err_cl,
                             anchor_label, [ident], root, diag, hint,
                             {"alias": prefix, "table": tbl_name, "column": name,
                              "suggestions": sugg})
        # 前缀不是真实表名 → 别名（尝试解析别名绑定的表做列级建议）
        if def_seg:
            err_cl = _both(def_seg.label)
            bound_table = _resolve_alias_table(def_seg, prefix)
            sugg: List[str] = []
            if bound_table:
                bt = schema.get("table_lower_to_name", {}).get(bound_table.lower())
                if bt:
                    sugg = _col_suggestions(schema, bt, name)
            # 跨表提示：该列若存在于其他表（别名未绑定），给出候选表
            other_tables = [t for t in _tables_with_col(schema, name)
                            if not (bt and t.lower() == bt.lower())][:2] \
                if bound_table else []
            root = (f"别名 {prefix} 由 {def_seg.label} 定义，其表缺少列 {name} "
                    f"（或 JOIN 错表）"
                    + (f"；相近列：{', '.join(sugg)}" if sugg else "")
                    + (f"；该列存在于 {', '.join(other_tables)}" if other_tables else ""))
            diag = (f"no such column: {ident}。根因：定义别名 {prefix} 的 "
                    f"{def_seg.label}（{_seg_excerpt(def_seg)}）没有提供列 "
                    f"{name}。报错点在 {anchor_label or '?'}。优先重写 "
                    f"{def_seg.label} 而非 {anchor_label or '?'}。"
                    + (f"相近列名：{', '.join(sugg)}。" if sugg else "")
                    + (f"该列存在于表 {', '.join(other_tables)}。" if other_tables else ""))
            hint = (f"Root cause clause: {def_seg.label}. Alias {prefix} "
                    f"introduced there lacks column {name}; the error merely "
                    f"surfaces at {anchor_label or '?'}. Rewrite the "
                    f"{def_seg.label} clause first."
                    + (f" Similar columns: {', '.join(sugg)}." if sugg else "")
                    + (f" Column {name} exists in: {', '.join(other_tables)}." if other_tables else ""))
            return _make_loc(True, CAT_SCHEMA_REF, "error_trace", err_cl,
                             anchor_label, [ident], root, diag, hint,
                             {"alias": prefix, "column": name,
                              "bound_table": bound_table, "suggestions": sugg,
                              "other_tables": other_tables})
        # 别名未定义
        err_cl = list(dict.fromkeys([anchor_label, "FROM", "JOIN"]))
        err_cl = [c for c in err_cl if c]
        root = f"别名/表 {prefix} 未在 FROM/JOIN 定义"
        diag = (f"no such column: {ident}。别名/表 {prefix} 未在 FROM/JOIN 中"
                f"定义：可能漏 JOIN 表 {prefix} 或别名拼写错。报错点 "
                f"{anchor_label or '?'}。")
        hint = (f"Root cause clause: FROM/JOIN. Qualifier {prefix} is not "
                f"defined: likely a missing JOIN of table {prefix} or a typo in "
                f"the alias. Error surfaces at {anchor_label or '?'}.")
        return _make_loc(True, CAT_SCHEMA_REF, "error_trace", err_cl,
                         anchor_label, [ident], root, diag, hint,
                         {"alias": prefix, "column": name, "alias_undefined": True})

    # 无前缀裸列
    with_col = _tables_with_col(schema, name)
    err_cl = [anchor_label] if anchor else ["SELECT"]
    sugg = []
    if len(with_col) == 1:
        t = with_col[0]
        refd = _referenced_tables(branches, schema)
        if not any(x.lower() == t.lower() for x in refd):
            err_cl = list(dict.fromkeys(err_cl + ["FROM", "JOIN"]))
            root = f"列 {name} 只在表 {t} 存在，但该表未 JOIN"
            diag = (f"no such column: {name}。该列属于表 {t}，但查询未 JOIN 该表"
                    f"（或引用处缺限定 {t}.{name}）。报错点 {anchor_label or '?'}；"
                    f"优先补 JOIN {t}。")
            hint = (f"Root cause clause: FROM/JOIN. Column {name} exists only in "
                    f"table {t}, which is not joined. Add the join or qualify as "
                    f"{t}.{name}.")
            return _make_loc(True, CAT_SCHEMA_REF, "error_trace", err_cl,
                             anchor_label, [name], root, diag, hint,
                             {"column": name, "table": t})
        root = f"列 {name} 属于表 {t}，引用处缺限定"
        diag = (f"no such column: {name}。该列属于表 {t}（已 JOIN），引用处"
                f"（{anchor_label or '?'}）缺表限定：应写 {t}.{name}。")
        hint = (f"Root cause clause: {anchor_label or '?'}. Column {name} "
                f"belongs to table {t}; qualify the reference as {t}.{name}.")
        return _make_loc(True, CAT_SCHEMA_REF, "error_trace", err_cl,
                         anchor_label, [name], root, diag, hint,
                         {"column": name, "table": t})
    if len(with_col) >= 2:
        err_cl = list(dict.fromkeys(err_cl + ["FROM", "JOIN"]))
        root = f"列 {name} 在多个表（{', '.join(with_col[:4])}）存在，缺限定"
        diag = (f"no such column: {name}。该列在 {', '.join(with_col[:4])} 等表"
                f"中都存在，引用处缺表限定。")
        hint = (f"Root cause clause: {anchor_label or '?'} / FROM/JOIN. Column "
                f"{name} exists in multiple tables; add a table qualifier.")
        return _make_loc(True, CAT_SCHEMA_REF, "error_trace", err_cl,
                         anchor_label, [name], root, diag, hint,
                         {"column": name, "tables": with_col})
    # 不存在
    refd = _referenced_tables(branches, schema)
    if refd:
        sugg = _similar([c for t in refd for c in tables.get(t, [])], name)
    has_cte = any(s.label == "WITH" for s in _segments_flat(branches))
    root = f"列 {name} 在 schema 中不存在" + (
        f"；相近列：{', '.join(sugg)}" if sugg else "")
    diag = (f"no such column: {name}。任何表都不存在该列"
            + (f"（相近列：{', '.join(sugg)}）" if sugg else "")
            + ("；若列来自 CTE，检查 CTE 定义" if has_cte else "")
            + f"。报错点 {anchor_label or '?'}。")
    hint = (f"Root cause clause: {anchor_label or '?'}. Column {name} does not "
            f"exist in the schema." + (f" Similar columns: {', '.join(sugg)}." if sugg else ""))
    return _make_loc(True, CAT_SCHEMA_REF, "schema_probe", err_cl, anchor_label,
                     [name], root, diag, hint,
                     {"column": name, "suggestions": sugg})


def _loc_no_table(sql, db_path, error, payload, branches):
    name = payload.get("ident") or ""
    if not name:
        return _default_loc(error, CAT_SCHEMA_REF, "fallback", sql, error[:120])
    rx = _bare_identifier_regex(name)
    segs = [s for s in _segments_flat(branches) if s.label in ("FROM", "JOIN")
            and rx.search(s.text)]
    if not segs:
        containing = _clauses_containing(branches, rx)
        anchor = _last_in_logical_order(containing) if containing else None
        segs = [anchor] if anchor else []
    if not segs:
        return _default_loc(error, CAT_SCHEMA_REF, "fallback", sql,
                            f"no such table: {name}（未定位到引用子句）")
    err_cl = _dedupe_labels(segs)
    schema = _schema(db_path) if db_path else {"table_list": []}
    tbl_exists = name.lower() in {t.lower() for t in schema.get("table_list", [])}
    if tbl_exists:
        root = f"表 {name} 存在但引用解析失败（大小写/引号/CTE 遮蔽）"
        diag = (f"no such table: {name}。schema 中存在该表，但引用未解析：检查"
                f"大小写、引号，或 CTE 同名遮蔽。引用位置 "
                f"{_clause_names(segs)}。")
        hint = (f"Root cause clause: {err_cl[0]}. Table {name} exists in schema "
                f"but the reference fails to resolve (case/quoting/CTE shadowing).")
        return _make_loc(True, CAT_SCHEMA_REF, "schema_probe", err_cl, err_cl[0],
                         [name], root, diag, hint, {"table": name})
    sugg = _similar(schema["table_list"], name) if schema.get("table_list") else []
    root = f"表 {name} 不存在" + (f"；相近表：{', '.join(sugg)}" if sugg else "")
    diag = (f"no such table: {name}。" + (f"相近表名：{', '.join(sugg)}。" if sugg else "")
            + f"引用位置 {_clause_names(segs)}。")
    hint = (f"Root cause clause: {err_cl[0]}. Table {name} does not exist."
            + (f" Similar tables: {', '.join(sugg)}." if sugg else ""))
    return _make_loc(True, CAT_SCHEMA_REF, "schema_probe", err_cl, err_cl[0],
                     [name], root, diag, hint,
                     {"table": name, "suggestions": sugg})


def _loc_no_func(sql, db_path, error, payload, branches):
    fn = payload.get("func") or ""
    rx = re.compile(r"(?<![A-Za-z0-9_$])" + re.escape(fn) + r"\s*\(", re.IGNORECASE)
    containing = _clauses_containing(branches, rx)
    if not containing and fn:
        # 运算符形式（如 x REGEXP y）按标识符定位
        containing = _clauses_containing(branches, _bare_identifier_regex(fn))
    anchor = _last_in_logical_order(containing) if containing else None
    err_cl = [anchor.label] if anchor else []
    diag = (f"no such function: {fn}()。该函数在 SQLite 中不存在（拼写错或其他"
            f"方言函数）。位置 {anchor.label if anchor else '?'}。")
    hint = (f"Root cause clause: {anchor.label if anchor else '?'}. Function "
            f"{fn}() does not exist in SQLite; check spelling or use a "
            f"SQLite-supported function.")
    return _make_loc(bool(err_cl), CAT_SCHEMA_REF, "error_trace", err_cl,
                     anchor.label if anchor else None, [fn],
                     f"函数 {fn} 不存在", diag, hint, {"func": fn})


def _loc_collation(sql, db_path, error, payload, branches):
    ident = payload.get("ident") or ""
    segs = _segments_flat(branches)
    order_segs = [s for s in segs if s.label == "ORDER BY" and ident.lower() in s.text.lower()]
    anchor = order_segs[0] if order_segs else None
    err_cl = ["ORDER BY"] if anchor else ["WHERE"]
    diag = (f"no such collation sequence: {ident}。COLLATE 使用了未注册的排序规则"
            f"（SQLite 内置 NOCASE/BINARY/RTRIM）。")
    hint = (f"Root cause clause: {err_cl[0]}. Collation {ident} is not "
            f"registered; use a built-in collation (NOCASE/BINARY/RTRIM).")
    return _make_loc(True, CAT_SCHEMA_REF, "error_trace", err_cl,
                     anchor.label if anchor else None, [ident],
                     f"排序规则 {ident} 不存在", diag, hint, {"collation": ident})


def _loc_order_range(sql, db_path, error, payload, branches):
    nth = payload.get("nth") or "?"
    diag = (f"ORDER BY 引用序号 {nth} 超出 SELECT 输出列数。根因子句 = ORDER BY"
            f"（或 SELECT 列被误删）。")
    hint = (f"Root cause clause: ORDER BY. The {nth} ORDER BY term references "
            f"a column index out of range; check the SELECT list length.")
    return _make_loc(True, CAT_SCHEMA_REF, "error_trace", ["ORDER BY"], "ORDER BY",
                     [], f"ORDER BY 序号 {nth} 越界", diag, hint, {})


def _loc_order_no_match(sql, db_path, error, payload, branches):
    diag = ("ORDER BY 引用的列不在结果集中（列名与 SELECT 输出/别名不匹配）。"
            "根因子句 = ORDER BY，注意核对 SELECT 列名/别名。")
    hint = ("Root cause clause: ORDER BY. The ORDER BY expression does not "
            "match any column in the result set; align it with the SELECT list.")
    return _make_loc(True, CAT_SCHEMA_REF, "error_trace", ["ORDER BY"], "ORDER BY",
                     [], "ORDER BY 列不匹配结果集", diag, hint, {})


def _loc_no_solution(sql, db_path, error, payload, branches):
    diag = ("SQLite 无法为查询生成执行计划（no query solution），常见于 "
            "JOIN/子查询引用关系矛盾。重点检查 FROM/JOIN。")
    hint = ("Root cause clause: FROM/JOIN. SQLite could not build a query plan; "
            "check FROM/JOIN/subquery references.")
    return _make_loc(True, CAT_SCHEMA_REF, "error_trace", ["FROM", "JOIN"], None,
                     [], "查询无可行执行计划", diag, hint, {})


def _loc_misuse_agg(sql, db_path, error, payload, branches):
    fn = payload.get("func") or ""
    rx = re.compile(r"(?<![A-Za-z0-9_$])" + re.escape(fn) + r"\s*\(", re.IGNORECASE)
    containing = _clauses_containing(branches, rx)
    if not containing and fn:
        # 报错形式 "misuse of aggregate: <col>"（无括号）：按标识符定位
        containing = _clauses_containing(branches, _bare_identifier_regex(fn))
    cond_ctx = [s for s in containing if s.label in ("WHERE", "HAVING", "JOIN")]
    anchor = (_last_in_logical_order(cond_ctx) if cond_ctx else
              (_last_in_logical_order(containing) if containing else None))
    err_cl = [anchor.label] if anchor else ["SELECT"]
    flat = _segments_flat(branches)
    has_group = any(s.label == "GROUP BY" for s in flat)
    anchor_is_where = anchor is not None and anchor.label in ("WHERE", "HAVING", "JOIN")
    if anchor_is_where:
        diag = (f"misuse of aggregate: {fn}()。聚合函数出现在 {anchor.label}："
                f"聚合不能用于 {anchor.label}（条件过滤应改用 HAVING 或先做子查询"
                f"聚合）。")
        hint = (f"Root cause clause: {anchor.label}. Aggregate {fn}() is not "
                f"allowed in {anchor.label}; move the condition to HAVING or a "
                f"subquery.")
    else:
        diag = (f"misuse of aggregate: {fn}()。聚合函数使用位置非法：常见原因 = "
                f"聚合嵌套、聚合用于 GROUP BY/HAVING 不当，或 SELECT 混合聚合与"
                f"非聚合列"
                + ("" if has_group else "（该查询没有 GROUP BY）")
                + f"。位置 {anchor.label if anchor else '?'}。")
        hint = (f"Root cause clause: {anchor.label if anchor else 'SELECT'}. "
                f"Aggregate {fn}() is misused"
                + ("" if has_group else " — the query has no GROUP BY")
                + "; check aggregate nesting/placement.")
    return _make_loc(True, CAT_LOGICAL, "error_trace", err_cl,
                     anchor.label if anchor else None, [fn],
                     f"聚合函数 {fn}() 使用不当（GROUP BY 缺失/位置非法）",
                     diag, hint, {"func": fn, "has_group_by": has_group})


def _loc_agg_forbidden(sql, db_path, error, payload, branches):
    where = payload.get("where") or ""
    label = None
    for cand in ("WHERE", "GROUP BY", "HAVING", "ORDER BY", "JOIN"):
        if cand.lower() in where:
            label = cand
            break
    label = label or "WHERE"
    diag = (f"aggregate functions are not allowed in {where}。聚合不能用在 "
            f"{label}（改用 HAVING 或先做子查询聚合）。")
    hint = (f"Root cause clause: {label}. Aggregates are not allowed in "
            f"{where}; move the condition to HAVING or a subquery.")
    return _make_loc(True, CAT_LOGICAL, "error_trace", [label], label, [],
                     f"聚合函数非法用于 {where}", diag, hint, {"where": where})


def _loc_ambiguous(sql, db_path, error, payload, branches):
    ident = payload.get("ident") or ""
    rx = _identifier_regex(ident)
    containing = _clauses_containing(branches, rx)
    anchor = _last_in_logical_order(containing) if containing else None
    err_cl = list(dict.fromkeys([anchor.label if anchor else None, "FROM", "JOIN"]))
    err_cl = [c for c in err_cl if c]
    schema = _schema(db_path) if db_path else {"tables": {}}
    name = ident.rsplit(".", 1)[-1]
    with_col = _tables_with_col(schema, name)
    if len(with_col) == 1:
        diag = (f"ambiguous column name: {ident}。该列其实只在表 {with_col[0]} "
                f"存在：引用处（{anchor.label if anchor else '?'}）应写 "
                f"{with_col[0]}.{name}。")
        hint = (f"Root cause clause: {anchor.label if anchor else '?'}. Column "
                f"{name} exists only in {with_col[0]}; qualify as "
                f"{with_col[0]}.{name}.")
        root = f"列 {name} 只存在于 {with_col[0]}，引用处缺限定"
    elif len(with_col) >= 2:
        diag = (f"ambiguous column name: {ident}。表 {', '.join(with_col[:4])} 均"
                f"有该列：引用处（{anchor.label if anchor else '?'}）需加表限定。")
        hint = (f"Root cause clause: {anchor.label if anchor else '?'}. Column "
                f"{name} exists in multiple tables ({', '.join(with_col[:4])}); "
                f"add a qualifier.")
        root = f"列 {name} 在 {', '.join(with_col[:4])} 均存在，缺限定"
    else:
        diag = (f"ambiguous column name: {ident}（多表/CTE 同名或别名重复）。"
                f"检查引用处 {anchor.label if anchor else '?'} 的限定。")
        hint = (f"Root cause clause: {anchor.label if anchor else '?'}. "
                f"Disambiguate column {name} with a table qualifier.")
        root = f"列 {name} 引用歧义"
    return _make_loc(True, CAT_LOGICAL, "error_trace", err_cl,
                     anchor.label if anchor else None, [ident], root, diag, hint,
                     {"column": name, "tables": with_col})


def _loc_window_forbidden(sql, db_path, error, payload, branches):
    where = payload.get("where") or ""
    label = None
    for cand in ("WHERE", "GROUP BY", "HAVING", "ORDER BY", "JOIN"):
        if cand.lower() in where:
            label = cand
            break
    label = label or "WHERE"
    diag = (f"window functions are not allowed in {where}。窗口函数不能直接用在"
            f"{label}（需子查询/CTE 先物化窗口结果）。")
    hint = (f"Root cause clause: {label}. Window functions are not allowed in "
            f"{where}; wrap the window computation in a subquery/CTE first.")
    return _make_loc(True, CAT_LOGICAL, "error_trace", [label], label, [],
                     f"窗口函数非法用于 {where}", diag, hint, {"where": where})


def _loc_alias_misuse(sql, db_path, error, payload, branches):
    diag = ("misuse of aliased identifier：别名在聚合/HAVING 等位置非法使用。"
            "检查 SELECT 别名在 GROUP BY/HAVING/WHERE 中的引用。")
    hint = ("Root cause clause: SELECT/GROUP BY/HAVING. An aliased identifier "
            "is misused (aggregate/alias semantics).")
    return _make_loc(True, CAT_LOGICAL, "error_trace",
                     ["SELECT", "GROUP BY", "HAVING"], None, [],
                     "别名使用不当", diag, hint, {})


def _loc_agg_window(sql, db_path, error, payload, branches):
    fn = payload.get("func") or ""
    rx = re.compile(r"(?<![A-Za-z0-9_$])" + re.escape(fn) + r"\s*\(", re.IGNORECASE)
    containing = _clauses_containing(branches, rx)
    anchor = _last_in_logical_order(containing) if containing else None
    err_cl = [anchor.label] if anchor else ["SELECT"]
    diag = (f"{fn} 不能与窗口帧（OVER …）同用：聚合与窗口语义冲突。检查 "
            f"OVER 子句是否多余或嵌套。")
    hint = (f"Root cause clause: {anchor.label if anchor else 'SELECT'}. "
            f"Aggregate {fn} cannot be combined with a window frame.")
    return _make_loc(True, CAT_LOGICAL, "error_trace", err_cl,
                     anchor.label if anchor else None, [fn],
                     f"聚合与窗口帧冲突（{fn}）", diag, hint, {"func": fn})


def _loc_distinct_agg(sql, db_path, error, payload, branches):
    diag = ("DISTINCT 聚合必须恰好一个参数（如 COUNT(DISTINCT x)）。检查聚合"
            "参数列表。根因子句 = SELECT。")
    hint = ("Root cause clause: SELECT. A DISTINCT aggregate must have exactly "
            "one argument (e.g. COUNT(DISTINCT x)).")
    return _make_loc(True, CAT_LOGICAL, "error_trace", ["SELECT"], "SELECT", [],
                     "DISTINCT 聚合参数个数错误", diag, hint, {})


def _loc_wrong_args(sql, db_path, error, payload, branches):
    fn = payload.get("func") or ""
    rx = re.compile(r"(?<![A-Za-z0-9_$])" + re.escape(fn) + r"\s*\(", re.IGNORECASE)
    containing = _clauses_containing(branches, rx)
    anchor = _last_in_logical_order(containing) if containing else None
    err_cl = [anchor.label] if anchor else ["SELECT"]
    diag = (f"wrong number of arguments to function {fn}()：函数参数个数不符。"
            f"检查 {fn} 的参数列表。")
    hint = (f"Root cause clause: {anchor.label if anchor else 'SELECT'}. "
            f"Function {fn}() called with the wrong number of arguments.")
    return _make_loc(True, CAT_LOGICAL, "error_trace", err_cl,
                     anchor.label if anchor else None, [fn],
                     f"函数 {fn} 参数个数错误", diag, hint, {"func": fn})


def _loc_having_nonagg(sql, db_path, error, payload, branches):
    diag = ("HAVING clause on a non-aggregate query：查询没有聚合/GROUP BY，"
            "HAVING 应改为 WHERE（或补聚合）。根因子句 = HAVING。")
    hint = ("Root cause clause: HAVING. The query has no aggregate/GROUP BY; "
            "replace HAVING with WHERE (or add aggregation).")
    return _make_loc(True, CAT_LOGICAL, "error_trace", ["HAVING"], "HAVING", [],
                     "HAVING 用在非聚合查询", diag, hint, {})


def _loc_order_before_union(sql, db_path, error, payload, branches):
    diag = ("ORDER BY clause should come after UNION not before：ORDER BY 只能"
            "出现在整个集合运算之后（或加括号包裹分支）。根因子句 = ORDER BY。")
    hint = ("Root cause clause: ORDER BY. Move the ORDER BY after the UNION "
            "(or wrap branches in parentheses).")
    return _make_loc(True, CAT_SYNTAX, "syntax_token", ["ORDER BY"], "ORDER BY",
                     [], "ORDER BY 位置错误（UNION 前）", diag, hint, {})


def _loc_on_without_join(sql, db_path, error, payload, branches):
    diag = ("a JOIN clause is required before ON：FROM 之后直接出现了 ON 条件，"
            "缺 JOIN 关键字/表。根因子句 = FROM。")
    hint = ("Root cause clause: FROM. An ON condition appears without a "
            "preceding JOIN; add the JOIN keyword and table.")
    return _make_loc(True, CAT_SYNTAX, "syntax_token", ["FROM"], "FROM", [],
                     "ON 前缺 JOIN", diag, hint, {})


def _loc_using_col(sql, db_path, error, payload, branches):
    col = payload.get("col") or ""
    segs = [s for s in _segments_flat(branches) if s.label == "JOIN"]
    anchor = segs[0] if segs else None
    diag = (f"cannot join using column {col}：JOIN USING 要求的列 {col} 不在两张"
            f"表中同时存在。改用 ON t1.{col} = t2.<实际列> 或换列。根因子句 = JOIN。")
    hint = (f"Root cause clause: JOIN. USING column {col} is not present in both "
            f"tables; switch to an ON condition with the correct columns.")
    return _make_loc(True, CAT_SCHEMA_REF, "error_trace", ["JOIN"],
                     "JOIN" if anchor else None, [col],
                     f"USING 列 {col} 不在两表中同时存在", diag, hint,
                     {"column": col})


def _loc_escape_char(sql, db_path, error, payload, branches):
    rx = _word_regex("ESCAPE")
    containing = _clauses_containing(branches, rx)
    anchor = _last_in_logical_order(containing) if containing else None
    err_cl = [anchor.label] if anchor else ["WHERE"]
    diag = ("ESCAPE expression must be a single character：LIKE ... ESCAPE 后必须是"
            "单字符。检查 ESCAPE 后面的表达式。")
    hint = (f"Root cause clause: {err_cl[0]}. The ESCAPE expression must be a "
            f"single character.")
    return _make_loc(True, CAT_SYNTAX, "syntax_token", err_cl,
                     anchor.label if anchor else None, [],
                     "ESCAPE 表达式非法", diag, hint, {})


def _loc_unknown_join(sql, db_path, error, payload, branches):
    op = payload.get("op") or ""
    diag = (f"unknown join type: {op}。JOIN 类型关键字非法（可能 OUTER 单独出现）。"
            "根因子句 = JOIN。")
    hint = (f"Root cause clause: JOIN. Unknown join type {op}; fix the JOIN "
            f"keyword sequence.")
    return _make_loc(True, CAT_DATA, "error_trace", ["JOIN"], "JOIN", [],
                     f"非法 JOIN 类型 {op}", diag, hint, {"op": op})


def _loc_distinct_window(sql, db_path, error, payload, branches):
    diag = ("DISTINCT is not supported for window functions：窗口函数不支持 "
            "DISTINCT 参数。去掉 DISTINCT 或改用子查询去重。根因子句 = SELECT。")
    hint = ("Root cause clause: SELECT. Window functions do not support "
            "DISTINCT; deduplicate via a subquery instead.")
    return _make_loc(True, CAT_LOGICAL, "error_trace", ["SELECT"], "SELECT", [],
                     "窗口函数不支持 DISTINCT", diag, hint, {})


def _loc_bindings(sql, db_path, error, payload, branches):
    diag = ("Incorrect number of bindings：SQL 含未绑定的 ? 占位符（生成物残留）。"
            "把 ? 替换为字面值或删除。")
    hint = ("Root cause clause: SELECT. The SQL contains unbound ? placeholders; "
            "replace them with literal values.")
    return _make_loc(True, CAT_SYNTAX, "syntax_token", ["SELECT"], "SELECT", [],
                     "未绑定占位符", diag, hint, {})


def _loc_non_select_sql(sql, db_path, error, payload, branches):
    diag = ("候选不是查询语句（CREATE TABLE/DDL 等），无法做子句级定位；修复链应"
            "直接丢弃该候选。")
    hint = ""
    return _default_loc(error, CAT_OTHER, "fallback", sql,
                        "非 SELECT 语句（DDL），不做子句定位")


def _loc_circular_ref(sql, db_path, error, payload, branches):
    ident = payload.get("ident") or ""
    has_cte = any(s.label == "WITH" for s in _segments_flat(branches))
    err_cl = ["WITH"] if has_cte else ["FROM"]
    diag = (f"circular reference: {ident}：CTE/子查询循环引用自身。检查 "
            f"{'CTE 定义' if has_cte else 'FROM/JOIN 引用'}。")
    hint = (f"Root cause clause: {err_cl[0]}. Circular reference to {ident}; "
            f"check the {'CTE' if has_cte else 'FROM/JOIN'} definition.")
    return _make_loc(True, CAT_SCHEMA_REF, "error_trace", err_cl, None, [ident],
                     f"循环引用 {ident}", diag, hint, {"ident": ident})


def _loc_datatype(sql, db_path, error, payload, branches):
    diag = ("datatype mismatch：运算/比较两端类型不兼容（字符串 vs 数值等）。"
            "按 EXPO Table 14 归因 JOIN/WHERE：检查 JOIN 键或 WHERE 条件两端的"
            "列类型/单位。")
    hint = ("Root cause clause: WHERE/JOIN. Datatype mismatch in a comparison; "
            "check the column types on both sides of the JOIN key or WHERE "
            "condition.")
    return _make_loc(True, CAT_DATA, "error_trace", ["WHERE", "JOIN"], None, [],
                     "类型不匹配（JOIN/WHERE 比较）", diag, hint, {})


def _loc_constraint(sql, db_path, error, payload, branches):
    ctype = payload.get("type") or ""
    diag = (f"{ctype} constraint failed：按 EXPO Table 14 归因 JOIN/WHERE——"
            f"检查 JOIN/WHERE 引入或过滤的数据是否违反约束语义。")
    hint = (f"Root cause clause: WHERE/JOIN. {ctype} constraint failure; inspect "
            f"the JOIN/WHERE conditions involved.")
    return _make_loc(True, CAT_DATA, "error_trace", ["WHERE", "JOIN"], None, [],
                     f"{ctype} 约束失败", diag, hint, {"constraint": ctype})


def _loc_outer_join(sql, db_path, error, payload, branches):
    segs = [s for s in _segments_flat(branches) if s.label == "JOIN"]
    diag = ("SQLite 不支持 RIGHT/FULL OUTER JOIN。改写为 LEFT JOIN 并交换表序，"
            "或改用子查询/UNION。根因子句 = JOIN。")
    hint = ("Root cause clause: JOIN. SQLite does not support RIGHT/FULL OUTER "
            "JOIN; rewrite as LEFT JOIN with swapped table order.")
    return _make_loc(True, CAT_DATA, "error_trace", ["JOIN"], "JOIN", [],
                     "RIGHT/FULL OUTER JOIN 不支持", diag, hint, {})


def _loc_near_token(sql, db_path, error, payload, branches):
    token = payload.get("token") or ""
    seg = _find_token_segment(branches, token) if branches else None
    if seg is None:
        # token 未命中任何段：取最后一段（解析中断处通常靠后）或首段
        flat = _segments_flat(branches) if branches else []
        seg = flat[-1] if flat else None
    err_cl = [seg.label] if seg else []
    if not seg:
        return _default_loc(error, CAT_SYNTAX, "fallback", sql,
                            f"near \"{token}\": 语法错误（无子句可定位）")
    diag = (f"near \"{token}\": syntax error。语法错误位于 {seg.label} 附近："
            f"「{_seg_excerpt(seg)}」。检查关键字拼写/逗号/括号配对。")
    hint = (f"Root cause clause: {seg.label}. Syntax error near "
            f"\"{token}\" (segment: {_seg_excerpt(seg, 40)}).")
    return _make_loc(True, CAT_SYNTAX, "syntax_token", err_cl, seg.label,
                     [token], f"{seg.label} 附近语法错（token: {token}）",
                     diag, hint, {"token": token})


def _loc_incomplete(sql, db_path, error, payload, branches):
    flat = _segments_flat(branches) if branches else []
    seg = flat[-1] if flat else None
    err_cl = [seg.label] if seg else []
    diag = ("incomplete input：SQL 被截断（括号/引号/关键字不闭合）。检查查询"
            f"尾部（最后一段：{_seg_excerpt(seg) if seg else '?'}）。")
    hint = ("Root cause clause: (query tail). The SQL is truncated with "
            "unclosed parentheses/quotes/keywords; check the end of the query.")
    return _make_loc(True, CAT_SYNTAX, "syntax_token", err_cl,
                     seg.label if seg else None, [], "SQL 截断/未闭合",
                     diag, hint, {})


def _loc_syntax_generic(sql, db_path, error, payload, branches):
    diag = f"语法错误（{error[:120]}）。根因子句无法精确定位，请检查整句语法。"
    hint = "Syntax error; check keyword spelling, commas and parentheses."
    return _make_loc(False, CAT_SYNTAX, "fallback", [], None, [],
                     f"语法错误（未定位）", diag, hint, {})


def _count_select_exprs(seg: Optional[Segment], sql: str) -> int:
    """粗略统计 SELECT 段顶层表达式数（SELECT 关键字后到段尾，统计顶层逗号 +1）。"""
    if seg is None:
        return 0
    t = seg.text
    t = re.sub(r"^\s*SELECT\s+(DISTINCT\s+|ALL\s+)?", "", t, flags=re.I)
    depth = 0
    commas = 0
    i = 0
    n = len(t)
    while i < n:
        c = t[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif c == "," and depth == 0:
            commas += 1
        elif c in "'\"`[":
            i = _skip_quote(t, i, c)
            continue
        i += 1
    return commas + 1


def _skip_quote(t: str, i: int, q: str) -> int:
    n = len(t)
    i += 1
    while i < n:
        if t[i] == q:
            if q in "'\"`" and i + 1 < n and t[i + 1] == q:
                i += 2
                continue
            return i + 1
        i += 1
    return n


def _loc_union_cols(sql, db_path, error, payload, branches):
    counts = {}
    for br, segs in enumerate(branches):
        sel = next((s for s in segs if s.label == "SELECT"), None)
        counts[f"branch{br + 1}"] = _count_select_exprs(sel, sql)
    diag = ("UNION/INTERSECT/EXCEPT 两侧 SELECT 列数不一致："
            + "，".join(f"分支{k + 1}={v}" for k, v in enumerate(counts.values()))
            + "。按 EXPO Table 15 在集合运算边界切开分别核对各分支 SELECT 列数。")
    hint = ("Root cause clause: UNION branches. Column counts differ across "
            "set-operation branches ("
            + ", ".join(f"branch{i + 1}={v}" for i, v in enumerate(counts.values()))
            + ").")
    return _make_loc(True, CAT_SETOP, "error_trace", ["UNION"], None, [],
                     "集合运算两侧列数不一致", diag, hint, {"counts": counts})


def _loc_subselect_cols(sql, db_path, error, payload, branches):
    n, m = payload.get("n"), payload.get("m")
    segs = _segments_flat(branches)
    in_seg = None
    for s in segs:
        if re.search(r"\bIN\s*\(", s.text, re.I):
            in_seg = s
            break
    label = in_seg.label if in_seg else ("WHERE" if any(
        s.label == "WHERE" for s in segs) else "SELECT")
    diag = (f"sub-select returns {n} columns - expected {m}：子查询输出列数与 "
            f"外部期望不符（常见于 IN/EXISTS/标量子查询）。位置 {label}。")
    hint = (f"Root cause clause: {label}. A subquery returns {n} columns where "
            f"{m} are expected.")
    return _make_loc(True, CAT_SETOP, "error_trace", [label], label, [],
                     f"子查询列数不匹配（{n} vs {m}）", diag, hint,
                     {"n": n, "m": m})


def _loc_compound_terms(sql, db_path, error, payload, branches):
    diag = ("too many terms in compound select：UNION 分支过多。检查是否误用"
            "逗号/OR 导致分支爆炸。")
    hint = ("Root cause clause: UNION. Too many compound-select terms; check "
            "for accidental commas/OR in place of UNION.")
    return _make_loc(True, CAT_SETOP, "error_trace", ["UNION"], None, [],
                     "UNION 分支过多", diag, hint, {})


# ---------------------------------------------------------------------------
# Algorithm 1：结果错误分支（clause-wise incremental execution + 10 类 diff）
# ---------------------------------------------------------------------------

def _normalize_row(row: Sequence[Any]) -> Tuple[Any, ...]:
    out = []
    for v in row:
        if isinstance(v, bytes):
            try:
                v = v.decode("utf-8", errors="ignore")
            except Exception:
                v = repr(v)
        elif isinstance(v, (int, float, str, type(None))):
            v = v
        else:
            v = str(v)
        out.append(v)
    return tuple(out)


def classify_diff(pred_cols: Sequence[str], pred_rows: Sequence[Sequence[Any]],
                  gold_cols: Sequence[str], gold_rows: Sequence[Sequence[Any]]) -> Optional[str]:
    """EXPO Table 12/13：两结果集 → 10 类 diff_type 之一；完全一致返回 None。"""
    pcols = list(pred_cols)
    gcols = list(gold_cols)
    if len(pcols) != len(gcols):
        return "col_count"
    if pcols != gcols:
        return "col_name"
    p_rows = [_normalize_row(r) for r in pred_rows]
    g_rows = [_normalize_row(r) for r in gold_rows]
    if not p_rows and g_rows:
        return "row_emptied"
    if p_rows and not g_rows:
        return "row_created"
    if p_rows == g_rows:
        return None  # 完全一致
    pc = Counter(p_rows)
    gc = Counter(g_rows)
    if pc == gc:
        return "row_order"
    if set(pc) == set(gc):
        return "row_dedup"
    if set(pc) <= set(gc) and all(pc[r] <= gc[r] for r in pc):
        return "row_subset"
    if set(gc) <= set(pc) and all(gc[r] <= pc[r] for r in gc):
        return "row_superset"
    if set(pc).isdisjoint(set(gc)):
        return "row_disjoint"
    return "row_partial"


def _exec(sql: str, db_path: str, row_cap: int = 100_000) -> Optional[Dict[str, Any]]:
    """执行单条 SQL，返回 {"cols": [...], "rows": [...], "ok": bool, "error": str}。"""
    try:
        uri = Path(db_path).resolve().as_uri()
        con = sqlite3.connect(f"{uri}?mode=ro", uri=True, timeout=5)
        con.text_factory = lambda b: b.decode(errors="ignore")
        ticks = [0]

        def _handler() -> int:
            ticks[0] += 1000
            return 1 if ticks[0] >= 5_000_000 else 0

        con.set_progress_handler(_handler, 1000)
        try:
            cur = con.execute(sql)
            rows = []
            total = 0
            for row in cur.fetchmany(row_cap + 1):
                total += 1
                if len(rows) < row_cap:
                    rows.append(list(row))
            cols = [d[0] for d in cur.description] if cur.description else []
            return {"ok": True, "cols": cols, "rows": rows,
                    "truncated": total > row_cap, "error": None}
        finally:
            con.close()
    except Exception as exc:
        return {"ok": False, "cols": [], "rows": [], "truncated": False,
                "error": str(exc)}


def incremental_analyze(sql: str, db_path: str,
                        gold_cols: Sequence[str],
                        gold_rows: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    """EXPO Algorithm 1（A.2）：子句级增量执行定位 C_err。

    - 预处理：SELECT 换成 SELECT *（有 GROUP BY 时 SQLite 宽松语义仍可执行，
      与论文"SELECT+GROUP BY 作为一个单元"的差异见下方 note）；
    - 按逻辑执行顺序逐步拼接：FROM/JOIN → WHERE → GROUP BY → HAVING →
      SELECT（恢复原投影）→ ORDER BY → LIMIT；
    - d1 = diff(Rgold, R1)（只看 col_count）；di = diff(R_{i-1}, R_i)；
      D = diff(Rgold, Rpred)；ci 判错当且仅当 di == D（非 None）。

    与论文的实现差异（记录于 details.notes）：
      1) 论文在 GROUP BY 存在时把 SELECT+GROUP BY 作为一个单元（SELECT * 违反
         分组约束）；SQLite 允许 bare columns，故本实现仍逐步分离，di 反映
         各步独立引入的差异。
      2) 顶层含 UNION/INTERSECT/EXCEPT 时不分解（返回 not_localized，
         Table 15 要求按分支独立分析，留待需要时扩展）。
    """
    branches = split_sql_clauses(sql)
    if branches is None:
        return {"ok": False, "method": "incremental", "err_clauses": [],
                "diffs": {}, "D": None, "note": "无法切分子句"}
    if len(branches) > 1:
        return {"ok": False, "method": "incremental", "err_clauses": [],
                "diffs": {}, "D": None,
                "note": "顶层集合运算暂不分解（Table 15 需按分支独立分析）"}
    segs = branches[0]
    # 提取各段文本（按文本顺序；WITH 前缀独立前置）
    parts: Dict[str, str] = defaultdict(str)
    order_seen: List[str] = []
    with_text = ""
    for seg in segs:
        if seg.label == "WITH":
            with_text = seg.text.strip()
            continue
        if seg.label == "FROM" or seg.label == "JOIN":
            parts["FROMJOIN"] += "\n" + seg.text
            if "FROMJOIN" not in order_seen:
                order_seen.append("FROMJOIN")
        else:
            parts[seg.label] = seg.text.strip()
            if seg.label not in order_seen:
                order_seen.append(seg.label)
    if "FROMJOIN" not in parts:
        # 无 FROM（SELECT 1 之类）：只做 SELECT 步
        steps = [("SELECT", parts.get("SELECT", ""))]
    else:
        fromjoin = parts["FROMJOIN"].strip()
        pre = (with_text + "\n" if with_text else "")
        steps = []
        steps.append(("FROM/JOIN", f"{pre}SELECT * {fromjoin}"))
        if "WHERE" in parts:
            steps.append(("WHERE", f"{pre}SELECT * {fromjoin} {parts['WHERE']}"))
        if "GROUP BY" in parts:
            steps.append(("GROUP BY",
                          f"{pre}SELECT * {fromjoin} {parts.get('WHERE', '')} "
                          f"{parts['GROUP BY']}"))
        if "HAVING" in parts:
            steps.append(("HAVING",
                          f"{pre}SELECT * {fromjoin} {parts.get('WHERE', '')} "
                          f"{parts.get('GROUP BY', '')} {parts['HAVING']}"))
        sel_sql = f"{pre}{parts.get('SELECT', '*')} {fromjoin} " \
                  f"{parts.get('WHERE', '')} {parts.get('GROUP BY', '')} " \
                  f"{parts.get('HAVING', '')}"
        steps.append(("SELECT", sel_sql.strip()))
        if "ORDER BY" in parts:
            steps.append(("ORDER BY", (sel_sql + " " + parts["ORDER BY"]).strip()))
        if "LIMIT" in parts:
            steps.append(("LIMIT",
                          (sel_sql + " " + parts.get("ORDER BY", "") + " "
                           + parts["LIMIT"]).strip()))

    r_pred = _exec(sql, db_path)
    diffs: Dict[str, Optional[str]] = {}
    errors: Dict[str, str] = {}
    prev = None
    for label, q in steps:
        r = _exec(q, db_path)
        if not r["ok"]:
            errors[label] = r["error"]
            diffs[label] = None
            prev = r
            continue
        if prev is None:
            # d1 = diff(Rgold, R1)，只看 col_count
            d1 = "col_count" if len(r["cols"]) != len(list(gold_cols)) else None
            diffs[label] = d1
        else:
            if prev.get("ok"):
                diffs[label] = classify_diff(r["cols"], r["rows"],
                                             prev["cols"], prev["rows"])
            else:
                diffs[label] = None
        prev = r
    if not r_pred["ok"]:
        return {"ok": False, "method": "incremental", "err_clauses": [],
                "diffs": diffs, "D": None, "error": r_pred["error"],
                "note": "原句执行失败（应走执行错误分支）"}
    D = classify_diff(r_pred["cols"], r_pred["rows"], list(gold_cols), list(gold_rows))
    if D is None:
        return {"ok": True, "method": "incremental", "err_clauses": [],
                "diffs": diffs, "D": None,
                "note": "R_pred 与 R_gold 一致，无差异可定位"}
    err_clauses = [label for label, d in diffs.items() if d == D]
    # 补充归因：D 为列级差异（col_count/col_name）时，投影差异由 SELECT 引入——
    # d_SELECT = diff(R_*步, R_SELECT) 因 * 替换的列数变化常被掩成 col_count，
    # 而 SELECT 步的列与 R_pred 完全一致，故 D 为列级时 SELECT 必是来源之一。
    if D in ("col_count", "col_name") and "SELECT" in diffs and \
            diffs["SELECT"] != D and "SELECT" not in err_clauses:
        err_clauses.append("SELECT")
    err_clauses = sorted(set(err_clauses),
                         key=lambda c: _CLAUSE_RANK.get(c, 99))
    diagnosis = (f"最终差异 D={D}（{DIFF_CLAUSE_HINT.get(D, '')}）。"
                 f"逐步差异："
                 + "；".join(f"{l}:{d or '-'}" for l, d in diffs.items())
                 + f"。C_err = {err_clauses or '（空：差异由单步无法归因）'}。")
    return {"ok": bool(err_clauses), "method": "incremental",
            "err_clauses": err_clauses, "diffs": diffs, "D": D,
            "diagnosis": diagnosis,
            "notes": [
                "GROUP BY 存在时论文将 SELECT+GROUP BY 合并为一个单元；SQLite "
                "宽松 bare-column 语义下本实现逐步分离。",
                "顶层 UNION/INTERSECT/EXCEPT 未分解（Table 15 分支独立分析未实现）。",
            ]}


# ---------------------------------------------------------------------------
# 修复 prompt 注入文本
# ---------------------------------------------------------------------------

def render_repair_hint(loc: Dict[str, Any], max_chars: int = 260) -> str:
    """把定位结果渲染为可注入修复 prompt 的英文提示段（空定位返回空串）。"""
    if not loc or not loc.get("ok") or not loc.get("err_clauses"):
        return ""
    clauses = "+".join(loc["err_clauses"])
    body = loc.get("hint_en") or loc.get("root_cause") or ""
    if re.match(r"root cause clause", body, re.IGNORECASE):
        hint = f"[Error localization] {body}"
    else:
        hint = f"[Error localization] Root-cause clause(s): {clauses}. {body}"
    if len(hint) > max_chars:
        hint = hint[: max_chars - 3].rstrip() + "…"
    return hint


# ---------------------------------------------------------------------------
# CLI：离线诊断扫描
# ---------------------------------------------------------------------------

def _load_items(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        data = data["items"]
    if not isinstance(data, list):
        raise ValueError(f"items 结构异常（期望 list）: {path}")
    return data


def run_scan(args: argparse.Namespace) -> None:
    """对候选池执行失败样本批量定位 + 统计 + 50 条人工抽查样本。"""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "exec_cache.jsonl"
    t0 = time.perf_counter()

    items = _load_items(Path(args.items))
    if args.limit:
        items = items[: args.limit]
    n_candidates = sum(len(it.get("candidates") or []) for it in items)

    # 恢复执行缓存：(sql, db) -> outcome
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    cache[(rec["sql"], rec["db"])] = rec["outcome"]
                except Exception:
                    continue
    print(f"[expo] cache loaded: {len(cache)} entries", file=sys.stderr)

    engine = AP.ExecutionEngine(args.threads, args.query_timeout,
                                args.max_vm_steps, args.row_cap)
    # 收集唯一 (sql, db) 任务
    tasks: Dict[Tuple[str, str], None] = {}
    gold_by_q: Dict[int, Tuple[str, str, str]] = {}  # dataset_index -> (db, gold_sql)
    for it in items:
        db_id = it.get("db_id", "")
        db = Path(args.db_root) / db_id / f"{db_id}.sqlite"
        if not Path(db).is_file():
            continue
        for c in it.get("candidates") or []:
            sql = (c.get("sql") or "").strip()
            if sql:
                tasks[(sql, str(db))] = None
        g = (it.get("gold_sql") or "").strip()
        if g:
            gold_by_q[it.get("dataset_index", it.get("di"))] = (str(db), g, db_id)
    todo = [(s, d) for (s, d) in tasks if (s, d) not in cache]
    print(f"[expo] unique (sql,db) tasks: {len(tasks)}; to execute: {len(todo)}",
          file=sys.stderr)
    exec_wall = 0.0
    if todo:
        # 分块执行 + 逐块落缓存（断点续跑：中断后重跑只补缺口）
        CHUNK = 2000
        for ci in range(0, len(todo), CHUNK):
            chunk = todo[ci:ci + CHUNK]
            engine.run(chunk, phase="expo_diag")
            exec_wall += engine._stats["expo_diag"]["wall_seconds"]
            with open(cache_path, "a", encoding="utf-8") as fh:
                for (s, d) in chunk:
                    outcome = engine.get(s, d)
                    fh.write(json.dumps({"sql": s, "db": d, "outcome": outcome},
                                        ensure_ascii=False) + "\n")
            print(f"[expo] executed {ci + len(chunk)}/{len(todo)} "
                  f"({exec_wall:.0f}s)", file=sys.stderr)
        print(f"[expo] execution done in {exec_wall:.1f}s", file=sys.stderr)

    # 逐候选归因：本地化失败样本
    records: List[Dict[str, Any]] = []
    gold_rows_cache: Dict[int, Optional[Dict[str, Any]]] = {}
    stats_cat: Dict[str, int] = defaultdict(int)
    stats_method: Dict[str, int] = defaultdict(int)
    stats_clause: Dict[str, int] = defaultdict(int)
    stats_model: Dict[str, Counter] = defaultdict(Counter)
    stats_nonquery = 0
    fallback_msgs: Counter = Counter()
    n_ok = n_fail = 0

    for it in items:
        db_id = it.get("db_id", "")
        db = Path(args.db_root) / db_id / f"{db_id}.sqlite"
        if not Path(db).is_file():
            continue
        for c in it.get("candidates") or []:
            sql = (c.get("sql") or "").strip()
            if not sql:
                continue
            outcome = cache.get((sql, str(db))) or engine.get(sql, str(db))
            model = c.get("model") or "unknown"
            if outcome["ok"]:
                n_ok += 1
                stats_model[model]["ok"] += 1
                continue
            n_fail += 1
            stats_model[model]["fail"] += 1
            loc = localize(sql, str(db), outcome.get("error"),
                           outcome.get("error_type"))
            stats_cat[loc["category"]] += 1
            stats_method[loc["method"]] += 1
            brs = split_sql_clauses(sql)
            non_query = (brs is None or not any(
                s.label in ("SELECT", "FROM", "JOIN") for b in brs for s in b))
            if non_query:
                stats_nonquery += 1
            if loc.get("ok"):
                for cl in loc["err_clauses"]:
                    stats_clause[cl] += 1
                stats_model[model]["localized"] += 1
            else:
                fallback_msgs[(outcome.get("error") or "")[:120]] += 1
            records.append({
                "dataset_index": it.get("dataset_index", it.get("di")),
                "db_id": db_id,
                "question": (it.get("question") or "")[:200],
                "gold_sql": it.get("gold_sql") or "",
                "model": model,
                "parse_success": bool(c.get("parse_success")),
                "non_query": bool(non_query),
                "sql": sql,
                "error_type": outcome.get("error_type"),
                "error": outcome.get("error"),
                "loc": {k: v for k, v in loc.items() if k != "diagnosis"},
                "diagnosis": loc.get("diagnosis", ""),
            })

    # ---- Algorithm 1 冒烟样本（错误结果分支，P2A-3 前置验证）----
    incr_stats: Dict[str, Any] = {"attempted": 0, "run": 0, "localized": 0,
                                  "diff_dist": Counter(), "clause_dist": Counter(),
                                  "notes": []}
    incr_records: List[Dict[str, Any]] = []
    if args.incremental_sample > 0:
        # 每题取最多 1 个"可执行但错"的候选（去重后），打乱取 N
        import random
        cand_wrong: List[Tuple[Any, str, str]] = []
        for it in items:
            db_id = it.get("db_id", "")
            db = Path(args.db_root) / db_id / f"{db_id}.sqlite"
            if not Path(db).is_file():
                continue
            gold = gold_by_q.get(it.get("dataset_index", it.get("di")))
            if not gold:
                continue
            if it.get("dataset_index", it.get("di")) not in gold_rows_cache:
                g = _exec(gold[1], str(db))
                gold_rows_cache[it.get("dataset_index", it.get("di"))] = \
                    g if g and g["ok"] else None
            gres = gold_rows_cache[it.get("dataset_index", it.get("di"))]
            if not gres:
                continue
            seen = set()
            for c in it.get("candidates") or []:
                sql = (c.get("sql") or "").strip()
                if not sql or sql in seen:
                    continue
                seen.add(sql)
                outcome = cache.get((sql, str(db))) or engine.get(sql, str(db))
                if not outcome["ok"]:
                    continue
                if classify_diff(outcome.get("cols", []), outcome.get("rows", []),
                                 gres["cols"], gres["rows"]) is not None:
                    cand_wrong.append((it, sql, str(db)))
                    break
        random.Random(args.seed).shuffle(cand_wrong)
        for it, sql, db in cand_wrong[: args.incremental_sample]:
            incr_stats["attempted"] += 1
            gres = gold_rows_cache[it.get("dataset_index", it.get("di"))]
            res = incremental_analyze(sql, db, gres["cols"], gres["rows"])
            incr_stats["run"] += 1
            if res.get("ok") and res.get("err_clauses"):
                incr_stats["localized"] += 1
            if res.get("D"):
                incr_stats["diff_dist"][res["D"]] += 1
            for cl in res.get("err_clauses", []):
                incr_stats["clause_dist"][cl] += 1
            incr_records.append({
                "dataset_index": it.get("dataset_index", it.get("di")),
                "db_id": it.get("db_id"),
                "sql": sql, "gold_sql": it.get("gold_sql"),
                "res": {k: (dict(v) if isinstance(v, Counter) else v)
                        for k, v in res.items()},
            })
        incr_stats["diff_dist"] = dict(incr_stats["diff_dist"])
        incr_stats["clause_dist"] = dict(incr_stats["clause_dist"])

    # ---- 50 条人工抽查样本（按类别分层）----
    sample = _stratified_sample(records, args.sample_size)
    with open(out_dir / "manual_sample_50.json", "w", encoding="utf-8") as fh:
        json.dump(sample, fh, ensure_ascii=False, indent=2)

    # ---- 统计汇总 ----
    n_localized = sum(1 for r in records if r["loc"]["ok"])
    stats = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": "src/expo_localize.py run_scan",
            "items_file": str(Path(args.items).resolve()),
            "db_root": str(Path(args.db_root).resolve()),
            "limit": args.limit,
        },
        "pool": {
            "n_items": len(items),
            "n_candidates": n_candidates,
            "n_unique_tasks": len(tasks),
            "n_executed": len(todo),
            "n_exec_ok": n_ok,
            "n_exec_fail": n_fail,
            "exec_fail_rate": round(n_fail / max(n_ok + n_fail, 1), 4),
        },
        "localization": {
            "n_localized": n_localized,
            "n_fail_records": n_fail,
            "success_rate": round(n_localized / max(n_fail, 1), 4),
            "n_non_query_failures": stats_nonquery,
            "category_dist": dict(sorted(stats_cat.items())),
            "method_dist": dict(sorted(stats_method.items())),
            "clause_dist": dict(sorted(stats_clause.items(),
                                       key=lambda kv: -kv[1])),
            "model_dist": {m: dict(c) for m, c in sorted(stats_model.items())},
            "top_unlocalized_errors": fallback_msgs.most_common(30),
        },
        "incremental_sample": incr_stats,
        "incremental_records_file": "incremental_sample.json",
        "records_file": "localize_records.jsonl",
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    with open(out_dir / "localize_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    with open(out_dir / "localize_records.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "incremental_sample.json", "w", encoding="utf-8") as fh:
        json.dump(incr_records, fh, ensure_ascii=False, indent=2)

    print("\n=== EXPO 定位离线诊断 ===")
    print(f"候选池: {len(items)} 题 / {n_candidates} 候选 / "
          f"{len(tasks)} 唯一 (sql,db)；执行失败 {n_fail} "
          f"({stats['pool']['exec_fail_rate']:.1%})")
    print(f"定位成功率: {n_localized}/{n_fail} = "
          f"{stats['localization']['success_rate']:.1%}")
    print(f"错误类分布: {stats['localization']['category_dist']}")
    print(f"方法分布: {stats['localization']['method_dist']}")
    print(f"根因子句分布(top10): "
          f"{list(stats['localization']['clause_dist'].items())[:10]}")
    print(f"产物: {out_dir}/localize_stats.json | localize_records.jsonl | "
          f"manual_sample_50.json | incremental_sample.json")


def _stratified_sample(records: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    import random
    # 只对"已定位"记录分层（人工可读性抽查对象 = 定位诊断）；
    # 未定位/引擎层错误由 stats.top_unlocalized_errors 另行覆盖。
    loc_recs = [r for r in records if r["loc"].get("ok") and r["loc"].get("err_clauses")]
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in loc_recs:
        by_cat[r["loc"]["category"]].append(r)
    cats = sorted(by_cat, key=lambda c: -len(by_cat[c]))
    per_cat = max(3, n // max(len(cats), 1))
    sample: List[Dict[str, Any]] = []
    for c in cats:
        pool = by_cat[c]
        random.Random(42).shuffle(pool)
        sample.extend(pool[:per_cat])
        if len(sample) >= n:
            break
    if len(sample) > n:
        sample = sample[:n]
    out = []
    for r in sample:
        out.append({
            "dataset_index": r["dataset_index"],
            "db_id": r["db_id"],
            "question": r["question"],
            "model": r["model"],
            "gold_sql": r["gold_sql"],
            "sql": r["sql"],
            "error": r["error"],
            "category": r["loc"]["category"],
            "method": r["loc"]["method"],
            "err_clauses": r["loc"]["err_clauses"],
            "anchor_clause": r["loc"]["anchor_clause"],
            "root_cause": r["loc"]["root_cause"],
            "hint_en": r["loc"]["hint_en"],
            "diagnosis": r["diagnosis"],
            "details": r["loc"].get("details", {}),
        })
    return out


def _cmd_one(args: argparse.Namespace) -> None:
    sql = Path(args.sql).read_text(encoding="utf-8") if os.path.isfile(args.sql) \
        else args.sql
    loc = localize(sql, args.db, args.error, args.error_type)
    print(json.dumps(loc, ensure_ascii=False, indent=2))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EXPO-SQL 错误子句定位（零训练）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_one = sub.add_parser("one", help="单条 SQL 定位")
    p_one.add_argument("--sql", required=True, help="错误 SQL 文本或文件路径")
    p_one.add_argument("--db", default=None, help="数据库文件路径")
    p_one.add_argument("--error", default=None, help="错误信息（缺省自动执行获取）")
    p_one.add_argument("--error-type", default=None)
    p_one.set_defaults(func=_cmd_one)

    p_scan = sub.add_parser("scan", help="候选池离线诊断扫描")
    p_scan.add_argument("--items", required=True)
    p_scan.add_argument("--db-root", required=True,
                        help="BIRD dev_databases 根目录（<db_id>/<db_id>.sqlite）")
    p_scan.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "expo_diag"))
    p_scan.add_argument("--threads", type=int, default=16)
    p_scan.add_argument("--query-timeout", type=float, default=30.0)
    p_scan.add_argument("--max-vm-steps", type=int, default=5_000_000)
    p_scan.add_argument("--row-cap", type=int, default=100_000)
    p_scan.add_argument("--limit", type=int, default=None, help="只处理前 N 题（冒烟）")
    p_scan.add_argument("--sample-size", type=int, default=50)
    p_scan.add_argument("--incremental-sample", type=int, default=100,
                        help="Algorithm 1 冒烟样本数（0=关）")
    p_scan.add_argument("--seed", type=int, default=42)
    p_scan.set_defaults(func=run_scan)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
