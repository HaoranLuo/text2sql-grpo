#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SEED 式自动 evidence 的共享核心（纯 stdlib，纯 CPU 离线，无 LLM 调用）。

对应 tmp_idea_research/bird_gen_scan_report.md 的 R3 建议，复刻
tmp_idea_research/bird_gen_scan/code/SEED/make_evidence.py 的三件套构造，
但把 SEED 的 3 次 LLM 调用（schema 摘要 / 关键词抽取 / evidence 生成）
替换为确定性规则：

  1. 库结构摘要（SEED: generate_schema + read_schema_description +
     concat_schema_and_desc）：sqlite_master 的 CREATE TABLE DDL + 每列
     注释（BIRD database_description CSV）/ 主外键标记。
  2. 样例数据行（SEED: get_sample_query_result）：每表前 N 行，以
     col=value 形式渲染。
  3. question 关键词→列名/值映射（SEED: keyword_extract_llm +
     编辑距离查库值）：词面匹配（含单复数变体）+ difflib 模糊匹配列名；
     值接地（规范化后大小写不敏感相等 / 包含 / 模糊 ≥0.88）查库内真实值，
     等价于 SEED 的 "closest in edit distance" 提示。

设计约定：
  - 确定性：同一 db + question 输入，输出恒定（含乱序去重值排序）。
  - 长度预算：evidence 单条硬上限 MAX_EVIDENCE_CHARS，按 摘要→关键词→
    样例 优先级裁剪，保证 prompt 增量可控（BIRD 实测 prompt 最长 2386
    token，+evidence 后仍远低于 3072 截断）。
  - 只读 sqlite（mode=ro），不写库、不联网。
"""
from __future__ import annotations

import csv
import difflib
import os
import re
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 参数（由两个入口脚本覆写）
# ---------------------------------------------------------------------------

MAX_EVIDENCE_CHARS = 1800          # 单条 evidence 硬上限（字符）
MAX_COL_COMMENT_CHARS = 80         # 列注释截断
MAX_CELL_CHARS = 32                # 样例单元格截断
MAX_VALUE_CHARS = 60               # 值索引单值截断
SAMPLE_ROWS = 3                    # 每表样例行数
SAMPLE_TABLES = 4                  # 样例段最多表数
MAX_VALUES_PER_COL = 60            # 值索引每列最多 distinct 值
VALUE_SCAN_ROWS = 20000            # 值索引每列最多扫描行
MAX_VALUE_IDX_CHARS = 40           # 值接地索引单值上限（长自由文本不入索引，防噪声）
KEYWORD_FUZZY_COL = 0.62           # 关键词→列名模糊匹配阈值
KEYWORD_FUZZY_VAL = 0.88           # 关键词→库值模糊匹配阈值
SCHEMA_CAP_CHARS = 1100            # 摘要段预算
SAMPLE_CAP_CHARS = 600             # 样例段预算
KEYWORD_CAP_CHARS = 520            # 关键词段预算
MIN_VALUE_TOK_LEN = 3              # 值接地最小 token 长（含 4 位年份）

STOPWORDS = set("""
a an the is are was were be been being of to in on at for with by from and or not
no nor what which who whom whose how many much most more most less least each every
all any some such than then that this these those it its their there here where when
why per our your above below between into over under again further once out up down
find get give show display list count sum average avg min max total number value
values information detail details record records data database table tables column
columns question answer queries query please would could should can will may does
do did done has have had having year month day date time first last
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_WS_RE = re.compile(r"\s+")
_NONALNUM_RE = re.compile(r"[^a-z0-9]")


def tokenize(text: str) -> List[str]:
    """小写分词（字母数字下划线 token）。"""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> List[str]:
    """内容分词：去停用词，供相关性/匹配使用。"""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def token_variants(tok: str) -> List[str]:
    """单复数变体：'songs'→['songs','song']；'song'→['song','songs']。"""
    out = [tok]
    if tok.endswith("s") and len(tok) >= 4:
        out.append(tok[:-1])
    elif not tok.endswith("s") and len(tok) >= 3:
        out.append(tok + "s")
    return out


def norm_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def norm_value(value: Any) -> str:
    """值规范化：去全部非字母数字 → 小写（'3-D Man'→'3dman'）。"""
    return _NONALNUM_RE.sub("", str(value or "").lower())


def truncate(text: Any, n: int) -> str:
    s = "" if text is None else str(text)
    s = _WS_RE.sub(" ", s).strip()
    if len(s) <= n:
        return s
    return s[: max(n - 3, 1)] + "..."


def _col_type_class(decl_type: str) -> str:
    d = (decl_type or "").lower()
    if any(k in d for k in ("char", "text", "clob", "blob", "binary")):
        return "text"
    if any(k in d for k in ("date", "time")):
        return "date"
    return "number"


def _is_numeric_token(tok: str) -> bool:
    return re.fullmatch(r"[\d.]+", tok) is not None


# ---------------------------------------------------------------------------
# 库 bundle：DDL 解析 + 注释 + 样例行 + 值索引（每库一次，缓存复用）
# ---------------------------------------------------------------------------

class DbBundle:
    """一个 sqlite 库的离线元数据包。"""

    def __init__(self) -> None:
        self.tables: Dict[str, Dict[str, Any]] = {}
        # 列名 token 索引：token -> [(table, col)]（列名/原名 token）
        self.col_name_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # 表名 token 索引：token -> [table]
        self.table_name_index: Dict[str, List[str]] = defaultdict(list)
        # 注释 token 索引：token -> [(table, col)]
        self.comment_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # 值索引（规范化）：norm -> [(value, table, col)]
        self.value_exact: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        # 值首字符桶：first char(norm) -> [(value, table, col)]（模糊检索）
        self.value_buckets: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        # 值 token 索引（原始值分词，len≥2）：token -> [(value, table, col)]
        # 例：'3-D Man' → 'man' -> ('3-D Man', superhero, superhero_name)
        self.value_token_index: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    def add_table(self, name: str, columns: List[Dict[str, Any]],
                  sample_rows: List[Dict[str, Any]]) -> None:
        self.tables[name] = {
            "columns": columns,          # [{name,type,cls,pk,fk,comment}]
            "sample_rows": sample_rows,  # [{col: value}]
        }
        for tok in content_tokens(name):
            self.table_name_index[tok].append(name)
        for c in columns:
            for tok in content_tokens(c["name"]):
                self.col_name_index[tok].append((name, c["name"]))
            for tok in content_tokens(c.get("comment", "")):
                self.comment_index[tok].append((name, c["name"]))

    def add_values(self, table: str, col: str, values: List[str]) -> None:
        for v in values:
            if len(v) > MAX_VALUE_IDX_CHARS:
                continue  # 长自由文本（AboutMe/Body 类）不做值接地
            nv = norm_value(v)
            if len(nv) < 2:
                continue
            self.value_exact[nv].append((v, table, col))
            self.value_buckets[nv[0]].append((v, table, col))
            for tok in tokenize(v):
                if len(tok) >= 2:
                    self.value_token_index[tok].append((v, table, col))


def build_bundle_from_sqlite(db_path: str,
                             comments: Optional[Dict[str, Dict[str, str]]] = None,
                             tables_meta: Optional[Dict[str, Any]] = None,
                             ) -> DbBundle:
    """只读打开 sqlite，构建 DbBundle。

    comments: {table_lower: {col_lower: comment_text}}（BIRD database_description）
    tables_meta: Spider tables.json 单库条目（补充 PK/FK/类型），可 None。
    """
    comments = comments or {}
    meta_cols: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if tables_meta:
        tnames = tables_meta.get("table_names_original") or []
        cnames = tables_meta.get("column_names_original") or []
        ctypes = tables_meta.get("column_types") or []
        pk_idx = set(tables_meta.get("primary_keys") or [])
        fk_pairs = tables_meta.get("foreign_keys") or []
        fk_by_col: Dict[Tuple[int, int], str] = {}
        for pair in fk_pairs:
            # 两种格式：[(from_col_idx, to_col_idx)] 或 [(t1,c1,t2,c2)]
            if len(pair) == 2:
                fi, ti = pair
                t1, c1 = cnames[fi][0], cnames[fi][1]
                t2, c2 = cnames[ti][0], cnames[ti][1]
                fk_by_col[(t1, c1)] = f"{tnames[t2]}.{c2}"
            else:
                t1, c1, t2, c2 = pair
                fk_by_col[(t1, c1)] = f"{tnames[t2]}.{cnames[c2][1]}"
        for i, (ti, ci) in enumerate(cnames):
            if ti < 0:
                continue
            meta_cols[(tnames[ti], ci)] = {
                "type": (ctypes[i] if i < len(ctypes) else ""),
                "pk": i in pk_idx,
                "fk": fk_by_col.get((ti, ci), None),
            }

    bundle = DbBundle()
    uri = f"file:{os.path.abspath(db_path).replace(os.sep, '/')}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY name").fetchall()
        # 先建主键映射：DDL 里 FK 未写目标列时（REFERENCES t 不带列），
        # PRAGMA foreign_key_list 的 to 为 None，用目标表 PK 兜底。
        pk_map: Dict[str, str] = {}
        for tname, _ddl in rows:
            tcols = con.execute(f'PRAGMA table_info("{tname}")').fetchall()
            tpks = [r[1] for r in tcols if r[5] > 0]
            if len(tpks) == 1:
                pk_map[tname] = tpks[0]
        for tname, _ddl in rows:
            cols = con.execute(f'PRAGMA table_info("{tname}")').fetchall()
            pks = {r[1] for r in cols if r[5] > 0}
            fks: Dict[str, str] = {}
            for fk in con.execute(f'PRAGMA foreign_key_list("{tname}")').fetchall():
                # (id, seq, table, from, to, on_update, on_delete, match)
                fk_to = fk[4] if fk[4] is not None else pk_map.get(fk[2], "?")
                fks[fk[3]] = f"{fk[2]}.{fk_to}"
            col_specs = []
            for r in cols:
                cname, ctype = r[1], r[2]
                meta = meta_cols.get((tname, cname), {})
                col_specs.append({
                    "name": cname,
                    "type": meta.get("type") or ctype,
                    "cls": _col_type_class(meta.get("type") or ctype),
                    "pk": cname in pks or bool(meta.get("pk")),
                    "fk": fks.get(cname) or meta.get("fk"),
                    "comment": (comments.get(tname.lower(), {}).get(cname.lower(), "")
                                or comments.get(tname.lower(), {}).get(cname, "")),
                })
            # 样例行（rowid 序，确定性）
            sample_rows: List[Dict[str, Any]] = []
            try:
                cur = con.execute(
                    f'SELECT * FROM "{tname}" LIMIT {SAMPLE_ROWS}')
                colnames = [d[0] for d in cur.description]
                for row in cur.fetchall():
                    sample_rows.append({c: v for c, v in zip(colnames, row)})
            except sqlite3.Error:
                pass
            bundle.add_table(tname, col_specs, sample_rows)
            # 值索引（仅 text 类列，限制扫描行数与 distinct 数）
            for spec in col_specs:
                if spec["cls"] != "text":
                    continue
                try:
                    sql = (f'SELECT DISTINCT v FROM (SELECT substr("{spec["name"]}",1,{MAX_VALUE_CHARS}) '
                           f'AS v FROM "{tname}" WHERE "{spec["name"]}" IS NOT NULL '
                           f'ORDER BY "{spec["name"]}" LIMIT {VALUE_SCAN_ROWS}) LIMIT {MAX_VALUES_PER_COL}')
                    vals = [v[0] for v in con.execute(sql).fetchall()
                            if v[0] not in (None, "")]
                    bundle.add_values(tname, spec["name"], vals)
                except sqlite3.Error:
                    pass
    finally:
        con.close()
    return bundle


def read_bird_comments(desc_dir: str, db_path: str) -> Dict[str, Dict[str, str]]:
    """读 BIRD database_description/*.csv → {table: {col: comment}}。

    CSV 头：original_column_name,column_name,column_description,data_format,
    value_description（部分库无 value_description 列）。
    """
    comments: Dict[str, Dict[str, str]] = {}
    if not desc_dir or not os.path.isdir(desc_dir):
        return comments
    for fname in sorted(os.listdir(desc_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        table = os.path.splitext(fname)[0]
        try:
            with open(os.path.join(desc_dir, fname), "r", encoding="utf-8",
                      errors="ignore", newline="") as fh:
                reader = csv.DictReader(fh)
                cmap: Dict[str, str] = {}
                for row in reader:
                    col = (row.get("original_column_name") or "").strip()
                    if not col:
                        col = (row.get("column_name") or "").strip()
                    if not col:
                        continue
                    desc = (row.get("column_description") or "").strip()
                    if not desc:
                        # 部分库 column_description 为空，退而求其次用 value_description
                        desc = (row.get("value_description") or "").strip()
                    fmt = (row.get("data_format") or "").strip()
                    if fmt and fmt.lower() not in ("text", "integer", "float",
                                                   "number", "real", "bool",
                                                   "boolean", "datetime", "date"):
                        desc = f"[fmt={fmt}] {desc}" if desc else f"[fmt={fmt}]"
                    if desc:
                        cmap[col] = desc
                if cmap:
                    comments.setdefault(table.lower(), {}).update(
                        {k.lower(): v for k, v in cmap.items()})
        except Exception:
            continue
    return comments


# ---------------------------------------------------------------------------
# 值接地（SEED get_sample_query_result 的离线等价物）
# ---------------------------------------------------------------------------

def compute_value_hits(qtokens: List[str], bundle: DbBundle,
                       matched_tokens: Optional[set] = None,
                       ) -> Dict[str, List[Tuple[int, str, str]]]:
    """每个 question token 查库内真实值。

    返回 {tok: [(priority, value, "table.col")]}：
      12 = 规范化相等；11 = 规范化包含；10 = 模糊（≥KEYWORD_FUZZY_VAL）。
    matched_tokens：已命中列名的 token 跳过（避免 "id" 类噪声）。
    """
    matched_tokens = matched_tokens or set()
    hits: Dict[str, List[Tuple[int, str, str]]] = {}
    for tok in qtokens:
        if tok in matched_tokens or len(tok) < MIN_VALUE_TOK_LEN:
            continue
        if _is_numeric_token(tok) and len(tok) < 4:
            continue
        ntok = norm_value(tok)
        if len(ntok) < 2:
            continue
        found: List[Tuple[int, str, str]] = []
        seen_refs: set = set()

        def add_found(prio: int, v: str, ref: str) -> None:
            if ref in seen_refs:
                return
            if len(found) >= 3:
                return
            seen_refs.add(ref)
            found.append((prio, v, ref))

        # 4a 规范化相等
        for v, t, c in bundle.value_exact.get(ntok, []):
            add_found(12, v, f"`{t}`.`{c}`")
        # 4b 值 token 包含（'man' → '3-D Man'）
        if len(found) < 3:
            for v, t, c in bundle.value_token_index.get(tok, []):
                add_found(11, v, f"`{t}`.`{c}`")
        # 4c 规范化子串（同首字符桶；长度守卫防 'you'⊂'youngest' 类噪声）
        if len(found) < 3 and ntok[0] in bundle.value_buckets:
            for v, t, c in bundle.value_buckets[ntok[0]]:
                nv = norm_value(v)
                if (len(nv) >= 3 and ntok in nv) or (len(nv) >= 4 and nv in ntok):
                    add_found(11, v, f"`{t}`.`{c}`")
        # 4d 模糊（≥KEYWORD_FUZZY_VAL，仅 ≥4 字符 token）
        if not found and len(tok) >= 4 and ntok[0] in bundle.value_buckets:
            best: List[Tuple[float, str, str, str]] = []
            for v, t, c in bundle.value_buckets[ntok[0]]:
                r = difflib.SequenceMatcher(None, ntok, norm_value(v)).ratio()
                if r >= KEYWORD_FUZZY_VAL:
                    best.append((r, v, t, c))
            best.sort(key=lambda x: -x[0])
            for r, v, t, c in best[:3]:
                add_found(10, v, f"`{t}`.`{c}`")
        if found:
            hits[tok] = found
    return hits


# ---------------------------------------------------------------------------
# 相关性打分与三件套拼装
# ---------------------------------------------------------------------------

def table_relevance(qtokens: List[str], bundle: DbBundle, tname: str,
                    value_hits: Dict[str, List[Tuple[int, str, str]]]) -> int:
    """表与 question 的相关分：表名/列名 token 命中 + 注释命中 + 值命中。"""
    tinfo = bundle.tables[tname]
    tname_toks = set(content_tokens(tname))
    col_toks = set()
    for c in tinfo["columns"]:
        col_toks.update(content_tokens(c["name"]))
    score = 0
    comment_tables: Dict[str, int] = defaultdict(int)
    for tok in qtokens:
        for vt in token_variants(tok):
            if vt in tname_toks:
                score += 3
            if vt in col_toks:
                score += 2
            for t, _c in bundle.comment_index.get(vt, []):
                comment_tables[t] = max(comment_tables[t], 1)
        if tok in value_hits and any(
                ref.startswith(f"`{tname}`.")
                for _p, _v, ref in value_hits[tok]):
            score += 4
    return score + comment_tables.get(tname, 0)


def _col_spec_line(spec: Dict[str, Any], with_comment: bool) -> str:
    marks = []
    if spec["pk"]:
        marks.append("PK")
    if spec["fk"]:
        marks.append(f"FK→{spec['fk']}")
    parts = [f"{spec['name']}:{spec['type']}"]
    if marks:
        parts.append("(" + ",".join(marks) + ")")
    if with_comment and spec.get("comment"):
        parts.append('"' + truncate(spec["comment"], MAX_COL_COMMENT_CHARS) + '"')
    return " ".join(parts)


def build_schema_section(bundle: DbBundle,
                         rel: Dict[str, int]) -> str:
    """第 1 件套：库结构摘要（全表；注释仅对相关表保留，超预算再降级）。"""
    order = sorted(bundle.tables, key=lambda t: (-rel[t], t))
    lines_full = []
    lines_bare = []
    for t in order:
        specs = bundle.tables[t]["columns"]
        with_c = rel[t] > 0
        line = f"{t}({', '.join(_col_spec_line(s, with_c) for s in specs)})"
        if with_c:
            lines_full.append(line)
        else:
            lines_bare.append(line)
    ordered = lines_full + lines_bare
    text = "\n".join(ordered)
    if len(text) <= SCHEMA_CAP_CHARS:
        return text
    # 降级 1：无关表只留列名（去类型/注释）
    compact = []
    for t in order:
        specs = bundle.tables[t]["columns"]
        if rel[t] > 0:
            compact.append(f"{t}({', '.join(_col_spec_line(s, True) for s in specs)})")
        else:
            compact.append(f"{t}({', '.join(s['name'] for s in specs)})")
    text = "\n".join(compact)
    if len(text) <= SCHEMA_CAP_CHARS:
        return text
    # 降级 2：只保留相关表 + 兜底前 8 表
    kept = []
    for t in order:
        if rel[t] <= 0:
            continue
        specs = bundle.tables[t]["columns"]
        kept.append(f"{t}({', '.join(_col_spec_line(s, False) for s in specs)})")
    return "\n".join(kept) if kept else "\n".join(compact[:8])


def build_sample_section(bundle: DbBundle, rel: Dict[str, int]) -> str:
    """第 2 件套：关键表样例数据行（top-SAMPLE_TABLES 相关表 × SAMPLE_ROWS 行）。"""
    cand = sorted(bundle.tables, key=lambda t: (-rel[t], t))
    # 小库（≤6 表）全量给样例；大库取相关 top-K + 兜底非空表
    if len(cand) <= 6:
        picked = cand
    else:
        picked = [t for t in cand if rel[t] > 0][:SAMPLE_TABLES]
        if len(picked) < 2:
            picked = cand[:SAMPLE_TABLES]
    picked = picked[:SAMPLE_TABLES]
    lines: List[str] = []
    budget = SAMPLE_CAP_CHARS
    for t in picked:
        rows = bundle.tables[t]["sample_rows"]
        if not rows:
            continue
        for i, row in enumerate(rows[:SAMPLE_ROWS]):
            cells = ", ".join(
                f"{c}={truncate(v, MAX_CELL_CHARS)}"
                for c, v in row.items())
            line = f"{t} #{i + 1}: {cells}"
            if sum(len(x) + 1 for x in lines) + len(line) > budget:
                break
            lines.append(line)
    return "\n".join(lines)


def build_keyword_section(qtokens: List[str], bundle: DbBundle,
                          value_hits: Dict[str, List[Tuple[int, str, str]]],
                          rel: Dict[str, int]) -> str:
    """第 3 件套：关键词→列名/表名映射 + 值接地提示。"""
    hits: List[Tuple[int, str]] = []  # (priority, text)

    def add(prio: int, text: str) -> None:
        hits.append((prio, text))

    matched_tokens: set = set()

    for tok in qtokens:
        if tok in STOPWORDS or len(tok) < 2:
            continue
        # 1) 列名 token 精确命中（含单复数变体）
        cols: List[Tuple[str, str]] = []
        seen = set()
        for vt in token_variants(tok):
            for t, c in bundle.col_name_index.get(vt, []):
                if (t, c) in seen:
                    continue
                seen.add((t, c))
                cols.append((t, c))
        for t, c in cols[:3]:
            add(10, f"{tok} -> `{t}`.`{c}`")
            matched_tokens.add(tok)
        # 2) 表名 token 命中（无列命中时）
        tables: List[str] = []
        if not cols:
            tables = [t for vt in token_variants(tok)
                      for t in bundle.table_name_index.get(vt, [])]
            for t in tables[:2]:
                add(9, f"{tok} ~ table `{t}`")
                matched_tokens.add(tok)
        # 3) 注释 token 命中（语义线索；已命中表名则跳过，避免列级噪声）
        if not cols and not tables:
            for vt in token_variants(tok):
                for t, c in bundle.comment_index.get(vt, [])[:2]:
                    cmt = truncate(
                        next((s["comment"] for s in bundle.tables[t]["columns"]
                              if s["name"] == c), ""), 50)
                    add(8, f"{tok} ~ `{t}`.`{c}` ({cmt})")
                    matched_tokens.add(tok)

    # 4) 列名模糊匹配（≥KEYWORD_FUZZY_COL 且同首字符，限 2 条，仅未精确命中的 token）
    col_names = [(t, s["name"]) for t in bundle.tables
                 for s in bundle.tables[t]["columns"]]
    for tok in qtokens:
        if tok in matched_tokens or len(tok) < 3:
            continue
        cands = sorted(
            ((difflib.SequenceMatcher(None, tok, norm_name(c)).ratio(), t, c)
             for t, c in col_names if norm_name(c)[:1] == tok[:1]),
            reverse=True)[:5]
        n = 0
        for ratio, t, c in cands:
            if ratio < KEYWORD_FUZZY_COL or n >= 2:
                break
            add(6, f"{tok} ≈ `{t}`.`{c}` ({ratio:.2f})")
            n += 1

    # 5) 值接地
    for tok, found in value_hits.items():
        for prio, v, ref in found[:3]:
            op = "=" if prio >= 12 else "~" if prio == 11 else "≈"
            add(prio, f"{tok} {op} '{v}' in {ref}")

    # 6) 相关表提示
    top_tables = [t for t in sorted(bundle.tables, key=lambda t: -rel[t])
                  if rel[t] > 0][:6]
    if top_tables:
        add(20, "tables likely needed: " + ", ".join(top_tables))

    hits.sort(key=lambda x: (-x[0], x[1]))
    lines: List[str] = []
    seen_lines: set = set()
    for _p, text in hits:
        if text in seen_lines:
            continue
        seen_lines.add(text)
        lines.append(text)
    out, used = [], 0
    for line in lines:
        if used + len(line) + 1 > KEYWORD_CAP_CHARS:
            continue
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


def build_evidence(question: str, bundle: DbBundle) -> str:
    """三件套拼装 + 总长硬上限裁剪。

    预算优先级：摘要 > 关键词映射 > 样例行（与 SEED 的 schema summary
    先行、sample SQL 次之的构造顺序一致；超预算先裁样例段）。
    """
    qtokens = content_tokens(question)
    # 先算 schema 命中 token（列名/表名精确），值接地跳过这些词
    # （'user' 已指向 users 表时不再去 AboutMe 自由文本里找值）。
    schema_matched: set = set()
    for tok in qtokens:
        if len(tok) < 2:
            continue
        if any(bundle.col_name_index.get(vt) for vt in token_variants(tok)) or \
                any(bundle.table_name_index.get(vt) for vt in token_variants(tok)):
            schema_matched.add(tok)
    value_hits = compute_value_hits(qtokens, bundle, schema_matched)
    rel = {t: table_relevance(qtokens, bundle, t, value_hits)
           for t in bundle.tables}
    schema_sec = build_schema_section(bundle, rel)
    sample_sec = build_sample_section(bundle, rel)
    keyword_sec = build_keyword_section(qtokens, bundle, value_hits, rel)

    # 优先级：schema > keyword > sample（样例段在超预算时最先被裁）
    sections = [
        ("Schema summary", schema_sec),
        ("Keyword map", keyword_sec),
        ("Sample rows", sample_sec),
    ]
    parts: List[str] = []
    used = 0
    for title, body in sections:
        if not body:
            continue
        head = f"[{title}]\n"
        avail = MAX_EVIDENCE_CHARS - used - len(head)
        if avail <= 0:
            break
        if len(body) > avail:
            cut = avail - 3
            if cut <= 0:
                break
            body = body[:cut]
        parts.append(head + body)
        used += len(head) + len(body)
    return "\n\n".join(parts).strip()


# ---------------------------------------------------------------------------
# 报告工具
# ---------------------------------------------------------------------------

def evidence_stats(evidences: Dict[str, str]) -> Dict[str, Any]:
    lens = [len(v) for v in evidences.values()]
    empty = sum(1 for v in evidences.values() if not v.strip())
    return {
        "n": len(evidences),
        "empty": empty,
        "chars_min": min(lens) if lens else 0,
        "chars_med": sorted(lens)[len(lens) // 2] if lens else 0,
        "chars_max": max(lens) if lens else 0,
    }


if __name__ == "__main__":
    print("gen_evidence_core: shared module, import from gen_bird_evidence.py "
          "or gen_spider_evidence.py")
