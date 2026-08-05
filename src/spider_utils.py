"""
Spider dataset utilities: loader, read-only database executor, custom metrics,
checkpoint helpers, config validation, and agent-output validation.

This module does NOT implement official Spider EM or EX metrics.
It provides basic custom evaluation suitable for a pre-training baseline.

Design principles:
- Prefer DDL from sqlite_master over tables.json.
- Use read-only SQLite connections, independently opened and closed per query.
- Reject non-SELECT statements.
- Never modify the model or existing test files.
- Each SQL executed only once; full rows returned for comparison, capped rows saved.
- Checkpoint and config-validation functions are in this module so they can be
  tested without importing the agent (which requires torch / GPU).

Run built-in self-tests (no GPU/model needed):
    python spider_utils.py
"""

import collections
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Row caps
# ---------------------------------------------------------------------------
MAX_SAVED_ROWS = 1000          # rows saved to JSON for inspection
FULL_ROWS_HARD_LIMIT = 100_000  # safety cap for in-memory full rows

# Per-query progress-handler cap (SQLite VM instructions, NOT real-time seconds).
PROGRESS_HANDLER_INTERVAL = 1000
MAX_VM_STEPS = 5_000_000


# ===================================================================
# SpiderLoader
# ===================================================================


class SpiderLoader:
    """Load Spider dev.json questions, gold SQL, and DDL schemas."""

    def __init__(self, spider_dir: str) -> None:
        self.spider_dir = Path(spider_dir)
        self._dev_path = self.spider_dir / "dev.json"
        self._tables_path = self.spider_dir / "tables.json"
        self._database_dir = self.spider_dir / "database"

        self._dev_data: Optional[List[Dict[str, Any]]] = None
        self._tables_data: Optional[Dict[str, Any]] = None

    @property
    def dev_path(self) -> Path:
        return self._dev_path

    def _load_dev(self) -> List[Dict[str, Any]]:
        if self._dev_data is not None:
            return self._dev_data
        if not self._dev_path.exists():
            raise FileNotFoundError(f"Spider dev.json not found: {self._dev_path}")
        with open(self._dev_path, "r", encoding="utf-8") as fh:
            self._dev_data = json.load(fh)
        return self._dev_data  # type: ignore[return-value]

    def load_dev(
        self,
        limit: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Dict[str, Any]]:
        all_items = self._load_dev()
        total = len(all_items)
        if start_index >= total:
            return []
        end = (
            min(start_index + limit, total)
            if limit is not None
            else total
        )
        result: List[Dict[str, Any]] = []
        for idx in range(start_index, end):
            item = dict(all_items[idx])
            item["dataset_index"] = idx
            result.append(item)
        return result

    def _load_tables(self) -> Dict[str, Any]:
        if self._tables_data is not None:
            return self._tables_data
        if not self._tables_path.exists():
            raise FileNotFoundError(f"Spider tables.json not found: {self._tables_path}")
        with open(self._tables_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        if isinstance(raw, list):
            self._tables_data = {}
            for entry in raw:
                db_id = entry.get("db_id")
                if db_id is None:
                    continue
                if db_id in self._tables_data:
                    raise ValueError(f"Duplicate db_id '{db_id}' in tables.json list")
                self._tables_data[db_id] = entry
            return self._tables_data

        if isinstance(raw, dict):
            first_val = next(iter(raw.values()), None)
            if isinstance(first_val, dict) and "db_id" in first_val:
                self._tables_data = raw
                return self._tables_data  # type: ignore[return-value]
            raise TypeError(
                "tables.json is a dict but values do not look like "
                "per-database entries (missing db_id key)."
            )
        raise TypeError(
            f"Unsupported tables.json type: {type(raw).__name__}. Expected list or dict."
        )

    def load_tables_json(self, db_id: str) -> Dict[str, Any]:
        tables = self._load_tables()
        if db_id not in tables:
            raise KeyError(
                f"db_id '{db_id}' not found in tables.json "
                f"(looked in {len(tables)} entries)"
            )
        return tables[db_id]  # type: ignore[no-any-return]

    def get_ddl_from_db(self, db_id: str) -> Optional[str]:
        db_path = self._database_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            return None
        try:
            uri = db_path.resolve().as_uri()
            conn = sqlite3.connect(f"{uri}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.Error:
            return None
        try:
            rows = conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            ddl_parts: List[str] = []
            for row in rows:
                name = row["name"]  # type: ignore[index]
                raw_sql = row["sql"]  # type: ignore[index]
                if raw_sql:
                    ddl_parts.append(raw_sql.strip())
                else:
                    ddl_parts.append(
                        self._reconstruct_table_ddl(conn, name)  # type: ignore[arg-type]
                    )
            return "\n\n".join(ddl_parts) if ddl_parts else None
        except sqlite3.Error:
            return None
        finally:
            conn.close()

    @staticmethod
    def _reconstruct_table_ddl(conn: sqlite3.Connection, table_name: str) -> str:
        cols = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        fks = conn.execute(f'PRAGMA foreign_key_list("{table_name}")').fetchall()
        col_defs: List[str] = []
        for c in cols:
            name = c[1]
            col_type = c[2].upper() if c[2] else "TEXT"
            pk = " PRIMARY KEY" if c[5] else ""
            col_defs.append(f'"{name}" {col_type}{pk}')
        if fks:
            for fk in fks:
                col_defs.append(
                    f'FOREIGN KEY ("{fk[3]}") REFERENCES "{fk[2]}"("{fk[4]}")'
                )
        return f'CREATE TABLE "{table_name}" (\n  ' + ",\n  ".join(col_defs) + "\n);"

    def get_ddl_from_tables_json(self, db_id: str) -> Optional[str]:
        try:
            entry = self.load_tables_json(db_id)
        except (KeyError, FileNotFoundError):
            return None
        table_names = entry.get("table_names_original", [])
        col_names_orig = entry.get("column_names_original", [])
        col_types = entry.get("column_types", [])
        primary_keys = set(entry.get("primary_keys", []))
        foreign_keys = entry.get("foreign_keys", [])

        table_cols: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(len(table_names))
        }
        for col_idx, (table_idx, col_name) in enumerate(col_names_orig):
            if table_idx == -1:
                continue
            col_type = col_types[col_idx] if col_idx < len(col_types) else "text"
            col_type = _normalize_spider_type(col_type)
            is_pk = col_idx in primary_keys
            table_cols[table_idx].append({"name": col_name, "type": col_type, "is_pk": is_pk})

        fk_map: Dict[int, Tuple[str, str]] = {}
        for child_idx, parent_idx in foreign_keys:
            if parent_idx < len(col_names_orig) and child_idx < len(col_names_orig):
                ref_table_idx, ref_col_name = col_names_orig[parent_idx]
                child_table_idx, _child_col_name = col_names_orig[child_idx]
                if ref_table_idx < len(table_names):
                    fk_map[child_idx] = (table_names[ref_table_idx], ref_col_name)

        ddl_parts: List[str] = []
        for table_idx in range(len(table_names)):
            table_name = table_names[table_idx]
            col_defs: List[str] = []
            for col_info in table_cols[table_idx]:
                pk_suffix = " PRIMARY KEY" if col_info["is_pk"] else ""
                col_defs.append(f'"{col_info["name"]}" {col_info["type"]}{pk_suffix}')
            for col_idx, (ref_table, ref_col) in fk_map.items():
                if col_idx < len(col_names_orig):
                    ct_idx, child_col = col_names_orig[col_idx]
                    if ct_idx == table_idx:
                        col_defs.append(
                            f'FOREIGN KEY ("{child_col}") REFERENCES "{ref_table}"("{ref_col}")'
                        )
            ddl_parts.append(
                f'CREATE TABLE "{table_name}" (\n  ' + ",\n  ".join(col_defs) + "\n);"
            )
        return "\n\n".join(ddl_parts) if ddl_parts else None

    def get_ddl_with_source(self, db_id: str) -> Tuple[str, str]:
        ddl = self.get_ddl_from_db(db_id)
        if ddl:
            return ddl, "sqlite_master"
        ddl = self.get_ddl_from_tables_json(db_id)
        if ddl:
            return ddl, "tables_json"
        raise RuntimeError(
            f"Cannot obtain DDL for db_id='{db_id}': "
            f"database file not found and tables.json entry missing."
        )

    def format_ddl(self, db_id: str) -> str:
        ddl, _source = self.get_ddl_with_source(db_id)
        return ddl

    @property
    def database_dir(self) -> Path:
        return self._database_dir


# ===================================================================
# DatabaseExecutor
# ===================================================================


class DatabaseExecutor:
    """Execute SQL against Spider SQLite databases (read-only, single-execution)."""

    def __init__(
        self,
        spider_dir: str,
        full_rows_limit: int = FULL_ROWS_HARD_LIMIT,
        saved_rows_limit: int = MAX_SAVED_ROWS,
    ) -> None:
        self._database_dir = Path(spider_dir) / "database"
        self.full_rows_limit = full_rows_limit
        self.saved_rows_limit = saved_rows_limit

    _DANGEROUS_KEYWORDS = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH)\b",
        flags=re.IGNORECASE,
    )
    _ALLOWED_PREFIXES = ("SELECT", "WITH", "EXPLAIN", "PRAGMA")

    @classmethod
    def _is_safe_sql(cls, sql: str) -> bool:
        stripped = sql.strip()
        if not stripped:
            return False
        upper = stripped[:50].upper().lstrip()
        if not any(upper.startswith(p) for p in cls._ALLOWED_PREFIXES):
            return False
        if cls._DANGEROUS_KEYWORDS.search(stripped):
            return False
        return True

    @staticmethod
    def _make_progress_handler():
        state = {"ticks": 0}
        def handler() -> int:
            state["ticks"] += PROGRESS_HANDLER_INTERVAL
            if state["ticks"] >= MAX_VM_STEPS:
                return 1
            return 0
        return handler

    def execute(self, db_id: str, sql: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "success": False,
            "error": None,
            "error_type": None,
            "full_rows": [],
            "full_rows_truncated": False,
            "saved_rows": [],
            "saved_rows_truncated": False,
            "row_count": 0,
            "execution_seconds": 0.0,
        }
        if not self._is_safe_sql(sql):
            result["error"] = "SQL rejected: non-read statement or disallowed keyword"
            result["error_type"] = "safety_rejection"
            return result
        db_path = self._database_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            result["error"] = f"Database file not found: {db_path}"
            result["error_type"] = "db_not_found"
            return result
        try:
            uri = db_path.resolve().as_uri()
            conn = sqlite3.connect(f"{uri}?mode=ro", uri=True)
            conn.row_factory = lambda _cursor, row: list(row)
        except sqlite3.Error as exc:
            result["error"] = f"Failed to open database: {exc}"
            result["error_type"] = "connection_error"
            return result
        try:
            conn.set_progress_handler(self._make_progress_handler(), PROGRESS_HANDLER_INTERVAL)
            start = time.perf_counter()
            try:
                cursor = conn.execute(sql)
            except sqlite3.Warning:
                # Multi-statement SQL: sqlite3 raises a Warning, not an Error.
                # Treat as execution failure (counted, not fatal).
                result["error"] = "SQL rejected: multi-statement input"
                result["error_type"] = "multi_statement"
                return result

            # Batch-read rows to cap memory — never keep more than limits
            _FETCH_BATCH = 1000
            full_rows: List[List[Any]] = []
            saved_rows: List[List[Any]] = []
            total_count = 0

            while True:
                batch = cursor.fetchmany(_FETCH_BATCH)
                if not batch:
                    break
                for row in batch:
                    total_count += 1
                    if len(full_rows) < self.full_rows_limit:
                        full_rows.append(list(row))
                    if len(saved_rows) < self.saved_rows_limit:
                        saved_rows.append(list(row))

            result["execution_seconds"] = round(time.perf_counter() - start, 4)
            result["success"] = True
            result["row_count"] = total_count
            result["full_rows"] = _safe_rows_for_json(full_rows)
            result["full_rows_truncated"] = total_count > self.full_rows_limit
            result["saved_rows"] = _safe_rows_for_json(saved_rows)
            result["saved_rows_truncated"] = total_count > self.saved_rows_limit
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "interrupt" in msg:
                result["error"] = (
                    f"Query interrupted after exceeding {MAX_VM_STEPS} "
                    f"SQLite VM steps (not a wall-clock timeout). Original: {exc}"
                )
                result["error_type"] = "query_interrupted"
            else:
                result["error"] = str(exc)
                result["error_type"] = "sqlite_error"
        except sqlite3.Error as exc:
            result["error"] = str(exc)
            result["error_type"] = "sqlite_error"
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        return result


# ===================================================================
# Metrics
# ===================================================================

_SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS", "NULL",
    "AS", "ON", "JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "CROSS",
    "FULL", "NATURAL", "USING", "GROUP", "BY", "ORDER", "ASC", "DESC",
    "HAVING", "LIMIT", "OFFSET", "UNION", "INTERSECT", "EXCEPT",
    "ALL", "ANY", "EXISTS", "BETWEEN", "LIKE", "DISTINCT", "CASE",
    "WHEN", "THEN", "ELSE", "END", "WITH", "RECURSIVE", "VALUES",
    "INSERT", "INTO", "UPDATE", "SET", "DELETE", "CREATE", "TABLE",
    "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "INDEX", "DROP",
    "ALTER", "ADD", "COLUMN", "COUNT", "SUM", "AVG", "MIN", "MAX",
    "CAST", "COALESCE", "IFNULL", "NULLIF", "ABS", "ROUND", "LENGTH",
    "UPPER", "LOWER", "SUBSTR", "REPLACE", "TRIM", "INSTR",
    "INTEGER", "TEXT", "REAL", "BLOB", "VARCHAR", "CHAR", "FLOAT",
    "DOUBLE", "DATE", "TIME", "DATETIME", "TIMESTAMP", "BOOLEAN",
    "DEFAULT", "CHECK", "UNIQUE", "CONSTRAINT", "CASCADE", "RESTRICT",
    "OVER", "PARTITION", "ROWS", "RANGE", "PRECEDING", "FOLLOWING",
    "CURRENT", "ROW", "UNBOUNDED", "FIRST", "LAST", "FILTER",
    "WINDOW", "LAG", "LEAD", "RANK", "ROW_NUMBER", "DENSE_RANK",
    "NTILE", "PERCENT_RANK", "CUME_DIST",
}


def normalize_sql(sql: str) -> str:
    s = sql.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(";").strip()
    tokens = re.split(r"(\b\w+\b)", s)
    result_tokens: List[str] = []
    for tok in tokens:
        if tok.upper() in _SQL_KEYWORDS:
            result_tokens.append(tok.lower())
        else:
            result_tokens.append(tok)
    s = "".join(result_tokens)
    s = re.sub(r"\s*\(\s*", "(", s)
    s = re.sub(r"\s*\)\s*", ")", s)
    return s.strip()


def compute_normalized_sql_string_match(
    predicted_sql: str, gold_sql: str
) -> Dict[str, Any]:
    pred_norm = normalize_sql(predicted_sql)
    gold_norm = normalize_sql(gold_sql)
    return {
        "match": pred_norm == gold_norm,
        "predicted_normalized": pred_norm,
        "gold_normalized": gold_norm,
        "note": (
            "Basic whitespace/keyword normalization only. "
            "This is NOT Spider official EM. "
            "Spider EM requires the official structured evaluator."
        ),
    }


# ---------------------------------------------------------------------------
# Internal helpers for row/value conversion
# ---------------------------------------------------------------------------

def _safe_row_for_json(row: List[Any]) -> List[Any]:
    result: List[Any] = []
    for val in row:
        if val is None:
            result.append(None)
        elif isinstance(val, (int, float, str)):
            result.append(val)
        elif isinstance(val, bytes):
            try:
                result.append(val.decode("utf-8"))
            except UnicodeDecodeError:
                result.append(val.hex())
        else:
            result.append(str(val))
    return result


def _safe_rows_for_json(rows: List[List[Any]]) -> List[List[Any]]:
    return [_safe_row_for_json(r) for r in rows]


def _normalize_value_for_comparison(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (int, str)):
        return val
    if isinstance(val, float):
        return round(val, 6)
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except UnicodeDecodeError:
            return val.hex()
    return str(val)


def _has_order_by(sql: str) -> bool:
    cleaned = re.sub(r"'[^']*'", "", sql)
    cleaned = re.sub(r'"[^"]*"', "", cleaned)
    return bool(re.search(r"\bORDER\s+BY\b", cleaned, flags=re.IGNORECASE))


def _rows_to_counter(rows: List[List[Any]]) -> "collections.Counter[Tuple[Any, ...]]":
    return collections.Counter(
        tuple(_normalize_value_for_comparison(v) for v in row)
        for row in rows
    )


# ---------------------------------------------------------------------------
# Public comparison / summary
# ---------------------------------------------------------------------------

def compare_execution_results(
    predicted_rows: List[List[Any]],
    gold_rows: List[List[Any]],
    gold_sql: str = "",
) -> Dict[str, Any]:
    pred_count = len(predicted_rows)
    gold_count = len(gold_rows)
    preserve_order = _has_order_by(gold_sql) if gold_sql else False
    base = {
        "predicted_row_count": pred_count,
        "gold_row_count": gold_count,
        "preserved_order": preserve_order,
        "note": (
            "Custom execution comparison only. "
            "This is NOT Spider official EX. is_official_spider_metric: false"
        ),
    }
    if pred_count != gold_count:
        return {**base, "match": False,
                "match_reason": f"Row count mismatch: predicted={pred_count}, gold={gold_count}",
                "failure_stage": "row_count"}
    if pred_count == 0:
        return {**base, "match": True, "match_reason": "Both results are empty", "failure_stage": None}
    if predicted_rows and gold_rows:
        pred_cols = len(predicted_rows[0])
        gold_cols = len(gold_rows[0])
        if pred_cols != gold_cols:
            return {**base, "match": False,
                    "match_reason": f"Column count mismatch: predicted={pred_cols}, gold={gold_cols}",
                    "failure_stage": "column_count"}
    if preserve_order:
        for i, (pr, gr) in enumerate(zip(predicted_rows, gold_rows)):
            pn = [_normalize_value_for_comparison(v) for v in pr]
            gn = [_normalize_value_for_comparison(v) for v in gr]
            if pn != gn:
                return {**base, "match": False,
                        "match_reason": f"Row {i} differs (order preserved): predicted={pn}, gold={gn}",
                        "failure_stage": "row_value_order_preserved"}
        return {**base, "match": True,
                "match_reason": "All rows match (order preserved, ORDER BY detected)",
                "failure_stage": None}
    else:
        pc = _rows_to_counter(predicted_rows)
        gc = _rows_to_counter(gold_rows)
        if pc == gc:
            return {**base, "match": True,
                    "match_reason": "All rows match (order ignored, duplicate counts preserved, no ORDER BY)",
                    "failure_stage": None}
        only_pred = pc - gc
        only_gold = gc - pc
        detail: List[str] = []
        # Safe sort key: rows may contain None (SQL NULL), which cannot be
        # compared with int/str. Convert to a comparable sentinel first.
        # count may also be None (execution failure) — map to -1.
        def _safe_sort_key(item):
            row, count = item
            return (tuple(str(v) if v is None else v for v in row),
                    count if count is not None else -1)
        if only_pred:
            detail.append(f"rows only/more in predicted: {sorted(only_pred.items(), key=_safe_sort_key)[:5]}")
        if only_gold:
            detail.append(f"rows only/more in gold: {sorted(only_gold.items(), key=_safe_sort_key)[:5]}")
        return {**base, "match": False,
                "match_reason": "Row mismatch (order ignored, duplicate counts preserved). " + "; ".join(detail),
                "failure_stage": "row_value_set"}


def compute_summary(
    items: List[Dict[str, Any]],
    requested_indices: Optional[Any] = None,
) -> Dict[str, Any]:
    if requested_indices is not None:
        index_set = set(requested_indices)
        filtered = [it for it in items if it.get("dataset_index") in index_set]
    else:
        filtered = list(items)
    total = len(filtered)
    if total == 0:
        return {
            "total_requested": 0, "total_completed": 0,
            "parse_success_count": 0, "parse_success_rate": 0.0,
            "prediction_execution_success_count": 0, "prediction_execution_success_rate": 0.0,
            "gold_execution_success_count": 0,
            "custom_execution_match_count": 0, "custom_execution_match_rate": 0.0,
            "normalized_sql_string_match_count": 0, "normalized_sql_string_match_rate": 0.0,
            "average_generation_seconds": 0.0, "total_wall_seconds": 0.0,
            "evaluator_type": "custom_execution_match", "is_official_spider_metric": False,
        }
    parse_ok = sum(1 for it in filtered if it.get("parse_success"))
    pred_ex_ok = sum(1 for it in filtered if it.get("prediction_execution_success"))
    gold_ex_ok = sum(1 for it in filtered if it.get("gold_execution_success"))
    custom_match = sum(1 for it in filtered if it.get("custom_execution_match"))
    sql_str_match = sum(1 for it in filtered if it.get("normalized_sql_string_match"))
    gen_times = [it["generation_seconds"] for it in filtered if "generation_seconds" in it]
    return {
        "total_requested": total, "total_completed": total,
        "parse_success_count": parse_ok, "parse_success_rate": round(parse_ok / total, 4),
        "prediction_execution_success_count": pred_ex_ok,
        "prediction_execution_success_rate": round(pred_ex_ok / total, 4),
        "gold_execution_success_count": gold_ex_ok,
        "custom_execution_match_count": custom_match,
        "custom_execution_match_rate": round(custom_match / total, 4),
        "normalized_sql_string_match_count": sql_str_match,
        "normalized_sql_string_match_rate": round(sql_str_match / total, 4),
        "average_generation_seconds": (
            round(sum(gen_times) / len(gen_times), 4) if gen_times else 0.0
        ),
        "total_wall_seconds": 0.0,
        "evaluator_type": "custom_execution_match",
        "is_official_spider_metric": False,
    }


def _normalize_spider_type(spider_type: str) -> str:
    mapping = {
        "text": "TEXT", "number": "REAL", "integer": "INTEGER",
        "time": "TEXT", "boolean": "INTEGER", "float": "REAL",
        "real": "REAL", "varchar": "TEXT", "char": "TEXT",
        "int": "INTEGER", "date": "TEXT", "datetime": "TEXT", "timestamp": "TEXT",
    }
    return mapping.get(spider_type.lower(), "TEXT")


# ===================================================================
# Run-config, checkpoint, and validation helpers
# (No torch / GPU dependency — safe to import and test anywhere.)
# ===================================================================

EVALUATOR_TYPE = "custom_execution_match"


def build_run_config(
    spider_dir: str,
    start_index: int,
    limit: int,
    model_path: str,
    max_new_tokens: int,
    evaluator_type: str = EVALUATOR_TYPE,
) -> Dict[str, Any]:
    """Build the immutable run-config dict."""
    return {
        "spider_dir": str(Path(spider_dir).resolve()),
        "start_index": start_index,
        "limit": limit,
        "model_path": str(Path(model_path).resolve()),
        "max_new_tokens": max_new_tokens,
        "evaluator_type": evaluator_type,
    }


def validate_resume_config(
    stored: Dict[str, Any],
    current: Dict[str, Any],
) -> None:
    """
    Compare stored checkpoint run_config with current config.
    Calls sys.exit(1) on any mismatch — does not return.
    """
    mismatches: List[str] = []
    for key in sorted(set(list(stored.keys()) + list(current.keys()))):
        sv = stored.get(key)
        cv = current.get(key)
        if sv != cv:
            mismatches.append(f"  {key}: checkpoint={sv!r}  vs  current={cv!r}")
    if mismatches:
        print(
            "ERROR: --resume config mismatch. "
            "Cannot merge results from different experiments.\n"
            "Mismatched fields:"
        )
        for m in mismatches:
            print(m)
        print(
            "\nTo resume, re-run with the EXACT same --spider-dir, --limit, "
            "--start-index, --model-path, and --max-new-tokens."
        )
        sys.exit(1)


def check_duplicate_indices(items: List[Dict[str, Any]]) -> None:
    """Raise ValueError if any dataset_index appears more than once in *items*."""
    seen: Dict[int, int] = {}
    for i, it in enumerate(items):
        di = it.get("dataset_index")
        if di is None:
            continue
        if di in seen:
            raise ValueError(
                f"Duplicate dataset_index={di} found in items "
                f"at positions {seen[di]} and {i}. Checkpoint may be corrupted."
            )
        seen[di] = i


def validate_checkpoint_integrity(
    checkpoint: Dict[str, Any],
    requested_indices: Any,  # iterable of int
) -> None:
    """
    Comprehensive checkpoint integrity check.

    Checks performed:
    1. completed_indices is a list/set with no duplicates.
    2. Every item has a ``dataset_index`` key.
    3. No duplicate ``dataset_index`` across items.
    4. ``completed_indices`` == set of ``dataset_index`` from items.
    5. All indices in ``completed_indices`` belong to *requested_indices*.

    Raises ValueError with a clear message on any violation.
    """
    req_set = set(requested_indices) if requested_indices is not None else None

    # 1. completed_indices
    ci_raw = checkpoint.get("completed_indices", [])
    if isinstance(ci_raw, set):
        ci_list = sorted(ci_raw)
    elif isinstance(ci_raw, list):
        ci_list = list(ci_raw)
    else:
        raise ValueError(
            f"completed_indices has unexpected type {type(ci_raw).__name__}"
        )
    ci_set = set(ci_list)
    if len(ci_set) != len(ci_list):
        raise ValueError("completed_indices contains duplicate values")

    # 2. items
    items: List[Dict[str, Any]] = checkpoint.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"items has unexpected type {type(items).__name__}")

    item_indices: Set[int] = set()
    for i, it in enumerate(items):
        di = it.get("dataset_index")
        if di is None:
            raise ValueError(f"Item at position {i} is missing 'dataset_index'")
        if not isinstance(di, int):
            raise ValueError(
                f"Item at position {i} has non-integer dataset_index={di!r}"
            )
        if di in item_indices:
            raise ValueError(
                f"Duplicate dataset_index={di} in items at or before position {i}"
            )
        item_indices.add(di)

    # 3. completed_indices must equal item_indices
    if ci_set != item_indices:
        only_ci = ci_set - item_indices
        only_items = item_indices - ci_set
        parts: List[str] = []
        if only_ci:
            parts.append(f"in completed_indices but not in items: {sorted(only_ci)}")
        if only_items:
            parts.append(f"in items but not in completed_indices: {sorted(only_items)}")
        raise ValueError(
            "completed_indices and items dataset_index sets do not match. " + "; ".join(parts)
        )

    # 4. All indices must belong to requested_indices
    if req_set is not None:
        outside = ci_set - req_set
        if outside:
            raise ValueError(
                f"Indices in checkpoint but not in requested_indices: {sorted(outside)}. "
                f"Checkpoint may be from a different run (different --start-index or --limit)."
            )


def validate_agent_candidate(gen_result: Any) -> Dict[str, Any]:
    """
    Validate the structure of ``agent.generate()`` output.

    Returns a dict::

        {
            "valid": bool,
            "error": str | None,
            "raw_response": str,
            "sql": str,
            "parse_success": bool,
            "parse_method": str | None,
            "generation_seconds": float,
        }

    Never raises.  Every field is populated with a safe default on failure.
    """
    def _fail(msg: str) -> Dict[str, Any]:
        return {
            "valid": False, "error": msg,
            "raw_response": "", "sql": "",
            "parse_success": False, "parse_method": None,
            "generation_seconds": 0.0,
        }

    if not isinstance(gen_result, dict):
        return _fail("agent.generate() did not return a dict")

    candidates = gen_result.get("candidates")
    if not isinstance(candidates, list) or len(candidates) == 0:
        return _fail("gen_result['candidates'] is missing, empty, or not a list")

    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return _fail("candidates[0] is not a dict")

    raw_response = candidate.get("raw_response")
    if not isinstance(raw_response, str):
        raw_response = str(raw_response) if raw_response is not None else ""

    sql = candidate.get("sql")
    parse_success = candidate.get("parse_success", False)
    parse_method = candidate.get("parse_method")

    if sql is None or not isinstance(sql, str) or sql.strip() == "":
        sql = ""
        parse_success = False
        if parse_method is None:
            parse_method = "empty_or_invalid_sql"

    metadata = gen_result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    generation_seconds = metadata.get("generation_seconds", 0.0)
    try:
        generation_seconds = float(generation_seconds)
    except (TypeError, ValueError):
        generation_seconds = 0.0

    return {
        "valid": True, "error": None,
        "raw_response": raw_response,
        "sql": sql,
        "parse_success": bool(parse_success),
        "parse_method": parse_method,
        "generation_seconds": float(generation_seconds),
    }


def save_checkpoint(
    output_dir: Path,
    checkpoint: Dict[str, Any],
    run_config: Dict[str, Any],
) -> None:
    """Atomically write checkpoint (including run_config) to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_dir / "checkpoint.json.tmp"
    cp_path = output_dir / "checkpoint.json"
    payload = dict(checkpoint)
    payload["completed_indices"] = sorted(list(checkpoint.get("completed_indices", set())))
    payload["run_config"] = run_config
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, cp_path)


def load_checkpoint(output_dir: Path) -> Dict[str, Any]:
    """
    Return checkpoint dict with ``completed_indices`` as a set.

    Returns ``{"completed_indices": set(), "items": [], "run_config": None}``
    when no checkpoint file exists.
    """
    cp_path = output_dir / "checkpoint.json"
    if not cp_path.exists():
        return {"completed_indices": set(), "items": [], "run_config": None}
    try:
        with open(cp_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"Checkpoint file is corrupted and cannot be parsed: {cp_path}\n{exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"Checkpoint file does not contain a JSON object: {cp_path}"
        )
    ci = data.get("completed_indices", [])
    if not isinstance(ci, list):
        raise ValueError("Checkpoint 'completed_indices' is not a list")

    # Validate each value and detect duplicates BEFORE set conversion
    for i, val in enumerate(ci):
        if not isinstance(val, int):
            raise ValueError(
                f"completed_indices[{i}] is not an integer: {val!r}"
            )
    seen_ci: Dict[int, int] = {}
    for i, val in enumerate(ci):
        if val in seen_ci:
            raise ValueError(
                f"Duplicate value {val} in completed_indices "
                f"at positions {seen_ci[val]} and {i}"
            )
        seen_ci[val] = i

    data["completed_indices"] = set(ci)
    data.setdefault("run_config", None)
    data.setdefault("items", [])
    return data


# ===================================================================
# Built-in self-tests (no GPU / model needed)
# ===================================================================

def _run_tests() -> int:
    """
    Run self-tests for spider_utils.

    Tests exercise the real checkpoint / config-validation / agent-candidate
    functions (no mocks), plus the loader / executor / metrics from earlier
    rounds.
    """
    failures: List[str] = []
    passed = 0

    def check(description: str, condition: bool) -> None:
        nonlocal passed
        if condition:
            passed += 1
        else:
            failures.append(f"FAIL: {description}")
            print(f"  FAIL: {description}")

    print("=== spider_utils self-tests ===\n")

    # ================================================================
    # Test 1: tables.json list format
    # ================================================================
    print("--- Test 1: tables.json list format ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        tables_list = [
            {"db_id": "test_db", "table_names_original": ["t1"],
             "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
             "column_types": ["text", "number", "text"],
             "primary_keys": [1], "foreign_keys": []},
            {"db_id": "other_db", "table_names_original": ["x"],
             "column_names_original": [[-1, "*"], [0, "val"]],
             "column_types": ["text", "number"],
             "primary_keys": [], "foreign_keys": []},
        ]
        (tmpdir / "tables.json").write_text(json.dumps(tables_list), encoding="utf-8")
        (tmpdir / "dev.json").write_text("[]", encoding="utf-8")
        (tmpdir / "database").mkdir(exist_ok=True)
        loader = SpiderLoader(str(tmpdir))
        entry = loader.load_tables_json("test_db")
        check("t1: lookup test_db", entry["db_id"] == "test_db")
        check("t1: table_names_original", entry["table_names_original"] == ["t1"])
        try:
            loader.load_tables_json("nonexistent")
            check("t1: missing db_id raises KeyError", False)
        except KeyError as e:
            check("t1: missing db_id raises KeyError", "nonexistent" in str(e))

    # ================================================================
    # Test 2: tables.json dict format (compat)
    # ================================================================
    print("--- Test 2: tables.json dict format (compat) ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        tables_dict = {
            "dict_db": {"db_id": "dict_db", "table_names_original": ["a"],
                        "column_names_original": [[-1, "*"], [0, "col"]],
                        "column_types": ["text", "number"],
                        "primary_keys": [], "foreign_keys": []},
        }
        (tmpdir / "tables.json").write_text(json.dumps(tables_dict), encoding="utf-8")
        (tmpdir / "dev.json").write_text("[]", encoding="utf-8")
        (tmpdir / "database").mkdir(exist_ok=True)
        loader2 = SpiderLoader(str(tmpdir))
        check("t2: dict lookup", loader2.load_tables_json("dict_db")["db_id"] == "dict_db")

    # ================================================================
    # Test 3: SQLite read-only SELECT
    # ================================================================
    print("--- Test 3: SQLite read-only SELECT ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        db_dir = tmpdir / "database" / "test_db"
        db_dir.mkdir(parents=True)
        conn = sqlite3.connect(str(db_dir / "test_db.sqlite"))
        conn.execute("CREATE TABLE t (id INTEGER, val TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'hello'), (2, 'world')")
        conn.commit(); conn.close()
        executor = DatabaseExecutor(str(tmpdir))
        r = executor.execute("test_db", "SELECT * FROM t ORDER BY id")
        check("t3: success", r["success"] is True)
        check("t3: row_count", r["row_count"] == 2)
        check("t3: full_rows", r["full_rows"] == [[1, "hello"], [2, "world"]])
        check("t3: not truncated", r["full_rows_truncated"] is False)

    # ================================================================
    # Test 3b: fetchmany truncation with low limits
    # ================================================================
    print("--- Test 3b: fetchmany truncation ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        db_dir = tmpdir / "database" / "test_trunc"
        db_dir.mkdir(parents=True)
        conn = sqlite3.connect(str(db_dir / "test_trunc.sqlite"))
        conn.execute("CREATE TABLE t (id INTEGER, val TEXT)")
        for i in range(50):
            conn.execute("INSERT INTO t VALUES (?, ?)", (i, f"row_{i}"))
        conn.commit(); conn.close()

        # Low limits: full=15, saved=5
        executor_low = DatabaseExecutor(str(tmpdir), full_rows_limit=15, saved_rows_limit=5)
        r = executor_low.execute("test_trunc", "SELECT * FROM t")
        check("t3b: success", r["success"] is True)
        check("t3b: row_count=50", r["row_count"] == 50)
        check("t3b: full_rows_truncated=True", r["full_rows_truncated"] is True)
        check("t3b: saved_rows_truncated=True", r["saved_rows_truncated"] is True)
        check("t3b: full_rows len <= 15", len(r["full_rows"]) == 15)
        check("t3b: saved_rows len <= 5", len(r["saved_rows"]) == 5)
        check("t3b: full_rows has first row", r["full_rows"][0] == [0, "row_0"])
        check("t3b: saved_rows has first row", r["saved_rows"][0] == [0, "row_0"])

    # ================================================================
    # Test 4: DROP / INSERT rejected
    # ================================================================
    print("--- Test 4: DROP / INSERT rejected ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        db_dir = tmpdir / "database" / "test_db"
        db_dir.mkdir(parents=True)
        conn = sqlite3.connect(str(db_dir / "test_db.sqlite"))
        conn.execute("CREATE TABLE t (x INTEGER)"); conn.commit(); conn.close()
        executor = DatabaseExecutor(str(tmpdir))
        r = executor.execute("test_db", "DROP TABLE t")
        check("t4: DROP rejected", r["success"] is False and r["error_type"] == "safety_rejection")
        r2 = executor.execute("test_db", "INSERT INTO t VALUES (1)")
        check("t4: INSERT rejected", r2["success"] is False and r2["error_type"] == "safety_rejection")

    # ================================================================
    # Test 5: ORDER BY preserves row order
    # ================================================================
    print("--- Test 5: ORDER BY preserves order ---")
    cmp = compare_execution_results(
        [["A", 1], ["B", 2]], [["B", 2], ["A", 1]],
        gold_sql="SELECT name, val FROM t ORDER BY name",
    )
    check("t5: order mismatch detected", cmp["match"] is False)
    check("t5: preserved_order=True", cmp["preserved_order"] is True)
    check("t5: failure_stage", cmp["failure_stage"] == "row_value_order_preserved")
    cmp2 = compare_execution_results(
        [["A", 1], ["B", 2]], [["A", 1], ["B", 2]],
        gold_sql="SELECT name, val FROM t ORDER BY name",
    )
    check("t5: correct order matches", cmp2["match"] is True)

    # ================================================================
    # Test 6: no ORDER BY — Counter preserves duplicates
    # ================================================================
    print("--- Test 6: no ORDER BY, duplicate preservation ---")
    cmp3 = compare_execution_results(
        [["A"], ["B"], ["A"]], [["A"], ["A"], ["B"]],
        gold_sql="SELECT name FROM t",
    )
    check("t6: order ignored, same multiset -> match", cmp3["match"] is True)
    cmp4 = compare_execution_results(
        [["A"], ["A"], ["B"]], [["A"], ["B"], ["B"]],
        gold_sql="SELECT name FROM t",
    )
    check("t6: different multiset -> mismatch", cmp4["match"] is False)
    check("t6: failure_stage=row_value_set", cmp4["failure_stage"] == "row_value_set")

    # ================================================================
    # Test 7: normalized_sql_string_match note
    # ================================================================
    print("--- Test 7: normalized_sql_string_match ---")
    sc = compute_normalized_sql_string_match(
        "SELECT a FROM t WHERE x > 1", "select a from t where x > 1"
    )
    check("t7: keyword casing normalised -> match", sc["match"] is True)
    check("t7: note mentions NOT official EM", "NOT Spider official EM" in sc["note"])
    sc2 = compute_normalized_sql_string_match("SELECT abc FROM t", "select ABC from t")
    check("t7: identifier casing NOT normalised -> mismatch", sc2["match"] is False)

    # ================================================================
    # Test 8: validate_resume_config — same config passes
    # ================================================================
    print("--- Test 8: validate_resume_config: same config ---")
    cfg = build_run_config(
        spider_dir="/fake/spider", start_index=0, limit=100,
        model_path="/fake/model", max_new_tokens=512,
    )
    cfg_same = build_run_config(
        spider_dir="/fake/spider", start_index=0, limit=100,
        model_path="/fake/model", max_new_tokens=512,
    )
    # Same config must NOT raise SystemExit
    try:
        validate_resume_config(cfg, cfg_same)
        check("t8: same config passes (no SystemExit)", True)
    except SystemExit:
        check("t8: same config passes (no SystemExit)", False)

    # ================================================================
    # Test 9: validate_resume_config — limit mismatch → SystemExit(1)
    # ================================================================
    print("--- Test 9: validate_resume_config: limit mismatch ---")
    cfg_diff_limit = build_run_config(
        spider_dir="/fake/spider", start_index=0, limit=50,
        model_path="/fake/model", max_new_tokens=512,
    )
    try:
        validate_resume_config(cfg, cfg_diff_limit)
        check("t9: limit mismatch raises SystemExit", False)
    except SystemExit as e:
        check("t9: limit mismatch raises SystemExit", True)
        check("t9: exit code is 1", e.code == 1)

    # ================================================================
    # Test 10: validate_resume_config — model_path mismatch → SystemExit(1)
    # ================================================================
    print("--- Test 10: validate_resume_config: model_path mismatch ---")
    cfg_diff_model = build_run_config(
        spider_dir="/fake/spider", start_index=0, limit=100,
        model_path="/other/model", max_new_tokens=512,
    )
    try:
        validate_resume_config(cfg, cfg_diff_model)
        check("t10: model_path mismatch raises SystemExit", False)
    except SystemExit as e:
        check("t10: model_path mismatch raises SystemExit", True)
        check("t10: exit code is 1", e.code == 1)

    # ================================================================
    # Test 11: validate_checkpoint_integrity — all-consistent passes
    # ================================================================
    print("--- Test 11: validate_checkpoint_integrity: consistent ---")
    cp_ok = {
        "completed_indices": [0, 1, 2],
        "items": [
            {"dataset_index": 0}, {"dataset_index": 1}, {"dataset_index": 2},
        ],
    }
    try:
        validate_checkpoint_integrity(cp_ok, requested_indices={0, 1, 2, 3})
        check("t11: consistent checkpoint passes", True)
    except ValueError as e:
        check(f"t11: consistent checkpoint passes (got: {e})", False)

    # ================================================================
    # Test 12: validate_checkpoint_integrity — ci/items mismatch
    # ================================================================
    print("--- Test 12: validate_checkpoint_integrity: ci/items mismatch ---")
    cp_bad = {
        "completed_indices": [0, 1, 2],
        "items": [{"dataset_index": 0}, {"dataset_index": 1}],  # missing 2
    }
    try:
        validate_checkpoint_integrity(cp_bad, requested_indices={0, 1, 2})
        check("t12: ci/items mismatch raises ValueError", False)
    except ValueError:
        check("t12: ci/items mismatch raises ValueError", True)

    # ================================================================
    # Test 13: validate_checkpoint_integrity — index outside requested
    # ================================================================
    print("--- Test 13: validate_checkpoint_integrity: index outside requested ---")
    cp_outside = {
        "completed_indices": [0, 5],
        "items": [{"dataset_index": 0}, {"dataset_index": 5}],
    }
    try:
        validate_checkpoint_integrity(cp_outside, requested_indices={0, 1, 2})
        check("t13: index outside requested raises ValueError", False)
    except ValueError:
        check("t13: index outside requested raises ValueError", True)

    # ================================================================
    # Test 14: validate_checkpoint_integrity — item missing dataset_index
    # ================================================================
    print("--- Test 14: validate_checkpoint_integrity: missing dataset_index ---")
    cp_missing_key = {
        "completed_indices": [0],
        "items": [{"no_index": 0}],
    }
    try:
        validate_checkpoint_integrity(cp_missing_key, requested_indices={0})
        check("t14: missing dataset_index raises ValueError", False)
    except ValueError as e:
        check("t14: missing dataset_index raises ValueError", "missing 'dataset_index'" in str(e))

    # ================================================================
    # Test 15: validate_checkpoint_integrity — duplicate in items
    # ================================================================
    print("--- Test 15: validate_checkpoint_integrity: duplicate in items ---")
    cp_dup = {
        "completed_indices": [0],
        "items": [{"dataset_index": 0}, {"dataset_index": 0}],
    }
    try:
        validate_checkpoint_integrity(cp_dup, requested_indices={0})
        check("t15: duplicate in items raises ValueError", False)
    except ValueError:
        check("t15: duplicate in items raises ValueError", True)

    # ================================================================
    # Test 16: save_checkpoint / load_checkpoint roundtrip
    # ================================================================
    print("--- Test 16: save / load checkpoint roundtrip ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cp = {"completed_indices": {3, 7}, "items": [{"dataset_index": 3}, {"dataset_index": 7}]}
        rc = build_run_config("/s", 0, 10, "/m", 256)
        save_checkpoint(tmpdir, cp, rc)
        loaded = load_checkpoint(tmpdir)
        check("t16: completed_indices roundtrip", loaded["completed_indices"] == {3, 7})
        check("t16: items count", len(loaded["items"]) == 2)
        check("t16: run_config.spider_dir", loaded["run_config"]["spider_dir"] == rc["spider_dir"])
        check("t16: run_config.limit", loaded["run_config"]["limit"] == 10)

    # ================================================================
    # Test 17: load_checkpoint — corrupted JSON raises ValueError
    # ================================================================
    print("--- Test 17: load_checkpoint corrupted JSON ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "checkpoint.json").write_text("not json{{{", encoding="utf-8")
        try:
            load_checkpoint(tmpdir)
            check("t17: corrupted JSON raises ValueError", False)
        except ValueError as e:
            check("t17: corrupted JSON raises ValueError", "corrupted" in str(e).lower())

    # ================================================================
    # Test 17b: load_checkpoint rejects duplicate completed_indices
    # ================================================================
    print("--- Test 17b: load_checkpoint duplicate completed_indices ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        rc = build_run_config("/s", 0, 10, "/m", 256)
        cp_dup_ci = {
            "completed_indices": [0, 0, 1],  # duplicate 0
            "items": [
                {"dataset_index": 0}, {"dataset_index": 1},
            ],
            "run_config": rc,
        }
        (tmpdir / "checkpoint.json").write_text(
            json.dumps(cp_dup_ci), encoding="utf-8"
        )
        try:
            load_checkpoint(tmpdir)
            check("t17b: duplicate in completed_indices raises ValueError", False)
        except ValueError as e:
            check(
                "t17b: duplicate in completed_indices raises ValueError",
                "duplicate" in str(e).lower() or "Duplicate" in str(e),
            )

        # Also test non-integer in completed_indices
        cp_bad_type = {
            "completed_indices": [0, "x", 1],
            "items": [{"dataset_index": 0}, {"dataset_index": 1}],
            "run_config": rc,
        }
        (tmpdir / "checkpoint.json").write_text(
            json.dumps(cp_bad_type), encoding="utf-8"
        )
        try:
            load_checkpoint(tmpdir)
            check("t17b: non-int in completed_indices raises ValueError", False)
        except ValueError as e:
            check(
                "t17b: non-int in completed_indices raises ValueError",
                "not an integer" in str(e).lower() or "integer" in str(e).lower(),
            )

    # ================================================================
    # Test 18: check_duplicate_indices detects duplicates
    # ================================================================
    print("--- Test 18: check_duplicate_indices ---")
    try:
        check_duplicate_indices([{"dataset_index": 0}, {"dataset_index": 1}, {"dataset_index": 0}])
        check("t18: duplicate indices raises ValueError", False)
    except ValueError:
        check("t18: duplicate indices raises ValueError", True)
    try:
        check_duplicate_indices([{"dataset_index": 0}, {"dataset_index": 1}])
        check("t18: no duplicates passes", True)
    except ValueError:
        check("t18: no duplicates passes", False)

    # ================================================================
    # Test 19: validate_agent_candidate — normal output
    # ================================================================
    print("--- Test 19: validate_agent_candidate: normal ---")
    normal = {
        "model": "test", "model_path": "/m",
        "candidates": [{"candidate_id": 0, "raw_response": "reasoning...",
                         "sql": "SELECT 1", "parse_success": True, "parse_method": "sql_code_block"}],
        "metadata": {"generation_seconds": 3.5},
    }
    v = validate_agent_candidate(normal)
    check("t19: valid=True", v["valid"] is True)
    check("t19: sql=SELECT 1", v["sql"] == "SELECT 1")
    check("t19: parse_success=True", v["parse_success"] is True)
    check("t19: generation_seconds=3.5", v["generation_seconds"] == 3.5)

    # ================================================================
    # Test 20: validate_agent_candidate — candidates is empty list
    # ================================================================
    print("--- Test 20: validate_agent_candidate: empty candidates ---")
    v2 = validate_agent_candidate({"candidates": []})
    check("t20: valid=False", v2["valid"] is False)
    check("t20: error mentions empty", "empty" in v2["error"].lower() or "missing" in v2["error"].lower())

    # ================================================================
    # Test 21: validate_agent_candidate — candidate missing fields
    # ================================================================
    print("--- Test 21: validate_agent_candidate: missing fields ---")
    v3 = validate_agent_candidate({"candidates": [{}]})
    check("t21: valid=True (handles missing fields gracefully)", v3["valid"] is True)
    check("t21: sql empty string", v3["sql"] == "")
    check("t21: parse_success=False", v3["parse_success"] is False)

    # ================================================================
    # Test 22: validate_agent_candidate — sql is None
    # ================================================================
    print("--- Test 22: validate_agent_candidate: sql=None ---")
    v4 = validate_agent_candidate({"candidates": [{"sql": None, "raw_response": "x",
                                                    "parse_success": True}]})
    check("t22: sql=None -> parse_success=False", v4["parse_success"] is False)
    check("t22: sql=None -> empty string", v4["sql"] == "")

    # ================================================================
    # Test 23: validate_agent_candidate — not a dict
    # ================================================================
    print("--- Test 23: validate_agent_candidate: not a dict ---")
    v5 = validate_agent_candidate("not a dict")
    check("t23: not dict -> valid=False", v5["valid"] is False)

    # ================================================================
    # Test 24: validate_agent_candidate — missing metadata
    # ================================================================
    print("--- Test 24: validate_agent_candidate: missing metadata ---")
    v6 = validate_agent_candidate({
        "candidates": [{"raw_response": "x", "sql": "SELECT 1", "parse_success": True}],
    })
    check("t24: missing metadata -> valid=True", v6["valid"] is True)
    check("t24: generation_seconds defaults to 0", v6["generation_seconds"] == 0.0)

    # ================================================================
    # Test 25: ddl_source recording
    # ================================================================
    print("--- Test 25: ddl_source recording ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        db_dir = tmpdir / "database" / "src_db"
        db_dir.mkdir(parents=True)
        conn = sqlite3.connect(str(db_dir / "src_db.sqlite"))
        conn.execute("CREATE TABLE x (a INTEGER PRIMARY KEY)"); conn.commit(); conn.close()
        (tmpdir / "tables.json").write_text("[]", encoding="utf-8")
        (tmpdir / "dev.json").write_text("[]", encoding="utf-8")
        loader = SpiderLoader(str(tmpdir))
        ddl, source = loader.get_ddl_with_source("src_db")
        check("t25: from DB -> sqlite_master", source == "sqlite_master")
        check("t25: DDL non-empty", len(ddl) > 0)
        tables_list2 = [
            {"db_id": "json_only", "table_names_original": ["y"],
             "column_names_original": [[-1, "*"], [0, "b"]],
             "column_types": ["text", "number"], "primary_keys": [], "foreign_keys": []},
        ]
        (tmpdir / "tables.json").write_text(json.dumps(tables_list2), encoding="utf-8")
        loader2 = SpiderLoader(str(tmpdir))
        ddl2, source2 = loader2.get_ddl_with_source("json_only")
        check("t25: no DB -> tables_json", source2 == "tables_json")

    # ================================================================
    # Test 26: compute_summary filters by requested_indices
    # ================================================================
    print("--- Test 26: compute_summary with requested_indices ---")
    items = [
        {"dataset_index": 0, "parse_success": True, "prediction_execution_success": True,
         "gold_execution_success": True, "custom_execution_match": True,
         "normalized_sql_string_match": True, "generation_seconds": 1.0},
        {"dataset_index": 1, "parse_success": True, "prediction_execution_success": True,
         "gold_execution_success": True, "custom_execution_match": False,
         "normalized_sql_string_match": False, "generation_seconds": 2.0},
        {"dataset_index": 2, "parse_success": False, "prediction_execution_success": False,
         "gold_execution_success": True, "custom_execution_match": False,
         "normalized_sql_string_match": False, "generation_seconds": 3.0},
    ]
    s1 = compute_summary(items, requested_indices={0, 1})
    check("t26: filtered total=2", s1["total_requested"] == 2)
    check("t26: filtered match=1", s1["custom_execution_match_count"] == 1)
    s2 = compute_summary(items)
    check("t26: all total=3", s2["total_requested"] == 3)

    # ================================================================
    # Report
    # ================================================================
    print()
    total = passed + len(failures)
    if failures:
        print(f"=== {passed}/{total} passed, {len(failures)} FAILED ===")
        for f in failures:
            print(f"  {f}")
        return 1
    else:
        print(f"=== All {passed} tests passed ===")
        return 0


if __name__ == "__main__":
    sys.exit(_run_tests())
