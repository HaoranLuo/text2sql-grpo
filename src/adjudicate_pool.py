#!/usr/bin/env python3
"""
src/adjudicate_pool.py — B1 旗舰实验「多实例 vav 裁决器」（纯 CPU，无 GPU 依赖）。

输入
  outputs/eval_pool_b1/items.json（另一 agent 生产）：
    [{"dataset_index": 0, "di": 0, "db_id": "...", "question": "...",
      "gold_sql": "...", "candidates": [{"model": "sft_phase1"|"sft_v2",
      "sql": "...", "parse_success": true, "sample_idx": 0}, ...]}, ...]

流程（每题，全部 CPU）
  1. 去重：SQL 文本 strip+lower+空白折叠 归一化合并重复候选，记录 dup 信息。
  2. 执行：每个唯一 SQL 在每个实例文件上执行（sqlite3 只读 URI；16 线程并行、
     每查询独立连接（查询在工作线程内打开，无跨线程连接共享）；进度回调
     （VM 步数上限）+ 墙钟 watchdog（conn.interrupt()）双重剪枝；跨候选执行
     缓存按 (sql, db_path) 键）；产出每候选「实例签名向量」= 每个实例上的
     规范化包语义结果（行排序 canonical 字符串，重复行保留，列序保留；
     执行失败 → 该实例签名 = ERROR）。
  3. 裁决臂（全部离线重算，同一批候选）：
     arm_maj        SQL 文本多数票（去重组按总票数加权，平票取 sample_idx 最小）
     arm_vav1       只用原始 <db_id>.sqlite 一个实例的签名分组（现行做法基线）
     arm_vav_multi2 用前 2 个实例文件（sorted 顺序）的签名向量分组
     arm_vav_multi_all 用全部实例的签名向量分组（主治疗）
     每个臂 × 池 {v1_only（只 sft_phase1）, v2_only, both}——池消融是候选子集，
     零额外执行。
  4. 判定（每题每臂每池）：臂胜者 SQL 与 gold 的官方语义 exec match——
     复用 test-suite-sql-eval exec_eval.py 的 eval_exec_match 机制：
     postprocess → remove_distinct（keep_distinct=False，与官方默认一致）→
     order_matters = gold 含 'order by' → 每个实例上 result_eq（bag semantics
     + 列置换容忍）→ 全实例等价才算对；replace_cur_year 在执行前替换。
  5. 输出：outputs/adjudicate_b1/summary.json（臂×池准确率矩阵 + 执行统计）
     + 每臂每池 items_<arm>_<pool>.json（predicted_sql 与 scripts/eval_official.sh
     兼容；空胜者按 AGENTS.md 规则写 "SELECT 1" 不跳过）。

关键语义纪律（与官方/finer 对齐）
  - 空结果（0 行）是合法签名（"SUCCESS_VALUES:"），不是错误；
    语法错/执行错 → 该实例签名 = "ERROR"。
  - 分组用包语义（行排序，重复行保留，列序保留，无列置换容忍——避免过度等价，
    与 FINER 的 header-agnostic 行内排序+行集合去重刻意不同，见 vav_voting.py
    对比）；官方判定用 result_eq（含 order_matters 与列置换容忍）。
  - choose_group_vav 语义照抄（finer_port/vav_voting.py）：
    只 SUCCESS 组（这里 = 向量所有实例分量均成功执行的候选才能入组，与
    run_vav_voting 只对成功候选分组一致）；跳空组（全向量无任何值）/全零组
    （数值型值全部 |x|<1e-12，非数值 token 丢弃，与 FINER 一致）；size 最大；
    平票取 key 字符串最大；全被过滤 → fallback 最大 SUCCESS 组；再无 → NO_RESULTS。
  - 每题某臂 NO_RESULTS → 胜者回退 = 同池 arm_maj 胜者（summary 记录
    fallback_maj 计数）；池为空 → no_pool（判错，predicted_sql 写 SELECT 1）。
  - 某 db 无实例文件 → 判定「空实例集合恒真」——镜像官方 eval_exec_match
    （db_paths 为空时循环不执行直接返回 1），summary.dataset_stats 显式计数。

用法
  # HPC 全量（默认 1034 题、全部实例）：
  python src/adjudicate_pool.py
  # 或显式：
  python src/adjudicate_pool.py --items outputs/eval_pool_b1/items.json \
      --out-dir outputs/adjudicate_b1 --spider-dir data/spider_data \
      --threads 16 --query-timeout 30
  # 冒烟：--limit 20；实例数上限策略：--max-instances 8
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ITEMS = PROJECT_ROOT / "outputs" / "eval_pool_b1" / "items.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "adjudicate_b1"
DEFAULT_SPIDER_DIR = PROJECT_ROOT / "data" / "spider_data"

ARMS = ["arm_maj", "arm_vav1", "arm_vav_multi2", "arm_vav_multi_all"]
POOLS = ["v1_only", "v2_only", "both"]

ERROR_SIG = "ERROR"
SUCCESS_PREFIX = "SUCCESS_VALUES:"
_SEP_ROW = "\x1e"  # 行间分隔（json 编码字符串不会裸出现控制字符）
_SEP_VAL = "\x1f"  # 行内值分隔
_TRUNC_MARK = "TRUNC"

# ===================================================================
# SQL 文本归一化 / 官方清洗管线
# ===================================================================


def normalize_for_dedup(sql: Any) -> str:
    """去重键：strip + lower + 空白折叠（任务规格，非 spider_utils.normalize_sql）。"""
    return " ".join((sql or "").strip().lower().split())


def clean_pred_for_official(sql: Any) -> str:
    """镜像 scripts/eval_official.sh 对 predicted_sql 的清洗：
    strip → rstrip(';') → 去掉 -- 行注释 → 空白折叠单空格。"""
    s = (sql or "").strip().rstrip(";")
    s = re.sub(r"^\s*--.*$", "", s, flags=re.MULTILINE)
    return " ".join(s.split())


def clean_gold_for_official(sql: Any) -> str:
    """镜像 eval_official.sh 对 gold 的处理：仅 strip + rstrip(';')（不折叠）。"""
    return (sql or "").strip().rstrip(";")


def postprocess(query: str) -> str:
    """官方 exec_eval.postprocess：修掉比较运算符间多余空格。"""
    return query.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")


try:
    import sqlparse  # type: ignore

    def remove_distinct(s: str) -> str:
        """官方 parse.remove_distinct：token 级剔除 DISTINCT（不动字符串字面量）。"""
        if not s:
            return s
        try:
            toks = [t.value for t in list(sqlparse.parse(s)[0].flatten())]
            return "".join(t for t in toks if t.lower() != "distinct")
        except Exception:
            return _DISTINCT_RE.sub("", s)

except ImportError:  # 退化：正则（字符串字面量含 distinct 一词会被误删，仅兜底）
    _DISTINCT_RE = re.compile(r"\bDISTINCT\b", flags=re.IGNORECASE)

    def remove_distinct(s: str) -> str:
        return _DISTINCT_RE.sub("", s)


_CUR_YEAR_RE = re.compile(r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)", flags=re.IGNORECASE)


def replace_cur_year(query: str) -> str:
    """官方 exec_eval.replace_cur_year：YEAR(CURDATE()) → 2020（确定化）。"""
    return _CUR_YEAR_RE.sub("2020", query)


def official_transform(sql: Any, is_pred: bool, keep_distinct: bool) -> str:
    """官方评估会执行的最终 SQL 文本：
    eval_official.sh 清洗 → postprocess → remove_distinct（keep_distinct=False）→
    replace_cur_year。"""
    s = clean_pred_for_official(sql) if is_pred else clean_gold_for_official(sql)
    s = postprocess(s)
    if not keep_distinct:
        s = remove_distinct(s)
    return replace_cur_year(s)


# ===================================================================
# 官方 result_eq 移植（test-suite-sql-eval exec_eval.py，逐函数忠实照抄）
# ===================================================================


def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    return set(s1) == set(s2)


def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(rng: random.Random, tab1_sets_by_columns: List[set], result2: List[Tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)
    # 采样 20 行收紧置换空间（只删必然无效的映射，剪枝是 sound 的，
    # 正确性确定；rng 固定种子保证可复现）
    for _ in range(20):
        random_tab2_row = rng.choice(result2)
        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    """官方语义：bag semantics + 列置换容忍；order_matters=True 时行序也须一致。"""
    if len(result1) == 0 and len(result2) == 0:
        return True
    if len(result1) != len(result2):
        return False
    num_cols = len(result1[0])
    if len(result2[0]) != num_cols:
        return False
    if not quick_rej(result1, result2, order_matters):
        return False
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]
    for perm in get_constraint_permutation(rng, tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False


# 官方 get_constraint_permutation 使用 random.choice；固定种子保证 rerun 稳定
# （剪枝 sound，正确性与种子无关）。模块级单例。
rng = random.Random(0)


# ===================================================================
# 分组签名（包语义：行排序、重复行保留、列序保留；无列置换容忍）
# ===================================================================


def _sig_value(v: Any) -> List[Any]:
    """值 → 带类型标签的编码（int/float/str/None/bytes 互不误并，可 json 往返）。"""
    if v is None:
        return ["N"]
    if isinstance(v, bool):
        return ["B", 1 if v else 0]
    if isinstance(v, int):
        return ["I", v]
    if isinstance(v, float):
        return ["F", v]
    if isinstance(v, (bytes, bytearray)):
        try:
            v = bytes(v).decode("utf-8")
        except UnicodeDecodeError:
            v = bytes(v).hex()
        return ["S", str(v)]
    return ["S", str(v)]


def rows_to_group_signature(rows: List[List[Any]], truncated: bool = False) -> str:
    """成功结果 → 'SUCCESS_VALUES:<sig>'；sig = 排序后的行 canonical 字符串
    （每行 = _SEP_VAL 连接的带类型值 json；行间 _SEP_ROW 连接；重复行保留）。
    truncated=True（行数超 row_cap）追加 _TRUNC_MARK，避免与未截断签名误并。"""
    row_strings = [
        _SEP_VAL.join(
            json.dumps(_sig_value(v), ensure_ascii=False, separators=(",", ":"))
            for v in row
        )
        for row in rows
    ]
    sig = SUCCESS_PREFIX + _SEP_ROW.join(sorted(row_strings))
    if truncated:
        sig += _SEP_VAL + _TRUNC_MARK
    return sig


def outcome_signature(outcome: Dict[str, Any]) -> str:
    """执行 outcome → 单实例分组签名（失败 → ERROR）。"""
    if not outcome.get("ok"):
        return ERROR_SIG
    return rows_to_group_signature(outcome.get("rows") or [], outcome.get("truncated", False))


def parse_signature_values(sig: Any) -> List[Tuple[str, Any]]:
    """从 'SUCCESS_VALUES:<sig>' 提取 (tag, value) 列表（跨组件用，供空/零判定）。"""
    if not isinstance(sig, str) or not sig.startswith(SUCCESS_PREFIX):
        return []
    body = sig[len(SUCCESS_PREFIX):]
    out: List[Tuple[str, Any]] = []
    for row_str in body.split(_SEP_ROW):
        if row_str == "":
            continue
        for tok in row_str.split(_SEP_VAL):
            if tok == _TRUNC_MARK or tok == "":
                continue
            try:
                decoded = json.loads(tok)
            except Exception:
                continue
            if isinstance(decoded, list) and len(decoded) >= 1:
                out.append((str(decoded[0]), decoded[1] if len(decoded) > 1 else None))
    return out


def _vector_numeric_values(key: Tuple[str, ...]) -> List[float]:
    """FINER 语义：只收集数值型值（int/float），非数值 token 直接丢弃。"""
    nums: List[float] = []
    for comp in key:
        for tag, val in parse_signature_values(comp):
            if tag in ("I", "F") and val is not None:
                try:
                    nums.append(float(val))
                except (TypeError, ValueError):
                    continue
    return nums


def vector_is_empty(key: Tuple[str, ...]) -> bool:
    """跳空组：整个向量没有任何值 token（与 FINER _is_empty_group 一致）。"""
    return len(parse_signature_values_for_empty(key)) == 0


def parse_signature_values_for_empty(key: Tuple[str, ...]) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    for comp in key:
        out.extend(parse_signature_values(comp))
    return out


def vector_is_all_zero(key: Tuple[str, ...]) -> bool:
    """全零组（FINER 语义）：至少一个数值型值且所有数值型值 |x| < 1e-12。"""
    nums = _vector_numeric_values(key)
    return len(nums) > 0 and all(abs(x) < 1e-12 for x in nums)


def choose_group_vav(groups: Dict[Tuple[str, ...], Dict[str, Any]]) -> Optional[Tuple[str, ...]]:
    """choose_group_vav 语义照抄（finer_port/vav_voting.py，推广到签名向量 key）：
    1) 只 SUCCESS_VALUES 组——入组前置条件已保证（含 ERROR 分量的候选不入组）；
    2) 硬跳过空组与全零组；
    3) 取剩余组 size 最大者，平票按 str(key) 最大者；
    4) 全被过滤 → fallback 最大 SUCCESS 组；
    5) 无组 → None（等价 NO_RESULTS，调用方回退 arm_maj）。"""
    if not groups:
        return None
    filtered = [
        (k, m) for k, m in groups.items()
        if not vector_is_empty(k) and not vector_is_all_zero(k)
    ]
    if filtered:
        return max(filtered, key=lambda km: (int(km[1].get("size", 0)), str(km[0])))[0]
    return max(groups.items(), key=lambda km: (int(km[1].get("size", 0)), str(km[0])))[0]


# ===================================================================
# 执行引擎：16 线程池、每查询独立只读连接、进度回调 + 墙钟 watchdog 剪枝、
# 跨候选缓存按 (sql, db_path) 键
# ===================================================================


def _failure(error: str, error_type: str) -> Dict[str, Any]:
    return {"ok": False, "rows": [], "row_count": 0, "truncated": False,
            "error": error, "error_type": error_type, "seconds": 0.0}


_EMPTY_SQL_OUTCOME = _failure("Empty SQL", "empty_sql")


def _query_thread(sql: str, db_path: str, holder: Dict[str, Any],
                  row_cap: int, max_vm_steps: int) -> None:
    """在专用线程内执行一次查询：连接在此线程内打开/关闭（无跨线程连接共享，
    interrupt() 从 watchdog 线程调用是 sqlite3 官方支持的线程安全接口）。"""
    conn = None
    start = time.perf_counter()
    try:
        if not os.path.exists(db_path):
            holder["result"] = _failure(f"Database file not found: {db_path}", "db_missing")
            return
        uri = Path(db_path).resolve().as_uri()
        conn = sqlite3.connect(f"{uri}?mode=ro", uri=True)
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
        except sqlite3.Warning:
            holder["result"] = _failure("SQL rejected: multi-statement input", "multi_statement")
            return
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
            "ok": True, "rows": rows, "row_count": total,
            "truncated": total > row_cap, "error": None, "error_type": None,
            "seconds": round(time.perf_counter() - start, 4),
        }
    except sqlite3.OperationalError as exc:
        if "interrupt" in str(exc).lower():
            holder["result"] = _failure(
                f"Query interrupted after {max_vm_steps} SQLite VM steps: {exc}",
                "interrupted")
        else:
            holder["result"] = _failure(str(exc), "sqlite_error")
    except Exception as exc:
        holder["result"] = _failure(f"{type(exc).__name__}: {exc}", "sqlite_error")
    finally:
        holder["conn"] = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


class ExecutionEngine:
    """并行执行器。results 字典 = 跨候选执行缓存，键 (sql, db_path)。
    每个 (sql, db_path) 全局只执行一次；'hits' = 计划任务命中已有结果的次数。"""

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
            return _EMPTY_SQL_OUTCOME
        return self._results.get((sql, db_path))

    def _execute_one(self, sql: str, db_path: str) -> Dict[str, Any]:
        holder: Dict[str, Any] = {"conn": None, "result": None}
        t = threading.Thread(
            target=_query_thread, args=(sql, db_path, holder, self.row_cap, self.max_vm_steps),
            daemon=True)
        t.start()
        t.join(self.query_timeout)
        if not t.is_alive():
            return holder["result"] if holder["result"] is not None else \
                _failure("Worker finished without result", "worker_error")
        # 墙钟预算耗尽 → interrupt 杀查询（即使底层报 interrupted 也归 timeout）
        if holder["conn"] is not None:
            try:
                holder["conn"].interrupt()
            except Exception:
                pass
        t.join(5.0)
        if t.is_alive():
            return _failure(
                f"Wall-clock timeout after {self.query_timeout}s and worker unresponsive after interrupt",
                "worker_hang")
        return _failure(f"Wall-clock timeout after {self.query_timeout}s", "timeout")

    def run(self, tasks: List[Tuple[str, str]], phase: str) -> None:
        stats = {"tasks": len(tasks), "hits": 0, "executed": 0, "failures": 0,
                 "interrupted": 0, "timeouts": 0, "truncated": 0, "wall_seconds": 0.0}
        start = time.perf_counter()
        todo: List[Tuple[str, str]] = []
        for sql, db_path in tasks:
            if (sql or "").strip() == "":
                continue  # 空 SQL 永远不触库（get() 直接返回合成失败）
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
                        outcome = _failure(f"Worker crash: {exc}", "worker_error")
                    self._results[(sql, db_path)] = outcome
                    stats["executed"] += 1
                    if not outcome["ok"]:
                        stats["failures"] += 1
                        if outcome.get("error_type") == "interrupted":
                            stats["interrupted"] += 1
                        elif outcome.get("error_type") == "timeout":
                            stats["timeouts"] += 1
                    if outcome.get("truncated"):
                        stats["truncated"] += 1
                    done += 1
                    if done % 2000 == 0:
                        print(f"  [{phase}] {done}/{len(todo)} executions ...", file=sys.stderr)
        stats["wall_seconds"] = round(time.perf_counter() - start, 2)
        self._stats[phase] = stats


# ===================================================================
# 实例枚举
# ===================================================================


def list_instances(db_dir: str, db_id: str, max_instances: Optional[int]) -> List[str]:
    """官方语义：os.listdir 中 basename 含 '.sqlite' 的文件全枚举（本实现排序
    保证确定性，且把原始 <db_id>.sqlite 固定放首位——sorted 序它本来就第一）。
    max_instances 非空时截断（成本上限策略）。"""
    if not db_dir or not os.path.isdir(db_dir):
        return []
    names = sorted(
        n for n in os.listdir(db_dir)
        if ".sqlite" in n and os.path.isfile(os.path.join(db_dir, n)))
    original = f"{db_id}.sqlite"
    if original in names:
        names = [original] + [n for n in names if n != original]
    paths = [os.path.join(db_dir, n) for n in names]
    if max_instances:
        paths = paths[: max(0, int(max_instances))]
    return paths


# ===================================================================
# 单题裁决
# ===================================================================

_ARM_K = {"arm_maj": 0, "arm_vav1": 1, "arm_vav_multi2": 2, "arm_vav_multi_all": -1}


def _dedupe(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """候选去重：key=normalize_for_dedup(sql)；合并 count/min_sample_idx/models；
    sql_text 取组内 sample_idx 最小者的原文。"""
    entries: List[Dict[str, Any]] = []
    by_key: Dict[str, Dict[str, Any]] = {}
    for i, c in enumerate(candidates):
        text = c.get("sql") or ""
        key = normalize_for_dedup(text)
        idx = c.get("sample_idx", i)
        e = by_key.get(key)
        if e is None:
            e = {"key": key, "sql_text": text, "count": 0,
                 "min_sample_idx": idx if isinstance(idx, int) else i,
                 "models": set()}
            by_key[key] = e
            entries.append(e)
        e["count"] += 1
        e["models"].add(c.get("model"))
        if isinstance(idx, int) and idx < e["min_sample_idx"]:
            e["min_sample_idx"] = idx
            e["sql_text"] = text
    return entries


def _judge_winner(pred_text: Optional[str], gold_text: str, instances: List[str],
                  engine: ExecutionEngine, keep_distinct: bool) -> Dict[str, Any]:
    """官方语义 exec match（全实例包语义等价）。winner None（no_pool）恒判错。
    order_matters 在 gold 变换后文本上判定（官方同款：'order by' in g_str.lower()）。"""
    gold_sql = official_transform(gold_text, is_pred=False, keep_distinct=keep_distinct)
    order_matters = "order by" in gold_sql.lower()
    if pred_text is None:
        return {"correct": False, "gold_exec_error": False, "order_matters": order_matters}
    pred_sql = official_transform(pred_text, is_pred=True, keep_distinct=keep_distinct)
    correct = True
    gold_err = False
    for inst in instances:
        g_out = engine.get(gold_sql, inst)
        p_out = engine.get(pred_sql, inst)
        if g_out is None or not g_out["ok"]:
            # 官方此处直接 assert 崩溃；批处理改为记录并判错，绝不 crash
            gold_err = True
            correct = False
            break
        if p_out is None or not p_out["ok"]:
            correct = False
            break
        g_rows = [tuple(r) for r in g_out["rows"]]
        p_rows = [tuple(r) for r in p_out["rows"]]
        eq = result_eq(g_rows, p_rows, order_matters=order_matters)
        if g_out["truncated"] or p_out["truncated"]:
            # 截断（超 row_cap）保守处理：总数不等必不等；等则按已存前缀多重集判定
            if g_out["row_count"] != p_out["row_count"]:
                eq = False
        if not eq:
            correct = False
            break
    return {"correct": correct, "gold_exec_error": gold_err, "order_matters": order_matters}


def adjudicate_question(item: Dict[str, Any], engine: ExecutionEngine,
                        db_instances: List[str],
                        model_v1: str, model_v2: str) -> Dict[str, Any]:
    """对一题完成 去重 → 签名向量 → 4 臂 × 3 池 裁决（胜者选择）。
    返回 {"item": ..., "entries": ..., "num_candidates": ..., "num_unique_candidates": ...,
          "num_instances": ..., "results": {(arm, pool): record}}。
    判定所需执行由 main 统一在 phase 2 并行补齐（本函数只读执行缓存）。"""
    db_id = item.get("db_id", "")
    candidates = item.get("candidates") or []
    instances = db_instances

    entries = _dedupe(candidates)

    # ---- 每唯一 SQL 的实例签名向量（原始文本执行，缓存已由 phase 1 填充）----
    sigs_per_entry: List[List[str]] = []
    for e in entries:
        if not (e["sql_text"] or "").strip():
            sigs = [ERROR_SIG] * len(instances)
        else:
            sigs = [outcome_signature(engine.get(e["sql_text"], inst))
                    for inst in instances]
        sigs_per_entry.append(sigs)

    # ---- 池成员 ----
    def in_pool(model: Any, pool: str) -> bool:
        if pool == "both":
            return True
        if pool == "v1_only":
            return model == model_v1
        return model == model_v2

    pool_entry_votes: Dict[str, Dict[int, int]] = {p: defaultdict(int) for p in POOLS}
    pool_cands: Dict[str, int] = {p: 0 for p in POOLS}
    for c in candidates:
        model = c.get("model")
        for p in POOLS:
            if in_pool(model, p):
                pool_cands[p] += 1
                # 定位该候选所属去重组
                for ei, e in enumerate(entries):
                    if normalize_for_dedup(c.get("sql")) == e["key"]:
                        pool_entry_votes[p][ei] += 1
                        break

    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for arm in ARMS:
        k = _ARM_K[arm]
        for pool in POOLS:
            if arm == "arm_maj":
                rec = _arm_maj(pool, entries, pool_entry_votes)
            else:
                rec = _arm_vav(arm, k, pool, entries, sigs_per_entry,
                               pool_entry_votes, instances)
            winner_text = rec["text"]
            # empty_winner = 有胜者但胜者 SQL 文本为空（多数票选中了空文本组）；
            # no_pool（池内无候选）不算 empty_winner，有独立 source 标签
            rec["empty_winner"] = (winner_text == "")
            # is_correct / gold_exec_error / order_matters 由 main 在 phase 2
            # 执行补齐后统一判定填充（此处只选胜者，不执行 SQL）
            results[(arm, pool)] = rec

    return {
        "item": item,
        "entries": entries,
        "num_candidates": len(candidates),
        "num_unique_candidates": len(entries),
        "num_instances": len(instances),
        "results": results,
    }


def _arm_maj(pool: str, entries: List[Dict[str, Any]],
             pool_entry_votes: Dict[str, Dict[int, int]]) -> Dict[str, Any]:
    """SQL 文本多数票：去重组按池内总票数加权；平票取 min_sample_idx 最小者。"""
    votes = pool_entry_votes[pool]
    if not votes:
        return {"source": "no_pool", "text": None, "votes": 0,
                "group_key": None, "group_size": 0, "instances_used": 0,
                "vav_grouped": 0, "vav_excluded": 0}
    best_idx = max(votes.items(), key=lambda kv: (kv[1], -entries[kv[0]]["min_sample_idx"]))[0]
    # max by (votes, -min_idx) 等价于 votes 最大、平票 min_idx 最小
    return {"source": "maj", "text": entries[best_idx]["sql_text"],
            "votes": votes[best_idx], "group_key": None, "group_size": votes[best_idx],
            "instances_used": 0, "vav_grouped": 0, "vav_excluded": 0}


def _arm_vav(arm: str, k: int, pool: str, entries: List[Dict[str, Any]],
             sigs_per_entry: List[List[str]],
             pool_entry_votes: Dict[str, Dict[int, int]],
             instances: List[str]) -> Dict[str, Any]:
    """单/多实例 vav：只用实例子集 [0:k] 的签名向量分组（k=-1 = 全部），
    choose_group_vav 语义；NO_RESULTS → 回退同池 arm_maj。"""
    votes = pool_entry_votes[pool]
    subset = instances if k < 0 else instances[:k]
    n_used = len(subset)
    groups: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    grouped = 0
    excluded = 0
    for ei, cnt in sorted(votes.items()):
        sigs = sigs_per_entry[ei][:n_used]
        if any(s == ERROR_SIG for s in sigs):
            excluded += cnt
            continue
        grouped += cnt
        key = tuple(sigs)
        g = groups.setdefault(key, {"size": 0, "members": []})
        g["size"] += cnt
        g["members"].append(ei)
    chosen = choose_group_vav(groups)
    if chosen is None:
        # NO_RESULTS → 回退 arm_maj（同池）
        fallback = _arm_maj(pool, entries, pool_entry_votes)
        if fallback["source"] == "no_pool":
            return {"source": "no_pool", "text": None, "votes": 0, "group_key": None,
                    "group_size": 0, "instances_used": n_used,
                    "vav_grouped": grouped, "vav_excluded": excluded}
        fallback["source"] = "fallback_maj"
        fallback["instances_used"] = n_used
        fallback["group_key"] = None
        fallback["vav_grouped"] = grouped
        fallback["vav_excluded"] = excluded
        return fallback
    members = groups[chosen]["members"]
    best_member = min(members, key=lambda ei: (entries[ei]["min_sample_idx"], entries[ei]["key"]))
    return {"source": "vav", "text": entries[best_member]["sql_text"],
            "votes": groups[chosen]["size"], "group_key": str(chosen),
            "group_size": groups[chosen]["size"], "instances_used": n_used,
            "vav_grouped": grouped, "vav_excluded": excluded}


# ===================================================================
# 主流程
# ===================================================================


def _load_items(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        data = data["items"]
    if not isinstance(data, list):
        raise ValueError(f"items.json 结构异常（期望 list 或 {{'items': [...]}}）: {path}")
    return data


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="B1 多实例 vav 裁决器（纯 CPU）")
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--spider-dir", type=Path, default=DEFAULT_SPIDER_DIR)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--query-timeout", type=float, default=30.0,
                    help="单查询墙钟 watchdog 秒数（超时 interrupt 剪枝）")
    ap.add_argument("--max-vm-steps", type=int, default=5_000_000,
                    help="单查询 SQLite VM 步数上限（进度回调剪枝）")
    ap.add_argument("--row-cap", type=int, default=100_000,
                    help="单结果内存行数上限（超出截断，签名加 TRUNC 标记）")
    ap.add_argument("--max-instances", type=int, default=None,
                    help="每库实例文件数上限（成本上限策略；默认 None = 全部）")
    ap.add_argument("--keep-distinct", action="store_true",
                    help="官方判定保留 DISTINCT（默认 False，与官方一致）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-v1", default="sft_phase1")
    ap.add_argument("--model-v2", default="sft_v2")
    ap.add_argument("--limit", type=int, default=None, help="只裁决前 N 题（冒烟）")
    args = ap.parse_args(argv)

    global rng
    rng = random.Random(args.seed)

    items = _load_items(args.items)
    if args.limit:
        items = items[: args.limit]
    print(f"[adjudicate] {len(items)} 题，实例上限 {args.max_instances or '全部'}，"
          f"线程 {args.threads}，查询超时 {args.query_timeout}s", file=sys.stderr)

    database_dir = args.spider_dir / "database"
    engine = ExecutionEngine(args.threads, args.query_timeout, args.max_vm_steps, args.row_cap)

    # ---- 实例枚举（按 db_id 缓存）----
    db_instances_cache: Dict[str, List[str]] = {}

    def instances_for(db_id: str) -> List[str]:
        if db_id not in db_instances_cache:
            db_instances_cache[db_id] = list_instances(
                str(database_dir / db_id), db_id, args.max_instances)
        return db_instances_cache[db_id]

    # ---- Phase 1: 所有唯一候选 SQL × 实例 并行执行（跨候选缓存）----
    phase1_tasks: List[Tuple[str, str]] = []
    for item in items:
        insts = instances_for(item.get("db_id", ""))
        for e in _dedupe(item.get("candidates") or []):
            text = (e["sql_text"] or "").strip()
            if not text:
                continue
            for inst in insts:
                phase1_tasks.append((text, inst))
    # 去重（set）后执行——(sql, db_path) 全局只跑一次
    phase1_tasks = list(set(phase1_tasks))
    print(f"[adjudicate] phase1: {len(phase1_tasks)} 个唯一 (sql, db_path) 任务", file=sys.stderr)
    t0 = time.perf_counter()
    engine.run(phase1_tasks, phase="grouping")
    print(f"[adjudicate] phase1 完成: {engine._stats['grouping']}", file=sys.stderr)

    # ---- 每题裁决（胜者选择，不执行 SQL）----
    per_question: List[Dict[str, Any]] = []
    for qi, item in enumerate(items):
        per_question.append(adjudicate_question(
            item, engine, instances_for(item.get("db_id", "")),
            args.model_v1, args.model_v2))
        if (qi + 1) % 50 == 0 or qi + 1 == len(items):
            print(f"[adjudicate] 裁决 {qi + 1}/{len(items)} 题 "
                  f"({(time.perf_counter() - t0):.1f}s)", file=sys.stderr)

    # ---- Phase 2: 判定所需（gold 变换后 + 各臂胜者变换后）SQL × 实例 并行执行 ----
    phase2_tasks: List[Tuple[str, str]] = []
    for qc in per_question:
        item = qc["item"]
        insts = instances_for(item.get("db_id", ""))
        gold_t = official_transform(item.get("gold_sql") or "", is_pred=False,
                                    keep_distinct=args.keep_distinct)
        for inst in insts:
            phase2_tasks.append((gold_t, inst))
        for (arm, pool), rec in qc["results"].items():
            if rec["text"] is None:
                continue
            pred_t = official_transform(rec["text"], is_pred=True,
                                        keep_distinct=args.keep_distinct)
            for inst in insts:
                phase2_tasks.append((pred_t, inst))
    phase2_tasks = list(set(phase2_tasks))
    print(f"[adjudicate] phase2: {len(phase2_tasks)} 个唯一判定任务 "
          f"（其中命中 phase1 缓存的不重跑）", file=sys.stderr)
    engine.run(phase2_tasks, phase="judgment")
    print(f"[adjudicate] phase2 完成: {engine._stats['judgment']}", file=sys.stderr)

    # ---- 重新判定（phase 2 已补齐所有 outcome，现在纯内存计算）----
    for qc in per_question:
        item = qc["item"]
        gold_raw = item.get("gold_sql") or ""
        insts = instances_for(item.get("db_id", ""))
        for (arm, pool), rec in qc["results"].items():
            j = _judge_winner(rec["text"], gold_raw, insts, engine, args.keep_distinct)
            rec["is_correct"] = j["correct"]
            rec["gold_exec_error"] = j["gold_exec_error"]
            rec["order_matters"] = j["order_matters"]

    # ---- 汇总 + 输出 ----
    cells: Dict[str, Dict[str, Dict[str, Any]]] = {}
    dataset_stats: Dict[str, Any] = {
        "total_questions": len(items),
        "questions_with_no_instances": 0,
        "questions_with_gold_exec_error": 0,
        "db_instance_count": {
            db: len(insts) for db, insts in db_instances_cache.items()},
        "db_instance_min": None,
        "db_instance_max": None,
        "db_instance_avg": None,
    }
    inst_counts = [len(v) for v in db_instances_cache.values()]
    if inst_counts:
        dataset_stats["db_instance_min"] = min(inst_counts)
        dataset_stats["db_instance_max"] = max(inst_counts)
        dataset_stats["db_instance_avg"] = round(sum(inst_counts) / len(inst_counts), 2)

    total_cands = 0
    unique_cands = 0
    max_dup = 0
    for qc in per_question:
        total_cands += qc["num_candidates"]
        unique_cands += qc["num_unique_candidates"]
        for e in qc["entries"]:
            max_dup = max(max_dup, e["count"] - 1)
        if qc["num_instances"] == 0:
            dataset_stats["questions_with_no_instances"] += 1
        if any(r["gold_exec_error"] for r in qc["results"].values()):
            dataset_stats["questions_with_gold_exec_error"] += 1
    dedup_stats = {
        "total_candidates": total_cands,
        "unique_after_dedup": unique_cands,
        "merged_duplicates": total_cands - unique_cands,
        "max_dup_count": max_dup,
    }

    for arm in ARMS:
        cells[arm] = {}
        for pool in POOLS:
            cell: Dict[str, Any] = {
                "total": len(items),
                "correct": 0,
                "accuracy": 0.0,
                "winner_sources": {"maj": 0, "vav": 0, "fallback_maj": 0, "no_pool": 0},
                "empty_winner": 0,
                "gold_exec_error": 0,
                "candidates_available": 0,
            }
            for qc in per_question:
                rec = qc["results"][(arm, pool)]
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
            cells[arm][pool] = cell

    total_wall = engine._stats.get("grouping", {}).get("wall_seconds", 0.0) + \
        engine._stats.get("judgment", {}).get("wall_seconds", 0.0)
    summary = {
        "meta": {
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_items": str(args.items),
            "output_dir": str(args.out_dir),
            "spider_dir": str(args.spider_dir),
            "threads": args.threads,
            "query_timeout_seconds": args.query_timeout,
            "max_vm_steps": args.max_vm_steps,
            "row_cap": args.row_cap,
            "max_instances_cap": args.max_instances,
            "keep_distinct": args.keep_distinct,
            "seed": args.seed,
            "model_v1": args.model_v1,
            "model_v2": args.model_v2,
            "arms": ARMS,
            "pools": POOLS,
            "semantics": (
                "grouping: bag semantics, row-sorted canonical, column order kept, "
                "no column permutation tolerance; judgment: official eval_exec_match "
                "mechanism (postprocess + remove_distinct + replace_cur_year + "
                "result_eq with order_matters and column permutation), all instances "
                "must match; NO_RESULTS falls back to same-pool arm_maj"),
        },
        "dataset_stats": dataset_stats,
        "dedup_stats": dedup_stats,
        "execution_stats": {
            "grouping_phase": engine._stats.get("grouping", {}),
            "judgment_phase": engine._stats.get("judgment", {}),
            "total_wall_seconds": round(total_wall, 2),
        },
        "accuracy_matrix": cells,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for arm in ARMS:
        for pool in POOLS:
            out_items = []
            for qc in per_question:
                item = qc["item"]
                rec = qc["results"][(arm, pool)]
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
            (args.out_dir / f"items_{arm}_{pool}.json").write_text(
                json.dumps(out_items, ensure_ascii=False, indent=1), encoding="utf-8")

    # 终端矩阵（主治疗臂加粗标注）
    print("\n=== accuracy matrix (correct / total) ===")
    for arm in ARMS:
        row = []
        for pool in POOLS:
            c = cells[arm][pool]
            tag = "*" if arm == "arm_vav_multi_all" and pool == "both" else " "
            row.append(f"{pool}={c['correct']}/{c['total']} ({c['accuracy']:.4f}){tag}")
        print(f"  {arm:20s} " + "  ".join(row))
    print(f"\nsummary -> {args.out_dir / 'summary.json'}")
    print(f"items   -> {args.out_dir / 'items_<arm>_<pool>.json'} (12 files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
