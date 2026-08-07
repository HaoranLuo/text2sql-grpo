"""
P1: vav 执行分组投票（FINER majority_voting.py 的本地化移植，纯 Python、无 GPU 依赖）。

来源语义：finer-sql/evaluation/majority_voting.py
  - normalize_execution_result：失败 → `ERROR: {err}`；成功空行 → `SUCCESS_ROWS_COUNT:{n}`；
    成功有行 → 每行值转 str 排序后 `|` 连接、行集合去重排序后 `;` 连接、截断 200 字符 →
    `SUCCESS_VALUES:{sig}`（header-agnostic：忽略列名列序）。
  - is_syntax_error：执行失败且错误信息不含 timeout/network/http/request/api/server/
    connection 等基础设施关键词 → 视为 SQL 语法错误（不参与投票分组）。
  - choose_group_vav：仅 `SUCCESS_VALUES:` 组；is_empty 或 is_all_zero（全部值可解析为
    数值且 |x|<1e-12）硬跳过；取 max(groups, key=(int(size), key字符串))；全被过滤则
    fallback 最大 SUCCESS_VALUES 组；再无 → `NO_RESULTS`；组内取第一条 SQL。

本地化差异（仅执行后端）：
  - FINER 走外部 HTTP API（结果 dict 带 ok/rows/row_count/error）；
    本实现直接复用 src/spider_utils.py 的 DatabaseExecutor（本地 SQLite 只读执行，
    结果 dict 带 success/full_rows/row_count/error/error_type），
    另加 `(db_id, normalize_sql)` 内存缓存——30 条候选高度重复，可省 60-70% 执行。
  - 分组口径与 FINER 完全一致（排序值集合，非 compare_execution_results 的
    ORDER BY 感知 multiset）——这是 FINER 官方 85.0% 复现的前提；
    compare_execution_results 仅由 eval_vav.py 作为「训练同口径」对照臂使用
    （见 PLAN §8 风险「奖励/评估口径漂移」）。

无模型即可运行自测：
    python finer_port/vav_voting.py
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 把项目 src/ 加入 sys.path，复用 spider_utils（DatabaseExecutor / normalize_sql）
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from spider_utils import DatabaseExecutor, normalize_sql  # noqa: E402

# ---------------------------------------------------------------------------
# 基础设施错误关键词（与 FINER is_syntax_error 一致）
# ---------------------------------------------------------------------------
_INFRASTRUCTURE_ERROR_INDICATORS = [
    "timeout",
    "network",
    "http",
    "request",
    "api",
    "server",
    "service unavailable",
    "timed out",
    "connection refused",
    "connection timeout",
    "database connection",
    "connection pool",
]

# 本地 DatabaseExecutor 的结构化 error_type 中属于基础设施问题的类型
_INFRA_ERROR_TYPES = {"db_not_found", "connection_error", "query_interrupted"}


# ===================================================================
# 结果归一化与语法错误判定（移植 majority_voting.py）
# ===================================================================

def _rows_to_value_tuples_agnostic(
    rows: Optional[List[Any]],
) -> Optional[List[Tuple[str, ...]]]:
    """每行 -> 行内值转 str 并排序的 tuple（忽略列名列序）。与 FINER 语义一致。"""
    if rows is None or not isinstance(rows, list):
        return None
    normalized_rows: List[Tuple[str, ...]] = []
    for row in rows:
        if isinstance(row, dict):
            values = ["" if v is None else v for v in row.values()]
        elif isinstance(row, (list, tuple)):
            values = ["" if v is None else v for v in row]
        else:
            values = [row]
        values_sorted = sorted(str(v) for v in values)
        normalized_rows.append(tuple(values_sorted))
    return normalized_rows


def normalize_execution_result(
    result: Dict[str, Any],
    gt_sql: Optional[str] = None,
) -> str:
    """
    把一次执行结果归一化为 header-agnostic 的分组签名。

    result 是 DatabaseExecutor.execute 的返回 dict：
        {"success": bool, "error": str|None, "full_rows": [[...], ...],
         "row_count": int, "full_rows_truncated": bool}

    gt_sql 为前向兼容保留（FINER 同签名）；FINER 实测在分组里用严格
    multiset 语义会掉 1 EX，故本实现默认保持排序值集合语义，不做严格化。
    """
    if not result.get("success", False):
        error = result.get("error") or "Unknown error"
        return f"ERROR: {error}"

    rows = result.get("full_rows") or []
    if not rows:
        # 成功但零行：生成空 SUCCESS_VALUES 签名。
        # vav 的 is_empty 会先过滤它，但 fallback（所有组被过滤）会选回它——
        # 空结果可能是正确答案（如无匹配行查询），必须保留为候选项。
        # （旧实现返回 SUCCESS_ROWS_COUNT:0 → vav 永远匹配不到 → NO_RESULTS
        #   丢掉了"条件无匹配"这类题的正确预测，官方评估 -8.7%）
        return "SUCCESS_VALUES:"

    vals = _rows_to_value_tuples_agnostic(rows)
    row_strings = ["|".join(map(str, row)) for row in vals]
    signature = ";".join(sorted(set(row_strings)))
    return f"SUCCESS_VALUES:{signature[:200]}"


def is_syntax_error(result: Dict[str, Any]) -> bool:
    """失败且错误信息不含基础设施关键词 → 视为 SQL 语法错误（不参与分组投票）。"""
    if result.get("success", False):
        return False
    error_type = str(result.get("error_type") or "").lower()
    if error_type in _INFRA_ERROR_TYPES:
        return False
    error = str(result.get("error") or "").lower()
    if any(indicator in error for indicator in _INFRASTRUCTURE_ERROR_INDICATORS):
        return False
    return True


def results_equal(
    true_rows: Optional[List[Any]],
    pred_rows: Optional[List[Any]],
) -> bool:
    """header-agnostic 行集合相等（自评 MV 正确性判定，与 FINER 一致）。"""
    true_vals = _rows_to_value_tuples_agnostic(true_rows)
    pred_vals = _rows_to_value_tuples_agnostic(pred_rows)
    if true_vals is None or pred_vals is None:
        return False
    return set(true_vals) == set(pred_vals)


# ===================================================================
# vav 分组选择（移植 majority_voting.choose_group_vav）
# ===================================================================

def _parse_vals(key: str) -> List[str]:
    """从 `SUCCESS_VALUES:<sig>` 提取值 token（按 `;` 分割）。"""
    pfx = "SUCCESS_VALUES:"
    if not key.startswith(pfx):
        return []
    s = key[len(pfx):]
    if s == "":
        return []
    return [t.strip() for t in s.split(";") if t.strip() != ""]


_NUM_RE = re.compile(r"^[\s\-+]?(\d+(\.\d+)?)$")


def _to_num(tok: Any) -> Optional[float]:
    """值 token 转数值（兼容尾部 % 与正负号）；非数值返回 None。"""
    t = tok[:-1].strip() if isinstance(tok, str) and tok.strip().endswith("%") else tok
    if not isinstance(t, str):
        t = str(t)
    m = _NUM_RE.match(t)
    return float(m.group(1)) if m else None


def _is_empty_group(vals: List[str]) -> bool:
    return sum(1 for v in vals if v != "") == 0


def _is_all_zero_group(vals: List[str]) -> bool:
    """全部值可解析为数值且 |x| < 1e-12 → 退化组。

    注：与 FINER 一致，只对单列签名（不含 `|` 的值 token）判定数值——
    多列 `0|0` 不会命中（该 token 解析为 None），属于已知局限，保持移植忠实。
    """
    nums = [v for v in vals if v != ""]
    nums = [_to_num(n) for n in nums]
    nums = [x for x in nums if x is not None]
    return len(nums) > 0 and all(abs(x) < 1e-12 for x in nums)


def choose_group_vav(groups: Dict[str, Dict[str, Any]]) -> str:
    """
    vav 分组选择（FINER 原文语义）：
      1) 仅考虑 `SUCCESS_VALUES:` 组；
      2) 硬跳过空组与全零组（不参与投票）；
      3) 取剩余组中 size 最大者，平局按 key 字符串最大者；
      4) 若全被过滤，fallback 到原始最大的 SUCCESS_VALUES 组；
      5) 再无 → `NO_RESULTS`。

    groups: {result_key: {"size": int, ...}}（由 run_vav_voting 构造）。
    """
    if not groups:
        return "NO_RESULTS"

    sv_items = [
        (k, meta) for k, meta in groups.items() if k.startswith("SUCCESS_VALUES:")
    ]

    filtered = []
    for k, meta in sv_items:
        vals = _parse_vals(k)
        if _is_empty_group(vals):
            continue
        if _is_all_zero_group(vals):
            continue
        filtered.append((k, meta))

    if filtered:
        return max(filtered, key=lambda km: (int(km[1].get("size", 0)), km[0]))[0]

    if sv_items:
        return max(sv_items, key=lambda km: (int(km[1].get("size", 0)), km[0]))[0]

    return "NO_RESULTS"


# ===================================================================
# 单样本投票流水线
# ===================================================================

def _group_is_correct(
    items: List[Dict[str, Any]],
    gt_outcome: Optional[Dict[str, Any]],
) -> bool:
    """组是否与 gold 结果行集相等（组内取第一条的执行结果判定）。"""
    if not items or not gt_outcome or not gt_outcome.get("success"):
        return False
    first = items[0]["result"]
    if not first.get("success"):
        return False
    return results_equal(
        gt_outcome.get("full_rows") or [],
        first.get("full_rows") or [],
    )


def run_vav_voting(
    pred_results: List[Dict[str, Any]],
    gt_outcome: Optional[Dict[str, Any]],
    gt_sql: str = "",
    strategy: str = "vav",
) -> Dict[str, Any]:
    """
    对一个样本执行完整的「执行验证 → 分组 → 投票选择」流水线（纯函数，不执行 SQL）。

    pred_results: [{"sql": str, "result": <DatabaseExecutor outcome>, "index": int}, ...]
    gt_outcome:   gold SQL 的 DatabaseExecutor outcome（或 None）
    gt_sql:       gold SQL（前向兼容 + 记录用）
    strategy:     "vav"（默认，跳过退化组）或 "majority"（不过滤，纯 size 投票）

    返回（节选）：
      chosen_result / chosen_group_size / selected_sql / selected_sql_index
      majority_result / majority_group_size / majority_selected_sql
      majority_is_correct / is_sample_correct / degenerate_skip_applied
      num_syntax_errors / num_infrastructure_failures / num_valid_sqls_after_filtering
      result_groups: {key: {"size", "is_majority", "is_correct", "sqls"}}
    """
    # 1) 过滤语法错误（不参与投票）；基础设施失败单独计数
    syntax_error_count = 0
    infrastructure_failures: List[Dict[str, Any]] = []
    valid_pred_results: List[Dict[str, Any]] = []
    for pred in pred_results:
        if is_syntax_error(pred["result"]):
            syntax_error_count += 1
        else:
            valid_pred_results.append(pred)
            if not pred["result"].get("success", False):
                infrastructure_failures.append(pred)

    # 2) 只对执行成功的候选按归一化结果签名分组
    result_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pred in valid_pred_results:
        if pred["result"].get("success", False):
            normalized_result = normalize_execution_result(
                pred["result"], gt_sql=gt_sql
            )
            result_groups[normalized_result].append(pred)

    sorted_groups = sorted(
        result_groups.items(), key=lambda kv: len(kv[1]), reverse=True
    )

    if not sorted_groups:
        majority_result = "NO_RESULTS"
        majority_group: List[Dict[str, Any]] = []
    else:
        majority_result, majority_group = sorted_groups[0]

    # 3) 组元数据（含正确性标注）
    groups_meta: Dict[str, Dict[str, Any]] = {}
    for res_key, items in sorted_groups:
        groups_meta[res_key] = {
            "size": len(items),
            "is_majority": res_key == majority_result,
            "is_correct": _group_is_correct(items, gt_outcome),
            "sqls": [{"index": item["index"], "sql": item["sql"]} for item in items],
        }

    # 4) 选择策略
    strategy = (strategy or "vav").lower()
    if strategy == "majority":
        chosen_key = majority_result
    else:
        chosen_key = choose_group_vav(groups_meta)

    chosen_items = result_groups.get(chosen_key, [])
    if chosen_items:
        selected_sql = chosen_items[0]["sql"]
        selected_result = chosen_items[0]["result"]
        selected_index = chosen_items[0]["index"]
    else:
        selected_sql = ""
        selected_result = {"success": False, "error": "No valid SQLs"}
        selected_index = -1

    # 5) 自评正确性（header-agnostic 行集合相等，与 FINER 口径一致）
    is_correct = False
    if (
        selected_result.get("success", False)
        and gt_outcome is not None
        and gt_outcome.get("success", False)
    ):
        is_correct = results_equal(
            gt_outcome.get("full_rows") or [],
            selected_result.get("full_rows") or [],
        )

    majority_is_correct = False
    majority_selected_sql = ""
    majority_selected_index = -1
    if majority_result != "NO_RESULTS" and majority_result in groups_meta:
        majority_is_correct = bool(groups_meta[majority_result].get("is_correct"))
        _m_items = result_groups.get(majority_result, [])
        if _m_items:
            majority_selected_sql = _m_items[0]["sql"]
            majority_selected_index = _m_items[0]["index"]

    return {
        "num_predicted_sqls": len(pred_results),
        "num_syntax_errors": syntax_error_count,
        "num_infrastructure_failures": len(infrastructure_failures),
        "num_valid_sqls_after_filtering": sum(
            1 for p in valid_pred_results if p["result"].get("success", False)
        ),
        "result_groups": dict(groups_meta),
        "majority_result": majority_result,
        "majority_group_size": len(majority_group),
        "majority_selected_sql": majority_selected_sql,
        "majority_selected_sql_index": majority_selected_index,
        "majority_is_correct": majority_is_correct,
        "chosen_strategy": strategy,
        "chosen_result": chosen_key,
        "chosen_group_size": len(chosen_items),
        "degenerate_skip_applied": (
            chosen_key != majority_result and majority_result != "NO_RESULTS"
        ),
        "selected_sql": selected_sql,
        "selected_sql_index": selected_index,
        "selected_result": selected_result,  # 含 full_rows，仅内存使用，勿写入 JSON
        "is_sample_correct": is_correct,
    }


# ===================================================================
# 带缓存的执行器（复用 DatabaseExecutor，本地 SQLite）
# ===================================================================

class VavEvaluator:
    """
    vav 评估执行器：包装 DatabaseExecutor 并加 `(db_id, normalize_sql)` 内存缓存。

    30 条候选高度重复（同一模型同一 prompt 采样），缓存可省 60-70% 执行。
    """

    def __init__(self, executor: DatabaseExecutor) -> None:
        self.executor = executor
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def execute_cached(self, db_id: str, sql: str) -> Dict[str, Any]:
        """执行（带缓存）。空 SQL 直接返回合成失败结果，不触库。"""
        sql = (sql or "").strip()
        if not sql:
            return {
                "success": False,
                "error": "Empty SQL",
                "error_type": "empty_sql",
                "full_rows": [],
                "row_count": 0,
                "full_rows_truncated": False,
            }
        key = (db_id, normalize_sql(sql))
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        outcome = self.executor.execute(db_id, sql)
        self._cache[key] = outcome
        self.cache_misses += 1
        return outcome

    @property
    def cache_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        return round(self.cache_hits / total, 4) if total else 0.0


# ===================================================================
# 自测（无 GPU / 无模型）
# ===================================================================

def _run_self_tests() -> int:
    import json
    import sqlite3
    import tempfile

    failures: List[str] = []
    passed = 0

    def check(description: str, condition: bool) -> None:
        nonlocal passed
        if condition:
            passed += 1
        else:
            failures.append(description)
            print(f"  FAIL: {description}")

    print("=== finer_port.vav_voting self-tests ===\n")

    # --- normalize_execution_result ---
    r_fail = {"success": False, "error": 'near "select": syntax error', "error_type": "sqlite_error"}
    check("norm: failure -> ERROR: prefixed",
          normalize_execution_result(r_fail).startswith("ERROR: near"))
    r_empty = {"success": True, "full_rows": [], "row_count": 0, "full_rows_truncated": False}
    check("norm: empty success -> SUCCESS_VALUES: (空签名，vav fallback 可选中)",
          normalize_execution_result(r_empty) == "SUCCESS_VALUES:")
    r_rows = {"success": True, "full_rows": [[1, "a"], ["a", 1], [2, "b"]], "row_count": 3, "full_rows_truncated": False}
    # 行内值排序：[1,"a"] -> "1|a"，["a",1] -> "1|a"（去重），[2,"b"] -> "2|b"
    check("norm: row-internal sort + row dedup",
          normalize_execution_result(r_rows) == "SUCCESS_VALUES:1|a;2|b")
    r_long = {"success": True,
              "full_rows": [["v" + str(i) + "x" * 15] for i in range(30)],
              "row_count": 30, "full_rows_truncated": False}
    check("norm: signature truncated to 200 chars",
          len(normalize_execution_result(r_long)) <= len("SUCCESS_VALUES:") + 200)

    # --- is_syntax_error ---
    check("syntax: no such table is syntax error",
          is_syntax_error({"success": False, "error": "no such table: t", "error_type": "sqlite_error"}))
    check("syntax: timeout is NOT syntax error",
          not is_syntax_error({"success": False, "error": "statement timeout", "error_type": "sqlite_error"}))
    check("syntax: http is NOT syntax error",
          not is_syntax_error({"success": False, "error": "HTTP 500 from server", "error_type": "sqlite_error"}))
    check("syntax: db_not_found error_type is NOT syntax error",
          not is_syntax_error({"success": False, "error": "Database file not found: x", "error_type": "db_not_found"}))
    check("syntax: success is not syntax error", not is_syntax_error({"success": True}))

    # --- choose_group_vav ---
    groups1 = {
        "SUCCESS_VALUES:a": {"size": 3, "is_correct": True},
        "SUCCESS_VALUES:0": {"size": 5, "is_correct": False},  # 全零 → 硬跳过
        "ERROR: x": {"size": 9},  # 非 SUCCESS_VALUES → 不参与
    }
    check("vav: all-zero group skipped",
          choose_group_vav(groups1) == "SUCCESS_VALUES:a")
    groups2 = {
        "SUCCESS_VALUES:": {"size": 4},      # 空签名 → 硬跳过
        "SUCCESS_VALUES:abc": {"size": 2},
    }
    check("vav: empty-signature group skipped",
          choose_group_vav(groups2) == "SUCCESS_VALUES:abc")
    groups3 = {"SUCCESS_VALUES:aaa": {"size": 2}, "SUCCESS_VALUES:bbb": {"size": 2}}
    check("vav: size tie -> larger key wins",
          choose_group_vav(groups3) == "SUCCESS_VALUES:bbb")
    groups4 = {"SUCCESS_VALUES:0": {"size": 5}}
    check("vav: fallback to largest SUCCESS_VALUES when all filtered",
          choose_group_vav(groups4) == "SUCCESS_VALUES:0")
    check("vav: no SUCCESS_VALUES -> NO_RESULTS",
          choose_group_vav({"ERROR: x": {"size": 1}}) == "NO_RESULTS")
    check("vav: empty groups -> NO_RESULTS", choose_group_vav({}) == "NO_RESULTS")

    # --- run_vav_voting 端到端 ---
    ok1 = {"success": True, "full_rows": [["1"]], "row_count": 1, "full_rows_truncated": False}
    zero = {"success": True, "full_rows": [["0"]], "row_count": 1, "full_rows_truncated": False}
    err = {"success": False, "error": "no such column: x", "error_type": "sqlite_error"}
    gt = {"success": True, "full_rows": [["1"]], "row_count": 1, "full_rows_truncated": False}
    preds = [
        {"sql": "SELECT 1", "result": ok1, "index": 0},
        {"sql": "SELECT 1", "result": ok1, "index": 1},
        {"sql": "SELECT 0", "result": zero, "index": 2},
        {"sql": "SELECT 0", "result": zero, "index": 3},
        {"sql": "SELECT 0", "result": zero, "index": 4},
        {"sql": "SELECT bad", "result": err, "index": 5},
    ]
    vote = run_vav_voting(preds, gt, gt_sql="SELECT 1")
    check("vote: syntax error counted", vote["num_syntax_errors"] == 1)
    check("vote: majority (unfiltered) is zero group",
          vote["majority_result"] == "SUCCESS_VALUES:0")
    check("vote: majority self-match False",
          vote["majority_is_correct"] is False)
    check("vote: vav picks correct group (zero skipped)",
          vote["chosen_result"] == "SUCCESS_VALUES:1")
    check("vote: degenerate skip applied",
          vote["degenerate_skip_applied"] is True)
    check("vote: selected sql is first of group", vote["selected_sql"] == "SELECT 1")
    check("vote: self correct", vote["is_sample_correct"] is True)
    check("vote: valid count", vote["num_valid_sqls_after_filtering"] == 5)

    vote_none = run_vav_voting([{"sql": "SELECT bad", "result": err, "index": 0}], gt)
    check("vote: all syntax errors -> NO_RESULTS / empty selected",
          vote_none["chosen_result"] == "NO_RESULTS" and vote_none["selected_sql"] == "")

    # --- VavEvaluator 缓存（真实 SQLite） ---
    print("--- VavEvaluator cache (real SQLite) ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        db_dir = tmpdir / "database" / "test_db"
        db_dir.mkdir(parents=True)
        conn = sqlite3.connect(str(db_dir / "test_db.sqlite"))
        conn.execute("CREATE TABLE t (id INTEGER, val TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'hello'), (2, 'world')")
        conn.commit()
        conn.close()
        executor = DatabaseExecutor(str(tmpdir))
        evaluator = VavEvaluator(executor)
        # 注意：normalize_sql 只折叠空白 + 小写 SQL 关键字（不动标识符/字面量，
        # 保证缓存键保守、不会把不同语义的 SQL 误判为同键）。所以下面用
        # 空白/关键字大小写/尾分号差异的变体验证去重。
        r1 = evaluator.execute_cached("test_db", "SELECT * FROM t ORDER BY ID")
        r2 = evaluator.execute_cached("test_db", "  select   * from t order by ID ")
        r3 = evaluator.execute_cached("test_db", "select * from t order by ID;")
        check("cache: first exec ok", r1["success"] is True and r1["row_count"] == 2)
        check("cache: hit after normalize_sql dedup",
              evaluator.cache_hits == 2 and evaluator.cache_misses == 1)
        check("cache: empty sql never executes",
              evaluator.execute_cached("test_db", "  ").get("error") == "Empty SQL")
        check("cache: empty sql does not touch cache counts",
              evaluator.cache_hits == 2 and evaluator.cache_misses == 1)
        # 缓存结果与直接执行一致（JSON 序列化往返安全）
        json.dumps(r1)

    print()
    total = passed + len(failures)
    if failures:
        print(f"=== {passed}/{total} passed, {len(failures)} FAILED ===")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"=== All {passed} tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(_run_self_tests())
