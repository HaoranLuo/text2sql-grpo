#!/usr/bin/env python3
"""收集 few-shot 示例: 从 train 侧产物提取 3 条"执行正确且含单 <think>+SQL 块"的轨迹。

背景: 视角多样性实验（实验 2）——用 SFT v2 教案里的成功推理轨迹做 few-shot
示例, 构造第 6 视角 prompt。示例必须来自 train 侧产物（严禁 dev 轨迹: dev 问题
会出现在被评估的 dev 子集里, 直接用等于泄漏答案）。

HPC 产物结构调查结论（2026-08-14, ssh hpc 只读核实）:
  - outputs/eval_maj_diag_100/items.json: dev 前 100 条 vav 诊断产物, candidates
    只有 200 字符 raw_response_preview（未加 --save-full-responses）→ 无法还原
    完整 <think>+SQL 轨迹, 且是 dev 侧 → 不可用。
  - outputs/eval_5p_sft_v2/items.json: dev 1034 条 5p 投票产物, 只存
    predicted_sql/truncated/votes, 无任何原始响应文本 → 不可用。
  - data/sft_v2_mix.json: train 侧 SFT v2 教案（3764 QC + 664 gold = 4428 条,
    messages 格式, assistant 含 <think>+```sql）; 实测与 dev.json 问题重叠 = 0;
    4401 条可匹配 train_spider.json+train_others.json 的 gold（123 个 db 全部存在）
    → 唯一可用来源, 且可在 HPC 上执行验证"执行正确"。

验证口径: 候选 SQL 与 gold SQL 分别在只读 sqlite 上执行（复用 spider_utils
DatabaseExecutor, torch-free）, 双成功且均未截断后按 compare_execution_results
行集合比较（与 eval_5p_t10 / eval_maj_diag 同口径）。

用法（CPU only, HPC 上跑, 无需 GPU）:
    python src/collect_fs_examples.py \
        --source data/sft_v2_mix.json \
        --output outputs/fewshot_examples.json \
        --min-examples 3
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

# torch-free 依赖（spider_utils 仅用 sqlite3/json/re）
from spider_utils import DatabaseExecutor, compare_execution_results  # noqa: E402

DEFAULT_SOURCE = str(PROJECT / "data" / "sft_v2_mix.json")
DEFAULT_OUTPUT = str(PROJECT / "outputs" / "fewshot_examples.json")
DEFAULT_SPIDER = str(PROJECT / "data" / "spider_data")
DEFAULT_TRAIN_JSONS = [
    str(PROJECT / "data" / "spider_data" / "train_spider.json"),
    str(PROJECT / "data" / "spider_data" / "train_others.json"),
]
DEFAULT_DEV_JSON = str(PROJECT / "data" / "spider_data" / "dev.json")

# 难度分桶（JOIN 数量）: 0 = 易, 1-2 = 中, >=3 = 难
_JOIN_RE = re.compile(r"\bjoin\b", re.IGNORECASE)
# 单 <think> 块 + 单 ```sql 块 的格式检查
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_SQL_FENCE_RE = re.compile(r"```sql", re.IGNORECASE)


# ===================================================================
# 纯函数（可离线单测）
# ===================================================================

def norm_question(q: str) -> str:
    """问题文本归一化（训练 gold / dev 泄漏守卫共用同一口径）。"""
    return " ".join((q or "").strip().lower().split())


def count_joins(sql: str) -> int:
    """JOIN 关键字数量（难度口径: 0=易, 1-2=中, >=3=难）。"""
    return len(_JOIN_RE.findall(sql or ""))


def difficulty_of(sql: str) -> str:
    j = count_joins(sql)
    if j == 0:
        return "easy"
    if j <= 2:
        return "medium"
    return "hard"


def has_single_think_sql(response: str) -> bool:
    """要求恰好一个 <think> 块和一个 ```sql 围栏（格式检查, 防示例拼接错位）。"""
    r = response or ""
    if not r.strip():
        return False
    return (len(_THINK_OPEN_RE.findall(r)) == 1
            and bool(re.search(r"</think>", r, re.IGNORECASE))
            and len(_SQL_FENCE_RE.findall(r)) == 1)


def extract_question_from_user(content: str) -> Optional[str]:
    """从 build_prompt 格式的 user content 中抽取 Question 行。"""
    m = re.search(r"Question:\s*\n(.*?)\n", content or "")
    return m.group(1).strip() if m else None


def extract_sql_from_response(response: str) -> Optional[str]:
    """取第一个 ```sql 围栏内的 SQL（与 ReasoningGeneratorAgent.extract_sql 同源格式）。"""
    m = re.search(r"```sql\s*(.*?)```", response or "", re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def parse_source(data: Any, source_path: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    把来源文件归一化为候选条目列表 [{idx, question, response, db_id, cand}].

    支持两种产物结构（读文件结构后实测确认）:
      1) sft 教案风格 [{messages: [{role, content}, ...]}]（data/sft_v2_mix.json）
      2) vav 风格 items.json [item{question, candidates[{raw_response, ...}]}]
         要求 candidate 存了完整 raw_response（maj_diag 只有 200 字符 preview →
         直接报错; 5p 投票产物只有 predicted_sql → 直接报错）。
    """
    entries: List[Dict[str, Any]] = []
    kind = "unknown"
    if isinstance(data, dict):
        # 允许 {"examples": [...]} / {"items": [...]} 包装
        if isinstance(data.get("examples"), list):
            data = data["examples"]
        elif isinstance(data.get("items"), list):
            data = data["items"]
        else:
            data = []
    if not isinstance(data, list):
        raise ValueError(f"{source_path}: 期望 list 或 dict 包装的产物, 实际 "
                         f"{type(data).__name__}")

    for i, e in enumerate(data):
        if not isinstance(e, dict):
            continue
        if "messages" in e:
            kind = kind if kind != "unknown" else "sft_messages"
            question, response = None, None
            for m in e["messages"]:
                if m.get("role") == "user" and question is None:
                    question = extract_question_from_user(m.get("content") or "")
                elif m.get("role") == "assistant":
                    response = m.get("content") or ""
            if question and response:
                entries.append({"idx": i, "question": question,
                                "response": response, "db_id": None, "cand": None})
        elif isinstance(e.get("candidates"), list) and "question" in e:
            kind = kind if kind != "unknown" else "vav_items"
            if not any("raw_response" in c for c in e["candidates"]):
                raise ValueError(
                    f"{source_path}: candidates 只有 raw_response_preview（200 字符）, "
                    "无法还原完整 <think>+SQL 轨迹 → 该产物不可用")
            for c in e["candidates"]:
                if not (c.get("raw_response") or "").strip():
                    continue
                # 预过滤明显错误候选, 执行验证阶段再复核
                if not c.get("execution_success") or not c.get("correct_vav"):
                    continue
                entries.append({"idx": i, "question": e["question"],
                                "response": c["raw_response"],
                                "db_id": e.get("db_id"), "cand": c})
        elif "predicted_sql" in e and "question" not in e:
            # eval_5prompt_agent / eval_5p_t10 投票产物: 只有投票后 SQL, 无原始响应
            raise ValueError(
                f"{source_path}: 投票产物只存 predicted_sql, 无原始响应文本 → 不可用")
    return entries, kind


def build_train_gold_index(train_jsons: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """问题归一化文本 → [(db_id, gold SQL), ...]。

    Spider 同一问题文本可能出现在多个数据库(train_spider/train_others 实测存在),
    因此一题多 gold 全部保留, 验证时逐一尝试(任一匹配即视为正确)。
    """
    index: Dict[str, List[Tuple[str, str]]] = {}
    for p in train_jsons:
        path = Path(p)
        if not path.exists():
            print(f"[collect][warn] train gold 文件不存在, 跳过: {p}")
            continue
        for t in json.loads(path.read_text(encoding="utf-8")):
            key = norm_question(t.get("question") or "")
            if not key:
                continue
            val = (t.get("db_id") or "", (t.get("query") or t.get("sql") or "").strip())
            index.setdefault(key, [])
            if val not in index[key]:
                index[key].append(val)
    return index


def load_dev_questions(dev_json: str) -> set:
    if not Path(dev_json).exists():
        print(f"[collect][warn] dev.json 不存在, 泄漏守卫失效风险: {dev_json}")
        return set()
    dev = json.loads(Path(dev_json).read_text(encoding="utf-8"))
    return {norm_question(d.get("question") or "") for d in dev}


def verify_correct(
    executor: DatabaseExecutor,
    db_id: str,
    sql: str,
    gold_sql: str,
) -> Tuple[bool, str]:
    """候选 SQL 与 gold SQL 双执行, 双成功且未截断后行集合比较（同 eval 口径）。"""
    cand_r = executor.execute(db_id, sql)
    if not cand_r.get("success"):
        return False, f"candidate exec failed: {str(cand_r.get('error'))[:80]}"
    if cand_r.get("full_rows_truncated"):
        return False, "candidate full_rows truncated"
    gold_r = executor.execute(db_id, gold_sql)
    if not gold_r.get("success"):
        return False, "gold exec failed (skip)"
    if gold_r.get("full_rows_truncated"):
        return False, "gold full_rows truncated"
    cmp = compare_execution_results(
        cand_r["full_rows"], gold_r["full_rows"], gold_sql=gold_sql)
    if not cmp.get("match"):
        return False, f"rows mismatch: {cmp.get('match_reason', '')[:80]}"
    return True, "ok"


def pick_examples(correct: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    易/中/难各一条; 桶内优先 (db 不重复, 响应最短, idx 最小)。

    桶为空时从其余正确候选中补齐（最近难度优先: 易→中→难 之外按
    与目标桶的 join 距离, 保持确定性）, 保证实验规格 3 条尽量成立。
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for e in correct:
        buckets[e["difficulty"]].append(e)
    used_dbs: List[str] = []
    selected: List[Dict[str, Any]] = []

    def pool_key(e: Dict[str, Any], prefer_dist: int) -> Tuple[Any, ...]:
        dist = abs(e["join_count"] - prefer_dist)
        return (e["db_id"] in used_dbs, dist, len(e["response"]), e["idx"])

    dist_of = {"easy": 0, "medium": 1, "hard": 2}
    for diff in ("easy", "medium", "hard"):
        pool = list(buckets[diff])
        if not pool:
            continue
        pool.sort(key=lambda e: pool_key(e, dist_of[diff]))
        e = pool[0]
        if e["db_id"]:
            used_dbs.append(e["db_id"])
        selected.append(e)
    # 桶缺失时补齐（去重后按最近难度/最短响应排序）
    while len(selected) < 3:
        rest = [e for e in correct if all(e is not s for s in selected)]
        if not rest:
            break
        rest.sort(key=lambda e: pool_key(e, 1))
        e = rest[0]
        if e["db_id"]:
            used_dbs.append(e["db_id"])
        selected.append(e)
    return selected


# ===================================================================
# 主流程
# ===================================================================

def run_collection(
    source: str,
    train_jsons: Optional[List[str]] = None,
    dev_json: str = DEFAULT_DEV_JSON,
    spider_dir: str = DEFAULT_SPIDER,
    min_examples: int = 3,
    max_response_chars: int = 1200,
) -> Dict[str, Any]:
    """执行收集, 返回结果 dict（examples / 统计）。不写文件（写文件由调用方决定）。"""
    train_jsons = train_jsons or DEFAULT_TRAIN_JSONS
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    entries, kind = parse_source(data, source)
    print(f"[collect] source={source} kind={kind} raw_entries={len(entries)}")

    train_gold = build_train_gold_index(train_jsons)
    dev_questions = load_dev_questions(dev_json)
    executor = DatabaseExecutor(spider_dir)

    dev_excluded = 0
    no_gold = 0
    fmt_fail = 0
    too_long = 0
    correct: List[Dict[str, Any]] = []

    for e in entries:
        # --- 泄漏守卫: 严禁 dev 轨迹 ---
        if norm_question(e["question"]) in dev_questions:
            dev_excluded += 1
            continue
        # --- 格式检查: 单 <think> + 单 ```sql ---
        if not has_single_think_sql(e["response"]):
            fmt_fail += 1
            continue
        # --- 上下文预算: 超长响应做示例会挤爆 1536 截断窗口 ---
        if len(e["response"]) > max_response_chars:
            too_long += 1
            continue
        # --- 对齐 train gold（拿 db_id + gold SQL 做执行验证; 一题多库逐一尝试）---
        key = norm_question(e["question"])
        golds = train_gold.get(key)
        if not golds:
            no_gold += 1
            continue
        e["sql"] = extract_sql_from_response(e["response"])
        if not e["sql"]:
            fmt_fail += 1
            continue
        matched = False
        for db_id, gold_sql in golds:
            ok, _reason = verify_correct(executor, db_id, e["sql"], gold_sql)
            if ok:
                e["db_id"] = db_id
                e["gold_sql"] = gold_sql
                matched = True
                break
        if not matched:
            continue
        e["join_count"] = count_joins(e["sql"])
        e["difficulty"] = difficulty_of(e["sql"])
        e["response_len"] = len(e["response"])
        e["verified"] = True
        correct.append(e)

    print(f"[collect] dev_excluded={dev_excluded} no_gold={no_gold} "
          f"fmt_fail={fmt_fail} too_long={too_long} exec_correct={len(correct)}")
    examples = pick_examples(correct)
    return {
        "source": source,
        "source_kind": kind,
        "n": len(examples),
        "min_examples": min_examples,
        "examples": examples,
        "stats": {
            "raw_entries": len(entries),
            "dev_overlap_excluded": dev_excluded,
            "no_train_gold": no_gold,
            "format_failed": fmt_fail,
            "too_long_excluded": too_long,
            "exec_correct": len(correct),
        },
    }


def write_output(result: Dict[str, Any], output: str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": result["source"],
        "source_kind": result["source_kind"],
        "n": result["n"],
        "stats": result["stats"],
        "examples": [
            {
                "question": e["question"],
                "db_id": e["db_id"],
                "difficulty": e["difficulty"],
                "join_count": e["join_count"],
                "response": e["response"],
                "sql": e["sql"],
                "gold_sql": e["gold_sql"],
                "response_len": e["response_len"],
                "source_entry_index": e["idx"],
                "verified": e.get("verified", True),
            }
            for e in result["examples"]
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[collect] wrote {result['n']} examples → {out}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="来源产物路径（sft 教案 messages 风格 或 vav items 风格）")
    parser.add_argument("--train-json", action="append", default=None,
                        help="train gold JSON（可多次指定; 默认 train_spider+train_others）")
    parser.add_argument("--dev-json", default=DEFAULT_DEV_JSON,
                        help="dev.json（泄漏守卫）")
    parser.add_argument("--spider-dir", default=DEFAULT_SPIDER,
                        help="spider_data 目录（database/ 子目录用于执行验证）")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="输出 fewshot_examples.json")
    parser.add_argument("--min-examples", type=int, default=3,
                        help="少于该数视为收集失败（exit 2）")
    parser.add_argument("--max-response-chars", type=int, default=1200,
                        help="示例响应长度硬上限（保护 1536 截断窗口）")
    args = parser.parse_args(argv)

    try:
        result = run_collection(
            source=args.source,
            train_jsons=args.train_json,
            dev_json=args.dev_json,
            spider_dir=args.spider_dir,
            min_examples=args.min_examples,
            max_response_chars=args.max_response_chars,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[collect] FAILED: {exc}", file=sys.stderr)
        return 1

    write_output(result, args.output)
    if result["n"] < args.min_examples:
        print(f"[collect] 只找到 {result['n']} 条可用轨迹 (< {args.min_examples}) → "
              f"few-shot 文件不完整, 由主控人工提供 --fewshot-file", file=sys.stderr)
        return 2
    for e in result["examples"]:
        print(f"[collect] 选中 [{e['difficulty']} join={e['join_count']}] "
              f"db={e['db_id']} len={e['response_len']} q={e['question'][:50]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
