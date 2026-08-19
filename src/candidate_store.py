"""中央候选库（candidate store）：生成一次，实验 N 次。

SQLite 后端，按 (di, model, lora, temperature, seed, prompt_hash) 索引每个评估
生成的候选 SQL 与执行结果。任何新评估先 query()——命中则跳过生成直接复用；
生成完成后 ingest() 入库。投票/过滤/maj@K 类实验从此可以在已有候选上离线重算。

用法:
    from candidate_store import ingest, query, prompt_hash
    hits = query(di, model="qwen2.5-coder-3b", lora="sft_v2", temperature=0.0, seed=0)
    ingest(items, model="qwen2.5-coder-3b", lora="sft_v2", temperature=0.0, seed=0)

items 中每题的候选结构约定（新评估脚本按此输出）:
    {
      "di": int, "question": str, "db_id": str,
      "candidates": [
        {"sql": str, "prompt": str(可空), "exec_ok": bool(可空),
         "result_hash": str(可空), "vote": int(该候选在本题的票数, 可空)}
      ]
    }
"""
import hashlib
import json
import sqlite3
import datetime
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
STORE_PATH = _PROJECT / "candidate_store" / "candidates.db"


def prompt_hash(text: str) -> str:
    """prompt 文本的短哈希（复用判定用，不同 prompt 不得混用同一缓存）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _norm(sql: str) -> str:
    return " ".join(sql.strip().lower().split())


def _conn() -> sqlite3.Connection:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(STORE_PATH), timeout=60)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute(
        """CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY,
            di INTEGER NOT NULL,
            model TEXT NOT NULL,
            lora TEXT,
            temperature REAL,
            seed INTEGER,
            prompt_hash TEXT,
            question TEXT,
            db_id TEXT,
            sql_text TEXT,
            norm_sql TEXT,
            exec_ok INTEGER,
            result_hash TEXT,
            vote INTEGER,
            created_at TEXT
        )"""
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_candidates_lookup "
        "ON candidates(di, model, lora, temperature, seed, prompt_hash)"
    )
    return c


def query(di, model, lora=None, temperature=0.0, seed=0, prompt_hash=None):
    """返回该 (di, 模型配置) 下已缓存的候选列表（sql/exec_ok/result_hash/vote）。无则 []。"""
    c = _conn()
    rows = c.execute(
        "SELECT sql_text, exec_ok, result_hash, vote FROM candidates "
        "WHERE di=? AND model=? AND IFNULL(lora,'')=IFNULL(?,'') AND temperature=? "
        "AND seed=? AND IFNULL(prompt_hash,'')=IFNULL(?,'') "
        "ORDER BY id",
        (di, model, lora, temperature, seed, prompt_hash),
    ).fetchall()
    c.close()
    return [
        {"sql": r[0], "exec_ok": bool(r[1]) if r[1] is not None else None,
         "result_hash": r[2], "vote": r[3]}
        for r in rows
    ]


def ingest(items, model, lora=None, temperature=0.0, seed=0):
    """把 items.json 内容入库。按 (配置, di, norm_sql) 去重；返回 (新增, 跳过)。"""
    c = _conn()
    added = skipped = 0
    now = datetime.datetime.now().isoformat()
    for it in items:
        di = it.get("di")
        if di is None or "candidates" not in it:
            continue
        ph = None
        for cand in it["candidates"]:
            if not isinstance(cand, dict) or not cand.get("sql"):
                continue
            sql = cand["sql"].strip()
            nsql = _norm(sql)
            p = cand.get("prompt")
            if p:
                ph = prompt_hash(p)
            exists = c.execute(
                "SELECT 1 FROM candidates WHERE di=? AND model=? AND IFNULL(lora,'')=IFNULL(?,'') "
                "AND temperature=? AND seed=? AND IFNULL(prompt_hash,'')=IFNULL(?,'') AND norm_sql=?",
                (di, model, lora, temperature, seed, ph, nsql),
            ).fetchone()
            if exists:
                skipped += 1
                continue
            c.execute(
                "INSERT INTO candidates (di, model, lora, temperature, seed, prompt_hash, "
                "question, db_id, sql_text, norm_sql, exec_ok, result_hash, vote, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (di, model, lora, temperature, seed, ph, it.get("question"),
                 it.get("db_id"), sql, nsql,
                 1 if cand.get("exec_ok") else 0 if cand.get("exec_ok") is False else None,
                 cand.get("result_hash"), cand.get("vote"), now),
            )
            added += 1
    c.commit()
    total = c.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    c.close()
    return {"added": added, "skipped": skipped, "total": total}


def stats():
    """库概况：总候选数、覆盖题数、按模型分布。"""
    c = _conn()
    total = c.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    nq = c.execute("SELECT COUNT(DISTINCT di) FROM candidates").fetchone()[0]
    by_model = c.execute(
        "SELECT model, IFNULL(lora,''), temperature, COUNT(*) FROM candidates "
        "GROUP BY model, lora, temperature ORDER BY 4 DESC LIMIT 10"
    ).fetchall()
    c.close()
    return {"total": total, "distinct_questions": nq, "by_model": by_model}
