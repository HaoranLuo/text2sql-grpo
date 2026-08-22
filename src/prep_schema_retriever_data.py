"""Build schema-retriever training/eval data (P1-2).

Produces jsonl files, one line per question:
{
  "question": str, "evidence": str, "db_id": str, "sql": str,
  "schema_items": [{"table": str, "text": str, "is_related": bool}, ...],
  "related_tables": [str, ...]   # original casing
}
- positives = tables referenced by the gold SQL (sqlglot parse, regex fallback,
  dictionary-matched against the db's real table names -> alias/case safe)
- negatives = all other tables of the same db (hard negatives, LitE-SQL style)
- schema item granularity = single table (name + column list + types + comments)

Training splits: BIRD train (+ Spider train_spider/train_others).
Eval splits    : BIRD dev_20240627 (+ Spider dev) kept separate for hold-out eval.

Run on the HPC from $BASE (defaults match the HPC layout):
    envs/reasoning3b/bin/python src/prep_schema_retriever_data.py \
        --do-train --do-eval --bird-train-json ... (defaults fine)
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever_common import build_table_doc, extract_related_tables  # noqa: E402

HPC_BASE = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b"


# --------------------------------------------------------------------------
# schema loading
# --------------------------------------------------------------------------
def load_tables_index(tables_json_path: str) -> Dict[str, dict]:
    with open(tables_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {d["db_id"]: d for d in raw}


def _read_csv(path: str) -> Optional[List[dict]]:
    for enc in ("utf-8-sig", "cp1252", "utf-8"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return None


def load_bird_descriptions(db_dir: str, db_ids: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """{db_id: {table_lower: {col_lower: column_description}}} from
    <db_dir>/<db_id>/database_description/*.csv"""
    index: Dict[str, Dict[str, Dict[str, str]]] = {}
    for db_id in db_ids:
        desc_dir = os.path.join(db_dir, db_id, "database_description")
        per_db: Dict[str, Dict[str, str]] = {}
        if os.path.isdir(desc_dir):
            for fn in sorted(os.listdir(desc_dir)):
                if not fn.lower().endswith(".csv"):
                    continue
                rows = _read_csv(os.path.join(desc_dir, fn))
                if not rows:
                    continue
                table_lower = os.path.splitext(fn)[0].lower().strip()
                cols = {}
                for row in rows:
                    col = (row.get("original_column_name") or "").strip().lower()
                    desc = (row.get("column_description") or "").strip()
                    desc = desc.replace("commonsense evidence:", "").strip()
                    if col:
                        cols[col] = desc
                if cols:
                    per_db[table_lower] = cols
        index[db_id] = per_db
    return index


def schema_to_docs(db_schema: dict, descriptions: Dict[str, Dict[str, str]]) -> List[dict]:
    """Build per-table schema items (original casing kept for display)."""
    tables = list(db_schema["table_names_original"])
    col_names = db_schema["column_names_original"]
    col_types = db_schema.get("column_types") or []
    # original column type strings
    type_of_col = {}
    for (t_idx, col), ctype in zip(col_names, col_types):
        if t_idx < 0:
            continue
        key = (tables[t_idx], col)
        type_of_col[key] = ctype

    docs = []
    for t_idx, table in enumerate(tables):
        cols = [c for (ti, c) in col_names if ti == t_idx]
        if not cols:
            continue
        desc = descriptions.get(table.lower(), {})
        col_types_d = {c: type_of_col.get((table, c), "") for c in cols}
        col_comments_d = {}
        for c in cols:
            d = desc.get(c.lower(), "")
            if d:
                col_comments_d[c] = d
        text = build_table_doc(table, cols, col_types_d, col_comments_d)
        docs.append({"table": table, "text": text})
    return docs


def build_dataset(
    qjson_paths: List[str],
    tables_index: Dict[str, dict],
    desc_index: Dict[str, Dict[str, Dict[str, str]]],
    use_evidence: bool,
) -> List[dict]:
    questions = []
    for path in qjson_paths:
        with open(path, "r", encoding="utf-8") as f:
            questions.extend(json.load(f))

    # cache schema docs per db
    db_docs_cache: Dict[str, List[dict]] = {}
    db_names_cache: Dict[str, List[str]] = {}
    records = []
    n_no_related = 0
    n_skip_missing_db = 0

    for q in questions:
        db_id = q.get("db_id") or q.get("database")
        sql = q.get("SQL") or q.get("query")
        question = q.get("question") or ""
        evidence = (q.get("evidence") or "").strip() if use_evidence else ""
        if not db_id or not question or not sql:
            continue
        schema = tables_index.get(db_id)
        if schema is None:
            n_skip_missing_db += 1
            continue
        if db_id not in db_docs_cache:
            docs = schema_to_docs(schema, desc_index.get(db_id, {}))
            db_docs_cache[db_id] = docs
            db_names_cache[db_id] = [t["table"].lower() for t in docs]
        docs = db_docs_cache[db_id]
        tables_lower = set(db_names_cache[db_id])

        related_lower = extract_related_tables(sql, tables_lower)
        if not related_lower:
            n_no_related += 1
            continue

        items = [dict(d, is_related=(d["table"].lower() in related_lower)) for d in docs]
        records.append(
            {
                "question": question,
                "evidence": evidence,
                "db_id": db_id,
                "sql": sql,
                "schema_items": items,
                "related_tables": [d["table"] for d in items if d["is_related"]],
            }
        )
    return records, n_no_related, n_skip_missing_db


def write_jsonl(records: List[dict], out_path: str, tag: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_pos = [len(r["related_tables"]) for r in records]
    n_tables = [len(r["schema_items"]) for r in records]
    print(
        f"[{tag}] {len(records)} records -> {out_path} | "
        f"avg related tables {sum(n_pos)/max(len(n_pos),1):.2f} | "
        f"avg db tables {sum(n_tables)/max(len(n_tables),1):.2f} | "
        f"pos tables / all tables: {sum(n_pos)} / {sum(n_tables)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--do-train", action="store_true")
    ap.add_argument("--do-eval", action="store_true")
    # BIRD train
    ap.add_argument("--bird-train-json", default=f"{HPC_BASE}/data/bird/bird_train/train/train.json")
    ap.add_argument("--bird-train-tables", default=f"{HPC_BASE}/data/bird/bird_train/train/train_tables.json")
    ap.add_argument("--bird-train-db-dir", default=f"{HPC_BASE}/data/bird/bird_train/train_databases")
    ap.add_argument("--bird-out", default=f"{HPC_BASE}/data/retriever_bird.jsonl")
    # Spider train
    ap.add_argument("--spider-json", default=f"{HPC_BASE}/data/spider_data/train_spider.json,{HPC_BASE}/data/spider_data/train_others.json")
    ap.add_argument("--spider-tables", default=f"{HPC_BASE}/data/spider_data/tables.json")
    ap.add_argument("--spider-out", default=f"{HPC_BASE}/data/retriever_spider.jsonl")
    # BIRD dev (eval)
    ap.add_argument("--eval-bird-json", default=f"{HPC_BASE}/data/bird/bird_dev/dev_20240627/dev.json")
    ap.add_argument("--eval-bird-tables", default=f"{HPC_BASE}/data/bird/bird_dev/dev_20240627/dev_tables.json")
    ap.add_argument("--eval-bird-db-dir", default=f"{HPC_BASE}/data/bird/bird_dev/dev_20240627/dev_databases")
    ap.add_argument("--eval-bird-out", default=f"{HPC_BASE}/data/retriever_eval_bird.jsonl")
    # Spider dev (eval)
    ap.add_argument("--eval-spider-json", default=f"{HPC_BASE}/data/spider_data/dev.json")
    ap.add_argument("--eval-spider-tables", default=f"{HPC_BASE}/data/spider_data/tables.json")
    ap.add_argument("--eval-spider-out", default=f"{HPC_BASE}/data/retriever_eval_spider.jsonl")
    args = ap.parse_args()

    if not args.do_train and not args.do_eval:
        ap.error("need --do-train and/or --do-eval")

    if args.do_train:
        # BIRD train (descriptions only for db ids present in train.json)
        bird_questions = json.load(open(args.bird_train_json, encoding="utf-8"))
        bird_db_ids = sorted({q["db_id"] for q in bird_questions})
        bird_tables = load_tables_index(args.bird_train_tables)
        bird_desc = load_bird_descriptions(args.bird_train_db_dir, bird_db_ids)
        records, n_norel, n_miss = build_dataset(
            [args.bird_train_json], bird_tables, bird_desc, use_evidence=True
        )
        print(f"[BIRD train] questions {len(bird_questions)} dbs {len(bird_db_ids)} | "
              f"no related tables skipped {n_norel} | db not found {n_miss}")
        write_jsonl(records, args.bird_out, "BIRD train")

        spider_tables = load_tables_index(args.spider_tables)
        records, n_norel, n_miss = build_dataset(
            args.spider_json.split(","), spider_tables, {}, use_evidence=False
        )
        print(f"[Spider train] no related tables skipped {n_norel} | db not found {n_miss}")
        write_jsonl(records, args.spider_out, "Spider train")

    if args.do_eval:
        bird_tables = load_tables_index(args.eval_bird_tables)
        bird_questions = json.load(open(args.eval_bird_json, encoding="utf-8"))
        bird_db_ids = sorted({q["db_id"] for q in bird_questions})
        bird_desc = load_bird_descriptions(args.eval_bird_db_dir, bird_db_ids)
        records, n_norel, n_miss = build_dataset(
            [args.eval_bird_json], bird_tables, bird_desc, use_evidence=True
        )
        print(f"[BIRD dev] questions {len(bird_questions)} | no related tables skipped "
              f"{n_norel} | db not found {n_miss}")
        write_jsonl(records, args.eval_bird_out, "BIRD dev eval")

        spider_tables = load_tables_index(args.eval_spider_tables)
        records, n_norel, n_miss = build_dataset(
            [args.eval_spider_json], spider_tables, {}, use_evidence=False
        )
        print(f"[Spider dev] no related tables skipped {n_norel} | db not found {n_miss}")
        write_jsonl(records, args.eval_spider_out, "Spider dev eval")


if __name__ == "__main__":
    main()
