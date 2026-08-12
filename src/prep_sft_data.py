#!/usr/bin/env python3
"""Phase 1 SFT data prep: distilled (reasoning+SQL) + gold mix -> chat-format SFT JSON.

Output schema per item:
    {"messages": [{"role": "user", "content": <eval-consistent prompt>},
                  {"role": "assistant", "content": <completion>}]}

- user prompt = ReasoningGeneratorAgent.build_prompt(question, ddl) — the exact
  prompt the eval agent sends at inference (train/inference consistency,
  HANDOFF 教训 1/2).
- assistant completion = distilled response (<think>...</think> + ```sql```)
  or gold SQL wrapped in a ```sql block.
- distilled input: reasoning_data JSONL (success=True records) or a JSON list
  (sft_filtered.json style items carrying db_id/question/text).
- gold input: the SFT-gold split list produced by the data-hygiene step (T0.2);
  accepts a JSON list of train indices or a list of {"db_id","question"} keys,
  optionally wrapped in {"indices": [...]} / {"sft_gold_indices": [...]}.
- mix ratio: --gold-frac (default 0.15 per approved D1).

Usage:
    python src/prep_sft_data.py \
        --distilled data/reasoning_data/deepseek-chat_spider_train_think.jsonl \
        --gold-split data_hygiene/sft_gold_split.json \
        --spider-dir data/spider_data \
        --gold-frac 0.15 \
        --output data/sft_phase1_mix.json
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from spider_utils import SpiderLoader
from reasoning_generator_agent import ReasoningGeneratorAgent

try:
    from transformers import AutoTokenizer
    _HAS_TOKENIZER = True
except Exception:
    AutoTokenizer = None
    _HAS_TOKENIZER = False

GOLD_SQL_TMPL = "```sql\n{query}\n```"
_SQL_BLOCK_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
MAX_PROMPT_TOKENS = 1536  # 与推理端 prompt 截断窗口一致(监管审查 P2 对齐)


def load_train_items(spider_dir: str) -> List[Dict[str, Any]]:
    with open(Path(spider_dir) / "train_spider.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _ddl_for(item: Dict[str, Any], loader: SpiderLoader) -> Optional[str]:
    ddl = item.get("ddl")
    if ddl:
        return ddl
    db_id = item.get("db_id", "")
    try:
        return loader.get_ddl_with_source(db_id)[0]
    except Exception:
        return None


def _completion_for(item: Dict[str, Any]) -> Optional[str]:
    """Distilled completion text, tolerant to several stored formats."""
    for key in ("response", "api_response"):
        val = (item.get(key) or "").strip()
        if val:
            return val
    # sft_filtered.json style: completion embedded in the "text" field
    text = item.get("text", "")
    m = _SQL_BLOCK_RE.search(text)
    if m:
        return f"```sql\n{m.group(1).strip()}\n```"
    marker = text.rfind("Answer:")
    if marker >= 0:
        tail = text[marker + len("Answer:"):].strip()
        if tail:
            return tail
    return None


def load_distilled(path: str, loader: SpiderLoader) -> List[Dict[str, str]]:
    """Return [{'question','ddl','completion'}], keeping only usable records."""
    out: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        content = fh.read()
    if content.lstrip().startswith("["):
        items = json.loads(content)
        items = items if isinstance(items, list) else [items]
    else:  # JSONL
        items = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    skipped = 0
    for item in items:
        if not isinstance(item, dict):
            skipped += 1
            continue
        if item.get("success") is False:
            skipped += 1
            continue
        question = item.get("question")
        if not question:
            skipped += 1
            continue
        ddl = _ddl_for(item, loader)
        if not ddl:
            skipped += 1
            continue
        completion = _completion_for(item)
        if not completion:
            skipped += 1
            continue
        # P2 修复(监管审查): 剔除无 SQL 块(API 截断)与双 <think> 畸形样本
        if not _SQL_BLOCK_RE.search(completion):
            skipped += 1
            continue
        if completion.count("<think>") > 1:
            skipped += 1
            continue
        out.append({"question": question, "ddl": ddl, "completion": completion})
    if skipped:
        print(f"  distilled: {len(out)} kept, {skipped} skipped")
    return out


def load_gold_examples(split_path: str, train_items: List[Dict],
                       loader: SpiderLoader) -> List[Dict[str, str]]:
    """Load SFT-gold examples referenced by the T0.2 hygiene split file."""
    with open(split_path, "r", encoding="utf-8") as fh:
        split = json.load(fh)
    if isinstance(split, dict):
        ids = (split.get("indices") or split.get("sft_gold_indices")
               or split.get("sft_gold") or [])
    else:
        ids = split
    out: List[Dict[str, str]] = []
    if ids and isinstance(ids[0], int):
        for i in ids:
            if i >= len(train_items):
                continue
            item = train_items[i]
            ddl = _ddl_for(item, loader)
            if not ddl:
                continue
            out.append({
                "question": item["question"], "ddl": ddl,
                "completion": GOLD_SQL_TMPL.format(query=item["query"]),
            })
    else:  # list of {"db_id","question"} keys
        key_map = {(it.get("db_id"), it.get("question")): it for it in train_items}
        for rec in ids:
            it = key_map.get((rec.get("db_id"), rec.get("question")))
            if not it:
                continue
            ddl = _ddl_for(it, loader)
            if not ddl:
                continue
            out.append({
                "question": it["question"], "ddl": ddl,
                "completion": GOLD_SQL_TMPL.format(query=it["query"]),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distilled", required=True)
    ap.add_argument("--gold-split", required=True, help="T0.2 SFT-gold split file")
    ap.add_argument("--spider-dir", default=str(PROJECT / "data" / "spider_data"))
    ap.add_argument("--gold-frac", type=float, default=0.15,
                    help="gold fraction of the final mix (approved D1 = 0.15)")
    ap.add_argument("--max-distilled", type=int, default=0, help="0 = all")
    ap.add_argument("--tokenizer-path",
                    default="/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/"
                            "models/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--output", default=str(PROJECT / "data" / "sft_phase1_mix.json"))
    args = ap.parse_args()

    if not _HAS_TOKENIZER:
        print("ERROR: transformers not available for prompt truncation")
        return 1
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                              local_files_only=True)

    loader = SpiderLoader(args.spider_dir)
    train_items = load_train_items(args.spider_dir)
    distilled = load_distilled(args.distilled, loader)
    if args.max_distilled > 0:
        distilled = distilled[: args.max_distilled]
    gold = load_gold_examples(args.gold_split, train_items, loader)

    n_gold = max(1, round(len(distilled) * args.gold_frac / (1 - args.gold_frac)))
    n_gold = min(n_gold, len(gold))
    print(f"distilled={len(distilled)} gold_available={len(gold)} "
          f"gold_mixed={n_gold} ({args.gold_frac:.0%} of mix)")

    records = []
    truncated = 0
    for rec in distilled:
        msgs, was_trunc = build_messages(rec, tokenizer)
        truncated += 1 if was_trunc else 0
        records.append(msgs)
    for rec in gold[:n_gold]:
        records.append(build_messages(rec, tokenizer)[0])
    if truncated:
        print(f"  prompt 截断到 {MAX_PROMPT_TOKENS} token: {truncated} 条")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} examples -> {out_path}")
    return 0


def truncate_user_prompt(user: str, tokenizer: Any,
                         max_prompt_tokens: int = MAX_PROMPT_TOKENS) -> str:
    """把 user 内容截断到模板化后 <= max_prompt_tokens, 与推理端窗口一致。"""
    def templated_len(content: str) -> int:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, add_generation_prompt=True)
        return len(ids)

    if templated_len(user) <= max_prompt_tokens:
        return user
    lo, hi = 0, len(user)
    while lo < hi:
        mid = (lo + hi) // 2
        if templated_len(user[:mid]) <= max_prompt_tokens:
            lo = mid + 1
        else:
            hi = mid
    return user[: max(0, lo - 1)].rstrip()


def build_messages(rec: Dict[str, str], tokenizer: Any) -> tuple:
    """Return (messages_dict, was_truncated)."""
    user = ReasoningGeneratorAgent.build_prompt(
        question=rec["question"], ddl_schema=rec["ddl"])
    truncated_user = truncate_user_prompt(user, tokenizer)
    return ({"messages": [
        {"role": "user", "content": truncated_user},
        {"role": "assistant", "content": rec["completion"]},
    ]}, truncated_user != user)


if __name__ == "__main__":
    sys.exit(main())
