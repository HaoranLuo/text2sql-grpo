"""Evaluate the schema retriever: recall@k / precision@k on gold tables (P1-2).

For each question: encode the query and all single-table schema items of its
db, rank by cosine similarity, then compute recall@k and precision@k of the
gold (related) tables for k in {3,5,8}, plus MRR and mean sim of gold tables.

Usage (HPC, from $BASE):
    envs/reasoning3b/bin/python src/eval_schema_retriever.py \
        --model checkpoints/schema_retriever/final \
        --eval-data data/retriever_eval_bird.jsonl --n 100 --seed 42 \
        --out-dir outputs/schema_retriever_eval
"""

import argparse
import json
import os
import random
import sys

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever_common import build_query_text, last_token_pool  # noqa: E402

HPC_BASE = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b"
KS = [3, 5, 8]


def encode_batch(model, tokenizer, texts, device, batch_size=64):
    embeds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = tokenizer(chunk, padding=True, truncation=True, max_length=512,
                            return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**enc)
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                hidden = out.hidden_states[-1]
            emb = last_token_pool(hidden, enc["attention_mask"])
            emb = torch.nn.functional.normalize(emb.float(), p=2, dim=-1)
            embeds.append(emb)
    return torch.cat(embeds, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-data", nargs="+", required=True)
    ap.add_argument("--out-dir", default=f"{HPC_BASE}/outputs/schema_retriever_eval")
    ap.add_argument("--n", type=int, default=0, help="max questions to evaluate (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model = model.model  # inner base model: BaseModelOutputWithPast, no lm_head logits

    records = []
    for path in args.eval_data:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    print(f"[eval] loaded {len(records)} records from {args.eval_data}", flush=True)

    rng = random.Random(args.seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    if args.n and args.n < len(order):
        order = order[: args.n]

    # encode queries once
    sel = [records[i] for i in order]
    queries = [build_query_text(r["question"], r.get("evidence", "")) for r in sel]
    q_embs = encode_batch(model, tokenizer, queries, device, args.batch_size)

    # encode table docs per db (cache)
    db_emb_cache = {}
    results = []
    for idx, (q_emb, rec) in enumerate(tqdm(zip(q_embs, sel), total=len(sel))):
        db_id = rec["db_id"]
        if db_id not in db_emb_cache:
            docs = [it["text"] for it in rec["schema_items"]]
            db_emb_cache[db_id] = (
                encode_batch(model, tokenizer, docs, device, args.batch_size),
                docs,
            )
        t_embs, docs = db_emb_cache[db_id]
        sims = (q_emb @ t_embs.T).cpu()  # [n_tables]
        ranked = torch.argsort(sims, descending=True).tolist()
        gold = [i for i, it in enumerate(rec["schema_items"]) if it["is_related"]]

        per_q = {"db_id": db_id, "question": rec["question"],
                 "gold_tables": [rec["schema_items"][g]["table"] for g in gold],
                 "ranked_tables": [rec["schema_items"][i]["table"] for i in ranked],
                 "ranked_sims": [round(float(sims[i]), 4) for i in ranked],
                 "n_tables": len(docs)}
        for k in KS:
            topk = set(ranked[:k])
            hit = len(topk.intersection(gold))
            per_q[f"recall@{k}"] = hit / len(gold) if gold else 1.0
            per_q[f"p@{k}"] = hit / k
        # MRR over gold tables
        mrr = 0.0
        for g in gold:
            try:
                mrr += 1.0 / (ranked.index(g) + 1)
            except ValueError:
                pass
        per_q["mrr"] = mrr / len(gold) if gold else 1.0
        per_q["mean_gold_sim"] = round(float(sims[gold].mean()), 4) if gold else None
        results.append(per_q)

    summary = {"model": args.model, "n_questions": len(results), "seed": args.seed,
               "data": args.eval_data, "k": KS}
    for k in KS:
        summary[f"recall@{k}"] = round(sum(r[f"recall@{k}"] for r in results) / len(results), 4)
        summary[f"p@{k}"] = round(sum(r[f"p@{k}"] for r in results) / len(results), 4)
    summary["mrr"] = round(sum(r["mrr"] for r in results) / len(results), 4)

    # per-db breakdown
    per_db = {}
    for r in results:
        per_db.setdefault(r["db_id"], []).append(r)
    summary["per_db"] = {
        db: {
            "n": len(rs),
            "avg_gold_tables": round(sum(len(r["gold_tables"]) for r in rs) / len(rs), 2),
            **{f"recall@{k}": round(sum(r[f"recall@{k}"] for r in rs) / len(rs), 4) for k in KS},
            **{f"p@{k}": round(sum(r[f"p@{k}"] for r in rs) / len(rs), 4) for k in KS},
        }
        for db, rs in sorted(per_db.items())
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "eval_details.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("[eval] SUMMARY", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_db"},
                     indent=2, ensure_ascii=False), flush=True)
    print(f"[eval] details -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
