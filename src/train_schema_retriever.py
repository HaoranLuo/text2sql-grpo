"""Fine-tune Qwen3-Embedding-0.6B as a schema retriever (P1-2).

Bi-encoder (question vs single-table schema item) with LitE-SQL's HN-SupCon
hard-negative contrastive loss:

  - positive  = one table referenced by the gold SQL (per training sample)
  - negatives = other tables of the SAME database (hard negatives), sampled up
                to --n-neg per sample at collate time
  - loss      = HardNegativeSuperConLoss (LitE-SQL fine-tune.py, verbatim logic),
                temperature 0.07, only negatives within --hn-threshold of the
                positive similarity enter the logsumexp

Reference: tmp_idea_research/bird_gen_scan/code/LitE-SQL/schema_retriever
(LitE-SQL uses column-level items; we use table-level per the P1-2 plan).

Full-parameter fine-tune (0.6B fits comfortably on A40 48G with bf16 autocast +
gradient checkpointing), fp32 master weights.
"""

import argparse
import json
import os
import random
import sys
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever_common import (  # noqa: E402
    build_query_text,
    hard_negative_supcon_loss,
    last_token_pool,
)

HPC_BASE = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b"


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
class SchemaRetrieverDataset(Dataset):
    def __init__(self, samples, tokenizer, max_q_len=256, max_doc_len=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_doc_len = max_doc_len
        self._cache = [None] * len(samples)  # lazy tokenization cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cached = self._cache[idx]
        if cached is not None:
            return cached
        s = self.samples[idx]
        q = self.tokenizer(
            build_query_text(s["question"], s["evidence"]),
            truncation=True, max_length=self.max_q_len,
        )["input_ids"]
        p = self.tokenizer(
            s["positive"], truncation=True, max_length=self.max_doc_len
        )["input_ids"]
        negs = [
            self.tokenizer(n, truncation=True, max_length=self.max_doc_len)["input_ids"]
            for n in s["negatives"]
        ]
        item = {"query": q, "positive": p, "negatives": negs}
        self._cache[idx] = item
        return item


def collate_fn(batch, pad_id, n_neg_limit):
    def left_pad(seq_list):
        max_len = max(len(s) for s in seq_list)
        ids, masks = [], []
        for s in seq_list:
            pad = max_len - len(s)
            ids.append([pad_id] * pad + s)
            masks.append([0] * pad + [1] * len(s))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

    q_ids, q_mask = left_pad([b["query"] for b in batch])
    p_ids, p_mask = left_pad([b["positive"] for b in batch])

    neg_ids_list, counts = [], []
    for b in batch:
        negs = b["negatives"]
        if len(negs) > n_neg_limit:
            negs = random.sample(negs, n_neg_limit)
        counts.append(len(negs))
        neg_ids_list.extend(negs)
    n_ids, n_mask = left_pad(neg_ids_list)

    return {
        "query": {"input_ids": q_ids, "attention_mask": q_mask},
        "positive": {"input_ids": p_ids, "attention_mask": p_mask},
        "negative": {"input_ids": n_ids, "attention_mask": n_mask},
        "neg_counts": torch.tensor(counts, dtype=torch.long),
    }


def load_training_samples(data_paths):
    """jsonl -> expanded samples: one sample per (question, related table).
    negatives = every OTHER table of the same db (hard negatives)."""
    samples = []
    n_records = 0
    n_skipped_no_neg = 0
    for path in data_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                n_records += 1
                items = rec["schema_items"]
                pos = [it for it in items if it["is_related"]]
                neg_texts = [it["text"] for it in items if not it["is_related"]]
                if not neg_texts or not pos:
                    n_skipped_no_neg += 1
                    continue
                for p in pos:
                    samples.append(
                        {
                            "question": rec["question"],
                            "evidence": rec.get("evidence", ""),
                            "positive": p["text"],
                            "negatives": neg_texts,
                        }
                    )
    return samples, n_records, n_skipped_no_neg


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def encode(model, batch, device):
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:  # causal-LM outputs only carry hidden_states tuple
        hidden = out.hidden_states[-1]
    return last_token_pool(hidden, batch["attention_mask"].to(device))


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += (p.grad.detach().norm(2).item()) ** 2
    return total ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-paths", nargs="+", required=True)
    ap.add_argument("--model-name-or-path", default=f"{HPC_BASE}/models/Qwen3-Embedding-0.6B")
    ap.add_argument("--out-dir", default=f"{HPC_BASE}/checkpoints/schema_retriever")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-neg", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--hn-threshold", type=float, default=0.1)
    ap.add_argument("--too-hard", action="store_true", default=True)
    ap.add_argument("--no-too-hard", dest="too_hard", action="store_false")
    ap.add_argument("--max-q-len", type=int, default=256)
    ap.add_argument("--max-doc-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1996)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--steps-limit", type=int, default=0, help="smoke test: stop after N steps")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    print(f"[train] device={device} name={torch.cuda.get_device_name(0)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Qwen3-Embedding-0.6B config declares Qwen3ForCausalLM; use the causal LM
    # wrapper (so checkpoints save/reload identically) but run the inner base
    # model for embeddings (no lm_head logits -> less memory).
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32)
    model.lm_head.requires_grad_(False)
    base = model.model
    base.gradient_checkpointing_enable()
    model.to(device)

    samples, n_records, n_skipped = load_training_samples(args.data_paths)
    print(f"[train] records={n_records} samples={len(samples)} "
          f"skipped(no neg/pos)={n_skipped}", flush=True)

    dataset = SchemaRetrieverDataset(samples, tokenizer, args.max_q_len, args.max_doc_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id, args.n_neg),
        num_workers=0,
    )

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "train_metrics.jsonl")
    metrics_f = open(metrics_path, "a", encoding="utf-8")

    print(f"[train] steps/epoch={len(loader)} epochs={args.epochs} "
          f"bs={args.batch_size} n_neg<={args.n_neg} lr={args.lr} "
          f"temp={args.temperature} hn_threshold={args.hn_threshold} "
          f"too_hard={args.too_hard}", flush=True)

    model.train()
    global_step = 0
    t0 = time.time()
    total_steps = args.epochs * len(loader)

    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", total=len(loader))
        for i, batch in enumerate(pbar):
            if args.steps_limit and global_step >= args.steps_limit:
                break
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                q_emb = encode(base, batch["query"], device)
                p_emb = encode(base, batch["positive"], device)
                n_emb = encode(base, batch["negative"], device)
                neg_embs = list(torch.split(n_emb, batch["neg_counts"].tolist(), dim=0))
                loss = hard_negative_supcon_loss(
                    q_emb, p_emb, neg_embs,
                    temperature=args.temperature,
                    hard_negative_threshold=args.hn_threshold,
                    too_hard_negative=args.too_hard,
                )
            loss.backward()
            gn = grad_norm(model)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if i % args.log_every == 0:
                msg = (f"[epoch {epoch} step {i}/{len(loader)}] loss={loss.item():.5f} "
                       f"grad_norm={gn:.3f}")
                print(msg, flush=True)
                metrics_f.write(json.dumps(
                    {"epoch": epoch, "step": i, "global_step": global_step,
                     "loss": float(loss.item()), "grad_norm": gn,
                     "elapsed_s": time.time() - t0}) + "\n")
                metrics_f.flush()
            if args.steps_limit and global_step >= args.steps_limit:
                break

        save_dir = os.path.join(args.out_dir, f"epoch-{epoch}")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"[train] saved {save_dir}", flush=True)
        if args.steps_limit and global_step >= args.steps_limit:
            break

    final_dir = os.path.join(args.out_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    metrics_f.close()
    print(f"[train] DONE final={final_dir} total_steps={global_step} "
          f"elapsed_h={ (time.time() - t0) / 3600:.2f}", flush=True)


if __name__ == "__main__":
    main()
