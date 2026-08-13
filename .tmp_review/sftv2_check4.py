# -*- coding: utf-8 -*-
# SFT v2 check part 4: question-cut detail, dev/test overlap (fixed), near-dupe scan, samples
import json, re, sys, collections
from difflib import SequenceMatcher
sys.path.insert(0, "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/src")
BASE = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b"

from transformers import AutoTokenizer
from prep_sft_data import load_distilled, load_gold_examples, build_messages

def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8-sig")]

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

loader = __import__("spider_utils", fromlist=["SpiderLoader"]).SpiderLoader(f"{BASE}/data/spider_data")
tok = AutoTokenizer.from_pretrained(f"{BASE}/models/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
train_items = load_json(f"{BASE}/data/spider_data/train_spider.json")
distill_raw = load_jsonl(f"{BASE}/data/reasoning_data/distill_v2_all_qc.jsonl")
distilled = load_distilled(f"{BASE}/data/reasoning_data/distill_v2_all_qc.jsonl", loader)
gold_ids = load_json(f"{BASE}/data_hygiene/sft_gold_v2.json")
gold = load_gold_examples(f"{BASE}/data_hygiene/sft_gold_v2.json", train_items, loader)
print(f"distilled={len(distilled)} gold={len(gold)}")

print("=" * 72)
print("QUESTION-CUT DETAIL")
qs = [r["question"] for r in distilled] + [g["question"] for g in gold]
mismatch_idx = []
for i, (rec, q) in enumerate(zip(distilled + gold, qs)):
    msgs, was_trunc = build_messages(rec, tok)
    u = msgs["messages"][0]["content"]
    if q.strip() and q.strip() not in u:
        mismatch_idx.append((i, was_trunc))
print(f"question not fully contained: {len(mismatch_idx)}; of which truncated={sum(1 for _,t in mismatch_idx if t)}")
for i, t in mismatch_idx:
    if not t:
        msgs, _ = build_messages((distilled + gold)[i], tok)
        u = msgs["messages"][0]["content"]
        print(f"  NON-truncated mismatch at {i}: q={qs[i][:120]!r}")
        # find where the prompt question section is
        mq = re.search(r"Question:\n(.*?)\n\nOptional", u, re.DOTALL)
        print(f"    prompt Question section: {mq.group(1)[:120]!r}" if mq else "    Question section not found!")
# show one truncated example fully
print("-- example truncated entry (index of first truncated):")
first_trunc = next((i for i, t in mismatch_idx if t), None)
if first_trunc is not None:
    rec = (distilled + gold)[first_trunc]
    msgs, _ = build_messages(rec, tok)
    u = msgs["messages"][0]["content"]
    print("full question:", qs[first_trunc][:200])
    print("user tail:", u[-300:])

print("=" * 72)
print("S6. DEV / TEST OVERLAP (fixed)")
dev = load_json(f"{BASE}/data/spider_data/dev.json")
test = load_json(f"{BASE}/data/spider_data/test.json")
print(f"dev.json={len(dev)} test.json={len(test)}")
mix_keys = set()
for r in distill_raw:
    mix_keys.add((r["db_id"], norm(r["question"])))
for i in gold_ids:
    it = train_items[i]
    mix_keys.add((it["db_id"], norm(it["question"])))
print(f"distinct mix (db_id, norm-q) keys: {len(mix_keys)}")
dev_keys = set((d["db_id"], norm(d["question"])) for d in dev)
test_keys = set((t["db_id"], norm(t["question"])) for t in test)
print(f"mix ∩ dev.json  = {len(mix_keys & dev_keys)}")
print(f"mix ∩ test.json = {len(mix_keys & test_keys)}")

print("-- near-dupe scan dev vs mix (same db, len ratio 0.7-1.3, quick_ratio>0.65):")
mix_qlist = list(mix_keys)
pairs = []
for d in dev:
    db, q = d["db_id"], d["question"]
    nq = norm(q)
    L = len(nq)
    for mdb, mq in mix_qlist:
        if mdb != db or mq == nq:
            continue
        Lm = len(mq)
        if not (0.7 * L <= Lm <= 1.3 * L):
            continue
        r = SequenceMatcher(None, nq, mq).quick_ratio()
        if r > 0.65:
            pairs.append((r, db, nq[:100], mq[:100]))
pairs.sort(reverse=True)
print(f"candidate near-dupes: {len(pairs)}")
for p in pairs[:12]:
    print(f"  {p[0]:.2f} [{p[1]}] | dev: {p[2]} | mix: {p[3]}")

print("-- v1 source multi-sql-block rate for context:")
v1 = load_jsonl(f"{BASE}/data/reasoning_data/deepseek-chat_spider_train_think.jsonl")
SQL_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
multi = sum(1 for r in v1 if len(SQL_RE.findall(r.get("response") or "")) > 1)
print(f"v1 jsonl lines={len(v1)} multi-sql-block={multi}")

print("=" * 72)
print("S8. SAMPLE 5 ENTRIES (quality read)")
mix = load_json(f"{BASE}/data/sft_v2_mix.json")
idxs = [0, 1500, 3763, 3764, 4427]
for i in idxs:
    e = mix[i]
    print(f"===== mix[{i}] {'(distilled)' if i < 3764 else '(gold)'}")
    print("USER:")
    print(e["messages"][0]["content"][:900])
    print("... (tail):")
    print(e["messages"][0]["content"][-400:])
    print("ASSISTANT:")
    print(e["messages"][1]["content"])
    print()
