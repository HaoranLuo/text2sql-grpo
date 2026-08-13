# -*- coding: utf-8 -*-
# SFT v2 check part 3: full prep recompute vs on-disk mix + dev/test overlap + near-dupe scan
import json, re, sys, collections
from difflib import SequenceMatcher
sys.path.insert(0, "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/src")
BASE = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b"

from spider_utils import SpiderLoader
from reasoning_generator_agent import ReasoningGeneratorAgent
from prep_sft_data import (load_distilled, load_gold_examples, build_messages,
                           GOLD_SQL_TMPL, MAX_PROMPT_TOKENS)

def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

print("=" * 72)
print("S5. FULL PREP RECOMPUTE AND COMPARE WITH ON-DISK MIX")
mix = load_json(f"{BASE}/data/sft_v2_mix.json")
print(f"mix len: {len(mix)}")
loader = SpiderLoader(f"{BASE}/data/spider_data")
train_items = load_json(f"{BASE}/data/spider_data/train_spider.json")
distilled = load_distilled(f"{BASE}/data/reasoning_data/distill_v2_all_qc.jsonl", loader)
gold = load_gold_examples(f"{BASE}/data_hygiene/sft_gold_v2.json", train_items, loader)
print(f"recomputed: distilled={len(distilled)} gold={len(gold)}")
import traceback
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{BASE}/models/Qwen2.5-Coder-3B-Instruct", local_files_only=True)
    records = []
    truncated = 0
    for rec in distilled:
        msgs, was_trunc = build_messages(rec, tok)
        truncated += 1 if was_trunc else 0
        records.append(msgs)
    for rec in gold:
        records.append(build_messages(rec, tok)[0])
    print(f"recomputed records={len(records)} truncated_prompts={truncated}")
    disk = json.dumps(records, ensure_ascii=False, indent=2)
    with open(f"{BASE}/data/sft_v2_mix.json", encoding="utf-8") as f:
        on_disk = f.read()
    print("on-disk file byte-identical to recomputed pipeline output:", disk == on_disk)
    # even if whitespace differs, compare semantic equality
    disk_obj = records
    same = True
    for i, (a, b) in enumerate(zip(disk_obj, mix)):
        if a != b:
            same = False
            print(f"  first diff at index {i}:")
            print(f"    recomputed user: {a['messages'][0]['content'][:120]!r}")
            print(f"    on-disk user   : {b['messages'][0]['content'][:120]!r}")
            break
    print("semantic equality of all entries:", same)
    # question-cut check
    qcut = 0
    qs = [r["question"] for r in distilled] + [g["question"] for g in gold]
    for i, e in enumerate(records):
        u = e["messages"][0]["content"]
        if qs[i].strip() and qs[i].strip() not in u:
            qcut += 1
            if qcut <= 5:
                print(f"  question not fully in user prompt (index {i}): q={qs[i][:60]!r}")
    print(f"entries where question text is not fully contained in user prompt: {qcut}")
    # how many user prompts got truncated
    print(f"truncated prompts: {truncated}")
except Exception as ex:
    traceback.print_exc()

print("=" * 72)
print("S6. DEV / TEST OVERLAP")
dev = load_json(f"{BASE}/data/spider_data/dev.json")
test = load_json(f"{BASE}/data/spider_data/test.json")
print(f"dev.json={len(dev)} test.json={len(test)}")
distill = load_distilled(f"{BASE}/data/reasoning_data/distill_v2_all_qc.jsonl", loader)
gold_ids = load_json(f"{BASE}/data_hygiene/sft_gold_v2.json")
mix_qs = [(r["question"], "distill") for r in distill] + \
         [(train_items[i]["question"], "gold") for i in gold_ids]
print(f"total mix questions: {len(mix_qs)}")
dev_keys = set((d["db_id"], norm(d["question"])) for d in dev)
test_keys = set((t["db_id"], norm(t["question"])) for t in test)
# need db_id per mix question: distill has db_id; gold via train_items
mix_keys = set()
for r in distill:
    mix_keys.add((r["db_id"], norm(r["question"])))
for i in gold_ids:
    it = train_items[i]
    mix_keys.add((it["db_id"], norm(it["question"])))
print(f"distinct mix (db_id, question) keys: {len(mix_keys)}")
inter_dev = mix_keys & dev_keys
inter_test = mix_keys & test_keys
print(f"mix ∩ dev.json  = {len(inter_dev)}  {sorted(inter_dev)[:10]}")
print(f"mix ∩ test.json = {len(inter_test)} {sorted(inter_test)[:10]}")

# near-duplicate fuzzy scan (dev vs mix, quick_ratio)
print("-- near-dupe scan dev vs mix (len ratio 0.7-1.3, quick_ratio>0.65):")
dev_qs = [(d["db_id"], d["question"]) for d in dev]
mix_qlist = list(mix_keys)
pairs = []
for db, q in dev_qs:
    nq = norm(q)
    L = len(nq)
    for mdb, mq in mix_qlist:
        if mdb != db:
            continue
        Lm = len(mq)
        if not (0.7 * L <= Lm <= 1.3 * L):
            continue
        r = SequenceMatcher(None, nq, mq).quick_ratio()
        if r > 0.65:
            pairs.append((r, nq[:90], mq[:90]))
pairs.sort(reverse=True)
print(f"candidate near-dupes: {len(pairs)}")
for p in pairs[:12]:
    print(f"  {p[0]:.2f} | {p[1]} | {p[2]}")
