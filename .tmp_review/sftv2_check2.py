# -*- coding: utf-8 -*-
# SFT v2 check part 2: multi-sql-block investigation + grpo/extra overlap pair
import json, re, sys, collections
sys.path.insert(0, "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/src")
BASE = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b"

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8-sig")]

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

SQL_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
distill = load_jsonl(f"{BASE}/data/reasoning_data/distill_v2_all_qc.jsonl")

print("=" * 72)
print("MULTI SQL BLOCK ENTRIES (97) - first 3 examples")
n = 0
for r in distill:
    m = SQL_RE.findall(r.get("response") or "")
    if len(m) > 1:
        n += 1
        if n <= 3:
            print(f"--- idx={r.get('index')} db={r['db_id']} q={r['question'][:80]}")
            print("RESPONSE:")
            print(r["response"][:1200])
            print()
print("total multi-block:", n)

print("=" * 72)
print("GRPO ∩ SAMPLED_EXTRA = 1 pair")
grpo = load_jsonl(f"{BASE}/data_hygiene/grpo_5500_manifest.jsonl")
g1500 = load_jsonl(f"{BASE}/data_hygiene/sft_gold_1500_manifest.jsonl")
extra_rows = [r for r in g1500 if r.get("origin") == "sampled_extra"]
grpo_nm = {}
for r in grpo:
    grpo_nm.setdefault((r["db_id"], norm(r["question"])), []).append(r)
for r in extra_rows:
    key = (r["db_id"], norm(r["question"]))
    if key in grpo_nm:
        print("extra:", json.dumps(r, ensure_ascii=False))
        for g in grpo_nm[key]:
            print("grpo :", json.dumps(g, ensure_ascii=False))
print()

print("=" * 72)
print("DO THE 97 MULTI-BLOCK ENTRIES SURVIVE INTO MIX? + DISTILL QUESTION SET vs GRPO (norm exact, full)")
mix = json.load(open(f"{BASE}/data/sft_v2_mix.json", encoding="utf-8"))
print("mix len:", len(mix))
multi_in_mix = 0
for e in mix[:3764]:
    a = e["messages"][1]["content"]
    if len(SQL_RE.findall(a)) > 1:
        multi_in_mix += 1
print("multi-sql-block assistants in mix distilled part:", multi_in_mix)
# check think count in mix distilled assistants
think0, think1, thinkgt1 = 0, 0, 0
for e in mix[:3764]:
    a = e["messages"][1]["content"]
    t = a.count("<think>")
    if t == 0: think0 += 1
    elif t == 1: think1 += 1
    else: thinkgt1 += 1
print(f"distilled assistants: think0={think0} think1={think1} think>1={thinkgt1}")
# gold part
gold_think0 = sum(1 for e in mix[3764:] if "<think>" not in e["messages"][1]["content"])
print(f"gold part: entries without <think> = {gold_think0} / {len(mix)-3764}")
print("mix[3764:] assistant has think at all:", sum(1 for e in mix[3764:] if "<think>" in e["messages"][1]["content"]))

# full distill vs grpo normalized exact intersection
distill_keys = set((r["db_id"], norm(r["question"])) for r in distill)
grpo_keys = set((r["db_id"], norm(r["question"])) for r in grpo)
inter = distill_keys & grpo_keys
print(f"distill question keys vs grpo question keys (normalized): intersection = {len(inter)}")
for k in sorted(inter):
    print("  ", k)
