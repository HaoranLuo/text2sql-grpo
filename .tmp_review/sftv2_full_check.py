# -*- coding: utf-8 -*-
# SFT v2 full data/leakage verification (read-only)
import json, re, sys, random, collections
sys.path.insert(0, "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/src")

BASE = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b"

def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8-sig")]

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

SQL_RE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)

print("=" * 72)
print("S1. INPUT COUNTS + QC FIELDS (distill_v2_all_qc.jsonl)")
qc1 = load_jsonl(f"{BASE}/data/reasoning_data/distill_v2_qc.jsonl")
qc2 = load_jsonl(f"{BASE}/data/reasoning_data/distill_v2b_qc.jsonl")
distill = load_jsonl(f"{BASE}/data/reasoning_data/distill_v2_all_qc.jsonl")
print(f"qc1={len(qc1)} qc2={len(qc2)} concat_expected={len(qc1)+len(qc2)} all_qc_actual={len(distill)}")
print(f"concat order match: {[r for r in distill] == qc1+qc2}")
print(f"key set: {sorted(distill[0].keys())}")
bad = {"no_extracted_sql": 0, "empty_extracted": 0, "extracted_mismatch": 0,
       "no_sql_block": 0, "multi_sql_block": 0, "think_gt1": 0, "think_0": 0,
       "success_not_true": 0, "reasoning_content_notnull": 0}
models = collections.Counter()
for r in distill:
    if "extracted_sql" not in r: bad["no_extracted_sql"] += 1; continue
    es = (r.get("extracted_sql") or "").strip()
    if not es: bad["empty_extracted"] += 1
    m = SQL_RE.findall(r.get("response") or "")
    if len(m) != 1: bad["no_sql_block" if len(m)==0 else "multi_sql_block"] += 1
    elif es != m[0].strip(): bad["extracted_mismatch"] += 1
    tc = (r.get("response") or "").count("<think>")
    if tc > 1: bad["think_gt1"] += 1
    elif tc == 0: bad["think_0"] += 1
    if r.get("success") is not True: bad["success_not_true"] += 1
    if r.get("reasoning_content"): bad["reasoning_content_notnull"] += 1
    models[r.get("model")] += 1
print("QC field problems:", dict(bad))
print("models:", dict(models))
print("all extracted_sql non-empty & match response block:", all(v == 0 for v in bad.values()))

print("=" * 72)
print("S2. DISTILL SOURCE MAPPING vs MANIFESTS (zero intersection with GRPO)")
grpo = load_jsonl(f"{BASE}/data_hygiene/grpo_5500_manifest.jsonl")
unassigned = load_jsonl(f"{BASE}/data_hygiene/unassigned_pool_1659_manifest.jsonl")
gold1500 = load_jsonl(f"{BASE}/data_hygiene/sft_gold_1500_manifest.jsonl")
print(f"manifest counts: grpo={len(grpo)} unassigned={len(unassigned)} gold1500={len(gold1500)}")

def manifest_sets(rows):
    exact, normed = set(), {}
    for r in rows:
        key = (r["db_id"], r["question"].strip())
        exact.add(key)
        normed[(r["db_id"], norm(r["question"]))] = r["question"]
    return exact, normed

grpo_ex, grpo_nm = manifest_sets(grpo)
una_ex, una_nm = manifest_sets(unassigned)
extra_rows = [r for r in gold1500 if r.get("origin") == "sampled_extra"]
ds_rows = [r for r in gold1500 if r.get("origin") != "sampled_extra"]
extra_ex, extra_nm = manifest_sets(extra_rows)
ds_ex, ds_nm = manifest_sets(ds_rows)
print(f"gold1500 composition: sampled_extra={len(extra_rows)} distilled_source={len(ds_rows)}")

# check grpo vs unassigned / extra sets are disjoint themselves
print("grpo ∩ unassigned (normalized):", len(set(grpo_nm) & set(una_nm)))
print("grpo ∩ extra (normalized):", len(set(grpo_nm) & set(extra_nm)))

def classify(db_id, q):
    qq = q.strip()
    if (db_id, qq) in grpo_ex: return "GRPO"
    if (db_id, qq) in una_ex: return "unassigned"
    if (db_id, qq) in extra_ex: return "sampled_extra"
    if (db_id, qq) in ds_ex: return "distilled_source"
    n = norm(q)
    if (db_id, n) in grpo_nm: return "GRPO~norm"
    if (db_id, n) in una_nm: return "unassigned~norm"
    if (db_id, n) in extra_nm: return "sampled_extra~norm"
    if (db_id, n) in ds_nm: return "distilled_source~norm"
    return "UNMATCHED"

src_counter = collections.Counter()
unmatched = []
for r in distill:
    c = classify(r["db_id"], r["question"])
    src_counter[c] += 1
    if c == "UNMATCHED":
        unmatched.append((r.get("index"), r["db_id"], r["question"][:80]))
print("distill source distribution:", dict(src_counter))
print("unmatched examples:", unmatched[:10], "total:", len(unmatched))

# independent random sample of 50 vs GRPO manifest
random.seed(20260814)
sample = random.sample(distill, 50)
hits = [r for r in sample if classify(r["db_id"], r["question"]).startswith("GRPO")]
print(f"random 50 sample: GRPO hits = {len(hits)}")
for r in sample:
    pass
# distinct questions
qkeys = set((r["db_id"], norm(r["question"])) for r in distill)
print(f"distinct (db_id, norm-question) in distill: {len(qkeys)} / {len(distill)}")
qcount = collections.Counter((r["db_id"], norm(r["question"])) for r in distill)
print("samples per question distribution:", dict(collections.Counter(qcount.values())))

print("=" * 72)
print("S3. GOLD v2 COMPOSITION")
gold_ids = load_json(f"{BASE}/data_hygiene/sft_gold_v2.json")
print(f"gold_v2 count={len(gold_ids)} unique={len(set(gold_ids))} min={min(gold_ids)} max={max(gold_ids)}")
gold1500_by_idx = {r["idx"]: r for r in gold1500}
comp = collections.Counter(gold1500_by_idx[i]["origin"] for i in gold_ids if i in gold1500_by_idx)
print("gold v2 origin composition:", dict(comp))
not_in_1500 = [i for i in gold_ids if i not in gold1500_by_idx]
print("gold ids not in gold1500 manifest:", not_in_1500[:10], "count:", len(not_in_1500))
train_spider = load_json(f"{BASE}/data/spider_data/train_spider.json")
print(f"train_spider.json len={len(train_spider)}; gold ids out of range: {sum(1 for i in gold_ids if i >= len(train_spider))}")
n_gold = round(len(distill) * 0.15 / (1 - 0.15))
print(f"n_gold formula: round({len(distill)}*0.15/0.85) = {n_gold}")
