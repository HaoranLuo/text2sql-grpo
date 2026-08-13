# -*- coding: utf-8 -*-
# SFT v2 data/leakage review script part 2
import json, sys, re, collections

BASE = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b"

def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

mix = load_json(f"{BASE}/data/sft_v2_mix.json")
gold = load_json(f"{BASE}/data_hygiene/sft_gold_v2.json")
print("gold[0..10]:", gold[:10])
print("gold type of elements:", type(gold[0]).__name__, "min:", min(gold), "max:", max(gold), "unique:", len(set(gold)))
print()

# full messages structure
e = mix[0]
msgs = e["messages"]
print("num messages:", len(msgs))
for m in msgs:
    c = m.get("content", "")
    print(f"  role={m.get('role')!r} content_len={len(c)}")
    print("  content head:", c[:120].replace(chr(10), " | "))
    print("  content tail:", c[-120:].replace(chr(10), " | "))
print()

# distribution of roles across all entries
role_counter = collections.Counter()
for entry in mix:
    msgs = entry["messages"]
    roles = tuple(m["role"] for m in msgs)
    role_counter[roles] += 1
print("role sequence distribution:")
for k, v in role_counter.most_common():
    print("  ", k, v)
print()

# top-level keys distribution
key_counter = collections.Counter()
for entry in mix:
    key_counter[tuple(sorted(entry.keys()))] += 1
print("entry top-level keys distribution:")
for k, v in key_counter.most_common():
    print("  ", k, v)
print()

# messages content distribution
content_counter = collections.Counter()
for entry in mix:
    for m in entry["messages"]:
        content_counter[type(m["content"]).__name__] += 1
print("content type distribution:", dict(content_counter))
print()

# extra fields besides messages
extra = collections.Counter()
for entry in mix:
    for k in entry.keys():
        if k != "messages":
            extra[k] += 1
print("extra top-level fields:", dict(extra))
print()

# check how many entries have multiple messages and what the last one looks like
mlen = collections.Counter(len(e["messages"]) for e in mix)
print("messages count per entry:", dict(mlen))
print()

# look at the assistant message of a few entries
for i in [0, 1, 2]:
    msgs = mix[i]["messages"]
    last = msgs[-1]
    print(f"--- mix[{i}] last msg role={last['role']!r} content head: {last['content'][:300].replace(chr(10),' | ')}")
    print(f"    content tail: {last['content'][-200:].replace(chr(10),' | ')}")
print()

# build_prompt reference: find definition in src
import subprocess
out = subprocess.run(["grep", "-rn", "Task Overview", f"{BASE}/src"], capture_output=True, text=True)
print("grep 'Task Overview' in src:")
print(out.stdout[:1000])
out2 = subprocess.run(["grep", "-rn", "def build_prompt", f"{BASE}/src"], capture_output=True, text=True)
print("grep 'def build_prompt' in src:")
print(out2.stdout[:2000])
