import json, sys

items = json.load(open(sys.argv[1]))
empty = [it for it in items if not (it.get("predicted_sql") or "").strip()]
print(f"empty={len(empty)}")
for it in empty[:3]:
    print(json.dumps({k: v for k, v in it.items() if k not in ("raw_model_response",)}, ensure_ascii=False, indent=1)[:1500])
