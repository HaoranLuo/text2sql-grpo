import json, os

base = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data"
dev = json.load(open(f"{base}/dev.json"))
dbs = sorted(set(d["db_id"] for d in dev))
missing = [db for db in dbs if not os.path.exists(f"{base}/database/{db}/{db}.sqlite")]
print(f"dev dbs: {len(dbs)}, missing original sqlite: {len(missing)}")
for db in missing:
    print(f"  MISSING: {db}")
    print(f"    dir contents: {os.listdir(f'{base}/database/{db}')[:6] if os.path.exists(f'{base}/database/{db}') else 'NO DIR'}")
