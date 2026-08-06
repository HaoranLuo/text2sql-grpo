import json, sys
items = json.load(open(sys.argv[1]))
print("items:", len(items))
di = sorted(it["dataset_index"] for it in items)
print("di range:", di[0], "-", di[-1])
print("unique di:", len(set(di)))
