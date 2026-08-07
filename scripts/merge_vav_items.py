import json, sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
merged = a + b
merged.sort(key=lambda x: x.get("dataset_index", x.get("di", 0)))
print("merged:", len(merged))
json.dump(merged, open(sys.argv[3], "w"), ensure_ascii=False)
