import json, sys
from collections import Counter

items = json.load(open(sys.argv[1]))
total = len(items)
empty = [it for it in items if not (it.get("predicted_sql") or "").strip()]
print(f"total={total}, empty_pred={len(empty)} ({len(empty)/total:.1%})")

# 空预测的原因统计
reasons = Counter()
for it in empty:
    reasons[it.get("match_reason", it.get("error", "unknown"))] += 1
for r, n in reasons.most_common(10):
    print(f"  {r}: {n}")

# 非空预测的 votes 分布（选中的组大小）
if not empty:
    votes = Counter(it.get("votes", 0) for it in items)
    print("votes dist:", dict(sorted(votes.items())))
