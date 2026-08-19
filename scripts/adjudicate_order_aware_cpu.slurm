#!/bin/bash
# T1-4 顺序敏感分组裁决（纯 CPU）：2 臂（基线 + order-aware）离线重算
# → 基线一致性核对 → 官方 EX 复评（关键统计 ③）
# 历史数据：adjudicate_pool 全量（同量级执行）约 14 分钟执行 + 3 分钟判定；
# 本作业 2 臂 + 2 次官方复评 + 一致性核对，超时上限 2 小时
#SBATCH --job-name=adj_ordaw
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/adj_ordaw_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/adj_ordaw_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE

echo "=== T1-4 顺序敏感分组裁决（全实例, 16 线程, 基线 + order-aware 两臂）==="
$BASE/envs/reasoning3b/bin/python $BASE/src/adjudicate_order_aware.py \
    --items $BASE/outputs/eval_pool_b1/items.json \
    --out-dir $BASE/outputs/adjudicate_order_aware \
    --spider-dir $BASE/data/spider_data \
    --threads 16

echo "=== 基线一致性核对：本 run 基线 vs adjudicate_b1 基线（应逐题一致）==="
$BASE/envs/reasoning3b/bin/python - <<'PY'
import json
base = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b"
old = json.load(open(f"{base}/outputs/adjudicate_b1/items_arm_vav_multi_all_both.json"))
new = json.load(open(f"{base}/outputs/adjudicate_order_aware/items_arm_vav_multi_all.json"))
diff = [(i, a["predicted_sql"], b["predicted_sql"]) for i, (a, b) in enumerate(zip(old, new))
        if a["predicted_sql"] != b["predicted_sql"] or a["is_correct"] != b["is_correct"]]
print(f"baseline consistency vs adjudicate_b1: {'OK' if not diff else 'MISMATCH'}"
      f" (rows={len(new)}, mismatches={len(diff)})")
if diff:
    for i, x, y in diff[:5]:
        print(f"  row {i}:\n    b1:   {x}\n    new:  {y}")
PY

echo "=== 官方复评（基线 + order-aware 两臂，eval_official.sh）==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/adjudicate_order_aware/items_arm_vav_multi_all.json \
    $BASE/outputs/official_adj_orderaware_arm_vav_multi_all
echo "--- OFFICIAL arm_vav_multi_all ---"
grep -A3 'EXECUTION ACCURACY' \
    $BASE/outputs/official_adj_orderaware_arm_vav_multi_all/official_result.txt | head -4

bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/adjudicate_order_aware/items_arm_orderaware.json \
    $BASE/outputs/official_adj_orderaware_arm_orderaware
echo "--- OFFICIAL arm_vav_multi_all_orderaware ---"
grep -A3 'EXECUTION ACCURACY' \
    $BASE/outputs/official_adj_orderaware_arm_orderaware/official_result.txt | head -4

echo "=== 官方 EX 对比写入 outputs/adjudicate_order_aware/official_ex_comparison.json ==="
$BASE/envs/reasoning3b/bin/python - <<'PY'
import json, re
base = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b"

def read_ex(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"^execution\s+([\d\.\s]+)$", txt, re.M)
    nums = [float(x) for x in m.group(1).split()]
    assert len(nums) == 5, f"unexpected execution line: {m.group(1)}"
    return {"easy": nums[0], "medium": nums[1], "hard": nums[2],
            "extra": nums[3], "all": nums[4]}

ex_base = read_ex(f"{base}/outputs/official_adj_orderaware_arm_vav_multi_all/official_result.txt")
ex_ordaw = read_ex(f"{base}/outputs/official_adj_orderaware_arm_orderaware/official_result.txt")
out = {
    "arm_vav_multi_all_official_ex": ex_base,
    "arm_vav_multi_all_orderaware_official_ex": ex_ordaw,
    "delta_ex_all": round(ex_ordaw["all"] - ex_base["all"], 4),
    "delta_ex_by_difficulty": {k: round(ex_ordaw[k] - ex_base[k], 4)
                               for k in ("easy", "medium", "hard", "extra")},
}
json.dump(out, open(f"{base}/outputs/adjudicate_order_aware/official_ex_comparison.json",
                    "w"), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2))
PY

echo "Done: $(date)"
