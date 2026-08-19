#!/bin/bash
# T1-3 执行失败修复链 pilot（gpudebug，≤100 题，禁止全量）
#
# 输入（pilot 混编子集，另行生成）：
#   outputs/repair_pilot/pilot_mix.json = 8 个全量触发题（di 99,460,469-480）
#     + 92 个头部题 —— 触发富集窗口，保证修复轮被真正走通；人口触发率另由
#     --scan-file 对全量 1034 做纯 CPU 扫描（8/1034 = 0.77%）。
# 输出：outputs/repair_pilot/{items_repair.json, items_baseline.json, summary.json}
# 官方复评（事后，CPU 节点）：eval_official.sh 分别跑 baseline 与 repair 两个
# items 文件，对比 execution accuracy。
#
# 冒烟：
#   sbatch --export=ALL,LIMIT=10,ITEMS=$BASE/outputs/repair_pilot/smoke_mix.json,\
# OUT=$BASE/outputs/repair_pilot_smoke scripts/repair_chain_pilot.slurm
#
# 时长估算（2026-08-15 实测口径）：引擎加载 ~1-2 min；全量扫描 ~0.6 min；
# 100 题胜者执行 ~4 s；修复生成 = 触发数 × 贪心 n=1（每题 ~2-5 s）→ 单切片充裕。
#SBATCH --job-name=repair_pilot
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/repair_pilot_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/repair_pilot_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

LIMIT=${LIMIT:-100}
ITEMS=${ITEMS:-$BASE/outputs/repair_pilot/pilot_mix.json}
OUT=${OUT:-$BASE/outputs/repair_pilot}
START=${START:-0}
SCAN_FILE=${SCAN_FILE:-$BASE/outputs/adjudicate_b1/items_arm_vav_multi_all_both.json}

echo "=== repair_chain pilot (limit=$LIMIT start=$START out=$OUT items=$ITEMS) ==="
$PYTHON $BASE/src/repair_chain.py \
    --items "$ITEMS" \
    --out-dir "$OUT" \
    --limit "$LIMIT" \
    --start-index "$START" \
    --lora sft_v2 \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-new-tokens 2048 \
    --max-prompt-tokens 2048 \
    --max-repair-rounds 1 \
    --scan-file "$SCAN_FILE" \
    --threads 8 \
    --checkpoint-every 5

echo "=== repair_chain done, 官方复评另行提交 ==="
echo "  bash scripts/eval_official.sh $OUT/items_baseline.json $OUT/official_baseline"
echo "  bash scripts/eval_official.sh $OUT/items_repair.json $OUT/official_repair"
echo "Done: $(date)"
