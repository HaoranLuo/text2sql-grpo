#!/bin/bash
# BIRD 判卷老师便宜适配 全链路编排（草稿——按 docs/BIRD_JUDGE_ADAPTATION_PLAN.md 执行）
#   ① 打标（cpu6348）→ ② 均衡数据准备（登录节点 CPU 直接跑）→ ③ prep 分组（cpu6348）
#   → ④ ORM 训练（aiaca40, 1 epoch）→ ⑤ 新判卷打分（gpudebug）→ ⑥ final + 官方 EX（cpu6348）
# 预注册判定: ⑥ 的 arm_orm_grouphead 官方 EX total > 56.26 算「适配成功」；
#             ≥ 58.26 算「强」（详见方案 §2，含 McNemar 配对检验）。
# 用法: nohup bash scripts/bird_judge_adapt_chain.sh DATA=bird_bal2 &
#   DATA ∈ {bird_bal2 默认主配方 | bird_bal1 | bird_bal3 | bird_raw 对照}
# 注意: 本脚本为草稿，未做 gpudebug MaxSubmitJobs 冲突处理（可参照 bird_chain.sh
#       wait_job_done 模式补齐）；首次执行建议逐步手工 sbatch 而非一键链跑。

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd "$BASE" || exit 1

DATA=${1:-bird_bal2}
case "$DATA" in
  bird_bal1|bird_bal2|bird_bal3|bird_raw) ;;
  *) echo "FATAL: DATA=$DATA 非法" >&2; exit 1 ;;
esac

LOG=logs/bird_judge_adapt.log
OUT_DIR=$BASE/outputs/bird_select_ormbird_${DATA}
CKPT=$BASE/checkpoints/orm_bird_${DATA}

wait_job_done() {
    local jid=$1
    while squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 45; done
}
job_state() {
    sacct -j "$1" -n -o State -X 2>/dev/null | tr -d ' ' | head -1
}
submit_ok() {
    local jid=$1 name=$2
    wait_job_done "$jid"
    local st
    st=$(job_state "$jid")
    echo "$(date '+%F %T') $name ($jid) -> $st" >> "$LOG"
    [ "$st" = "COMPLETED" ] || { echo "$(date '+%F %T') $name FAILED, abort" >> "$LOG"; exit 1; }
}

echo "$(date '+%F %T') BIRD judge adaptation start [$DATA]" >> "$LOG"

# ---- ① 打标 ----
j1=$(sbatch scripts/label_orm_bird_cpu.slurm | awk '{print $4}')
echo "$(date '+%F %T') LABEL submit -> $j1" >> "$LOG"
submit_ok "$j1" "LABEL"
[ -f data/orm_train_bird_cap12.json ] || { echo "LABEL 完成但缺 cap12 产物, abort" >> "$LOG"; exit 1; }

# ---- ② 均衡数据准备（登录节点 CPU，几秒级）----
if [ "$DATA" != "bird_raw" ]; then
    envs/reasoning3b/bin/python src/prep_orm_balanced.py \
        --in data/orm_train_bird_cap12.json --out-dir data/bird_orm \
        --stats-out data/bird_orm/orm_bal_stats.json
    echo "$(date '+%F %T') PREP_BAL done" >> "$LOG"
fi

# ---- ③ prep 分组（cpu6348，~14 min）----
j3=$(sbatch --export=ALL,PHASE=prep,OUT_DIR="$OUT_DIR" scripts/bird_select_cpu.slurm | awk '{print $4}')
echo "$(date '+%F %T') PREP submit -> $j3" >> "$LOG"
submit_ok "$j3" "PREP"
[ -f "$OUT_DIR/work/orm_payloads.json" ] || { echo "PREP 完成但缺 payloads, abort" >> "$LOG"; exit 1; }

# ---- ④ ORM 训练（aiaca40，~3h）----
j4=$(sbatch --export=ALL,DATA="$DATA" scripts/train_orm_bird_bal.slurm | awk '{print $4}')
echo "$(date '+%F %T') TRAIN submit -> $j4" >> "$LOG"
submit_ok "$j4" "TRAIN"
[ -f "$CKPT/eval_metrics.json" ] || { echo "TRAIN 完成但缺 eval_metrics.json, abort" >> "$LOG"; exit 1; }

# dev 软门（预注册 §2.3：rank_acc≥0.70 且 rank_acc_maj_wrong≥0.35 才继续官方 EX）
python3 - <<EOF >> "$LOG" 2>&1
import json
m = json.load(open("$CKPT/eval_metrics.json"))
ra = m.get("rank_acc") or 0
mw = m.get("rank_acc_maj_wrong")
ok = ra >= 0.70 and (mw is None or mw >= 0.35)
print(f"dev soft-gate: rank_acc={ra} rank_acc_maj_wrong={mw} -> {'PASS' if ok else 'CHECK'}")
EOF

# ---- ⑤ 新判卷打分（gpudebug，~10 min）----
j5=$(sbatch --export=ALL,ORM_CKPT="$CKPT",OUT_DIR="$OUT_DIR" scripts/bird_orm_score_adapted.slurm | awk '{print $4}')
echo "$(date '+%F %T') SCORE submit -> $j5" >> "$LOG"
submit_ok "$j5" "SCORE"
[ -f "$OUT_DIR/work/orm_scores.json" ] || { echo "SCORE 完成但缺 orm_scores.json, abort" >> "$LOG"; exit 1; }

# ---- ⑥ final 裁决 + 官方 EX（cpu6348，~2 min）----
j6=$(sbatch --export=ALL,PHASE=final,OUT_DIR="$OUT_DIR" scripts/bird_select_cpu.slurm | awk '{print $4}')
echo "$(date '+%F %T') FINAL submit -> $j6" >> "$LOG"
submit_ok "$j6" "FINAL"

# ---- 收割 + 预注册判定 ----
echo "$(date '+%F %T') === BIRD 官方执行准确率 ===" >> "$LOG"
python3 - <<EOF >> "$LOG" 2>&1
import json
d = json.load(open("$OUT_DIR/summary.json"))
o = d["official_exec_accuracy"]
vav = o["arm_vav"]["total"]
orm = o["arm_orm_grouphead"]["total"]
print(f"arm_vav={vav}  arm_orm_grouphead({DATA})={orm}")
if orm > 56.26:
    tag = "STRONG (>=58.26)" if orm >= 58.26 else "SUCCESS (>56.26)"
else:
    tag = "NOT PASS" if orm <= 56.26 else ""
print(f"preregistration: {tag}")
EOF
echo "$(date '+%F %T') chain done" >> "$LOG"
