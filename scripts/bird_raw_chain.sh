#!/bin/bash
# bird_raw 主配方补跑（决策表 §5.1：实测正率 0.1875 < 0.20 → 主配方 = bird_raw 自动 pos_weight）
# 打标产物 data/orm_train_bird_cap12.json 已存在（跳过打标与均衡准备），
# 三步：训练（aiaca40）→ 新判卷打分（gpudebug）→ final 官方 EX（cpu6348）
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b || exit 1
LOG=logs/bird_raw_chain.log
OUT_DIR=$PWD/outputs/bird_select_ormbird_bird_raw
CKPT=$PWD/checkpoints/orm_bird_bird_raw
echo "$(date '+%F %T') bird_raw chain start（主配方，决策表 p<0.20 触发）" >> "$LOG"

wait_done() {
    local jid=$1
    while squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 60; done
}
job_state() {
    sacct -j "$1" -n -o State -X 2>/dev/null | tr -d ' ' | head -1
}
submit_ok() {
    local jid=$1 name=$2
    wait_done "$jid"
    local st
    st=$(job_state "$jid")
    echo "$(date '+%F %T') $name ($jid) -> $st" >> "$LOG"
    [ "$st" = "COMPLETED" ] || { echo "$(date '+%F %T') $name FAILED, abort" >> "$LOG"; exit 1; }
}

[ -f data/orm_train_bird_cap12.json ] || { echo "cap12 缺失, abort" >> "$LOG"; exit 1; }

j1=$(sbatch --export=ALL,DATA=bird_raw scripts/train_orm_bird_bal.slurm | awk '{print $4}')
echo "$(date '+%F %T') TRAIN_RAW submit -> $j1" >> "$LOG"
submit_ok "$j1" "TRAIN_RAW"
[ -f "$CKPT/eval_metrics.json" ] || { echo "TRAIN_RAW 缺 eval_metrics.json, abort" >> "$LOG"; exit 1; }

j2=$(sbatch --export=ALL,ORM_CKPT="$CKPT",OUT_DIR="$OUT_DIR" scripts/bird_orm_score_adapted.slurm | awk '{print $4}')
echo "$(date '+%F %T') SCORE_RAW submit -> $j2" >> "$LOG"
submit_ok "$j2" "SCORE_RAW"
[ -f "$OUT_DIR/work/orm_scores.json" ] || { echo "SCORE_RAW 缺 orm_scores.json, abort" >> "$LOG"; exit 1; }

j3=$(sbatch --export=ALL,PHASE=final,OUT_DIR="$OUT_DIR" scripts/bird_select_cpu.slurm | awk '{print $4}')
echo "$(date '+%F %T') FINAL_RAW submit -> $j3" >> "$LOG"
submit_ok "$j3" "FINAL_RAW"

python3 - <<EOF >> "$LOG" 2>&1
import json
d = json.load(open("$OUT_DIR/summary.json"))
o = d["official_exec_accuracy"]
vav = o["arm_vav"]["total"]
orm = o["arm_orm_grouphead"]["total"]
print(f"bird_raw: arm_vav={vav} arm_orm_grouphead={orm}")
print("prereg(decision-table primary):", "PASS(>56.26)" if orm > 56.26 else "FAIL")
EOF
echo "$(date '+%F %T') bird_raw chain done" >> "$LOG"
