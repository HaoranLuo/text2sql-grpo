#!/bin/bash
# bird_raw 主配方续跑：SCORE（gpudebug）→ FINAL 官方 EX（cpu6348），避免依赖链的 awk 提取
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b || exit 1
LOG=logs/bird_raw_resume.log
OUT_DIR=$PWD/outputs/bird_select_ormbird_bird_raw
CKPT=$PWD/checkpoints/orm_bird_bird_raw
echo "$(date '+%F %T') bird_raw resume start" >> "$LOG"

wait_done() {
    local jid=$1
    while squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 60; done
}
job_state() {
    sacct -j "$1" -n -o State -X 2>/dev/null | tr -d ' ' | head -1
}

JID=$(sbatch --export=ALL,ORM_CKPT="$CKPT",OUT_DIR="$OUT_DIR" scripts/bird_orm_score_adapted.slurm | grep -oE '[0-9]+$')
echo "$(date '+%F %T') SCORE_RAW submit -> $JID" >> "$LOG"
[ -n "$JID" ] || { echo "sbatch parse failed, abort" >> "$LOG"; exit 1; }
wait_done "$JID"
ST=$(job_state "$JID")
echo "$(date '+%F %T') SCORE_RAW -> $ST" >> "$LOG"
[ "$ST" = "COMPLETED" ] || exit 1
[ -f "$OUT_DIR/work/orm_scores.json" ] || { echo "缺 orm_scores.json, abort" >> "$LOG"; exit 1; }

JID=$(sbatch --export=ALL,PHASE=final,OUT_DIR="$OUT_DIR" scripts/bird_select_cpu.slurm | grep -oE '[0-9]+$')
echo "$(date '+%F %T') FINAL_RAW submit -> $JID" >> "$LOG"
[ -n "$JID" ] || { echo "sbatch parse failed, abort" >> "$LOG"; exit 1; }
wait_done "$JID"
ST=$(job_state "$JID")
echo "$(date '+%F %T') FINAL_RAW -> $ST" >> "$LOG"
[ "$ST" = "COMPLETED" ] || exit 1

python3 - <<EOF >> "$LOG" 2>&1
import json
d = json.load(open("$OUT_DIR/summary.json"))
o = d["official_exec_accuracy"]
print(f"bird_raw: arm_vav={o['arm_vav']['total']} arm_orm_grouphead={o['arm_orm_grouphead']['total']}")
print("prereg primary(bird_raw):", "PASS(>56.26)" if o['arm_orm_grouphead']['total'] > 56.26 else "FAIL")
EOF
echo "$(date '+%F %T') bird_raw resume done" >> "$LOG"
