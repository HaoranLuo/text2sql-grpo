#!/bin/bash
# BIRD 三层管线链（BIRD dev 官方 EX）：
#   ① 生成 4 pass（gpudebug 切片，重提直至 fully_done）
#   ② prep 分组（cpu6348）
#   ③ ORM 组代表打分（gpudebug）
#   ④ final 裁决 + FINER 官方评估器（cpu6348）
# gpudebug QOS MaxSubmitJobs=1 → GPU 作业严格串行（wait→submit）。
# 用法: nohup bash scripts/bird_chain.sh &
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b || exit 1
LOG=logs/bird_chain.log
echo "$(date '+%F %T') bird chain start" >> "$LOG"

wait_job_done() {
    local jid=$1
    while squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 45; done
}

job_state() {
    sacct -j "$1" -n -o State -X 2>/dev/null | tr -d ' ' | head -1
}

# ---- ① 生成：切片重提直至 fully_done ----
N_GEN=0
while true; do
    N_GEN=$((N_GEN+1))
    if [ $N_GEN -gt 40 ]; then
        echo "$(date '+%F %T') GEN: 超过 40 次提交仍未完成，中止（人工介入）" >> "$LOG"
        exit 1
    fi
    jid=$(sbatch scripts/gen_bird_pool.slurm | awk '{print $4}')
    echo "$(date '+%F %T') GEN submit #$N_GEN -> $jid" >> "$LOG"
    wait_job_done "$jid"
    st=$(job_state "$jid")
    echo "$(date '+%F %T') GEN #$N_GEN ($jid) -> $st" >> "$LOG"
    if [ "$st" != "COMPLETED" ]; then
        echo "$(date '+%F %T') GEN job FAILED ($st)，检查 logs/bird_pool_$jid.err 后人工修复重提" >> "$LOG"
        exit 1
    fi
    done_flag=$(python3 -c "
import json,os
p='outputs/eval_pool_bird/summary.json'
if not os.path.exists(p): print('none')
else:
    d=json.load(open(p))
    print('1' if d.get('fully_done') else '0')
" 2>/dev/null)
    if [ "$done_flag" = "1" ]; then
        echo "$(date '+%F %T') GEN fully done after $N_GEN slices" >> "$LOG"
        break
    fi
    if [ "$done_flag" = "none" ]; then
        echo "$(date '+%F %T') GEN job completed but no summary.json — treat as failure" >> "$LOG"
        exit 1
    fi
    echo "$(date '+%F %T') GEN not fully done (slice $N_GEN), resubmit" >> "$LOG"
done

# ---- ② prep 分组（cpu6348）----
p1=$(sbatch scripts/bird_select_cpu.slurm | awk '{print $4}')
echo "$(date '+%F %T') PREP submit -> $p1" >> "$LOG"
wait_job_done "$p1"
st=$(job_state "$p1")
echo "$(date '+%F %T') PREP ($p1) -> $st" >> "$LOG"
[ "$st" = "COMPLETED" ] || { echo "$(date '+%F %T') PREP FAILED, abort" >> "$LOG"; exit 1; }
[ -f outputs/bird_select/work/orm_payloads.json ] || {
    echo "$(date '+%F %T') PREP 完成但缺少 orm_payloads.json, abort" >> "$LOG"; exit 1; }

# ---- ③ ORM 打分（gpudebug）----
s1=$(sbatch scripts/bird_orm_score.slurm | awk '{print $4}')
echo "$(date '+%F %T') SCORE submit -> $s1" >> "$LOG"
wait_job_done "$s1"
st=$(job_state "$s1")
echo "$(date '+%F %T') SCORE ($s1) -> $st" >> "$LOG"
[ "$st" = "COMPLETED" ] || { echo "$(date '+%F %T') SCORE FAILED, abort" >> "$LOG"; exit 1; }
[ -f outputs/bird_select/work/orm_scores.json ] || {
    echo "$(date '+%F %T') SCORE 完成但缺少 orm_scores.json, abort" >> "$LOG"; exit 1; }

# ---- ④ final 裁决 + 官方 EX（cpu6348）----
f1=$(sbatch --export=ALL,PHASE=final scripts/bird_select_cpu.slurm | awk '{print $4}')
echo "$(date '+%F %T') FINAL submit -> $f1" >> "$LOG"
wait_job_done "$f1"
st=$(job_state "$f1")
echo "$(date '+%F %T') FINAL ($f1) -> $st" >> "$LOG"
[ "$st" = "COMPLETED" ] || { echo "$(date '+%F %T') FINAL FAILED, abort" >> "$LOG"; exit 1; }

# ---- 收割 ----
echo "$(date '+%F %T') === BIRD 官方执行准确率 ===" >> "$LOG"
python3 -c "
import json
d=json.load(open('outputs/bird_select/summary.json'))
for arm, r in d['official_exec_accuracy'].items():
    c=r.get('counts') or {}
    print(f\"{arm}: simple={r['simple']} moderate={r['moderate']} challenging={r['challenging']} total={r['total']} counts={c}\")
" >> "$LOG" 2>&1
echo "$(date '+%F %T') bird chain done" >> "$LOG"
