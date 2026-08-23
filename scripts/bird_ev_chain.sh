#!/bin/bash
# BIRD evidence 增强口径全链路（自动证据生成池 → 已适配判卷 → 官方 EX）：
#   ① 生成 4 pass（gpudebug 切片，重提直至 fully_done；gen_bird_pool_ev.slurm
#      注入 --evidence-json，其余超参与原池完全一致）
#   ② prep 分组（cpu6348）
#   ③ ORM 组代表打分（gpudebug，ORM_CKPT=checkpoints/orm_bird_bird_bal2 适配判卷）
#   ④ final 裁决 + FINER 官方评估器（cpu6348）→ outputs/bird_select_bird_ev/
# gpudebug QOS MaxSubmitJobs=1 → GPU 作业严格串行（wait→submit）。
# 用法: nohup bash scripts/bird_ev_chain.sh > logs/bird_ev_chain.out 2>&1 &
# 日志: logs/bird_ev_chain.log
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b || exit 1
LOG=logs/bird_ev_chain.log
GEN_DIR=$PWD/outputs/eval_pool_bird_ev
OUT_DIR=$PWD/outputs/bird_select_bird_ev
ORM_CKPT=$PWD/checkpoints/orm_bird_bird_bal2
echo "$(date '+%F %T') bird ev chain start" >> "$LOG"

wait_job_done() {
    # 以 sacct 终态为准（比 squeue 更可靠：登录节点过载时 squeue 可能瞬时为空，
    # 导致误判 job 已结束）。RUNNING/PENDING/空记录都继续等；兜底 2h 超时防挂死。
    local jid=$1
    local waited=0
    local st=""
    while true; do
        st=$(sacct -j "$jid" -n -o State -X 2>/dev/null | tr -d ' ' | head -1)
        case "$st" in
            COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|BOOT_FAIL|NODE_FAIL|DEADLINE|SPECIAL_EXIT|PREEMPTED)
                break ;;
        esac
        sleep 30
        waited=$((waited+30))
        if [ "$waited" -ge 7200 ]; then
            echo "$(date '+%F %T') wait_job_done($jid) 兜底超时 2h，last_state=$st" >> "$LOG"
            break
        fi
    done
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
    jid=$(sbatch scripts/gen_bird_pool_ev.slurm | awk '{print $4}')
    echo "$(date '+%F %T') GEN submit #$N_GEN -> $jid" >> "$LOG"
    wait_job_done "$jid"
    st=$(job_state "$jid")
    echo "$(date '+%F %T') GEN #$N_GEN ($jid) -> $st" >> "$LOG"
    if [ "$st" != "COMPLETED" ]; then
        echo "$(date '+%F %T') GEN job FAILED ($st)，检查 logs/bird_pool_ev_$jid.err 后人工修复重提" >> "$LOG"
        exit 1
    fi
    done_flag=$(python3 -c "
import json,os
p='$GEN_DIR/summary.json'
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
[ -f "$GEN_DIR/items.json" ] || { echo "$(date '+%F %T') GEN 完成但缺 items.json, abort" >> "$LOG"; exit 1; }

# ---- ② prep 分组（cpu6348）----
p1=$(sbatch --export=ALL,PHASE=prep,ITEMS=$GEN_DIR/items.json,OUT_DIR=$OUT_DIR scripts/bird_select_cpu.slurm | awk '{print $4}')
echo "$(date '+%F %T') PREP submit -> $p1" >> "$LOG"
wait_job_done "$p1"
st=$(job_state "$p1")
echo "$(date '+%F %T') PREP ($p1) -> $st" >> "$LOG"
[ "$st" = "COMPLETED" ] || { echo "$(date '+%F %T') PREP FAILED, abort" >> "$LOG"; exit 1; }
[ -f "$OUT_DIR/work/orm_payloads.json" ] || {
    echo "$(date '+%F %T') PREP 完成但缺少 orm_payloads.json, abort" >> "$LOG"; exit 1; }

# ---- ③ ORM 打分（gpudebug，适配判卷 orm_bird_bird_bal2）----
s1=$(sbatch --export=ALL,ORM_CKPT=$ORM_CKPT,OUT_DIR=$OUT_DIR scripts/bird_orm_score_adapted.slurm | awk '{print $4}')
echo "$(date '+%F %T') SCORE submit -> $s1" >> "$LOG"
wait_job_done "$s1"
st=$(job_state "$s1")
echo "$(date '+%F %T') SCORE ($s1) -> $st" >> "$LOG"
[ "$st" = "COMPLETED" ] || { echo "$(date '+%F %T') SCORE FAILED, abort" >> "$LOG"; exit 1; }
[ -f "$OUT_DIR/work/orm_scores.json" ] || {
    echo "$(date '+%F %T') SCORE 完成但缺少 orm_scores.json, abort" >> "$LOG"; exit 1; }

# ---- ④ final 裁决 + 官方 EX（cpu6348）----
f1=$(sbatch --export=ALL,PHASE=final,ITEMS=$GEN_DIR/items.json,OUT_DIR=$OUT_DIR scripts/bird_select_cpu.slurm | awk '{print $4}')
echo "$(date '+%F %T') FINAL submit -> $f1" >> "$LOG"
wait_job_done "$f1"
st=$(job_state "$f1")
echo "$(date '+%F %T') FINAL ($f1) -> $st" >> "$LOG"
[ "$st" = "COMPLETED" ] || { echo "$(date '+%F %T') FINAL FAILED, abort" >> "$LOG"; exit 1; }

# ---- 收割 ----
echo "$(date '+%F %T') === BIRD 官方执行准确率（evidence 池 + 适配判卷）===" >> "$LOG"
python3 -c "
import json
d=json.load(open('$OUT_DIR/summary.json'))
for arm, r in d['official_exec_accuracy'].items():
    c=r.get('counts') or {}
    print(f\"{arm}: simple={r['simple']} moderate={r['moderate']} challenging={r['challenging']} total={r['total']} counts={c}\")
" >> "$LOG" 2>&1
echo "$(date '+%F %T') bird ev chain done" >> "$LOG"
