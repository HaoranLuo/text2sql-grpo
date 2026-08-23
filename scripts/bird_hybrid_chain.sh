#!/bin/bash
# BIRD 混合证据池全链路（prep 分组 → 已适配判卷打分 → final 官方 EX）：
#   池 = outputs/eval_pool_bird_hybrid/items.json（bird_hybrid_evidence.py 产出：
#   官方 evidence 非空题取原池、缺失 148 题取证据池，零生成成本）
#   ② prep 分组（cpu6348）→ ③ ORM 组代表打分（gpudebug，
#     ORM_CKPT=checkpoints/orm_bird_bird_bal2 适配判卷，只打一次/预注册）→
#   ④ final 裁决 + FINER 官方评估器（cpu6348）→ outputs/bird_select_bird_hybrid/
# gpudebug QOS MaxSubmitJobs=1 → GPU 作业严格串行（wait→submit）。
# 用法: nohup bash scripts/bird_hybrid_chain.sh > logs/bird_hybrid_chain.out 2>&1 &
# 日志: logs/bird_hybrid_chain.log
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b || exit 1
LOG=logs/bird_hybrid_chain.log
POOL_DIR=$PWD/outputs/eval_pool_bird_hybrid
OUT_DIR=$PWD/outputs/bird_select_bird_hybrid
ORM_CKPT=$PWD/checkpoints/orm_bird_bird_bal2
echo "$(date '+%F %T') bird hybrid chain start" >> "$LOG"

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

[ -f "$POOL_DIR/items.json" ] || { echo "$(date '+%F %T') 缺少 $POOL_DIR/items.json, abort" >> "$LOG"; exit 1; }

# ---- ② prep 分组（cpu6348）----
p1=$(sbatch --export=ALL,PHASE=prep,ITEMS=$POOL_DIR/items.json,OUT_DIR=$OUT_DIR scripts/bird_select_cpu.slurm | awk '{print $4}')
echo "$(date '+%F %T') PREP submit -> $p1" >> "$LOG"
wait_job_done "$p1"
st=$(job_state "$p1")
echo "$(date '+%F %T') PREP ($p1) -> $st" >> "$LOG"
[ "$st" = "COMPLETED" ] || { echo "$(date '+%F %T') PREP FAILED, abort" >> "$LOG"; exit 1; }
[ -f "$OUT_DIR/work/orm_payloads.json" ] || {
    echo "$(date '+%F %T') PREP 完成但缺少 orm_payloads.json, abort" >> "$LOG"; exit 1; }

# ---- ③ ORM 打分（gpudebug，适配判卷 orm_bird_bird_bal2；预注册：只打一次）----
s1=$(sbatch --export=ALL,ORM_CKPT=$ORM_CKPT,OUT_DIR=$OUT_DIR scripts/bird_orm_score_adapted.slurm | awk '{print $4}')
echo "$(date '+%F %T') SCORE submit -> $s1" >> "$LOG"
wait_job_done "$s1"
st=$(job_state "$s1")
echo "$(date '+%F %T') SCORE ($s1) -> $st" >> "$LOG"
[ "$st" = "COMPLETED" ] || { echo "$(date '+%F %T') SCORE FAILED, abort" >> "$LOG"; exit 1; }
[ -f "$OUT_DIR/work/orm_scores.json" ] || {
    echo "$(date '+%F %T') SCORE 完成但缺少 orm_scores.json, abort" >> "$LOG"; exit 1; }

# ---- ④ final 裁决 + 官方 EX（cpu6348）----
f1=$(sbatch --export=ALL,PHASE=final,ITEMS=$POOL_DIR/items.json,OUT_DIR=$OUT_DIR scripts/bird_select_cpu.slurm | awk '{print $4}')
echo "$(date '+%F %T') FINAL submit -> $f1" >> "$LOG"
wait_job_done "$f1"
st=$(job_state "$f1")
echo "$(date '+%F %T') FINAL ($f1) -> $st" >> "$LOG"
[ "$st" = "COMPLETED" ] || { echo "$(date '+%F %T') FINAL FAILED, abort" >> "$LOG"; exit 1; }

# ---- 收割 ----
echo "$(date '+%F %T') === BIRD 官方执行准确率（混合证据池 + 适配判卷）===" >> "$LOG"
python3 -c "
import json
d=json.load(open('$OUT_DIR/summary.json'))
for arm, r in d['official_exec_accuracy'].items():
    c=r.get('counts') or {}
    print(f\"{arm}: simple={r['simple']} moderate={r['moderate']} challenging={r['challenging']} total={r['total']} counts={c}\")
" >> "$LOG" 2>&1
echo "$(date '+%F %T') bird hybrid chain done" >> "$LOG"
