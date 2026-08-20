#!/bin/bash
# HPC 端接力链（bal2 判卷）：multi 打分完成 → 提交 unseen 打分 → 完成 → 提交 CPU 重算+官方复评
# gpudebug QOS MaxSubmitJobs=1，需串行。由主控 nohup 启动。
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b || exit 1
LOG=logs/v3_chain_bal2.log
SM=$1
[ -z "$SM" ] && { echo "usage: $0 <multi_score_job_id>" >&2; exit 1; }
echo "$(date '+%F %T') chain start (multi=$SM)" >> "$LOG"

wait_done() {
    local jid=$1
    while true; do
        local st
        st=$(squeue -j "$jid" -h -o '%T' 2>/dev/null)
        [ -z "$st" ] && break
        sleep 60
    done
    local end
    end=$(sacct -j "$jid" -n -o State -X 2>/dev/null | tr -d ' ' | head -1)
    if [ "$end" != "COMPLETED" ]; then
        echo "$(date '+%F %T') job $jid ended with state=$end, abort chain" >> "$LOG"
        exit 1
    fi
    echo "$(date '+%F %T') job $jid COMPLETED" >> "$LOG"
}

wait_done "$SM"
SU=$(sbatch --export=POOLS=unseen scripts/orm_v3_score_bal2.slurm | awk '{print $4}')
echo "$(date '+%F %T') unseen score submitted: $SU" >> "$LOG"
wait_done "$SU"
C=$(sbatch scripts/orm_v3_cpu_bal2.slurm | awk '{print $4}')
echo "$(date '+%F %T') cpu recompute submitted: $C" >> "$LOG"
