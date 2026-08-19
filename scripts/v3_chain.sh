#!/bin/bash
# HPC 端接力链：multi 打分(2099823)结束 → 提交 unseen 打分 → 结束 → 提交 CPU 重算
# 原因：gpudebug QOS MaxSubmitJobs=1，一次只能有一个 3090 作业在队列。
# 由主控在登录节点 nohup 启动。
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b || exit 1
LOG=logs/v3_chain.log
echo "$(date '+%F %T') chain start (multi=2099823, payloads=2099807)" >> "$LOG"

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

wait_done 2099823
wait_done 2099807
SU=$(sbatch --export=POOLS=unseen scripts/orm_v3_score.slurm | awk '{print $4}')
echo "$(date '+%F %T') unseen score submitted: $SU" >> "$LOG"
wait_done "$SU"
C=$(sbatch scripts/orm_v3_cpu.slurm | awk '{print $4}')
echo "$(date '+%F %T') cpu recompute submitted: $C" >> "$LOG"
