#!/bin/bash
# RetrySQL 后置链：训练结束 → ckpt-100 门 → 最终模型门 → 汇总对比（vs 基线 EX）
# 由主控 nohup 启动；基线 EX 由调用方传入（BASELINE_EX 数值，如 0.851）。
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b || exit 1
LOG=logs/retry_chain.log
TRAIN_JOB=$1
BASELINE_EX=$2
[ -z "$TRAIN_JOB" ] && { echo "usage: $0 <train_job_id> <baseline_ex>" >&2; exit 1; }
[ -z "$BASELINE_EX" ] && { echo "usage: $0 <train_job_id> <baseline_ex>" >&2; exit 1; }
echo "$(date '+%F %T') retry chain start (train=$TRAIN_JOB, baseline_ex=$BASELINE_EX)" >> "$LOG"

wait_done() {
    local jid=$1
    while true; do
        local st
        st=$(squeue -j "$jid" -h -o '%T' 2>/dev/null)
        [ -z "$st" ] && break
        sleep 120
    done
    local end
    end=$(sacct -j "$jid" -n -o State -X 2>/dev/null | tr -d ' ' | head -1)
    if [ "$end" != "COMPLETED" ]; then
        echo "$(date '+%F %T') job $jid ended with state=$end, abort" >> "$LOG"
        exit 1
    fi
    echo "$(date '+%F %T') job $jid COMPLETED" >> "$LOG"
}

run_gate() {
    local model=$1 out=$2
    local g
    g=$(sbatch --export=MODEL="$model",OUT="$out" scripts/gate_single_shot.slurm | awk '{print $4}')
    echo "$(date '+%F %T') gate submitted: $g ($model)" >> "$LOG"
    wait_done "$g"
    local ex
    ex=$(grep -A1 'EXECUTION ACCURACY' "$out/official/official_result.txt" | tail -1 | awk '{print $NF}')
    echo "$(date '+%F %T') gate result: $model EX=$ex" >> "$LOG"
}

wait_done "$TRAIN_JOB"
run_gate checkpoints/retry_cpt/checkpoint-100 outputs/gate_retry_ckpt100
run_gate checkpoints/retry_cpt outputs/gate_retry_final
echo "$(date '+%F %T') retry chain done (baseline=$BASELINE_EX)" >> "$LOG"
