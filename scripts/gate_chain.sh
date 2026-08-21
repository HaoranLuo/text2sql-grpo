#!/bin/bash
# RetrySQL 总链 v3：smoke(3090) → 基线门(gen+official) → 等 CPT 训练 → ckpt-100 门 → 终模门
# gpudebug QOS MaxSubmit=1：GPU 作业全部经本链串行提交，官方 EX 走 cpu6348。
# 用法: nohup bash scripts/gate_chain.sh <smoke_job> <train_job> &
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b || exit 1
LOG=logs/gate_chain.log
SMOKE=$1
TRAIN=$2
[ -z "$SMOKE" ] || [ -z "$TRAIN" ] && { echo "usage: $0 <smoke_job> <train_job>" >&2; exit 1; }
echo "$(date '+%F %T') gate chain v3 start (smoke=$SMOKE train=$TRAIN)" >> "$LOG"

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
        echo "$(date '+%F %T') job $jid ended with state=$end, abort" >> "$LOG"
        exit 1
    fi
    echo "$(date '+%F %T') job $jid COMPLETED" >> "$LOG"
}

run_gate() {
    local model=$1 out=$2
    local g o ex
    g=$(sbatch --export=MODEL="$model",OUT="$out" scripts/gate_single_shot.slurm | awk '{print $4}')
    echo "$(date '+%F %T') gate gen submitted: $g ($model)" >> "$LOG"
    wait_done "$g"
    o=$(sbatch --export=OUT="$out" scripts/gate_official.slurm | awk '{print $4}')
    echo "$(date '+%F %T') gate official submitted: $o" >> "$LOG"
    wait_done "$o"
    ex=$(grep -A1 'EXECUTION ACCURACY' "$out/official/official_result.txt" | tail -1 | awk '{print $NF}')
    echo "$(date '+%F %T') gate result: $model EX=$ex" >> "$LOG"
}

wait_done "$SMOKE"
run_gate checkpoints/sft_v3_merged outputs/gate_baseline_sftv3
wait_done "$TRAIN"
run_gate checkpoints/retry_cpt/checkpoint-100 outputs/gate_retry_ckpt100
run_gate checkpoints/retry_cpt outputs/gate_retry_final
echo "$(date '+%F %T') gate chain done" >> "$LOG"
