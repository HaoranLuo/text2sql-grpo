#!/bin/bash
# gate_chain v4：基线门已完成(B=0.635)；ckpt-100 一出现立即测（训练进行中）；训练结束即测终模；
# official 异步提交不阻塞（cpu6348 与 gpudebug 互不占坑），末尾统一收割。
# 用法: nohup bash scripts/gate_chain_v4.sh <train_job> &
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b || exit 1
LOG=logs/gate_chain.log
TRAIN=$1
[ -z "$TRAIN" ] && { echo "usage: $0 <train_job>" >&2; exit 1; }
echo "$(date '+%F %T') gate chain v4 start (train=$TRAIN, baseline EX=0.635)" >> "$LOG"

wait_job_done() {
    local jid=$1
    while squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 60; done
}

wait_file() {
    local f=$1
    while [ ! -f "$f" ]; do sleep 60; done
}

run_gate() {  # model out —— gen 完成后 official 异步提交，不等待
    local model=$1 out=$2 g o
    g=$(sbatch --export=MODEL="$model",OUT="$out" scripts/gate_single_shot.slurm | awk '{print $4}')
    echo "$(date '+%F %T') gate gen submitted: $g ($model)" >> "$LOG"
    wait_job_done "$g"
    o=$(sbatch --export=OUT="$out" scripts/gate_official.slurm | awk '{print $4}')
    echo "$(date '+%F %T') gate official async: $o ($out)" >> "$LOG"
}

# 1) ckpt-100 门：文件一出现就测（训练进行中）
wait_file checkpoints/retry_cpt/checkpoint-100/config.json
echo "$(date '+%F %T') checkpoint-100 出现，启动中途门" >> "$LOG"
run_gate checkpoints/retry_cpt/checkpoint-100 outputs/gate_retry_ckpt100

# 2) 终模门：训练结束即测
wait_job_done "$TRAIN"
ST=$(sacct -j "$TRAIN" -n -o State -X 2>/dev/null | tr -d ' ' | head -1)
echo "$(date '+%F %T') train $TRAIN -> $ST" >> "$LOG"
[ "$ST" = "COMPLETED" ] || { echo "$(date '+%F %T') train FAILED, abort（等监管 agent 修复）" >> "$LOG"; exit 1; }
run_gate checkpoints/retry_cpt outputs/gate_retry_final

# 3) 收割所有 gate_official 作业后汇总
sleep 30
while squeue -u jiahuiwang24 -h -o '%j' 2>/dev/null | grep -q gate_off; do sleep 60; done
echo "$(date '+%F %T') all officials done" >> "$LOG"
for d in outputs/gate_retry_ckpt100 outputs/gate_retry_final; do
    ex=$(grep -A1 'EXECUTION ACCURACY' "$d/official/official_result.txt" 2>/dev/null | tail -1 | awk '{print $NF}')
    echo "$(date '+%F %T') $d EX=$ex" >> "$LOG"
done
echo "$(date '+%F %T') gate chain v4 done" >> "$LOG"
