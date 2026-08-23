#!/bin/bash
# 检索器生成链守望：等链完成 → 自动跑 prep/score/final → 报数字
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
LOG=$BASE/logs/watch_ret.log
cd "$BASE" || exit 1
echo "$(date '+%F %T') watch start" >> "$LOG"
while true; do
  if squeue -u jiahuiwang24 -h -o '%j' 2>/dev/null | grep -q bird_pool_ret; then sleep 300; continue; fi
  N=$(python3 -c "import json; d=json.load(open('outputs/eval_pool_bird_ret/items.json')); print(len(d))" 2>/dev/null)
  echo "$(date '+%F %T') gen done, items=$N" >> "$LOG"
  if [ "$N" = "1534" ]; then
    echo "$(date '+%F %T') items complete, starting select pipeline" >> "$LOG"
    J=$(sbatch --export=ALL,PHASE=prep,OUT_DIR="$BASE/outputs/bird_select_bird_ret" scripts/bird_select_cpu.slurm | grep -oE '[0-9]+$')
    while squeue -j "$J" -h 2>/dev/null | grep -q .; do sleep 60; done
    S=$(sacct -j "$J" -n -o State -X 2>/dev/null | tr -d ' ' | head -1)
    echo "$(date '+%F %T') prep $J -> $S" >> "$LOG"
    [ "$S" = "COMPLETED" ] || exit 1
    J=$(sbatch --export=ALL,ORM_CKPT="$BASE/checkpoints/orm_bird_bird_bal2",OUT_DIR="$BASE/outputs/bird_select_bird_ret" scripts/bird_orm_score_adapted.slurm | grep -oE '[0-9]+$')
    while squeue -j "$J" -h 2>/dev/null | grep -q .; do sleep 60; done
    S=$(sacct -j "$J" -n -o State -X 2>/dev/null | tr -d ' ' | head -1)
    echo "$(date '+%F %T') score $J -> $S" >> "$LOG"
    [ "$S" = "COMPLETED" ] || exit 1
    J=$(sbatch --export=ALL,PHASE=final,OUT_DIR="$BASE/outputs/bird_select_bird_ret" scripts/bird_select_cpu.slurm | grep -oE '[0-9]+$')
    while squeue -j "$J" -h 2>/dev/null | grep -q .; do sleep 60; done
    S=$(sacct -j "$J" -n -o State -X 2>/dev/null | tr -d ' ' | head -1)
    echo "$(date '+%F %T') final $J -> $S" >> "$LOG"
    python3 -c "import json; d=json.load(open('outputs/bird_select_bird_ret/summary.json')); o=d['official_exec_accuracy']; print('vav', o['arm_vav']['total'], 'orm', o['arm_orm_grouphead']['total'])" >> "$LOG" 2>&1
    echo "$(date '+%F %T') WATCH_DONE" >> "$LOG"
    exit 0
  fi
  sleep 300
done
