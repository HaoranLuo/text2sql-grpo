#!/bin/bash
# RFT 5 源池集成全链路编排（训练线决定性一战）：
#   ① RFT 生成（gpudebug 切片，重提直至 fully_done，~4h）
#   ② 拼 5 源池（登录节点，秒级）
#   ③ prep 分组（cpu6348，~15min）
#   ④ ORM 分合并 partition（登录节点，秒级）→ 新增组代表打分（gpudebug，~10min）
#   ⑤ final 裁决 + 官方 EX（cpu6348，~5min）
#   ⑥ 对照收割：pool_pass_oracle + McNemar + 预注册判定
# gpudebug QOS MaxSubmitJobs=1 → GPU 作业严格串行（wait→submit）。
# 预注册（records/experiments.jsonl BIRD_RFT_pool5src）：
#   5 源 arm_orm_grouphead 官方 EX total > 60.37 判正向；>= 62.0 判强。
# 用法: nohup bash scripts/bird_rft_pool_chain.sh > logs/bird_rft_pool_chain.out 2>&1 &
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b || exit 1

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
LOG=logs/bird_rft_pool_chain.log
echo "$(date '+%F %T') rft pool chain start" >> "$LOG"

wait_job_done() {
    local jid=$1
    while squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 45; done
}
job_state() {
    sacct -j "$1" -n -o State -X 2>/dev/null | tr -d ' ' | head -1
}
check_state() {
    local jid=$1 name=$2
    wait_job_done "$jid"
    local st
    st=$(job_state "$jid")
    echo "$(date '+%F %T') $name ($jid) -> $st" >> "$LOG"
    if [ "$st" != "COMPLETED" ]; then
        echo "$(date '+%F %T') $name FAILED ($st)，检查 logs/ 后人工修复重提" >> "$LOG"
        exit 1
    fi
}

# ---- ① 生成：切片重提直至 fully_done ----
N_GEN=0
while true; do
    N_GEN=$((N_GEN+1))
    if [ $N_GEN -gt 30 ]; then
        echo "$(date '+%F %T') GEN: 超过 30 次提交仍未完成，中止（人工介入）" >> "$LOG"
        exit 1
    fi
    jid=$(sbatch scripts/gen_bird_rft_pool.slurm | awk '{print $4}')
    echo "$(date '+%F %T') GEN submit #$N_GEN -> $jid" >> "$LOG"
    wait_job_done "$jid"
    st=$(job_state "$jid")
    echo "$(date '+%F %T') GEN #$N_GEN ($jid) -> $st" >> "$LOG"
    if [ "$st" != "COMPLETED" ]; then
        echo "$(date '+%F %T') GEN job FAILED ($st)，检查 logs/bird_rft_pool_$jid.err 后人工修复重提" >> "$LOG"
        exit 1
    fi
    done_flag=$(python3 -c "
import json,os
p='$BASE/outputs/eval_pool_bird_rft/summary.json'
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

# ---- ② 拼 5 源池（登录节点）----
echo "$(date '+%F %T') MERGE start" >> "$LOG"
envs/reasoning3b/bin/python src/merge_pool_rft.py >> "$LOG" 2>&1 || {
    echo "$(date '+%F %T') MERGE FAILED, abort" >> "$LOG"; exit 1; }
[ -f $BASE/outputs/eval_pool_bird_5src/items.json ] || {
    echo "$(date '+%F %T') MERGE 完成但缺 items.json, abort" >> "$LOG"; exit 1; }
echo "$(date '+%F %T') MERGE done" >> "$LOG"

# ---- ③ prep 分组（cpu6348）----
OUT_DIR=$BASE/outputs/bird_select_5src
p1=$(sbatch --export=ALL,PHASE=prep,ITEMS=$BASE/outputs/eval_pool_bird_5src/items.json,OUT_DIR="$OUT_DIR" scripts/bird_select_cpu.slurm | awk '{print $4}')
echo "$(date '+%F %T') PREP submit -> $p1" >> "$LOG"
check_state "$p1" "PREP"
[ -f "$OUT_DIR/work/orm_payloads.json" ] || {
    echo "$(date '+%F %T') PREP 完成但缺少 orm_payloads.json, abort" >> "$LOG"; exit 1; }

# ---- ④ ORM 分合并 partition + 新增组代表打分 ----
echo "$(date '+%F %T') SCORE-PARTITION start" >> "$LOG"
envs/reasoning3b/bin/python src/bird_score_merge.py --phase partition \
    --old-work $BASE/outputs/bird_select_ormbird_bird_bal2/work \
    --new-work "$OUT_DIR/work" >> "$LOG" 2>&1 || {
    echo "$(date '+%F %T') SCORE-PARTITION FAILED, abort" >> "$LOG"; exit 1; }
need_n=$(python3 -c "
import json
d=json.load(open('$OUT_DIR/work/orm_scores_need.json'))
print(len(d['payloads']))
" 2>/dev/null)
echo "$(date '+%F %T') SCORE-PARTITION done: need=$need_n" >> "$LOG"
if [ "$need_n" -gt 0 ]; then
    s1=$(sbatch --export=ALL,NEW_WORK="$OUT_DIR/work" scripts/bird_rft_score_inc.slurm | awk '{print $4}')
    echo "$(date '+%F %T') SCORE-GPU submit -> $s1" >> "$LOG"
    check_state "$s1" "SCORE-GPU"
    [ -f "$OUT_DIR/work/orm_scores.json" ] || {
        echo "$(date '+%F %T') SCORE-GPU 完成但缺 orm_scores.json, abort" >> "$LOG"; exit 1; }
else
    [ -f "$OUT_DIR/work/orm_scores.json" ] || {
        echo "$(date '+%F %T') 无新增组代表但缺 orm_scores.json, abort" >> "$LOG"; exit 1; }
fi

# ---- ⑤ final 裁决 + 官方 EX（cpu6348）----
f1=$(sbatch --export=ALL,PHASE=final,ITEMS=$BASE/outputs/eval_pool_bird_5src/items.json,OUT_DIR="$OUT_DIR" scripts/bird_select_cpu.slurm | awk '{print $4}')
echo "$(date '+%F %T') FINAL submit -> $f1" >> "$LOG"
check_state "$f1" "FINAL"

# ---- ⑥ 对照收割 + 预注册判定 ----
echo "$(date '+%F %T') === HARVEST ===" >> "$LOG"
envs/reasoning3b/bin/python src/bird_pool_oracle.py >> "$LOG" 2>&1 || \
    echo "$(date '+%F %T') WARN oracle 脚本失败（不影响主判据）" >> "$LOG"
envs/reasoning3b/bin/python src/bird_mcnemar_rft.py >> "$LOG" 2>&1 || \
    echo "$(date '+%F %T') WARN mcnemar 脚本失败（不影响主判据）" >> "$LOG"
envs/reasoning3b/bin/python - <<EOF >> "$LOG" 2>&1
import json
base5 = "$OUT_DIR"
d5 = json.load(open(f"{base5}/summary.json"))
o5 = d5["official_exec_accuracy"]
o4 = json.load(open("$BASE/outputs/bird_select_ormbird_bird_bal2/summary.json"))["official_exec_accuracy"]
print("=== 官方 EX 对照 ===")
for arm in ("arm_vav", "arm_orm_grouphead"):
    a4, a5 = o4[arm]["total"], o5[arm]["total"]
    print(f"  {arm:22s} 4src={a4}  5src={a5}  delta={a5 - a4:+.2f}")
ex5 = o5["arm_orm_grouphead"]["total"]
tag = "STRONG (>=62.0)" if ex5 >= 62.0 else ("POSITIVE (>60.37)" if ex5 > 60.37 else "NOT PASS (<=60.37)")
print(f"preregistration [5src arm_orm_grouphead]: {ex5} -> {tag}")
EOF
echo "$(date '+%F %T') rft pool chain done" >> "$LOG"
