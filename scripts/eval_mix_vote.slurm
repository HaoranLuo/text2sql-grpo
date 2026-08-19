#!/bin/bash
# v1 + v2 跨 checkpoint 混合投票: sft_phase1(v1) + sft_v2(v2) 各 5 视角 = 10 票
# 主裁决 = vav 执行结果分组投票(mixvav), 简单多数(maj10)仅对照; 四路输出逐题
# (a5p / b5p / mixvav / maj10)。假设: 混合两个不同 checkpoint 恢复多样性,
# 超过 v1 5p 官方 70.1%。离线分析: 29 题两模型全票一致但都错; 30 题 v2 独有一
# 致错 v1 对; v1 5 对 + v2 5 错时简单多数 5:5 无解 → vav 裁决(平票回退 v1)。
# 时间: 1034×10=10340 次贪心生成; 5p 全量 5170 次实测 13620s(≈2.6s/次) → ≈7.6h;
#       上限 12h 留余量。显存: base+单 LoRA(与 5p 作业一致), A 跑完 unload 再挂 B。
#SBATCH --job-name=mix_v1v2
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/mix_v1v2_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/mix_v1v2_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== v1+v2 混合 10 票投票评估 (vav 主裁决, 贪心 T=0) ==="
$PYTHON $BASE/src/eval_mix_vote.py \
    --lora-a $BASE/checkpoints/sft_phase1 \
    --lora-b $BASE/checkpoints/sft_v2 \
    --output-dir $BASE/outputs/eval_mix_vote \
    --limit 1034 --temperature 0.0

echo "=== 官方口径 (predicted_sql = mixvav 主裁决胜者) ==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/eval_mix_vote/items.json \
    $BASE/outputs/official_eval_mix_vote

echo "Done: $(date)"
