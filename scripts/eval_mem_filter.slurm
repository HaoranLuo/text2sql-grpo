#!/bin/bash
# 文献弹药: memory 质心过滤投票 —— L7 选择缺口修复尝试 (checkpoints/sft_v2 LoRA)
#   Spider dev 前 100 条, 同 prompt 一次采样 n=16 条候选
#   计算候选两两相似度 (执行结果分组优先, 执行失败回退 sqlglot token Jaccard, 两种口径均落盘)
#   剔除质心相似度最低的 20% / 30% 离群候选后再执行分组投票, 对照基线 maj@16
#   --min-votes 3: 仅对基线票数<=3 的低一致争议题启用过滤 (高一致题不动, 防误杀)
#   summary 输出 votes 分层 (0/1/2/3/4/5/6+): baseline vs 过滤后正确率, 重点看低一致题
# 输出: outputs/eval_mem_filter/ (checkpoint.json / items.json / summary.json)
# 官方口径: items.json 的 predicted_sql 为基线臂结果, 可用 scripts/eval_official.sh 复评
#SBATCH --job-name=mem_filter
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/mem_filter_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/mem_filter_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

echo "=== memory 质心过滤投票 (n=16, drop-ratio 0.2/0.3, min-votes 3) ==="
envs/reasoning3b/bin/python src/eval_mem_filter.py \
    --model-path $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --lora-path $BASE/checkpoints/sft_v2 \
    --spider-dir $BASE/data/spider_data \
    --limit 100 \
    --n 16 \
    --drop-ratio 0.2,0.3 \
    --min-votes 3 \
    --temperature 1.0 \
    --output-dir $BASE/outputs/eval_mem_filter

echo "Done: $(date)"
