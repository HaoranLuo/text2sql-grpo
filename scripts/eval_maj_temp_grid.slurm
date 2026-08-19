#!/bin/bash
# 投票诊断曲线温度网格: sft_v2 检查点 (LoRA) × maj@{8,16,32,64} × 温度{0.7,1.0}
#   Spider dev 前 100 条, 同 prompt 每温度一次采样 K=64, 按前缀 K∈{8,16,32,64}
#   逐格计算 maj@K(双口径: vav + 项目自定义 compare_execution_results) +
#   Pass@K + 执行分组数 + 绝对多数命中率 + Self-BLEU 近似(maj_diag 同口径)
# 输出: outputs/eval_maj_temp_grid/ (checkpoint.json / items.json / summary.json)
# 官方口径: 由主控另跑 scripts/eval_official.sh(各温度档 max-K 的 predicted_sql)
# 时间预算: maj_diag(K=32×100×1 温度) 用 12h; 本实验 64×100×2 温度 ≈ 4× 生成长度,
#   给 24h; 超时靠 checkpoint 断点续跑(每 10 题存一次, 断点粒度=整题)
#SBATCH --job-name=maj_tgrid
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/maj_tgrid_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/maj_tgrid_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

echo "=== maj@K x temperature grid evaluation (sft_v2, LoRA) ==="
envs/reasoning3b/bin/python src/eval_maj_temp_grid.py \
    --lora-path $BASE/checkpoints/sft_v2 \
    --base-model $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --spider-dir $BASE/data/spider_data \
    --n 100 \
    --grid "8,16,32,64:0.7,1.0" \
    --output-dir $BASE/outputs/eval_maj_temp_grid

echo "=== 官方 test-suite 复评由主控另跑 (参考命令, 按温度档提取 predicted_sql) ==="
echo "# bash $BASE/scripts/eval_official.sh <per-temp items.json> $BASE/outputs/official_maj_temp_grid_<temp>"

echo "Done: $(date)"
