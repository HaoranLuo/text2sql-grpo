#!/bin/bash
# B1 大候选池生成: Spider dev 1034 × {sft_phase1, sft_v2} × n=16 采样(vLLM)
# gpudebug 50min 切片 + checkpoint 断点续跑(resume 混配保护在脚本内)
#
# 时长估算(2026-08-15 冒烟实测, gpudebug RTX3090, 10 题双模型 n=16):
#   sft_phase1: 28.0s/10题 (629 tok/s)   -> 全量生成约 48 min + 引擎 ~1 min ≈ 49 min
#   sft_v2:      7.8s/10题 (1693 tok/s)  -> 全量生成约 13 min + 引擎 ~1 min ≈ 14 min
#   合计生成 ~62 min(前 10 题偏易, dev 尾部更慢, 实际预计 70-90 min)
#   -> 50min 切片跑不完, 需切片: phase1 ≈ 1-2 片 + sft_v2 ≈ 1 片, 预计 2-3 个切片。
#   每 25 题写 checkpoint, 切片被杀最多丢 25 题; 重新 sbatch 本脚本即续跑。
#SBATCH --job-name=pool_b1
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/pool_b1_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/pool_b1_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

# 冒烟覆盖: sbatch --export=ALL,LIMIT=10,OUTPUT_DIR=$BASE/outputs/eval_pool_b1_smoke scripts/eval_pool_b1.slurm
LIMIT=${LIMIT:-1034}
OUTPUT_DIR=${OUTPUT_DIR:-$BASE/outputs/eval_pool_b1}

echo "=== eval_pool_b1 (limit=$LIMIT, out=$OUTPUT_DIR, slice=TIME_LIM 00:50:00) ==="
# 每模型独立进程：vLLM 0.11 EngineCore 子进程在 del llm 后显存回收有延迟，
# 同进程连开两个引擎会撞上 "Free memory on device" 检查；进程边界 = 干净显存。
# run_config 记录完整 --loras 列表（两进程一致，resume 混配保护不受影响）。
for LORA in sft_phase1 sft_v2; do
  echo "--- pass: $LORA ---"
  $PYTHON $BASE/src/eval_pool_b1.py \
      --limit $LIMIT \
      --output-dir $OUTPUT_DIR \
      --loras sft_phase1,sft_v2 \
      --only-lora $LORA \
      --n 16 --temperature 1.0 --top-p 1.0 --seed 0 \
      --max-new-tokens 2048 \
      --checkpoint-every 25
done

echo "Done: $(date)"
