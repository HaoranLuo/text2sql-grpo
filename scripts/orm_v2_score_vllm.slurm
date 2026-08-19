#!/bin/bash
# orm_selection_v2 阶段 2（vLLM 版）：GPU 打分两池全量候选，分数落盘 scores/scores_vllm.json
# 打分器 = orm_selection.VllmScorer（86.1/83.4 基线同款；vLLM 0.11 batch + LoRA）
# 用法: sbatch scripts/orm_v2_score_vllm.slurm
#SBATCH --job-name=ormv2_sv
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv2_sv_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv2_sv_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "$BASE/checkpoints/orm_b1/adapter_config.json" ]; then
    echo "FATAL: orm_b1 不存在" >&2
    exit 1
fi

echo "=== [multi] vLLM 打分（全量入组候选 + 组代表）==="
$PY src/orm_selection_v2.py \
    --items outputs/eval_pool_multi/items.json \
    --out-dir outputs/orm_selection_v2 \
    --baseline-items outputs/orm_selection_multi/items_arm_orm_grouphead.json \
    --orm-backend vllm --score-only \
    --base-model $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --orm-checkpoint $BASE/checkpoints/orm_b1 \
    --threads 16

echo "=== [unseen] vLLM 打分（全量入组候选 + 组代表）==="
$PY src/orm_selection_v2.py \
    --items outputs/eval_pool_unseen/items.json \
    --out-dir outputs/orm_selection_v2_unseen \
    --baseline-items outputs/orm_selection_unseen/items_arm_orm_grouphead.json \
    --orm-backend vllm --score-only \
    --base-model $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --orm-checkpoint $BASE/checkpoints/orm_b1 \
    --threads 16

echo "Done: $(date)"
