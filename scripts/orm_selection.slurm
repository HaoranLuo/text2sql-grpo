#!/bin/bash
# T2 收尾实验：ORM 集成选择（4 臂）→ 官方 test-suite 复评
# 前置：checkpoints/orm_b1 训练完成（scripts/train_orm.slurm 产出 adapter_config.json；
#       训练完成前只能用 STUB=1 冒烟，见下）
#
# 时间估算（全部离线/推理侧，不重新生成候选）：
#   ORM 打分：20,596 唯一候选 × 一次前向（prompt ~560 token ≈ 11.5M token），
#             vLLM batch 3B LoRA 预填，A40/3090 ≈ 5-15 min（含引擎加载 ~2 min、
#             top-k logprobs 取 Yes/No；HF+peft batch 约 30-75 min，不采用）
#   CPU 分组：全部唯一候选原始 SQL × 实例执行 ≈ 15 min
#             （实测参照：adjudicate_soft grouping_phase 760,777 任务 841.6s，16 线程）
#   官方复评：4 臂 × eval_official.sh ≈ 10 min
#   合计 ≈ 35-45 min；time 上限 4h 留足余量（引擎加载/排队/兜底重打分）
#
# 冒烟（无 ORM 模型，随机分数走通全流程；登录节点 CPU 可直接跑 src 脚本，
#       这里用 STUB=1 在 GPU 节点走同一入口，覆盖 slurm 环境本身）：
#   sbatch --export=ALL,STUB=1,LIMIT=30,OUT_DIR=$BASE/outputs/orm_selection_smoke_slurm scripts/orm_selection.slurm
#SBATCH --job-name=orm_select
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_selection_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/orm_selection_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/vllmenv/bin/python
STUB=${STUB:-0}
LIMIT=${LIMIT:-}
OUT_DIR=${OUT_DIR:-$BASE/outputs/orm_selection}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $BASE

echo "=== ORM 集成选择（4 臂：argmax/grouphead/tiebreak/hybrid）==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

EXTRA_ARGS=()
if [ "$STUB" = "1" ]; then
    echo "STUB=1: 随机分数替代 ORM（冒烟，不加载模型）"
    EXTRA_ARGS+=(--stub-scores random)
else
    if [ ! -f "$BASE/checkpoints/orm_b1/adapter_config.json" ]; then
        echo "FATAL: checkpoints/orm_b1 无 adapter_config.json（ORM 未训练完成）" >&2
        echo "       先跑 scripts/train_orm.slurm；或 sbatch --export=ALL,STUB=1 冒烟" >&2
        exit 1
    fi
fi
if [ -n "$LIMIT" ]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

echo "=== 步骤 1/2：ORM 打分 + 4 臂选择 + 官方语义判定（in-process）==="
$PYTHON $BASE/src/orm_selection.py \
    --items $BASE/outputs/eval_pool_b1/items.json \
    --out-dir "$OUT_DIR" \
    --spider-dir $BASE/data/spider_data \
    --orm-checkpoint $BASE/checkpoints/orm_b1 \
    --base-model $BASE/models/Qwen2.5-Coder-3B-Instruct \
    --merge-python $BASE/envs/reasoning3b/bin/python \
    --threads 16 \
    "${EXTRA_ARGS[@]}"

echo "=== 步骤 2/2：官方复评（4 臂，test-suite evaluator）==="
for ARM in arm_orm_argmax arm_orm_grouphead arm_orm_tiebreak arm_orm_hybrid; do
    bash $BASE/scripts/eval_official.sh \
        "$OUT_DIR"/items_${ARM}.json \
        $BASE/outputs/official_orm_selection_${ARM}
    echo "--- OFFICIAL $ARM ---"
    grep -A1 'EXECUTION ACCURACY' \
        $BASE/outputs/official_orm_selection_${ARM}/official_result.txt | tail -1
done

echo "=== in-process summary ==="
grep -A20 '"accuracy"' "$OUT_DIR"/summary.json | head -25 || true

echo "Done: $(date)"
