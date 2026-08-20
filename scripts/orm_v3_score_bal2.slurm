#!/bin/bash
# orm_selection_v3 bal2 判卷 步骤 2：GPU 打分（gpudebug 3090，QOS 单作业上限 1h）
# 与 orm_v3_score.slurm 唯一差别：--orm-checkpoint orm_bal2、--m8-temp 1.0
# （T=1.0 时 soft_mean = sigmoid(Δ) = grouphead 标准 P(Yes)，即 bal2 判卷 grouphead 分）
# POOLS env 控制跑哪些池（multi|unseen 分开提交避开 1h QOS 墙）
#SBATCH --job-name=ormv3s_bal2
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv3s_bal2_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv3s_bal2_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "$BASE/checkpoints/orm_bal2/adapter_config.json" ]; then
    echo "FATAL: orm_bal2 不存在" >&2
    exit 1
fi

for P in ${POOLS:-multi unseen}; do
    case "$P" in
        multi)
            ITEMS=outputs/eval_pool_multi/items.json
            OUT=outputs/orm_selection_v3_bal2
            ;;
        unseen)
            ITEMS=outputs/eval_pool_unseen/items.json
            OUT=outputs/orm_selection_v3_bal2_unseen
            ;;
        *)
            echo "FATAL: unknown pool $P" >&2
            exit 1
            ;;
    esac
    echo "=== [$P] bal2 判卷: partial(T=1.0) + m8(T=1.0,n=8) 打分 ==="
    $PY src/orm_selection_v3.py \
        --items "$ITEMS" \
        --out-dir "$OUT" \
        --score-only \
        --base-model $BASE/models/Qwen2.5-Coder-3B-Instruct \
        --orm-checkpoint $BASE/checkpoints/orm_bal2 \
        --merge-python $BASE/envs/reasoning3b/bin/python \
        --m8-samples 8 --m8-temp 1.0 --chunk-size 256
done

echo "Done: $(date)"
