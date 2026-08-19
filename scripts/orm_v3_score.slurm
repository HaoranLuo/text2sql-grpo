#!/bin/bash
# orm_selection_v3 步骤 2：GPU 打分（gpudebug 3090 短作业，QOS 单作业上限 1h）
#   ① Stage-A 部分证据打分（T=1.0，orm_selection.VllmScorer 原样）→ scores_partial_vllm.json
#   ② M8 采样打分（T=0.2，n=8）→ scores_m8_vllm.json
# 打分对象 = rankable 组代表（~6-8k/池）。依赖步骤 1 的 partial_payloads.json。
# POOLS env 控制跑哪些池（空格分隔，默认 multi unseen）；两池分开提交以避开 1h QOS 墙。
# 用法:
#   sbatch --dependency=afterok:<payloads_job> --export=POOLS=multi scripts/orm_v3_score.slurm
#   sbatch --dependency=afterok:<payloads_job> --export=POOLS=unseen scripts/orm_v3_score.slurm
#SBATCH --job-name=ormv3_score
#SBATCH --partition=gpudebug
#SBATCH --qos=gpudebug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv3_score_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/ormv3_score_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=$BASE/envs/vllmenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "$BASE/checkpoints/orm_b1/adapter_config.json" ]; then
    echo "FATAL: orm_b1 不存在" >&2
    exit 1
fi

for P in ${POOLS:-multi unseen}; do
    case "$P" in
        multi)
            ITEMS=outputs/eval_pool_multi/items.json
            OUT=outputs/orm_selection_v3
            ;;
        unseen)
            ITEMS=outputs/eval_pool_unseen/items.json
            OUT=outputs/orm_selection_v3_unseen
            ;;
        *)
            echo "FATAL: unknown pool $P" >&2
            exit 1
            ;;
    esac
    echo "=== [$P] partial(T=1.0) + m8(T=0.2,n=8) 打分 ==="
    $PY src/orm_selection_v3.py \
        --items "$ITEMS" \
        --out-dir "$OUT" \
        --score-only \
        --base-model $BASE/models/Qwen2.5-Coder-3B-Instruct \
        --orm-checkpoint $BASE/checkpoints/orm_b1 \
        --merge-python $BASE/envs/reasoning3b/bin/python \
        --m8-samples 8 --m8-temp 0.2 --chunk-size 256
done

echo "Done: $(date)"
