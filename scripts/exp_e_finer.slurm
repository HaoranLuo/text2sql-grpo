#!/bin/bash
#SBATCH --job-name=e_finer
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:30:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/e_finer_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/e_finer_%j.err

# E: FINER 式非负奖励（调研共识 v3）
#   - 奖励: finer (1.0 匹配 / 0.5+0.5×atomic 可执行但错 / 0 失败) 非负
#   - 数据: 2000 条 + --filter-gold（执行失败剔除）
#   - G=8, β=0.02, lr 3e-6, 100 步, checkpoint 每 25 步
#   - 依据: FINER-3B 同基座 Spider 85%（配方差距）；PaVeRL 非负奖励实证；
#           Think2SQL/CSC-SQL 超参共识
set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/e_finer
export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== E: FINER 式非负奖励 + 2000条 + G8 + 100步 ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) | $(date)"

$PYTHON $BASE/src/train_reasoning_grpo.py \
    --num-train 2000 --num-generations 8 --max-steps 100 \
    --reward-type finer --learning-rate 3e-6 --beta 0.02 --temperature 1.0 \
    --train-batch-size 16 --filter-gold \
    --output-dir "$OUT" --spider-dir "$SPIDER" --model-path "$MODEL"

echo "Training done: $(date)"

# 评估每个 checkpoint（25/50/75/100），取最优
for CKPT in $OUT/checkpoint-25 $OUT/checkpoint-50 $OUT/checkpoint-75 $OUT; do
    if [ -d "$CKPT" ] || [ -f "$CKPT/adapter_config.json" ]; then
        NAME=$(basename "$CKPT")
        echo "=== Eval $NAME ==="
        $PYTHON $BASE/src/evaluate_after_grpo.py \
            --lora-path "$CKPT" --limit 100 --spider-dir "$SPIDER" --model-path "$MODEL" \
            --output-dir $BASE/outputs/e_finer_$NAME --batch-size 8
    fi
done

echo ""
echo "=== E 结果汇总 ==="
$PYTHON -c "
import json, glob
for f in sorted(glob.glob('$BASE/outputs/e_finer_*/summary.json')):
    try:
        s = json.load(open(f))
        print(f'{f.split(\"/\")[-2]}: {s[\"custom_execution_match_rate\"]:.1%}')
    except Exception as e:
        print(f'{f}: {e}')
print('参考: 三级GRPO 50% | FINER论文3B 85%(官方口径)')
"
echo "Done: $(date)"
