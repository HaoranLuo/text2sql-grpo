#!/bin/bash
# =====================================================================
# P1#4 大规模 FINER 完整配方 GRPO (提交前准备; 本脚本只用于 preflight 校验, 勿 sbatch)
# 对照对象: T2.1/T2.2/T2.3 三连败 (500 条 / G4 / 40 步 / lr 1-3e-6 / β 0.02-0.04 /
#           sft_phase1 起点 / 无 warmup / grad_norm 默认 1.0)
# ---------------------------------------------------------------------
# 本配方与 T2.x 的关键差异 (为什么这次不同):
#  1) 数据: 500 → GRPO-manifest 5500 条全量不相交子集, 与 SFT-gold 1.5k 构造性零交集。
#     物化脚本 scripts/make_grpo_train_json.py 先 sha256 校验原始数据
#     (== split_record.md 记录值), 再按规范索引取行, 并断言与 SFT-gold 交集为空。
#     (8659 全量 = GRPO 5500 + SFT-gold 1500 + 未分配 1659; 本配方取已划定的 5500,
#      未吸收"未分配池"1659, 理由见脚本注释与报告)
#  2) 组大小: G4 → G16 (组内方差足、优势估计稳)。FINER 原文用 G32, 但 A40 48G 上
#     G32 × 4 提示/步 = 128 条完成 ≈ 2×T23 峰值(logits/KV/激活全翻倍) ≈ 70-80GB,
#     会 OOM; G32 × 2 提示/步虽可跑但 2000 步仅覆盖 0.73 epoch。故主方案 G16 × 4
#     提示/步 = 64 条完成/步 == T23 实测足迹(59.5s/步, 40 步无 OOM)。
#  3) 步数: 40 → 2000 步上限 (5500 条 × 4 提示/步 → 约 1.45 epoch; 40 步只见过 160 条)
#  4) 奖励: finer 非负稠密 1.0 / 0.5+0.5×atomic(Jaccard) / 0.0 (同 T2.2), 加
#     --filter-gold 剔除 gold 不可执行的噪声样本(FINER Step-1 执行过滤同款纪律)。
#     FINER 四分量 = format + exec + atomic + memory; 本项目三份量已有实现
#     (format: 解析失败即 0; exec: compare_execution_results; atomic: sqlglot Jaccard),
#     memory(教师轨迹 embedding 语义对齐)在 src/ 无实现、HPC 未装
#     chromadb/sentence-transformers → 降级为三份量, 影响与补齐方案见报告。
#  5) 梯度修复 (P7 结论, 已核对 TRL 0.17.0 源码 grpo_trainer.py):
#     - scale_rewards=none → advantages 不做组内 std 归一(源码: if self.scale_rewards:
#       advantages/(std+1e-4)); 零 std 组 → 全零优势, 无 NaN。
#     - loss_type=dr_grpo → token-mean 去长度归一化(已确认 0.17 支持分支)。
#     - DAPO 式静默组过滤: TRL 0.17.0 无此能力(loss_type 仅 bnpo/dr_grpo/grpo,
#       无零优势组过滤/clip-higher, 需 >=0.18), 由 RewardStdGuard 兜底:
#       reward_std 连续 3 次为 0 即提前停止(已在 train_reasoning_grpo.py 内)。
#  6) 优化: lr 4e-6(区间 1e-6~8e-6 中值; T2.x 1-3e-6 全部欠训) + warmup 0.1
#     + cosine 衰减 + max_grad_norm 0.1 (GRPO 论文配方; TRL 默认 warmup=0.0/
#     grad_norm=1.0/linear, 由新 driver src/train_grpo_finer_full.py 显式覆盖)
#  7) β: 0.04/0.02 → 0.01 (规格 0.01-0.02 下限, 更强锁定 SFT 先验防回吐;
#     β sweep 单卡串行成本见文件尾注释块)
#  8) 起点: sft_phase1 → checkpoints/sft_v3 (最新最强 SFT 起点)
#  9) checkpoint 早停: save_steps=200 → 10 个 ckpt 全量 dev(1034) 自定义口径评估,
#     按 custom_execution_match_rate 选最优, 官方口径只评最优 ckpt。
# ---------------------------------------------------------------------
# 时长估算 (依据 T23 实测: 64 条完成/步 ≈ 59.5s/步, 作业墙钟 3h46m):
#   数据物化 ~5min + 训练 2000 × ~60s ≈ 33h + 10 ckpt × ~61min dev 评估 ≈ 10h
#   + 官方评估 ~1h ≈ 总 ~44h; --time=72h 余量约 60% (QOS 1a40 MaxWall=7d)。
#   若 RewardStdGuard 触发早停, 全部时长相应缩短。
# ---------------------------------------------------------------------
# 提交前检查 (0 错 0 警才算过):
#   PREFLIGHT_SLURM=$BASE/scripts/train_grpo_finer_full.slurm \
#   PREFLIGHT_EXP=grpo_finer_full \
#   bash $BASE/scripts/preflight_check.sh --strict
#SBATCH --job-name=grpo_finer_full
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/grpo_finer_full_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/grpo_finer_full_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
MODEL=$BASE/models/Qwen2.5-Coder-3B-Instruct
SPIDER=$BASE/data/spider_data
OUT=$BASE/checkpoints/grpo_finer_full
TRAIN_JSON=$BASE/data/grpo_train_5500.json

export PATH="$BASE/envs/reasoning3b/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== P1#4 FINER 全量配方 GRPO (G16 × 5500 不相交, 自 sft_v3) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) | $(date)"
echo "trl: $($PYTHON -c 'import trl; print(trl.__version__)')"

# ---------------------------------------------------------------------
# 0. 数据物化: manifest(规范索引) → 训练 JSON
#    sha256 校验原始数据 + (db_id,question) 逐条核对 + SFT-gold 不相交断言 + 幂等
# ---------------------------------------------------------------------
$PYTHON $BASE/scripts/make_grpo_train_json.py \
    --manifests $BASE/data_hygiene/grpo_5500_manifest.jsonl \
    --sft-manifest $BASE/data_hygiene/sft_gold_1500_manifest.jsonl \
    --spider-dir $SPIDER \
    --out $TRAIN_JSON

# ---------------------------------------------------------------------
# 1. GRPO 训练 (RewardStdGuard 内建: reward_std 连续 3 次为 0 即提前停止)
#    G16 × 4 提示/步 = 64 条完成/步 == T23 已验证的 A40 显存足迹
# ---------------------------------------------------------------------
$PYTHON $BASE/src/train_grpo_finer_full.py \
    --train-file $TRAIN_JSON \
    --num-train 5500 \
    --num-generations 16 \
    --train-batch-size 4 \
    --max-steps 2000 \
    --save-steps 200 \
    --reward-type finer \
    --filter-gold \
    --loss-type dr_grpo \
    --scale-rewards none \
    --learning-rate 4e-6 \
    --beta 0.01 \
    --temperature 1.0 \
    --max-grad-norm 0.1 \
    --warmup-ratio 0.1 \
    --lr-scheduler-type cosine \
    --lora-init $BASE/checkpoints/sft_v3 \
    --output-dir "$OUT" \
    --spider-dir "$SPIDER" \
    --model-path "$MODEL"

echo "Training done: $(date)"

# ---------------------------------------------------------------------
# 2. checkpoint 早停: 每个 ckpt 全量 dev(1034) 自定义口径评估 → 选最优
# ---------------------------------------------------------------------
for CK in $(ls -d $OUT/checkpoint-* 2>/dev/null); do
    NAME=$(basename $CK)
    echo "--- eval $NAME ---"
    $PYTHON $BASE/src/evaluate_after_grpo.py \
        --lora-path "$CK" --model-path "$MODEL" --spider-dir "$SPIDER" \
        --limit 1034 --batch-size 8 \
        --output-dir $BASE/outputs/grpo_finer_full_$NAME
done

echo "=== 汇总各 ckpt 自定义口径, 选最优 (custom_execution_match_rate) ==="
BEST_RESULT=$($PYTHON - "$BASE/outputs" <<'PYEOF'
import glob, json, sys
base = sys.argv[1]
best_dir, best_rate = None, -1.0
for s in sorted(glob.glob(base + "/grpo_finer_full_checkpoint-*/summary.json")):
    d = json.load(open(s))
    r = d.get("custom_execution_match_rate", -1.0)
    print(f"{s}: {r:.4f}")
    if r > best_rate:
        best_rate, best_dir = r, s.rsplit("/", 1)[0]
if best_dir is None:
    sys.exit("未找到任何 checkpoint summary.json, 早停选择失败")
print("BEST " + best_dir)
PYEOF
)
echo "$BEST_RESULT"
BEST_EVAL_DIR=$(echo "$BEST_RESULT" | grep '^BEST ' | cut -d' ' -f2-)

# RewardStdGuard 可能在首个 save_steps 之前就早停 → 无 checkpoint-* 子目录,
# 此时退而评估 trainer.save_model 写入的最终 adapter (仍在 $OUT)。
if [ -z "$BEST_EVAL_DIR" ]; then
    echo "无 checkpoint 子目录 (早停于首个 save 之前), 评估最终 adapter"
    $PYTHON $BASE/src/evaluate_after_grpo.py \
        --lora-path "$OUT" --model-path "$MODEL" --spider-dir "$SPIDER" \
        --limit 1034 --batch-size 8 \
        --output-dir $BASE/outputs/grpo_finer_full_final
    BEST_EVAL_DIR=$BASE/outputs/grpo_finer_full_final
fi

echo "=== 官方口径 (最优 ckpt) ==="
bash $BASE/scripts/eval_official.sh \
    $BEST_EVAL_DIR/items.json \
    $BASE/outputs/official_grpo_finer_full_best

echo "Done: $(date)"

# =====================================================================
# 可选: β sweep (β=0.02 第二臂) — 单卡串行成本 ≈ 再 +44h (训练 33h + 评估 ~11h)。
# QOS 1a40 限 1 卡/用户, 无法并行; 建议拆成独立作业分别走 preflight+批准,
# 而不是在本作业内串行。启用时复制第 1 步训练命令, 仅改 --beta 与 --output-dir:
#   $PYTHON $BASE/src/train_grpo_finer_full.py \
#       --train-file $TRAIN_JSON --num-train 5500 --num-generations 16 \
#       --train-batch-size 4 --max-steps 2000 --save-steps 200 \
#       --reward-type finer --filter-gold --loss-type dr_grpo \
#       --scale-rewards none --learning-rate 4e-6 --beta 0.02 --temperature 1.0 \
#       --max-grad-norm 0.1 --warmup-ratio 0.1 --lr-scheduler-type cosine \
#       --lora-init $BASE/checkpoints/sft_v3 \
#       --output-dir $BASE/checkpoints/grpo_finer_full_b002 \
#       --spider-dir $SPIDER --model-path $MODEL
# =====================================================================
