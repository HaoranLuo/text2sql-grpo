#!/bin/bash
# ============================================================
# Pre-flight 检查脚本 — 训练提交前自动运行
# 用法: bash scripts/preflight_check.sh [--strict]
# 返回: 0=通过, 1=有警告(可继续), 2=有错误(禁止提交)
# ============================================================

BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
STRICT=0
[ "$1" == "--strict" ] && STRICT=1

ERRORS=0
WARNINGS=0

check() {
    local desc="$1" status="$2" detail="$3"
    if [ "$status" == "ERROR" ]; then
        ERRORS=$((ERRORS+1))
        echo "  [❌] $desc: $detail"
    elif [ "$status" == "WARN" ]; then
        WARNINGS=$((WARNINGS+1))
        echo "  [⚠️] $desc: $detail"
    else
        echo "  [✅] $desc"
    fi
}

echo "=============================================="
echo "Pre-flight Check — $(date)"
echo "=============================================="

# ------------------------------------------------------------
echo "[1/16] 生成长度一致性 (训练 vs 推理 vs 评估)"
# ------------------------------------------------------------
TRAIN_LEN=$(grep -oP 'max_completion_length=\K\d+' $BASE/src/train_reasoning_grpo.py)
INFER_LEN=$(grep -oP 'max_new_tokens: int = \K\d+' $BASE/src/reasoning_generator_agent.py)
EVAL_LEN=$(grep -oP 'max-new-tokens.*default=\K\d+' $BASE/src/evaluate_after_grpo.py)

if [ "$TRAIN_LEN" == "$INFER_LEN" ] && [ "$INFER_LEN" == "$EVAL_LEN" ]; then
    check "生成长度一致 ($TRAIN_LEN)" OK ""
else
    check "生成长度一致" ERROR "train=$TRAIN_LEN infer=$INFER_LEN eval=$EVAL_LEN — 必须相等!"
fi

# ------------------------------------------------------------
echo "[2/10] Prompt 构建函数共享"
# ------------------------------------------------------------
if grep -q 'ReasoningGeneratorAgent.build_prompt' $BASE/src/train_reasoning_grpo.py; then
    check "训练复用推理 prompt" OK ""
else
    check "训练复用推理 prompt" ERROR "训练未复用 agent 的 build_prompt!"
fi

# ------------------------------------------------------------
echo "[3/10] 奖励函数与评估逻辑对齐"
# ------------------------------------------------------------
if grep -q 'compare_execution_results' $BASE/src/train_reasoning_grpo.py; then
    check "奖励复用评估逻辑" OK ""
else
    check "奖励复用评估逻辑" ERROR "训练 reward 未用 compare_execution_results!"
fi

# ------------------------------------------------------------
echo "[4/16] tokenizer + model.config pad/eos 一致性 (Qwen默认不同!)"
# ------------------------------------------------------------
if grep -q 'tokenizer.pad_token = tokenizer.eos_token' $BASE/src/train_reasoning_grpo.py && grep -q 'model.config.pad_token_id' $BASE/src/train_reasoning_grpo.py; then
    check "训练 pad/eos 统一 (含 model.config)" OK ""
else
    check "训练 pad/eos 统一" ERROR "Qwen pad=151643 eos=151645 且 config 保留旧 pad! 需 tokenizer+config 都改"
fi
if grep -q 'self.tokenizer.pad_token = self.tokenizer.eos_token' $BASE/src/reasoning_generator_agent.py && grep -q 'base_model.config.pad_token_id' $BASE/src/reasoning_generator_agent.py; then
    check "推理 pad/eos 统一 (含 model.config)" OK ""
else
    check "推理 pad/eos 统一" ERROR "推理 agent 未统一 pad/eos + model.config!"
fi

# ------------------------------------------------------------
echo "[5/10] remove_unused_columns"
# ------------------------------------------------------------
if grep -q 'remove_unused_columns=False' $BASE/src/train_reasoning_grpo.py; then
    check "保留奖励所需列" OK ""
else
    check "保留奖励所需列" ERROR "remove_unused_columns 未设为 False!"
fi

# ------------------------------------------------------------
echo "[6/10] batch 整除 num_generations"
# ------------------------------------------------------------
BS=$(grep -oP 'per_device_train_batch_size=\K[^,]+' $BASE/src/train_reasoning_grpo.py)
NG=$(grep -oP 'num_generations=args.num_generations|num_generations=\K\d+' $BASE/src/train_reasoning_grpo.py)
# batch size 由 args.num_generations 决定，检查脚本参数
if [ "${PREFLIGHT_MODE:-grpo}" == "sft" ]; then
    check "batch 整除 num_generations" OK "SFT 作业不适用, 跳过"
elif grep -qE 'per_device_train_batch_size=.*args\.num_generations|args\.num_generations \* 4' $BASE/src/train_reasoning_grpo.py; then
    check "batch=k*num_generations (自动整除)" OK ""
else
    check "batch 整除 num_generations" WARN "手动设置需确认整除"
fi

# ------------------------------------------------------------
echo "[7/10] 环境版本"
# ------------------------------------------------------------
$PYTHON -c "
import torch, transformers, trl, peft
print(f'  torch={torch.__version__} transformers={transformers.__version__} trl={trl.__version__} peft={peft.__version__}')
" 2>/dev/null

# ------------------------------------------------------------
echo "[8/10] SLURM 分区有效性"
# ------------------------------------------------------------
# P1 修复(监管复审): 必须显式指定要校验的 slurm 文件, 禁止静默回退到 stale 旧文件
if [ -z "$PREFLIGHT_SLURM" ]; then
    check "PREFLIGHT_SLURM 未设置" ERROR "必须显式传 PREFLIGHT_SLURM=<提交的slurm脚本>, 防止校验错文件"
    SLURM_FILE="$BASE/scripts/NO_SUCH_FILE.slurm"
else
    SLURM_FILE="$PREFLIGHT_SLURM"
fi
PART=$(grep -oP '(?<=--partition=)\S+' "$SLURM_FILE" 2>/dev/null | head -1)
QOS=$(grep -oP '(?<=--qos=)\S+' "$SLURM_FILE" 2>/dev/null | head -1)
if sinfo -p "$PART" >/dev/null 2>&1; then
    check "分区 $PART 存在" OK ""
else
    check "分区 $PART 存在" ERROR "无效分区!"
fi

# ------------------------------------------------------------
echo "[9/10] 模型文件完整性"
# ------------------------------------------------------------
MODEL=$(grep -oP -- '--model-path\s+\S+|\$MODEL|\$MODEL_PATH' $BASE/scripts/*.slurm 2>/dev/null | head -1)
for M in Qwen2.5-Coder-3B-Instruct qwen2.5-coder-7b-instruct; do
    if [ -d "$BASE/models/$M" ] && [ "$(ls $BASE/models/$M/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
        check "模型 $M 完整" OK ""
    else
        check "模型 $M 完整" ERROR "缺少 safetensors!"
    fi
done

# ------------------------------------------------------------
echo "[10/10] 数据文件"
# ------------------------------------------------------------
for f in dev.json tables.json train_spider.json; do
    if [ -f "$BASE/data/spider_data/$f" ]; then
        check "数据 $f" OK ""
    else
        check "数据 $f" ERROR "缺失!"
    fi
done

# ------------------------------------------------------------
echo "[11/16] 奖励函数单元测试 (好答案 > 坏答案)"
# ------------------------------------------------------------
$PYTHON -c "
import sys
sys.path.insert(0, '$BASE/src')
from train_reasoning_grpo import create_reward_function
rf = create_reward_function('$BASE/data/spider_data', 'three_level')
good = rf(completions=['\`\`\`sql\nSELECT count(*) FROM head WHERE age > 56\n\`\`\`'],
          query=['SELECT count(*) FROM head WHERE age > 56'],
          db_id=['department_management'])[0]
bad = rf(completions=['\`\`\`sql\nSELECT * FROM nonexistent\n\`\`\`'],
         query=['SELECT count(*) FROM head WHERE age > 56'],
         db_id=['department_management'])[0]
if good > bad:
    print('  [✅] 奖励区分度: good=' + str(good) + ' > bad=' + str(bad))
else:
    print('  [❌] 奖励无区分度: good=' + str(good) + ' <= bad=' + str(bad))
    sys.exit(1)
" 2>/dev/null
if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS+1))
    echo "  [❌] 奖励区分度检查失败"
fi

# ------------------------------------------------------------
echo "[12/16] 数据泄漏检查 (train vs dev 重叠)"
# ------------------------------------------------------------
$PYTHON -c "
import json
SPIDER = '$BASE/data/spider_data'
with open(SPIDER + '/train_spider.json') as f:
    train = json.load(f)
with open(SPIDER + '/dev.json') as f:
    dev = json.load(f)
train_q = set((t['question'].strip().lower(), t['db_id']) for t in train)
dev_q = set((d['question'].strip().lower(), d['db_id']) for d in dev[:100])
overlap = train_q & dev_q
if len(overlap) == 0:
    print('  [✅] 无数据泄漏 (train全部 vs dev前100: 0 重叠)')
else:
    print('  [❌] 数据泄漏! ' + str(len(overlap)) + ' 条重叠')
    sys.exit(1)
" 2>/dev/null
if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS+1))
    echo "  [❌] 数据泄漏检查失败"
fi

# ------------------------------------------------------------
echo "[13/16] gold SQL 可执行性"
# ------------------------------------------------------------
$PYTHON -c "
import sys, json
sys.path.insert(0, '$BASE/src')
from spider_utils import DatabaseExecutor
SPIDER = '$BASE/data/spider_data'
executor = DatabaseExecutor(SPIDER)
with open(SPIDER + '/train_spider.json') as f:
    train = json.load(f)
fail = 0
for t in train[:100]:
    r = executor.execute(t['db_id'], t['query'])
    if not r['success']:
        fail += 1
if fail == 0:
    print('  [✅] gold SQL 100条全部可执行')
else:
    print('  [❌] ' + str(fail) + ' 条 gold 不可执行')
    sys.exit(1)
" 2>/dev/null
if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS+1))
    echo "  [❌] gold SQL 可执行性检查失败"
fi

# ------------------------------------------------------------
echo "[14/16] checkpoint 冲突检查 (防重复跑)"
# ------------------------------------------------------------
EXP="${PREFLIGHT_EXP:-grpo_7b_500d}"   # 实验名经环境变量传入, 默认保持旧行为
CONFLICT=0
[ -d "$BASE/checkpoints/$EXP" ] && CONFLICT=1
ls "$BASE"/outputs/eval_${EXP}*/summary.json >/dev/null 2>&1 && CONFLICT=1
if [ $CONFLICT -eq 1 ]; then
    check "已有相同实验输出" WARN "$EXP 已存在 — 确认是否覆盖重跑"
else
    check "无 checkpoint 冲突 ($EXP)" OK ""
fi

# ------------------------------------------------------------
echo "[15/16] TRL API 签名 + loss_type 路由检查 (版本变动)"
# ------------------------------------------------------------
$PYTHON -c "
from trl import GRPOTrainer
import inspect
params = list(inspect.signature(GRPOTrainer.__init__).parameters.keys())
required = ['reward_funcs', 'processing_class']
missing = [p for p in required if p not in params]
if not missing:
    print('  [✅] TRL API: reward_funcs + processing_class 存在')
else:
    print('  [❌] TRL API 变动, 缺失: ' + str(missing))
    sys.exit(1)
" 2>/dev/null
if [ $? -ne 0 ]; then
    ERRORS=$((ERRORS+1))
    echo "  [❌] TRL API 签名检查失败"
fi
TRL_VER=$($PYTHON -c "import trl; print(trl.__version__)" 2>/dev/null)
echo "  [ℹ️] 已装 TRL 版本: $TRL_VER"
# loss_type 路由校验: 已装源码支持分支 vs 提交脚本请求值(防 dapo 类漏检, 监管 P0 教训)
LOSS_BRANCHES=$(grep -oP 'self\.loss_type == "[a-z_0-9]+"' \
    $BASE/envs/reasoning3b/lib/python3.10/site-packages/trl/trainer/grpo_trainer.py \
    2>/dev/null | grep -oP '[a-z_0-9]+"$' | tr -d '"' | sort -u | tr '\n' ' ')
REQ_LOSS=$(grep -oP '(?<=--loss-type )[a-z_0-9]+' "$SLURM_FILE" 2>/dev/null | head -1)
if [ -z "$REQ_LOSS" ]; then REQ_LOSS="grpo"; fi
if echo " $LOSS_BRANCHES " | grep -q " $REQ_LOSS "; then
    check "loss_type=$REQ_LOSS 在已装 TRL 支持范围 ($(echo $LOSS_BRANCHES))" OK ""
else
    check "loss_type=$REQ_LOSS 不受已装 TRL 支持" ERROR "仅支持: $(echo $LOSS_BRANCHES) — 训练第一步必崩"
fi

# ------------------------------------------------------------
echo "[16/16] GPU 显存预估"
# ------------------------------------------------------------
MODEL_SIZE=$(du -sh "$BASE/models/qwen2.5-coder-7b-instruct" 2>/dev/null | cut -f1)
echo "  模型大小: $MODEL_SIZE (A40 48GB 应足够)"

# ------------------------------------------------------------
echo ""
echo "=============================================="
echo "结果: $ERRORS 错误, $WARNINGS 警告"
if [ $ERRORS -gt 0 ]; then
    echo ">>> 存在错误，禁止提交训练！"
    exit 2
elif [ $WARNINGS -gt 0 ] && [ $STRICT -eq 1 ]; then
    echo ">>> strict 模式：警告也阻止提交"
    exit 1
else
    echo ">>> 检查通过，可以提交"
    exit 0
fi
