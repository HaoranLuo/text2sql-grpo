# Text-to-SQL GRPO

用 **GRPO 强化学习**训练小模型（Qwen2.5-Coder 3B/7B）做 **Text-to-SQL**（自然语言→SQL），在 Spider 数据集上验证。

## 🏆 核心成果

| 模型 | 方法 | Match |
|------|------|:---:|
| Qwen2.5-Coder-7B | 零样本 + 3prompt投票 | **85%** |
| Qwen2.5-Coder-7B | 零样本 | 81% |
| Qwen2.5-Coder-3B | GRPO训练25步 + 投票 | **65%** |
| Qwen2.5-Coder-3B | GRPO三级奖励25步 | 50% |
| Qwen2.5-Coder-3B | 零样本基线 | 45% |

## 🐛 修复的 7 个静默 bug

1. 生成长度截断（256 vs 512）
2. 奖励顺序敏感 vs 评估不敏感
3. tokenizer pad/eos 不一致
4. 奖励黑客（stub SQL）
5. model.config.pad 保留旧值
6. **prompt 字符串 vs chat 格式**（根因）
7. 评估脚本多语句 SQL 崩溃

## 📁 结构

```
├── src/                     # 核心代码
│   ├── train_reasoning_grpo.py   GRPO训练（含三级/partial/原子奖励）
│   ├── reasoning_generator_agent.py  推理Agent
│   ├── evaluate_after_grpo.py  评估
│   └── atomic_ops.py       # FINER原子奖励
├── scripts/                # SLURM脚本 + pre-flight检查
├── records/                # 自动实验记录（jsonl）
├── charts/                 # 自动生成图表
├── FINAL_REPORT.md         # 最终汇报
├── EXPERIMENT_MATRIX.md    # 实验对比总表
└── PHASE2_PLAN.md          # 第二阶段方案
```

## 🚀 快速开始

```bash
# 训练前检查（16项自动）
bash scripts/preflight_check.sh

# GRPO 训练（三级奖励）
python src/train_reasoning_grpo.py \
    --num-train 100 --num-generations 8 --max-steps 25 \
    --reward-type three_level

# 评估
python src/evaluate_after_grpo.py \
    --lora-path checkpoints/xxx --limit 100 \
    --spider-dir data/spider_data --model-path models/xxx

# 记录实验 + 生成图表
python scripts/record_experiment.py --summary outputs/xxx/summary.json \
    --name xxx --charts
```

## 📊 实验记录

所有实验自动记录在 `records/experiments.jsonl`，图表在 `charts/`。

## 技术栈

- TRL 0.15.2 (GRPOTrainer) + LoRA (r=16)
- 多prompt投票推理增强
- FINER-SQL 原子奖励（sqlglot Jaccard）
- XJTLU HPC (A40/3090)

## 文档

- [最终汇报](FINAL_REPORT.md)
- [实验矩阵](EXPERIMENT_MATRIX.md)
- [实验规范](EXPERIMENT_STANDARDS.md)
- [GRPO检查清单](GRPO_CHECKLIST.md)
