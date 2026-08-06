# Text-to-SQL GRPO

用 **GRPO 强化学习**训练小模型（Qwen2.5-Coder 3B/7B）做 **Text-to-SQL**（自然语言 → SQL 查询），在 Spider 数据集上验证"训练 + 多 prompt 投票"的组合路径。

## 🏆 核心成果（全部经修复后代码验证，数字可信）

| 模型 | 方法 | Match | 增益 |
|------|------|:---:|:---:|
| **Qwen2.5-Coder-7B** | 零样本 + 3prompt 投票 | **85%** 🏆 | +4pp（零训练成本） |
| Qwen2.5-Coder-7B | 零样本基线 | 81% | — |
| **Qwen2.5-Coder-3B** | GRPO 25步(500条) + 5prompt 投票 | **71%** | **+26pp** |
| Qwen2.5-Coder-3B | GRPO 三级奖励 25步 | 50% | +5pp |
| Qwen2.5-Coder-3B | 零样本基线 | 45% | — |

## 🔑 关键发现

1. **训练 + 推理增强组合有效**：3B 从 45% → 训练(50%) → 5prompt 投票(71%)，累计 +26pp
2. **5p 是投票甜点**：视角数 3→5 提升 +5pp（65→70%），5→7 零提升（投票已收敛）
3. **投票鲁棒性**：训练数据量 100/500/2000/7000 条，投票后全部收敛 70~71%——投票抹平训练状态差异
4. **三级奖励是 3B 唯一有效奖励**：partial（行重叠分数）与原子（FINER）奖励均无提升
5. **25 步是训练甜点**：50/75 步过拟合（50% → 34%/36%）
6. **多prompt 视角 >> 随机采样**：85% vs 54%
7. **7B 投票上界 = 85%**：3p/5p/5p+仲裁 三种方案全部相同

## 📊 图表（charts/）

| 级别 | 文件 | 用途 |
|------|------|------|
| 汇报 | `experiment_summary_v2.png` | 一张图讲完 3B/7B 两条路径 |
| 汇报 | `training_reward_compare_v2.png` | 三级 vs partial 奖励训练对比 |
| 汇报 | `voting_curve_v2.png` | 投票视角数曲线（5p 甜点） |
| 分析 | `analysis_vote_agreement.png` | ⭐投票一致度 vs 正确率（有效性证据链） |
| 分析 | `analysis_data_volume.png` | 数据量 × 评估方式双线 |
| 分析 | `analysis_overfit.png` | checkpoint 过拟合曲线 |
| 分析 | `analysis_training_panel.png` | 训练过程 6 面板全景 |
| 分析 | `analysis_full_matrix.png` | 全部实验成绩分布 |

## 🗂️ 目录结构

```
├── src/                     # 核心代码
│   ├── train_reasoning_grpo.py    # GRPO训练（binary/three_level/partial/atomic奖励）
│   ├── reasoning_generator_agent.py  # 推理Agent（chat格式+注释剥离+quote-aware分号切分）
│   ├── evaluate_after_grpo.py     # 评估
│   ├── eval_5prompt_agent.py      # 通用多prompt投票评估（支持5/7视角、任意模型）
│   ├── atomic_ops.py              # FINER原子奖励（sqlglot Jaccard）
│   └── spider_utils.py            # Spider工具（执行匹配比较、checkpoint、summary）
├── scripts/                 # SLURM作业 + 工具
│   ├── preflight_check.sh         # 16项训练前检查
│   ├── record_experiment.py       # 实验自动记录（jsonl + 图表）
│   ├── plot_pretty_charts.py      # 汇报级图表
│   ├── plot_analysis_charts.py    # 分析级图表
│   └── exp_*.slurm / eval_*.slurm # 实验作业模板
├── records/                 # 实验自动记录（experiments.jsonl）
├── charts/                  # 图表输出
├── logs/                    # 训练日志存档（可复现曲线）
├── FINAL_REPORT.md          # 最终汇报（含全部结论与证据）
├── EXPERIMENT_MATRIX.md     # 实验对比总表
├── HANDOFF.md               # 会话交接文档（含监控重建方法）
└── GRPO_CHECKLIST.md        # 训练前检查清单
```

## 🚀 快速开始

```bash
# 训练前检查（16项自动）
bash scripts/preflight_check.sh

# GRPO 训练（三级奖励，500条，G4，25步）
python src/train_reasoning_grpo.py \
    --num-train 500 --num-generations 4 --max-steps 25 \
    --reward-type three_level --train-batch-size 8 \
    --output-dir checkpoints/xxx

# 评估（单prompt）
python src/evaluate_after_grpo.py \
    --lora-path checkpoints/xxx --limit 100

# 5prompt 投票评估（任意模型/数据量）
python src/eval_5prompt_agent.py \
    --lora-path checkpoints/xxx --output-dir outputs/eval_5p_xxx --n-prompts 5

# 记录实验 + 生成图表
python scripts/record_experiment.py --summary outputs/xxx/summary.json \
    --name xxx --charts
```

## 🔬 技术栈

- TRL 0.15.2 (GRPOTrainer) + LoRA (r=16, α=32)
- 多 prompt 投票推理增强（AgentScale-SQL 思想）
- 执行结果匹配评估（与 Spider test-suite 同思路）
- XJTLU HPC（A40 48GB / 3090 24GB，SLURM 调度）
- 自动监控：Claude cron 每 30 分钟检查作业/记录/推送（见 HANDOFF.md）

## 📄 文档索引

- [FINAL_REPORT.md](FINAL_REPORT.md) — 最终汇报（方法、结果、工程、结论）
- [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) — 30+ 实验对比总表
- [HANDOFF.md](HANDOFF.md) — 会话交接 + 自动监控重建方法
- [GRPO_CHECKLIST.md](GRPO_CHECKLIST.md) — 训练前检查清单
