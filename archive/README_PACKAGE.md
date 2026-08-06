# Text-to-SQL GRPO 项目完整打包

> 打包日期：2026-08-04
> 研究者：jiahuiwang24
> 平台：XJTLU HPC (A40/3090)

---

## 一、项目简介

用 **GRPO 强化学习**训练小模型（Qwen2.5-Coder-3B/7B）做 **Text-to-SQL**，在 Spider 数据集上对比训练前后效果。

**核心问题**：7B 零样本 81%，但所有 GRPO 训练都低于基线——最终定位到 **6 个配置 bug**，其中根因是 **prompt 格式错误**（TRL 对字符串 prompt 不应用 chat template，导致模型生成 2 token 就停止）。

## 二、核心结果

| 模型/方法 | Match | 备注 |
|-----------|:---:|------|
| **Qwen2.5-Coder-7B 零样本** | **81%** | 🏆 最佳结果 |
| DeepSeek-Coder-V2-Lite | 78% | MoE 16B |
| Qwen2.5-Coder-14B | 78% | 零样本 |
| 7B GRPO E-A (G=8) | 77% | 修复后仍有 loss=0 问题 |
| 7B GRPO (旧配置) | 76-79% | 受 6 个 bug 影响 |
| 3B 零样本 | 37% | — |
| 3B GRPO G=4 (修复版) | 35% | 有梯度但需更多训练 |

## 三、发现的 6 个 bug（全部已修复）

| # | Bug | 症状 | 修复 |
|---|-----|------|------|
| 1 | max_completion_length=256 vs 推理 512 | 训练截断 | 统一 512 |
| 2 | 奖励顺序敏感 vs 评估不敏感 | 训练信号错位 | 复用 compare_execution_results |
| 3 | tokenizer pad≠eos | 生成提前停 | pad=eos |
| 4 | 奖励黑客 stub SQL | 模型生成 SELECT 1 | schema 表引用检查 |
| 5 | model.config.pad 保留旧值 | padding 当 EOS | config.pad=eos |
| 6 | **prompt 字符串 vs chat 格式** | **生成 2 token 就停** | **messages 列表** |

## 四、目录结构

```
reasoning_generator_3b/
├── README_PACKAGE.md        ← 本文件
├── PROJECT_CONTEXT.md       ← 项目设计文档
├── EXPERIMENT_LOG.md        ← 全部 20+ 实验记录
├── 3B_PROGRESS_REPORT.md    ← 3B 阶段汇报
├── 7B_PROGRESS_REPORT.md    ← 7B 阶段汇报
├── EXPERIMENT_REPORT.md     ← 首次实验报告
├── RESEARCH_PLAN.md         ← 研究计划（论文调研）
├── EXPERIMENT_STANDARDS.md  ← 实验规范标准
├── GRPO_CHECKLIST.md        ← 训练前检查清单
├── ALL_RESULTS.json         ← 全部实验结果汇总
├── src/                     ← 核心代码
│   ├── reasoning_generator_agent.py  推理 Agent
│   ├── train_reasoning_grpo.py       GRPO 训练
│   ├── train_sft.py                  SFT 训练
│   ├── evaluate_after_grpo.py        评估
│   ├── spider_utils.py               Spider 工具
│   └── ...
├── scripts/                 ← SLURM 脚本（24个）
│   ├── preflight_check.sh   训练前自动检查
│   └── exp_*.slurm          各实验
├── .claude/skills/
│   └── grpo-preflight/      ← 检查 skill
└── tmp/                     ← 调试工具
```

## 五、关键代码文件

| 文件 | 用途 |
|------|------|
| `src/train_reasoning_grpo.py` | GRPO 训练主脚本（含全部修复） |
| `src/reasoning_generator_agent.py` | 推理 Agent（chat template 正确应用） |
| `src/spider_utils.py` | Spider 数据加载 + SQL 执行 + 评估逻辑 |
| `src/evaluate_after_grpo.py` | 训练后评估 |
| `src/train_sft.py` | SFT 冷启动 |
| `scripts/preflight_check.sh` | 16 项自动配置检查 |
| `scripts/exp_ea_g8.slurm` | E-A 实验（G=8 组统计修复） |

## 六、复现方法

```bash
# 1. 环境
ssh jiahuiwang24@login.hpc.xjtlu.edu.cn
cd ~/reasoning_generator_3b

# 2. 训练前检查（16 项自动）
bash scripts/preflight_check.sh

# 3. 跑实验
sbatch scripts/exp_ea_g8.slurm    # E-A: G=8 组统计修复

# 4. 评估
python src/evaluate_after_grpo.py \
    --lora-path checkpoints/grpo_ea_g8 \
    --limit 100 --spider-dir data/spider_data \
    --model-path models/qwen2.5-coder-7b-instruct
```

## 七、马拉松调研成果

深挖 4 个开源项目（CSC-SQL/FINER-SQL/Reasoning-SQL/Agentar-Scale-SQL）：
- 全部使用 G≥6、temp=1.0、8k-20k 数据
- 我们的 G=2 + 100条 → 组内 advantage 塌缩为噪声
- 5 个实验方案（E-A~E-E）已设计，E-A/E-B/E-C 已实施
