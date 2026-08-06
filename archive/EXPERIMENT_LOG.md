# 实验日志 — Text-to-SQL 推理生成器 GRPO

> **维护人**: jiahuiwang24
> **项目文档**: [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md) · [RESEARCH_PLAN.md](./RESEARCH_PLAN.md) · [EXPERIMENT_REPORT.md](./EXPERIMENT_REPORT.md) · [7B_PROGRESS_REPORT.md](./7B_PROGRESS_REPORT.md)
> **时间范围**: 2026-07-30 → 2026-08-03
> **日期说明**: 日期为尽力记录（SLURM 日志保存在 HPC 上，本地文件仅记录最终报告）。

---

## 0. 固定实验设置（适用于全部实验）

| 项目 | 值 |
|------|-----|
| 基准 | Spider，`dev.json` 前 100 条 |
| 训练数据来源 | Spider `train_spider.json`（前 N 条） |
| 评估协议 | 自定义：生成 SQL → 从 ```sql 代码块提取 → SQLite 执行 → 与金标准结果行比较（ORDER BY 感知） |
| 指标 | Parse（SQL 提取成功率）· Exec（预测 SQL 执行成功率）· Match（执行结果匹配率，**主指标**）。字符串精确匹配率亦跟踪（约 1%），不作主指标 |
| 生成提示词 | 训练与推理使用相同的 11 条指令提示词（仅 schema，不含金标准 SQL，sqlite 方言） |
| 平台 | XJTLU HPC；环境 `reasoning3b`（Python 3.10.20, torch 2.5.1+cu124, transformers 4.48.3, trl 0.15.2, peft 0.14.0, datasets 3.2.0） |
| LoRA（全部运行） | r=16, alpha=32, dropout=0.05，目标为全部 7 个线性模块（q, k, v, o, gate, up, down） |
| GRPO 基础超参数 | 温度 0.7，最大提示 1536，最大生成 256，梯度检查点开启，批大小为 `num_generations` 的整数倍 |
| GRPO 算法 | TRL `GRPOTrainer`（无 critic，组内相对优势，KL 正则化） |

**GPU 与分区**

| 实验系列 | GPU | 分区 |
|---|---|---|
| 3B 阶段（E1–E2） | NVIDIA GeForce RTX 3090（24GB） | `gpudebug` |
| 3B 激进扩量 / 全部 7B 运行及评估（E3–E15） | NVIDIA A40（48GB） | `aiaca40`（QoS `1a40`） |

**奖励类型**

| 类型 | 打分 | 用途 |
|---|---|---|
| `binary`（二值） | 结果行匹配 1.0，否则 0.0 | 原 Phase-3 设计；信号非常稀疏 |
| `three_level`（三级） | 行匹配 1.0；可执行但错误 0.1；不可执行 0.0 | Agentar-Scale-SQL 方案；提供部分得分 |

---

## 1. 结果总览（Spider dev-100 执行结果匹配率）

| # | 实验 | Match | Δ | 状态 |
|---|---|---|:---:|:---:|
| E1 | 3B 零样本基线 | **37%** | —（基线） | 已完成 |
| E2 | 3B + GRPO 40 步（100 条 / 2 候选 / binary） | **39%** | +2.0 | 已完成 |
| E3 | 3B + GRPO 100 步（1000 条 / 4 候选 / binary） | **33%** | −4.0 | 已完成（过拟合） |
| E4 | 7B 零样本基线 | **81%** | +44.0（对比 E1） | 已完成（当前最佳） |
| E5 | 7B + GRPO 25 步（100 条 / 2 候选 / binary） | **79%** | −2.0 | 已完成 |
| E5b | 7B + GRPO 100 步（100 条 / 2 候选 / binary） | **79%** | −2.0 | 已完成 |
| E6 | 7B + GRPO 25 步（100 条 / 2 候选 / three_level） | **77%** | −4.0 | 已完成 |
| E7 | 7B + GRPO 50 步（100 条 / 2 候选 / three_level） | **79%** | −2.0 | 已完成 |
| E8 | 7B + GRPO 调参（lr=1e-6, β=0.1, 25 步） | **79%** | −2.0 | 已完成 |
| E9 | CSC-SQL-7B（零样本） | **73%** | −8.0 | 已完成 |
| E10 | 7B + SFT（DeepSeek API，500 条） | **71%** | −10.0 | 已完成 |
| E11 | 7B + SFT + GRPO 25 步 | **78%** | +7.0（对比 E10） | 已完成 |
| E12 | 7B + SFT + GRPO 50 步 | **78%** | +7.0（对比 E10） | 已完成 |
| E13 | XiYanSQL-7B（零样本） | **44%** | −37.0 | 已完成（格式不匹配，结果不可比） |
| E14 | OmniSQL-7B（零样本） | **66%** | −15.0 | 已完成 |
| E15 | DeepSeek-Coder-V2-Lite（零样本） | 待定 | 待定 | 🔄 运行中 |

**一句话总结**: 没有任何微调方案超过 7B 零样本基线（**81%**）。7B 上的所有 RL/SFT 实验要么退化，要么仅恢复至基线附近——小数据（100 条）、二值奖励与 SFT 数据质量是疑似根因。

---

## 2. 按时间顺序的实验日志

> 说明：训练后 Parse/Exec 未单独存档的实验以"未记录"标注（与 7B_PROGRESS_REPORT.md 一致）；零样本评估无训练前后对比，"训练前"列为 N/A（无训练）。

### E1 — 3B 零样本基线

| 项目 | 值 |
|------|-----|
| 日期 | 2026-07-30 |
| 模型 | Qwen2.5-Coder-3B-Instruct |
| 基座 | Qwen2.5-Coder-3B-Instruct |
| 训练方式 | 预训练模型直接评估（Zero-shot） |
| 训练步数 | 0（未训练） |
| 训练数据量 | N/A（未训练） |
| 每步候选数 | N/A |
| 奖励类型 | N/A |
| 学习率 | N/A |
| Beta (KL) | N/A |
| 温度 | N/A |
| GPU | RTX 3090 |
| 分区 | gpudebug |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | N/A（无训练） | 98% | —（基线） |
| SQL执行成功率 (Exec) | N/A（无训练） | 59% | —（基线） |
| 执行结果匹配率 (Match) | N/A（无训练） | **37%** | —（基线） |

**分析**: 3B 基座零样本在 Spider dev-100 上匹配率仅 **37%**，SQL 提取成功率 98% 但执行成功率仅 59%，约四成生成 SQL 存在语法或执行错误。该结果确立了 3B 阶段的性能下限，是 E2/E3 的对比基线。

**证据文件**: `outputs/baseline_pretrain_100/`（`summary.json`）

---

### E2 — 3B + GRPO 40 步（二值奖励）

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-01 |
| 模型 | Qwen2.5-Coder-3B-Instruct + LoRA |
| 基座 | Qwen2.5-Coder-3B-Instruct |
| 训练方式 | GRPO |
| 训练步数 | 40 steps |
| 训练数据量 | 100 examples from train_spider.json |
| 每步候选数 | 2 candidates/group |
| 奖励类型 | binary |
| 学习率 | 5e-6 |
| Beta (KL) | 0.04 |
| 温度 | 0.7 |
| GPU | RTX 3090 |
| 分区 | gpudebug |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | 98%（E1 基线） | 98% | 0 |
| SQL执行成功率 (Exec) | 59%（E1 基线） | 64% | +5.0 |
| 执行结果匹配率 (Match) | 37%（E1 基线） | **39%** | +2.0 |

**分析**: 首个正向信号：GRPO 先教会模型生成语法合法的 SQL（执行成功率 +5.0%），再提升正确性（匹配率 **39%**，+2.0%）。训练动态健康（KL 0.0014→0.0004，奖励标准差 0.42→0.14，梯度范数 27.1→0.013），未出现过拟合；40 步不足以将可执行的 SQL 提升为正确的 SQL。

**证据文件**: `scripts/train_grpo.slurm`, `outputs/baseline_pretrain_100/`, `EXPERIMENT_REPORT.md`

---

### E3 — 3B + GRPO 100 步激进扩量（二值奖励）

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-01 → 2026-08-02 |
| 模型 | Qwen2.5-Coder-3B-Instruct + LoRA |
| 基座 | Qwen2.5-Coder-3B-Instruct |
| 训练方式 | GRPO |
| 训练步数 | 100 steps |
| 训练数据量 | 1000 examples from train_spider.json |
| 每步候选数 | 4 candidates/group |
| 奖励类型 | binary |
| 学习率 | 5e-6 |
| Beta (KL) | 0.04 |
| 温度 | 0.7 |
| GPU | A40 |
| 分区 | aiaca40 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | 98%（E1 基线） | 未记录 | — |
| SQL执行成功率 (Exec) | 59%（E1 基线） | 未记录 | — |
| 执行结果匹配率 (Match) | 37%（E1 基线） | **33%** | −4.0 |

**分析**: 二值稀疏奖励配合更大的数据量与步数导致过拟合：模型记忆训练查询，泛化能力反而下降（匹配率 **33%**，−4.0%）。该结果与文献共识一致——稀疏二值奖励是根因，直接催生了三级奖励方案（P0）。

**证据文件**: `scripts/train_3b_aggressive.slurm`, `outputs/grpo_3b_aggressive_100/`

---

### E4 — 7B 零样本基线

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-02（约） |
| 模型 | Qwen2.5-Coder-7B-Instruct |
| 基座 | Qwen2.5-Coder-7B-Instruct |
| 训练方式 | 预训练模型直接评估（Zero-shot） |
| 训练步数 | 0（未训练） |
| 训练数据量 | N/A（未训练） |
| 每步候选数 | N/A |
| 奖励类型 | N/A |
| 学习率 | N/A |
| Beta (KL) | N/A |
| 温度 | N/A |
| GPU | A40 |
| 分区 | aiaca40 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | N/A（无训练） | 100% | —（基线） |
| SQL执行成功率 (Exec) | N/A（无训练） | 97% | —（基线） |
| 执行结果匹配率 (Match) | N/A（无训练） | **81%** | —（基线） |

**分析**: 规模主导一切：7B 零样本匹配率 **81%**，较 3B 零样本（37%）提升 44 个百分点，超过本项目测得的任何微调增益。7B 生成的 SQL 几乎全部合法（执行成功率 97%），剩余 16 个百分点的差距位于语义层（错表、错列、错 JOIN），这正是 RL 应当解决的推理部分。该模型至今仍是项目最佳模型。

**证据文件**: `scripts/train_7b.slurm`（Step 1 基线）, `outputs/baseline_7b_100/`

---

### E5 — 7B + GRPO 25 步（二值奖励）

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-02（约） |
| 模型 | Qwen2.5-Coder-7B-Instruct + LoRA |
| 基座 | Qwen2.5-Coder-7B-Instruct |
| 训练方式 | GRPO |
| 训练步数 | 25 steps |
| 训练数据量 | 100 examples from train_spider.json |
| 每步候选数 | 2 candidates/group |
| 奖励类型 | binary |
| 学习率 | 5e-6 |
| Beta (KL) | 0.04 |
| 温度 | 0.7 |
| GPU | A40 |
| 分区 | aiaca40 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | 100%（E4 基线） | 未记录 | — |
| SQL执行成功率 (Exec) | 97%（E4 基线） | 未记录 | — |
| 执行结果匹配率 (Match) | 81%（E4 基线） | **79%** | −2.0 |

**分析**: 与 E3 相同的二值奖励退化模式在强基座上复现：RL 将策略推离其强先验，正确性无净增益（匹配率 **79%**，−2.0%）。证实"二值奖励在小数据上导致过拟合"的诊断与模型规模无关。

**证据文件**: `scripts/exp_reward.slurm`（Group B）, `outputs/grpo_7b_binary_25/`

---

### E5b — 7B + GRPO 100 步（二值奖励，train_7b.slurm 运行）

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-02（约） |
| 模型 | Qwen2.5-Coder-7B-Instruct + LoRA |
| 基座 | Qwen2.5-Coder-7B-Instruct |
| 训练方式 | GRPO |
| 训练步数 | 100 steps |
| 训练数据量 | 100 examples from train_spider.json |
| 每步候选数 | 2 candidates/group |
| 奖励类型 | binary |
| 学习率 | 5e-6 |
| Beta (KL) | 0.04 |
| 温度 | 0.7 |
| GPU | A40 |
| 分区 | aiaca40 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | 100%（E4 基线） | 未记录 | — |
| SQL执行成功率 (Exec) | 97%（E4 基线） | 未记录 | — |
| 执行结果匹配率 (Match) | 81%（E4 基线） | **79%** | −2.0 |

**分析**: 将二值奖励训练从 25 步延长至 100 步，结果与 E5 完全一致（匹配率 **79%**，−2.0%），说明在二值奖励 + 100 条样本下步数不是限制因素。该运行与 E5 相互印证：性能退化由奖励信号稀疏性驱动，而非训练时长。

**证据文件**: `scripts/train_7b.slurm`, `outputs/grpo_7b_100/`

---

### E6 — 7B + GRPO 25 步（三级奖励）

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-02（约） |
| 模型 | Qwen2.5-Coder-7B-Instruct + LoRA |
| 基座 | Qwen2.5-Coder-7B-Instruct |
| 训练方式 | GRPO |
| 训练步数 | 25 steps（checkpoint-25） |
| 训练数据量 | 100 examples from train_spider.json |
| 每步候选数 | 2 candidates/group |
| 奖励类型 | three_level |
| 学习率 | 5e-6 |
| Beta (KL) | 0.04 |
| 温度 | 0.7 |
| GPU | A40 |
| 分区 | aiaca40 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | 100%（E4 基线） | 未记录 | — |
| SQL执行成功率 (Exec) | 97%（E4 基线） | 未记录 | — |
| 执行结果匹配率 (Match) | 81%（E4 基线） | **77%** | −4.0 |

**分析**: 三级奖励（Agentar-Scale-SQL 的 P0 方案，1.0/0.1/0.0）未能在 100 条样本上拯救 7B：25 步的短训练矫正不足（匹配率 **77%**，−4.0%）。在数据极小时，密集奖励信号本身不足以带来收益。

**证据文件**: `scripts/exp_reward.slurm`（Group A checkpoint-25）, `outputs/grpo_7b_threelevel_25/`

---

### E7 — 7B + GRPO 50 步（三级奖励）

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-02（约） |
| 模型 | Qwen2.5-Coder-7B-Instruct + LoRA |
| 基座 | Qwen2.5-Coder-7B-Instruct |
| 训练方式 | GRPO |
| 训练步数 | 50 steps（final） |
| 训练数据量 | 100 examples from train_spider.json |
| 每步候选数 | 2 candidates/group |
| 奖励类型 | three_level |
| 学习率 | 5e-6 |
| Beta (KL) | 0.04 |
| 温度 | 0.7 |
| GPU | A40 |
| 分区 | aiaca40 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | 100%（E4 基线） | 未记录 | — |
| SQL执行成功率 (Exec) | 97%（E4 基线） | 未记录 | — |
| 执行结果匹配率 (Match) | 81%（E4 基线） | **79%** | −2.0 |

**分析**: 50 步优于 25 步（**79%** vs 77%），但仍低于基线（81%）。同一步数下三级奖励较二值奖励仅微弱恢复（E5/E5b 二值同为 79%），从未超过零样本基线——奖励密化解决了"学习崩塌"，但没有解决"超越起点"。

**证据文件**: `scripts/exp_reward.slurm`（Group A final）, `outputs/grpo_7b_threelevel_50/`

---

### E8 — 7B + GRPO 调参（lr=1e-6, β=0.1, 三级奖励）

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-02 → 2026-08-03 |
| 模型 | Qwen2.5-Coder-7B-Instruct + LoRA |
| 基座 | Qwen2.5-Coder-7B-Instruct |
| 训练方式 | GRPO |
| 训练步数 | 25 steps |
| 训练数据量 | 100 examples from train_spider.json |
| 每步候选数 | 2 candidates/group |
| 奖励类型 | three_level |
| 学习率 | 1e-6（为标准值的 1/5） |
| Beta (KL) | 0.10（为标准值的 2.5 倍） |
| 温度 | 0.7 |
| GPU | A40 |
| 分区 | aiaca40 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | 100%（E4 基线） | 未记录 | — |
| SQL执行成功率 (Exec) | 97%（E4 基线） | 未记录 | — |
| 执行结果匹配率 (Match) | 81%（E4 基线） | **79%** | −2.0 |

**分析**: 更强的 KL 正则化（β 0.04 → 0.10）与更低的学习率（5e-6 → 1e-6）在 100 条样本上没有带来任何变化（匹配率 **79%**，与未调参的 50 步运行相同）。这表明性能瓶颈是数据量与奖励信号密度，而非优化不稳定性或超参数漂移。

**证据文件**: `scripts/exp_grpo_tuned.slurm`, `outputs/grpo_7b_tuned_25/`

---

### E9 — CSC-SQL-7B 零样本评估（第三方 GRPO 预训练）

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-03（约） |
| 模型 | `cycloneboy/CscSQL-Grpo-Qwen2.5-Coder-7B-Instruct`（CSC-SQL-7B） |
| 基座 | Qwen2.5-Coder-7B-Instruct |
| 训练方式 | 预训练模型直接评估（Zero-shot） |
| 训练步数 | 0（未训练） |
| 训练数据量 | N/A（未训练） |
| 每步候选数 | N/A |
| 奖励类型 | N/A |
| 学习率 | N/A |
| Beta (KL) | N/A |
| 温度 | N/A |
| GPU | A40 |
| 分区 | aiaca40 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | N/A（无训练） | 未记录 | — |
| SQL执行成功率 (Exec) | N/A（无训练） | 未记录 | — |
| 执行结果匹配率 (Match) | N/A（无训练） | **73%** | −8.0（对比 E4） |

**分析**: 第三方 GRPO 预训练的 7B 在我们的提示词与评估协议下仍输给原版 Qwen2.5-Coder-7B 零样本（**73%** vs 81%，−8.0）。可能原因：其训练分布与我们的评估分布不一致，或我们的提示词模板未触发其调优行为。在自定义评测框架下，外部检查点不是越过基线的捷径。

**证据文件**: `scripts/eval_csc.slurm`, `tmp/download_csc.py`, `outputs/eval_csc_100/`

---

### E10 — 7B + SFT 冷启动（DeepSeek API，500 条）

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-03（约） |
| 模型 | Qwen2.5-Coder-7B-Instruct + LoRA（SFT） |
| 基座 | Qwen2.5-Coder-7B-Instruct |
| 训练方式 | SFT |
| 训练步数 | 2 epochs |
| 训练数据量 | 500 examples（DeepSeek API 生成，非 train_spider.json） |
| 每步候选数 | N/A（SFT 无候选组） |
| 奖励类型 | N/A（交叉熵损失） |
| 学习率 | 2e-5 |
| Beta (KL) | N/A |
| 温度 | N/A |
| GPU | A40 |
| 分区 | aiaca40 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | 100%（E4 基线） | 未记录 | — |
| SQL执行成功率 (Exec) | 97%（E4 基线） | 未记录 | — |
| 执行结果匹配率 (Match) | 81%（E4 基线） | **71%** | −10.0 |

**分析**: 对 API 生成数据的 SFT 主动伤害性能（匹配率 **71%**，−10.0%）：模仿 DeepSeek 的推理/SQL 格式将策略推离 7B 自身的强项，且 500 条未做正确性过滤的混合质量 API 样本劣于零样本。单独 SFT 不是可行的冷启动方案。

**证据文件**: `scripts/sft_grpo_pipeline.slurm`（Step 1）, `src/train_sft.py`, `scripts/run_api_gen.sh`, `outputs/eval_sft_100/`

---

### E11 — 7B + SFT + GRPO 25 步

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-03（约） |
| 模型 | Qwen2.5-Coder-7B-Instruct + SFT LoRA + GRPO LoRA |
| 基座 | Qwen2.5-Coder-7B-Instruct |
| 训练方式 | SFT + GRPO（在 SFT 模型上继续 GRPO） |
| 训练步数 | 25 steps（GRPO，checkpoint-25；SFT 阶段为 2 epochs） |
| 训练数据量 | 100 examples from train_spider.json（GRPO 阶段） |
| 每步候选数 | 2 candidates/group |
| 奖励类型 | three_level |
| 学习率 | 5e-6 |
| Beta (KL) | 0.04 |
| 温度 | 0.7 |
| GPU | A40 |
| 分区 | aiaca40 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | 未记录（E10 SFT 后） | 未记录 | — |
| SQL执行成功率 (Exec) | 未记录（E10 SFT 后） | 未记录 | — |
| 执行结果匹配率 (Match) | 71%（E10 SFT 后） | **78%** | +7.0 |

**分析**: GRPO 在 SFT 之后修复了大部分损伤：匹配率 **78%**（+7.0），接近零样本基线（81%）但仍差 3 个百分点。RL 作为修复机制有效，SFT 本身才是损伤来源。

**证据文件**: `scripts/sft_grpo_pipeline.slurm`, `outputs/eval_sft_grpo_25/`

---

### E12 — 7B + SFT + GRPO 50 步

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-03（约） |
| 模型 | Qwen2.5-Coder-7B-Instruct + SFT LoRA + GRPO LoRA |
| 基座 | Qwen2.5-Coder-7B-Instruct |
| 训练方式 | SFT + GRPO（在 SFT 模型上继续 GRPO） |
| 训练步数 | 50 steps（GRPO，final；SFT 阶段为 2 epochs） |
| 训练数据量 | 100 examples from train_spider.json（GRPO 阶段） |
| 每步候选数 | 2 candidates/group |
| 奖励类型 | three_level |
| 学习率 | 5e-6 |
| Beta (KL) | 0.04 |
| 温度 | 0.7 |
| GPU | A40 |
| 分区 | aiaca40 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | 未记录（E10 SFT 后） | 未记录 | — |
| SQL执行成功率 (Exec) | 未记录（E10 SFT 后） | 未记录 | — |
| 执行结果匹配率 (Match) | 71%（E10 SFT 后） | **78%** | +7.0 |

**分析**: 25 步与 50 步结果完全相同（匹配率 **78%**），出现平台期。SFT+GRPO 流水线整体不优于直接零样本（81%），在此数据规模下，流水线的复杂度没有换来收益。

**证据文件**: `scripts/sft_grpo_pipeline.slurm`, `outputs/eval_sft_grpo_50/`

---

### E13 — XiYanSQL-7B 零样本评估

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-03（约） |
| 模型 | `XGenerationLab/XiYanSQL-QwenCoder-7B-2502`（XiYanSQL-7B） |
| 基座 | Qwen2.5-Coder-7B-Instruct（QwenCoder 变体） |
| 训练方式 | 预训练模型直接评估（Zero-shot） |
| 训练步数 | 0（未训练） |
| 训练数据量 | N/A（未训练） |
| 每步候选数 | N/A |
| 奖励类型 | N/A |
| 学习率 | N/A |
| Beta (KL) | N/A |
| 温度 | N/A |
| GPU | A40 |
| 分区 | aiaca40 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | N/A（无训练） | 未记录 | — |
| SQL执行成功率 (Exec) | N/A（无训练） | 未记录 | — |
| 执行结果匹配率 (Match) | N/A（无训练） | **44%** | −37.0（对比 E4，不可比） |

**分析**: 格式不匹配导致结果不可比：XiYanSQL 使用自带的 schema 链接与输出格式，输出额外标记而非 ```sql 代码块，我们的解析器在大多数输出上失败。**44%** 主要反映解析失败而非模型质量，需完成提示词与格式适配后重新评估。

**证据文件**: `scripts/eval_all_models.slurm`, `tmp/dl_both.py`

---

### E14 — OmniSQL-7B 零样本评估

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-03（约） |
| 模型 | `seeklhy/OmniSQL-7B`（OmniSQL-7B） |
| 基座 | Qwen2.5-Coder-7B-Instruct |
| 训练方式 | 预训练模型直接评估（Zero-shot） |
| 训练步数 | 0（未训练） |
| 训练数据量 | N/A（未训练） |
| 每步候选数 | N/A |
| 奖励类型 | N/A |
| 学习率 | N/A |
| Beta (KL) | N/A |
| 温度 | N/A |
| GPU | A40 |
| 分区 | aiaca40 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | N/A（无训练） | 未记录 | — |
| SQL执行成功率 (Exec) | N/A（无训练） | 未记录 | — |
| 执行结果匹配率 (Match) | N/A（无训练） | **66%** | −15.0（对比 E4） |

**分析**: OmniSQL 在统一协议下的匹配率 **66%**，远高于 XiYanSQL（44%）但低于原版 Qwen 7B 零样本（81%）。与 E9 结论一致：外部 Text-to-SQL 检查点在我们的流水线中均输给基座模型。

**证据文件**: `scripts/eval_all_models.slurm`, `tmp/dl_both.py`

---

### E15 — DeepSeek-Coder-V2-Lite 零样本评估

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-03 |
| 模型 | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`（本地 `models/deepseek-coder-v2-lite`） |
| 基座 | DeepSeek-Coder-V2-Lite-Instruct（非 Qwen 基座） |
| 训练方式 | 预训练模型直接评估（Zero-shot） |
| 训练步数 | 0（未训练） |
| 训练数据量 | N/A（未训练） |
| 每步候选数 | N/A |
| 奖励类型 | N/A |
| 学习率 | N/A |
| Beta (KL) | N/A |
| 温度 | N/A |
| GPU | A40 |
| 分区 | aiaca40 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | N/A（无训练） | 待定（🔄 运行中） | 待定 |
| SQL执行成功率 (Exec) | N/A（无训练） | 待定（🔄 运行中） | 待定 |
| 执行结果匹配率 (Match) | N/A（无训练） | 待定（🔄 运行中） | 待定 |

**分析**: 该评估正在 HPC 上运行（`scripts/eval_dsv2.slurm`，`--batch-size 2`），用于补充外部非 Qwen 基座模型的参考点，结果待定。完成后将补全本表与第 1 节汇总表。

**证据文件**: `scripts/eval_dsv2.slurm`, `tmp/dl_dsv2.py`, `outputs/eval_dsv2_100/`

---

## 3. 超参数参考表（全部实验）

| 实验 | 训练方式 | 模型 | 步数 | 训练数据 | 候选/组 | 奖励 | lr | β | 温度 | GPU | 分区 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | Zero-shot | 3B | 0（未训练） | — | — | — | — | — | — | RTX 3090 | gpudebug |
| E2 | GRPO | 3B | 40 | 100 | 2 | binary | 5e-6 | 0.04 | 0.7 | RTX 3090 | gpudebug |
| E3 | GRPO | 3B | 100 | 1000 | 4 | binary | 5e-6 | 0.04 | 0.7 | A40 | aiaca40 |
| E4 | Zero-shot | 7B | 0（未训练） | — | — | — | — | — | — | A40 | aiaca40 |
| E5 | GRPO | 7B | 25 | 100 | 2 | binary | 5e-6 | 0.04 | 0.7 | A40 | aiaca40 |
| E5b | GRPO | 7B | 100 | 100 | 2 | binary | 5e-6 | 0.04 | 0.7 | A40 | aiaca40 |
| E6 | GRPO | 7B | 25 | 100 | 2 | three_level | 5e-6 | 0.04 | 0.7 | A40 | aiaca40 |
| E7 | GRPO | 7B | 50 | 100 | 2 | three_level | 5e-6 | 0.04 | 0.7 | A40 | aiaca40 |
| E8 | GRPO | 7B | 25 | 100 | 2 | three_level | **1e-6** | **0.10** | 0.7 | A40 | aiaca40 |
| E9 | Zero-shot | CSC-SQL-7B | 0（未训练） | — | — | — | — | — | — | A40 | aiaca40 |
| E10 | SFT | 7B | 2 epochs | 500（API 生成） | — | —（CE 损失） | 2e-5 | — | — | A40 | aiaca40 |
| E11 | SFT+GRPO | 7B+SFT | 25（GRPO） | 100 | 2 | three_level | 5e-6 | 0.04 | 0.7 | A40 | aiaca40 |
| E12 | SFT+GRPO | 7B+SFT | 50（GRPO） | 100 | 2 | three_level | 5e-6 | 0.04 | 0.7 | A40 | aiaca40 |
| E13 | Zero-shot | XiYanSQL-7B | 0（未训练） | — | — | — | — | — | — | A40 | aiaca40 |
| E14 | Zero-shot | OmniSQL-7B | 0（未训练） | — | — | — | — | — | — | A40 | aiaca40 |
| E15 | Zero-shot | DeepSeek-Coder-V2-Lite | 0（未训练） | — | — | — | — | — | — | A40 | aiaca40 |

通用配置：LoRA r=16 / alpha=32 / dropout=0.05 / 7 个目标模块；TRL 0.15.2 `GRPOTrainer`；最大提示 1536；最大生成 256；梯度检查点开启；bf16。表中"—"表示该行不适用。

---

## 4. 关键结论

1. **规模胜过微调**：3B → 7B 零样本跃升 +44 个百分点（37% → 81%），超过所有训练增益的总和。模型选择是主导杠杆，训练技巧（奖励工程、候选数、步数、正则）在容量面前是二阶因素。
2. **二值奖励在小数据上导致过拟合**：E3（−4.0）、E5（−2.0）、E5b（−2.0）均退化至基线以下，与文献共识一致。切换到三级奖励（E6–E8，77–79%）消除了崩塌式过拟合，但在 100 条样本下从未超过基线。
3. **100 条样本上的 RL 对强基座模型是净负**：所有 7B GRPO 运行均落在 77–79%，低于 81% 基线；E8 调参（lr 1e-6、β 0.1）结果不变，证明瓶颈是数据量与奖励信号密度，而非优化器设置或训练不稳定性。
4. **RL 是修复工具，不是冷启动**：SFT 造成 −10.0（E10，71%），GRPO 修复 +7.0（E11/E12，78%），但流水线总体不优于直接零样本（81%）。"先模仿后 RL"在此数据规模下收益有限。
5. **外部模型不构成捷径**：CSC-SQL（73%）、OmniSQL（66%）、XiYanSQL（44%，格式失配）在统一协议下均低于 7B 零样本基线（81%）；DeepSeek-Coder-V2-Lite（E15）结果待定。
6. **剩余空间在语义层**：7B 执行成功率 97% vs 匹配率 81%，中间 16 个百分点是语义错误（错表、错列、错 JOIN），即 RL 设计上应解决的"硬推理"部分——但 100 条样本的奖励信号不足以教会它。
7. **训练动态始终健康**：即使指标下降（E3、E5），KL 下降、奖励标准差下降、梯度范数趋近 0——优化器没有故障，问题出在奖励信号与数据上。

---

## 5. 状态汇总与下一步

- **当前最佳结果**: 7B 零样本基线 **81%**（E4），尚无方案超越。
- **实验状态**: E1–E14 共 15 项实验已完成（含 E5b）；E15（DeepSeek-Coder-V2-Lite 零样本）运行中，结果待补。
- **下一步计划（按 RESEARCH_PLAN 排序）**:
  - **P0**: 三级奖励在更大数据上重试（训练样本 100 → ≥500–1000 条），验证奖励信号密度假设；集成官方 Spider 评估器（`test-suite-sql-eval`，精确匹配 + 执行准确率）。
  - **P1**: 列级部分分奖励（PaVeRL-SQL 风格）；参考 FINER-SQL 引入原子操作奖励；高质量 SFT 数据（执行匹配过滤 + 教师模型更换为 DeepSeek V2 Lite）；为 XiYanSQL-7B 与 OmniSQL-7B 适配提示词/格式做公平对比。
  - **P2**: AutoLink 式 schema 链接（bge-large-en-v1.5 列检索）；多候选生成 + 执行投票（4–8 候选）；更大的数据切片（500 → 7000 训练行）后再给 RL 下结论。
  - **P3**: 过程奖励模型（RuCo-C 式 rubric）；Agentic GRPO（AGRO-SQL，POMDP + 数据工厂）。
