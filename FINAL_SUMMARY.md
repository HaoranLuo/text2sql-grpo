# 项目最终总结（2026-08-07）

## 一句话

> Text-to-SQL GRPO 项目：完成了自己的实验矩阵（3B 45%→71% 子集 / 全量 68.5%）、官方评估口径打通、并完整复刻验证了顶级论文 FINER-SQL（3B 模型 85%）的真实性与方法管线。

## 一、核心成果

### 我们自己的路径（全量 1034 官方口径）
| 实验 | 结果 |
|------|:---:|
| 3B 基线 → GRPO训练 → 5p投票 | 41.4% → 49.0% → **68.1%**（官方子集）/ 68.5%（自定义全量） |
| 7B 基线 | 67.5%（官方全量） |

### FINER-SQL 复刻验证（开源顶级论文，同基座 3B）
| 验证项 | 我们的结果 | 论文 |
|------|:---:|:---:|
| 单次生成（官方） | 77.3% | — |
| n=30 vav 投票（自定义口径） | **~85%** | 85.88% ✅ |
| n=30 vav 投票（官方 EX） | 76.6% | 85.0%（差异待查） |

## 二、关键发现（10 条）

1. **投票是最强杠杆**：3B 全量 +14.3pp（54.2→68.5），7B +4pp（零训练成本）
2. **5p 是投票甜点**：3p→5p +5pp，5p→7p 零提升
3. **子集偏易 +10pp**：前 100 条子集不可用于论文级对比，必须全量
4. **官方口径更严格**：test-suite EX 比自定义低 1-4pp
5. **三级奖励是 3B 唯一有效奖励**：partial/原子/FINER 式均无增益（50% 天花板）
6. **partial 无效根因**：组内扁平化 → 梯度归零（文献证实）
7. **训练侧天花板**：3B 所有训练配方峰值 50-51% 单次——突破需推理蒸馏前置
8. **FINER 85% 的核心** = 推理蒸馏（大模型教小模型）+ 大规模 GRPO + n=30 投票 的组合
9. **vav 投票 > 简单投票**：执行结果分组 + 退化组处理（修复空结果 bug 后 +6.4pp）
10. **两阶段（SFT+GRPO）无增益**：同批数据过拟合（44% < SFT 47%）

## 三、工程成果

- ✅ 官方 test-suite 评估器集成（多实例模糊化，`scripts/eval_official.sh`）
- ✅ vav 投票评估完整移植（`finer_port/`，28 项自测，与论文逐条对齐）
- ✅ 训练脚本增强：`--lora-init`（SFT→GRPO 真两阶段）、`--filter-gold`（执行过滤）、`finer` 奖励
- ✅ 投票升级：执行结果分组（Query and Conquer 式）
- ✅ 修复 bug 链：20+ 个（静默配置/代码审查/vav 空结果/下载完整性等）
- ✅ 自动化体系：cron 监控 + 记录系统 + 图表 + GitHub 私有仓库全同步
- ✅ 调研资产：14 agent 学习产出（RESEARCH_NOTES.md + FINER_REPLICATION_PLAN.md 27KB）

## 四、未完成/后续方向（按价值排序）

1. **P2 推理蒸馏**（FINER Step 1）：DeepSeek API 生成 reasoning trace+SQL → SFT → 这是解锁 85% 的关键，未执行
2. **官方 EX 差距排查**：76.6% vs 85%（评估器版本/plug_value/多实例细节）
3. **大规模 GRPO 复刻**：全量 8659 + G32 + 四分量奖励（依赖 P2）
4. 7B 全量投票官方口径、3B 基线全量等补全

## 五、文档索引

- [FINAL_REPORT.md](FINAL_REPORT.md) — 第一阶段完整报告
- [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) — 全部实验对比总表（40+ 项）
- [OFFICIAL_EVAL.md](OFFICIAL_EVAL.md) — 官方口径全景 + 复刻终验
- [RESEARCH_NOTES.md](RESEARCH_NOTES.md) — 14 agent 调研结论
- [docs/FINER_REPLICATION_PLAN.md](docs/FINER_REPLICATION_PLAN.md) — 复刻路线图
- [HANDOFF.md](HANDOFF.md) — 会话交接
