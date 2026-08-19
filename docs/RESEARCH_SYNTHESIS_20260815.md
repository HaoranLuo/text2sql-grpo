# MI-VAV 新颖性判定 + 实验改进路线图（2026-08-15 六路检索合成）

> 六路并行检索（3 路新颖性判定 + 3 路改进思路，共 ~140 条 query + 原文核验）的合成结论。
> 检索局限：纯 WebSearch/WebFetch + arXiv API；2026 年 6-8 月新预印本可能有遗漏；部分数字来自摘要级来源，写入论文前需取原文复核。

---

# 第一部分：新颖性判定

## 1.1 核心结论

**MI-VAV 的具体组合没有被占，但单点部件都有先例。** 论文必须把贡献锚定在组合上，不能声称单点机制全新。

| 部件 | 先例 |
|---|---|
| "按执行结果分组 + 最大组胜出" | **很早就有**：代码域 AlphaCode（Science 2022）/ CodeT（ICLR 2023）/ MBR-Exec（EMNLP 2022）；SQL 域 C3-SQL（2023）起，Maj/FMV 已成 2025-2026 标准基线（GradeSQL、CHASE-SQL 消融、Agentar consolidation 均有定义） |
| "多实例执行用于测试时选择" | 三篇近邻（机制均不同）：Li & Xie 2024（自造测试用例 + LLM 预测 gold 输出）、SISelection 2026（provenance 派生小实例成对决斗）、DPC ACL 2026（构造最小区分库 + SQL/Python 双范式） |
| **"官方 test-suite 全变体一致性用于测试时选择"** | **无先例**（两路独立检索交叉确认）——官方多实例库至今只用于最终评测 |

## 1.2 必须显式差异化的最近邻（按危险度排序）

| 论文 | 机制 | 与我们的差异（论文要写清） |
|---|---|---|
| **DPC**（ACL 2026 Main） | 构造 Minimal Distinguishing Database + SQL/Python 双范式验证 | 他们现场构造实例、要写 Python 对照解；我们用官方既有变体，零构造、与官方口径天然对齐 |
| **SISelection**（2026-05） | separating instances + provenance 成对决斗 | 面向 2-3 候选场景的对抗构造；我们是全变体包语义分组投票 |
| **R³-SQL**（ACL Findings 2026） | 执行分组 + 训练型组级排序 + 重采样 | 已占"执行分组+排序"发表位；单实例；训练型。它点名批评的"小组正确被大组错误压过"正是我们下一步要解决的问题（见路线图 T1-1） |
| **FINER-SQL vav**（ICDE 2026） | 单实例执行分组 + 值过滤启发式（跳空/全零组） | 我们的直接基线。**重要修正**：FINER 官方 85.0% 本身就含 n=30 采样 + vav 投票，不是单次解码；其 vav 是单实例。多实例一致性是我们可主张的增量。注：76.6% 是"我们的复现口径"不是 FINER 官方数，表述时严格区分 |
| Li & Xie 2024 | 自造测试用例 + LLM 预测 gold 期望输出 + 通过数打分 | 绝对判分非互分组；实例自造（约 60% 有错） |

## 1.3 可主张的差异化三点（related work 的 gap）

1. **实例来源**：官方 test-suite 既有变体（标准、可复现、零构造成本、与官方 TSA 口径对齐）——所有"构造实例"近邻都需要现场生成
2. **选择机制**：候选间跨实例互分组（agreement-based）——无需 LLM 预测 gold 输出、无需训练裁判、无需 provenance
3. **分组条件**：全称包语义（所有实例一致才同组）——比单实例等价类和"通过测试用例数"更严格
4. **附加**：3B 双 checkpoint 零训练集成；外部支持证据 SQLDriller（执行一致性 > LLM 一致性，后者在 3 个模型上甚至为负）

## 1.4 对比矩阵

| 方法 | 选择机制 | 执行实例 | 规模 | 训练型 | 官方口径 |
|---|---|---|---|---|---|
| C3-SQL (2023) | 执行分组+MV | 单实例 | API 模型 | 否 | Spider dev |
| FINER-SQL (ICDE 26) | 执行分组+vav 值过滤 | 单实例 | 3B | 是(GRPO) | Spider 85.0% dev |
| CSC-SQL (2025) | top-2 组→merge-revision | 单实例 | 3B | 是(GRPO) | BIRD 65.28% dev |
| DeepEye-SQL (SIGMOD 26) | 执行聚类占比+LLM 裁决 | 单实例 | ~30B | 否 | BIRD 73.5% dev |
| R³-SQL (ACL 26 Findings) | 分组+排序奖励+重采样 | 单实例 | 未披露 | 是 | BIRD 75.03% dev |
| SIRIUS-SQL (2026) | 执行一致+形式对判断门控 | 单实例 | 32B | 是(RL) | BIRD 75.88% dev |
| DPC (ACL 26) | MDD+SQL/Python 一致 | 构造多实例 | 多 LLM | 否 | +2.2% vs SC |
| SISelection (2026) | separating instances | 构造多实例 | 任意 | 否 | BIRD 子集 |
| **MI-VAV（本方法）** | **全 test-suite 实例一致分组+最大组** | **官方 25-60 变体** | **3B×2 ckpt** | **否** | **Spider dev 74.3%** |

## 1.5 风险与投稿前动作

- "最大组胜出"会面对 R³-SQL/CHASE-SQL 的同一批评——论文要主动讨论（用我们自己的实例数阶梯数据回应：严格分组后该批评的空间已被压缩，仍需验证）
- 投稿前用 arXiv API 按 "execution consistency" / "distinguishing database" / "test suite selection" 复核 2026 年 6-8 月新预印本
- DPC、SISelection 建议精读原文后再写入 related work

---

# 第二部分：实验改进路线图

## 2.1 Tier 1：纯推理侧、零训练、一周内可出数（★最优先）

| # | 改进 | 依据 | 预期 | 说明 |
|---|---|---|---|---|
| **T1-1** | **组级软排序替换"最大组胜出"** | R³-SQL（小组正确被大组压过）；FINER 数据显示选择器吃掉 ~6pp | +1~3pt | **可在已有 1034 题候选池上离线重算**——候选库红利，零生成成本，几小时出数 |
| **T1-2** | **平票/近票门控二次裁决** | SIRIUS-SQL（confidence-gated，Stage 2 免 LLM） | +0.5~1.5pt | 仅在 top-2 组差距小/大量 singleton 时触发结构化裁决 |
| **T1-3** | **执行失败修复链** | 三路报告共识（MAC-SQL Refiner +4.63 BIRD；2606.29733 "near-free win"；Agentar 条件化修复） | +1~3pt | 胜者/候选执行失败 → 带错误信息 1-2 轮低温修复 → 再执行验证；失败样例才付费；**无执行反馈的 intrinsic 自纠错已被证伪（ICLR 2024），必须接地** |
| **T1-4** | **难度门控预算重分配** | BAP-SQL（在 FINER-4B 骨架上验证：省 4.5-5% token 还 +3.4/3.6pp） | 难题 +1~2pt / 同精度省 5-15% | 简单题少采、难题多采；零成本启发式（schema 宽度/JOIN 数）+ 离线校准曲线 |
| **T1-5** | **骨架检索 few-shot + 难题自适应提示** | 同模型实证（Qwen2.5-Coder-3B + CoT 蒸馏 + 检索 = Spider dev 86.5%，待核实原文） | hard/extra +3~6 | 题目骨架+SQL 骨架双向量检索例题；难题用分解式 CoT |

## 2.2 Tier 2：一次性训练/数据改造（中期）

| # | 改进 | 依据 | 预期 | 说明 |
|---|---|---|---|---|
| T2-1 | **自训 3B ORM**（GradeSQL 式） | 唯一有"超过执行投票"实证的方向（Spider +0.93~2.10，BIRD +2.91~4.33）；自动标注零人工 | 推测 2~5pt（我们缺口更大） | 用自产候选 + gold 执行结果自动打标训练。**与"3B 不能当 critic"约束的澄清**：文献共识是"零样本 critic 不可靠、执行标签微调后的判别器可靠"——约束应解释为零样本，T2-1 不受限 |
| T2-2 | **修复记忆固化回灌 SFT/few-shot** | Crystallization（验证过的修复 episode 回灌：held-out 首答 +4.34pt，覆盖 44.4% 在线修复收益） | +1~2pt 长期复利 | 把"错误 SQL → 执行反馈 → 正确 SQL"按 database 聚类回灌 |
| T2-3 | **RetrySQL 式纠错轨迹蒸馏 + 难题过采样** | RetrySQL（1.5B 上 Spider +3.93；[BACK] 回溯标注）；难题过采样（hard/extra +5% 相对） | +3~4pt | 蒸馏数据混入"错误→回溯→纠正"轨迹；需全参继续预训练（LoRA 无效） |

## 2.3 Tier 3：需 pilot / 较重（视 T1 结果决定）

- **组代表两两 PK**（Agentar 锦标赛 +1.82，但需证据接地版；32B RL 选择器不可照搬）
- **Join 深度分解**（SchemaScope：join 跳数 ≥3 是头号难题杀手，强制分解 h≤2；难题 +2~4）
- **蒸馏 schema linker**（DTS-SQL 式 +3~5 但训练/推理开销大，后置）

## 2.4 明确不做（检索提供了否定证据）

| 项 | 否定证据 |
|---|---|
| 高温采样扩多样性 | COLM 2510.10885：parallel scaling 收益中等/反降；2606.29733：self-consistency 仅 +0.13pp/5× token（与我们实验一致） |
| 零样本 3B 当 critic | Huang et al. ICLR 2024 + TrustSQL：无外部反馈的自我评判不可靠（与项目约束一致） |
| 无执行反馈的自纠错 | 2310.01798：无效甚至变差 |
| 检索增强/RAG | COLM 2510.10885：检索常起反作用（加剧错误逻辑）；2606.29733：96.5% 召回 linker 与不用无差 |
| STaR 自举于"5% 无解"题 | 模型采样不出正确轨迹；必须外部强模型合成 |

## 2.5 建议执行顺序（与 B2 并行）

```
现在（零生成成本）: T1-1 + T1-2 在已有候选池上离线重算 → 几小时出数
同时（3090）:      T1-3 修复链 + T1-4 难度门控（小脚本，pilot 100 题）
A40 链:           E3×4 → sft_v3 → 5p×sft_v3（已排队）
sft_v3 出数后:     B2（三模型池 MI-VAV）→ 目标 75-76%
T1 收益兑现后:     T2-1 ORM 训练（用 T1 的平票/败局样本做训练集重点）
持续:              T2-2 修复记忆固化（随 T1-3 每次修复自动积累）
```

## 2.6 顶层逻辑一句话

**执行分组是对的（MI-VAV 基底成立），但"最大组胜出 + 硬等价"已被 2025-2026 多条工作证伪为瓶颈；文献共识路径 = 组级软排序 + 门控二次裁决 + （执行标签微调过的）ORM 终裁，外加"修复链 + 难度门控"两个免费引擎。所有 T1 项都可以在现有候选库上离线验证，不存在 GPU 排队问题。**

---

## 附：关键来源（去重后 Top 15）

- FINER-SQL: arXiv:2605.03465 / github.com/thanhdath/finer-sql（vav 源码）
- DPC: arXiv:2604.15163（ACL 2026 Main）｜SISelection: arXiv:2605.12319
- R³-SQL: arXiv:2604.25325（ACL Findings 2026）｜SIRIUS-SQL: arXiv:2606.01246
- GradeSQL: arXiv:2509.01308 / 2606.30851｜JudgeSQL: arXiv:2510.15560
- Agentar-Scale-SQL: arXiv:2509.24403｜CHASE-SQL: arXiv:2410.01943
- MAC-SQL: arXiv:2312.11242｜AC-SQL: arXiv:2410.22082
- BAP-SQL: arXiv:2608.02876｜CA-SQL: arXiv:2605.08057
- Crystallization: arXiv:2608.07213｜RetrySQL: arXiv:2507.02529
- SQLDriller（执行一致性>LLM 一致性实证）｜Li & Xie: arXiv:2401.02115
- 负证据: 2310.01798（无反馈自纠错无效）/ 2510.10885（测试时策略系统对比）/ 2606.29733（小模型技巧消融）
