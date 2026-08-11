# IMPROVEMENT_PLAN：未来 1 个月改进计划（综合可落地清单 + 批判审稿意见）

> 生成日期：2026-08-11
> 输入：① 可落地综合（Top-10 ROI 清单）② 批判综合（8 路调研交叉验证与风险审稿）③ `NEXT_STEPS.md`（2026-08-07）
> 项目锚点（以 `FINAL_SUMMARY.md` 为准）：3B 训练侧单次天花板 50%（官方）/ 5p vav 投票全量 68.5%（自定义）/ 68.1%（官方子集）；FINER-3B 发布权重复刻 = 单次官方 77.3% / vav 自定义 ~85%（说明基座与评估器能到该水平，缺的是训练配方）；推理蒸馏未做；官方评估器已集成；finer-sql 本地克隆完整；HPC 双卡 A40。
> 本文档是 `NEXT_STEPS.md` 的升级版路线图：与其冲突处（见 §6），以本文档为准。

---

## 1. 摘要（5 条核心建议）

1. **推理蒸馏 → SFT 是突破 3B 单次 50% 天花板的唯一已证路径**（FINER Step 1，DeepSeek-R1 教师 + SQLite 执行过滤 + gold 混合 10-20%），本月主战役第一优先级；SFT 后设单次官方口径门槛，<55% 先查数据再进 GRPO。
2. **动任何蒸馏/训练之前，先校准评估器**：用 FINER 官方权重在项目 harness 上跑，区分"评估端 vs 训练端"8.4pp 差距（76.6% vs 85%），并钉死"对外只用全量 1034 官方 test-suite 口径"纪律。
3. **GRPO 只作为蒸馏后的第二步**，且必须配稠密非负奖励 + 大步数 + 训练集/SFT 不相交划分（5.5k/1.5k）；裸 GRPO、纯结果奖励 GRPO 已被四路独立证据证伪，重跑只会重复负结果。
4. **投票管线升级是零训练成本的最大杠杆叠加**（失败类型感知修复 + 加权投票 + 置信度过滤 + 采样温度多样性），纠错信号只来自执行结果，绝不让 3B 自己当 critic。
5. **换基座先零训练 A/B**：Qwen3-Coder-30B-A3B（MoE 仅 3.3B 激活）与 Qwen3-4B-Base 跑现有投票管线，显著为正再 fork；切勿直接换通用 Qwen3-8B（代码任务崩塌先例）。

---

## 2. 改进建议优先级矩阵（10 项，按 ROI 排序）

> 批次：第一批（W1）= #1/#2/#3/#5；第二批（W2-3）= #4/#6/#7；第三批（W3-4）= #8/#9；#10 视前序结果启动。
> 证据强度：强 = 多轨独立证据 + 项目实证收敛；中 = 单/双源但可低成本验证；中低 = 单源待验（引用/投入前必须 A/B 或溯源）。

| # | 改进项 | 批次 | 类型 | 预期收益（量化） | 成本 | 依赖 | 证据强度 | 关键风险点 |
|---|--------|:---:|------|------------------|------|------|:---:|-----------|
| 1 | 【P0】推理蒸馏 → SFT（FINER Step 1） | W1 | 全新方向（未跑过，已复刻验证论文的第一步） | 单次生成官方口径 50% → **60-70%**（STaR-SQL 对 3B 的锚点） | ¥20-50 API（**可能低估 5-10 倍，先 50 题实测**）+ 1 天 GPU | 无（本地三件套已就绪） | 强（T1/T3/T7 + 项目 FINAL_SUMMARY 发现 7 四方收敛） | C12 划分冲突；L10 成本低估；空结果/超时污染（Arctic 教训） |
| 2 | 【P0】官方 EX 口径差异排查（76.6% vs 85%） | W1 | 已有方向延续（工程收尾） | 消除或确认 8.4pp 口径差；后续所有数字的地基 | 半天 | 无 | 强（项目实证 8.4pp 差，含难度分档系统性低 3-19pp） | 85% 可能是子集/自定义口径，非全量单次（C2） |
| 3 | 【P0】数据清洗三件套 + GRPO/SFT 不相交划分 | W1 | 已有方向延续（E 系列负结果根因修复） | 修复奖励饱和（**极可能是 2000 条 GRPO 无增益、44%<47% 的根因**），使 #4 变"有增益" | 0.5-1 天 | 无（先于 #4） | 强（jenishk20 奖励饱和实证 + 项目 E 系列） | 划分方案必须在生成蒸馏数据之前定死（C12） |
| 4 | 【P1】蒸馏后大规模 FINER 配方 GRPO + 零代码算法升级 | W2-3 | 延续（FINER 复刻）；算法项（DAPO/Dr.GRPO）全新 | 单次 60-70% → **68-76%**（官方口径）；配合投票冲 80%+ | 2-5 天 GPU（双卡 A40） | #1 + #2 + #3 | 强（FINER 消融：atomic 奖励 −3.26% 有意义；但项目前置条件是梯度修复） | C3 扁平化失败重演；C7 版本核对（trl 0.15.2 无 scale_rewards）；C6 冗长偏置 |
| 5 | 【P1】投票管线升级：失败类型感知修复 + 加权投票 + 置信度过滤 | W1 | 已有方向延续（最强杠杆扩展） | 全量投票 68.5%（自定义）→ **71-74%**（Query and Conquer 7B +10pp，3B 锚点 +2-5pp） | 2-4 天 | 无（复用现有执行器） | 强（投票 = 项目已验证最大杠杆 +14.3pp；T4/T5 多源） | M4：采样多样性是前提（greedy 投票收益会消失）；纠错信号只能来自执行结果 |
| 6 | 【P1】工程可靠性包：vLLM 前缀缓存 + rlprobe + 确定性报告 + 监控恢复 | W2 | 已有方向延续（自动化体系重建） | 吞吐提升 + 防浪费（一次静默失败 ≈ 1-2 GPU 天，本包成本 < 1 天）；所有报告数字可追溯 | 1-2 天 | 无；注意 trl==0.15.2 字段核对 | 强（项目实证：监控已停摆、40+ 实验靠人工盯 stdout；TRL+Liger 静默失败已知坑） | C7：提交前跑 grpo-preflight 技能 |
| 7 | 【P1】执行验证合成数据增强（SQL-PaLM 式） | W2-3 | 全新方向（训练侧数据增强，此前只做解码侧） | 单次 +2-4pp（叠加蒸馏） | 1-2 天 | #1（蒸馏后模型） | 中（SQL-PaLM 经典路径 + DataGpt-SQL-7B 84.8→87.2；3B 无直接证据，警惕 Mnemosyne 离域稀释 −7.8pp 反例） | 域外低质数据稀释；必须执行验证双门槛 |
| 8 | 【P1】教师 prompt 检索增强 + 难题 rationalization（RFT 式） | W3-4 | 全新方向（#1 的增量优化，非独立实验） | 蒸馏数据难题覆盖与质量（中高复杂度题 +2-5pp；简单 86% vs 复杂 ~40% 断层是主要增长点） | 1-2 天 | #1 数据管线 | 中（RFT 证据在数学/大模型；SQL 域靠 SAC-SQL 动机数据） | 可与 #7 合并实现 |
| 9 | 【P2】Qwen3 基座 A/B：30B-A3B 推理线 + 4B-Base 训练线 | W3-4 | 全新方向（有风险，先 A/B 决策） | 推理线零训练破 85% 上限，或 4B 训练线 79%+（SQL-R1 社区复现 55.51→79.59，同显存档位） | A/B：0.5-1 天零训练；4B 线约 1 周 | 无（与 #1-5 并行独立） | 中低（C10/L3：Qwen3-4B 数字单源复现 + 污染风险；L4：DeepEye-SQL 零训练逼近 SOTA 张力大） | 切勿换通用 Qwen3-8B（ACM/PortKit 崩塌先例：Qwen2.5-Coder-7B 50.2% vs Qwen3-8B 2.8%） |
| 10 | 【P2】ORM 验证器替换启发式投票（GradeSQL/STaR-SQL 路线） | 视前序 | 全新方向 | 选择层 +2-4pp（GradeSQL：Spider +2.10 vs 执行 BoN）；叠加重采样可突破管线上限 | 1-2 周（数据 2-3 天 + LoRA 3-5 天 + 评估） | #2 + #5 | 中（GradeSQL 开源 7B/14B/32B ORM 可直接评估；R3-SQL 未开源、缺口检测思想待自实现） | 任何选择器救不了池外答案（R3-SQL 上限）；#1-4 未达预期可延后 |

**应避免的坑（3 个，执行纪律）**：
1. **裸 GRPO / 纯结果奖励 GRPO**——已被三个独立来源证伪（slmsql-grpo-v2、Sparks of Tabular Reasoning、Reward-SQL naive outcome-only），我们自己 E 系列（partial/原子奖励小配方无效、50% 天花板）是同一现象的本地复现。GRPO 只作为蒸馏后的第二步，且必须配稠密奖励 + 大步数 + 不相交数据。
2. **3B 自己当教师（自生成数据做 STaR/拒绝采样/DPO）**——师生差距铁律：jenishk20 自生成 CoT 拒绝采样 −20、on-policy CoT-DPO −8.2；我们两阶段 SFT+GRPO 44%<47% 同因。自蒸馏只可作免费对照（预期大概率负）；若真用必须混合 10-20% gold 防 collapse 且每轮从基座重启。
3. **偏好优化 + 低质跨域合成数据**——(a) 评估主路径是投票，采样多样性是命根：自我采样 DPO 后 BoN 胜率上升但贪婪输出无提升、困难题 −4.1pp；(b) 离域低质合成数据会稀释（Mnemosyne-3B 混合 40k 后 Spider −7.8pp）：SynSQL 只取域内高质量子集 + 执行过滤，不做 2.5M 全量；Spider 1.0 榜单天花板 ~91%，预算投向难度分层与稳健性而非纯数量。

---

## 3. 依赖路线图（先做什么后做什么）

### 3.1 硬性 gate 纪律（每级输出的门槛，不达标不进下一级）

- **G1 口径纪律**：对外汇报只用"全量 1034 官方 test-suite"口径；n=30 / 子集 / vav 自定义口径数字（如 FINAL_SUMMARY 中 ~85%）仅限内部诊断。
- **G2 SFT 门槛**：蒸馏 SFT 后单次生成官方口径 ≥55% 才进 GRPO（目标 60-70%）；<55% 停跑，排查教师质量 / 数据泄漏 / 过拟合——低于门槛进 GRPO 是上一次失败（44%<47%）的复刻路径。
- **G3 数据划分纪律**：GRPO 训练集与 SFT/蒸馏数据不相交（5.5k / 1.5k 显式划分，混合比例 10% / 25% 两档小扫）；所有数据文件记 sha256，防 "Tuesday/Thursday" 漂移。
- **G4 提交前检查**：每次 GRPO/RL 训练提交前跑 `grpo-preflight` 技能（生成长度一致性、奖励/评估逻辑错位、数据泄漏、版本漂移）。
- **G5 数字引用纪律**：任何外部数字入文档前查 §5"待核"清单；无出处的数字（如 C9 的 3B CoT+检索 86.5%）不得引用。

### 3.2 时间线（四批并行，双卡 A40 + API）

**第一批（W1，约 1 周）——全部 ≤1 天、相互独立可并行，是后续 gate 前提：**

```
W1 并行启动：
  #2  评估器校准 + FINER 官方权重探针（M1）      —— 半天，最高优先
  #3  数据卫生 + 5.5k/1.5k 不相交划分决策         —— 0.5-1 天，先于任何蒸馏数据生成
  #9a Qwen3-30B-A3B + 4B-Base 零训练 A/B（投票管线）—— 0.5-1 天，同时回答"换基座 fork"与"本地免费教师"两问
  #6  工程恢复（cron 监控 + rlprobe 冒烟 + TRL 版本能力核对）—— 1-2 天
  L10 API 成本实测（50 题 token 账单）            —— 0.5 天
  #5  投票升级并行（复用现有 checkpoint，零训练，立即推 71% 上限）
W1 主战役：
  #1  蒸馏 SFT：API 生成 2h（后台，不占 GPU）→ 执行过滤 30min（SELECT read_only /
      非 SELECT sandbox_rollback，加空结果/超时剔除）→ SFT 40min → 官方口径评估 1h
      → G2 门槛检查（单次 ≥55%，目标 60-70%）
```

**第二批（W2-3，约 2 周）：**

```
  #4  大规模 GRPO（依赖 #1/#2/#3 达标 + #6 就绪）：
      不相交 5.5k 全量 + G16-32、lr 5e-6、1000-2000 步、每 100 步存 checkpoint + dev 早停
      四分量奖励（format + exec + atomic 操作级 Jaccard + memory[需 embedding+ChromaDB，
      暂缺可用前三分量]）+ 隐藏 CoT 选项（注意 C4：与 FINER 路线二选一，勿混用）
      梯度修复先行：loss_type="dapo"（token 级损失）+ scale_rewards=None（去 std）
      + 静默组过滤（对应 diag_zero_adv 诊断的零 advantage 组）+ beta ∈ {0,0.02,0.04} 小 sweep
      若重演"训练倒退"：先做配方审计（上次失败缺的正是"不相交 + 非负稠密 + 组内方差"三样）；
      BREAD 专家锚点分支（1-2 周）是最后手段不是首选，先记录为 fallback
  #6  工程包落地：vLLM V1（schema 前缀缓存省 60-80% prefill）、record_experiment.py 输出
      latest.json、StatGuard 式 No-Invent 报告纪律、slurm --requeue + SIGUSR1 checkpoint handler
  #7  执行验证合成增强（依赖 #1 模型）：每 gold 生成 1-3 替代 SQL，执行验证 + 相似度阈值双验证
      → 1.5-2x 数据量；只取 SynSQL 域内高质量子集
```

**第三批（W3-4，约 2 周）：**

```
  #8  教师检索增强（gold SQL cosine 检索最相似已验证 CoT 作 few-shot）+ 难题 rationalization
      （gold SQL 做 hint 反向生成推理，治 tail narrowing）——可与 #7 合并实现
  #9b Qwen3-4B-Base 训练线（仅当 #9a A/B 显著为正且论文主线允许换基座）；
      否则只保留 #9a 结果作支撑证据
  备选：BIRD 迁移评估（评估适配 1 天 + 重训 2-3 天，3B 预期 dev 60-70%）——
      与 #1-4 竞争同一份 API/GPU 预算，建议排下月；迁移前先用 SAR-Agent/GBV 类工具
      筛查 BIRD 标注错误（Mini-Dev 标注错误率 52.8%，错误 gold 会污染蒸馏数据）
```

**第四阶段（#10，视前序结果）：** ORM 验证器（LoRA 微调 7B 做二分类，Yes 概率排序选最高；可选叠加 R3-SQL 式"正确 SQL 缺失检测 → 定向重采样"）。依赖 #2 基线 + #5 候选池；若 #1-4 未达预期可延后。

**明确不做/延后**（多轨共识）：PRM 过程奖励、多智能体 RL（MARS-SQL 全套）、投机解码、MLflow/W&B、Spider 2.0、LLM-judge 奖励（除非已有可用 judge）。

---

## 4. 风险与坑（批判综合输出）

### 4.1 冲突清单与裁决（C1-C12）

| # | 冲突 | 裁决 / 动作 |
|---|------|-------------|
| C1【高危】 | STaR 自蒸馏：T1 说"免费有效" vs T3 说"必然退化"（自生成 CoT 拒绝采样 −20） | **STaR 从主链降为对照实验**；主链用强教师蒸馏（DeepSeek-R1 + gpt-4.1-mini/gpt-oss-120b 双教师池）；若用自蒸馏必须四前置（执行匹配过滤、gold-hint、每轮从基座重启、教师显著强于学生） |
| C2【高危】 | FINER 85% 数字口径三处打架（单次全量？n=30 子集+vav？官方复刻仅 76.6%） | **85% 只能当上限目标，不是可达基线**；先用 FINER 官方权重在项目 harness 上跑（M1）区分评估端/训练端；若 85% 真是全量单次，则"投票仅 +0.88pp"与项目经验矛盾，需复核 |
| C3【高危】 | "partial/原子奖励无效"（项目实证）vs PaVeRL 列级部分匹配推荐 | T4 推荐**缺少两个前置**：去 std 归一（scale_rewards=None）或 DAPO 静默组过滤、以及非负组合。前置未满足前，PaVeRL 改动降级为"梯度修复之后的 A/B"，不能作第一轮推荐 |
| C4【中危】 | SQL-R1"RL 隐藏 CoT" vs FINER"memory 奖励显式对齐已验证推理"（相反的训练信号） | 已选定 FINER 路线，**保持内部一致性，勿混入 SQL-R1 隐藏 CoT**；隐藏 CoT 仅记录为备选 |
| C5【中危】 | SQL-R1 正负四层奖励（±1/±2/±3）vs 非负奖励共识 + 项目 0.1 档伤害实证 | 若参考格式惩罚用小幅负分并加监控；**不整段照搬四层符号设计** |
| C6【中危】 | DAPO token 级损失 vs Dr.GRPO 去长度归一（MAD-GRPO 批评 + 2026 不可能定理） | "loss_type=dapo + scale_rewards=None"可叠加，但接受冗长偏置（长正确回答获更大正更新）；项目 max_completion_length=512 较短、风险可控——是取舍不是免费午餐 |
| C7【中危】 | "一行切换"承诺 vs pin trl==0.15.2 | **实际 TRL 默认 loss_type 是 "grpo"，dapo 需显式配置**；0.15.2 可能无 scale_rewards（RESEARCH_NOTES 记录 v0.16+ 才有）；完整 DAPO（dynamic sampling、Clip-Higher、Overlong 惩罚）无一行全开。先核对可用选项，排期按 1 天+ |
| C8【中危】 | "Spider 饱和应立即迁 BIRD" vs "Spider 蒸馏 P0 优先" | 设 gate：先跑 Spider P0（管线验证 + 可比数字），达标后转 BIRD；**不在 Spider 无限投入**（7000 条已近收益递减点）；BIRD 迁移管线 90% 复用 |
| C9【高危】 | T7 的"3B CoT+检索 Spider-dev 86.5%"与项目"3B 训练+投票峰值 71%"直接矛盾 | 该数字无出处、与轨道内其他数字冲突。**溯源核实后再引用，无果则删**（G5） |
| C10【中危】 | Qwen3-4B-Base 原生 55.51%（单社区复现）vs 项目 Qwen2.5-Coder-3B 同口径 41.4-45% | 同显存档位差 10-14pp，若成立换基座收益巨大；但单源、可能有数据污染（Qwen3 训练语料含 Spider 家族）、Qwen3-8B 崩塌前例。**先零训练 A/B（#9a），再决定 fork，切勿直接重训** |
| C11【中低】 | "LLM 自校验可行"（数学域/强验证器）vs "LLM 自批评系统性失败"（SQL 域） | **3B 线不采用自批评/自校验机制**；批评角色必须强于生成角色（7B/强 LLM/执行结果），或为确定性信号 |
| C12【低危】 | 蒸馏 gold 混合比例口径不一（10-20% vs 1:1~1:2）且与 GRPO 不相交冲突 | **显式划分 5.5k（GRPO 集）/ 1.5k（SFT gold 混合来源）**；混合比例做 10%/25% 两档小扫 |

### 4.2 低置信发现清单（按风险排序，引用/投入前必核）

| # | 发现 | 证据强度问题 | 验证/降险动作 |
|---|------|-------------|---------------|
| L1 | FINER 85.0/85.88/Recall@30=91.3 | 口径三重不明（C2），项目复刻差 8.4pp | 官方权重跑项目 harness（M1）；未验证前不作目标基线 |
| L2 | STaR-SQL +20pp | 与 T3 的 −20 反例矛盾；8B 结果 ≠ 3B；项目实验 A 未进关键发现 | 只做对照；执行过滤 + gold-hint + 基座重启四前置 |
| L3 | Qwen3-4B-Base 55.51→77.27→79.59 | 单一社区仓库；79.59 是否官方 test-suite 口径不明；污染风险 | 零训练 A/B + 官方口径复验（#9a） |
| L4 | DeepEye-SQL 零训练 BIRD-Test 75.07% / Spider-Test 89.8% | 零训练逼近训练管线 SOTA，张力大；LocalSQLAgent 仅 n=50 | 直接 A/B，别先信 |
| L5 | SIRIUS-SQL Spider-test 91.20% | 2026 新论文、贴近 91% 天花板 | 引用前读全文；已核 arXiv 2606.01246 |
| L6 | OPSD/PAINT 4-8x token 效率 | 2026 新范式、无 SQL 域验证、无 TRL 实现、需自写训练循环 | 观望，不进 P0/P1 |
| L7 | Self-Verified Distillation +16.7 AIME | 新论文、二手摘要、数学域、与 C11 冲突 | 仅保留"AND 门过滤思想"，SQL 域不实施 |
| L8 | T7 的"3B CoT+检索 86.5%" | 无出处、与项目数字矛盾（C9） | 溯源；无果则删 |
| L9 | GBV-SQL"修正 gold 后 96.5/97.6" | 修正标注后的数字不可入榜比较 | 只取"Spider 存在 gold 标注错误"结论（真实且有价值），用于 #3 与蒸馏过滤审计（M5） |
| L10 | 蒸馏 API 成本 ¥20-50 | 按 8659 题 × 长 schema 提示 × 多候选粗算应在数百元级，低估约 5-10 倍 | 先跑 50 题实测 token 账单再定预算 |
| L11 | SynSQL-2.5M 子集对 3B 的收益 | 证据集中在 ≤1.5B（SLM-SQL）和 7B（OmniSQL）；3B 无直接证据，有 Mnemosyne-3B 稀释反例 | 只做小规模 A/B，Spider 域内数据为主 |
| L12 | T2"完整 DAPO 零代码切换 + 默认 dapo" | TRL 默认 loss_type 是 grpo；dynamic sampling 需自写；版本 pin 0.15.2 | 版本核对后按实际能力排期（C7） |

### 4.3 遗漏方向（批判审稿补充的行动项）

- **M1【高价值】FINER 官方权重校准实验**：下载 `griffith-bigdata/FINER-SQL-3B-Spider` 权重跑项目 harness，是区分"评估端 vs 训练端"差距的确定性方法，半天完成，直接决定 85% 叙事的真伪。→ 已并入 #2。
- **M2【高价值】错误归因分析（error taxonomy）**：对当前 68.5%/71% 失败样本做错误类型分解（schema linking / JOIN / 聚合 / 语法 / 取值歧义 / 超时），直接决定下一份预算投数据、奖励还是 linking。SAC-SQL 动机数据（JOIN 错 30%、嵌套错 23.3%）与 SIRIUS 失败类型学是现成框架。→ 建议并入 #5 的失败类型感知修复，先跑一周期的归因报告。
- **M3【低成本】零训练 ICL/DC 基线**：DAIL-SQL 式示例选择 + 分解式提示（DC 分解提示 + 3-shot 对所有模型一致有效）半天可落地，同时提升单次生成与投票池质量。→ 与 #5 同批次。
- **M4【低成本】投票池多样性工程**：项目投票是 greedy，RESEARCH_NOTES 已建议 T=0.7-0.8；CSC-SQL caveat 是"模型输出高度一致时投票收益消失"。多样性是投票杠杆的前提，应作为 #5 的组成部分（共识集大小加权 + 温度采样）。
- **M5【中价值】Spider train gold 标注错误容错**：GBV-SQL 证明 Spider 存在 gold 错误；蒸馏执行过滤时"gold 执行失败/空结果"的样本单独审计（错误 gold 会污染蒸馏与 GRPO 数据）。→ 并入 #3/#7。
- **M6【轻量】难度分层汇报制度化**：论文需要 easy/medium/hard/extra 分解（FINER 有对照，项目缺失），并加 Spider-DK/Syn/Realistic 稳健性面。

---

## 5. 来源索引（论文 / 项目链接）

> 状态说明：**已核** = 本计划编写时经网络核实；**待核** = 来自调研笔记（`RESEARCH_NOTES.md` 14 agent 结论），链接或数据未经本次核实，引用前按 G5 溯源。

### 5.1 训练 / 蒸馏核心（T1/T3 轨道）

| 名称 | 链接 | 状态 |
|------|------|:---:|
| FINER-SQL（ICDE 2026，Spider-dev 85.0% / BIRD-dev 67.80%，3B） | [arXiv 2605.03465](https://arxiv.org/abs/2605.03465)；[github.com/thanhdath/finer-sql](https://github.com/thanhdath/finer-sql)（项目本地克隆 `.research_tmp/finer-sql/`） | 已核 |
| FINER-SQL-3B-Spider 官方权重（M1 校准实验用） | [huggingface.co/griffith-bigdata/FINER-SQL-3B-Spider](https://huggingface.co/griffith-bigdata/FINER-SQL-3B-Spider) | 已核 |
| STaR-SQL（ACL 2025；Llama-3.1-8B +20 点、ORM@16 86.6%） | [arXiv 2502.13550](https://arxiv.org/abs/2502.13550)（代码未确认开源） | 已核 |
| Think2SQL（GRPO 奖励密度；3B 变体 +11.8%） | [arXiv 2504.15077](https://arxiv.org/abs/2504.15077) | 已核 |
| SQL-R1（NeurIPS 2025；Spider 88.6% / BIRD 67.1%） | [arXiv 2504.08600](https://arxiv.org/abs/2504.08600)；[github.com/DataArcTech/SQL-R1](https://github.com/DataArcTech/SQL-R1) | 已核 |
| SLM-SQL（≤1.5B；SynSQL-Think-916k 配方） | [arXiv 2507.22478](https://arxiv.org/abs/2507.22478)；[github.com/CycloneBoy/slm_sql](https://github.com/CycloneBoy/slm_sql)；[SynsQL-Think-916k](https://huggingface.co/datasets/cycloneboy/SynsQL-Think-916k) | 已核 |
| SQL-PaLM（TMLR 2024；执行验证合成增强） | [arXiv 2306.00739](https://arxiv.org/abs/2306.00739) | 已核 |
| DataGpt-SQL-7B（无效 SQL 反馈样本 84.8→87.2） | [github.com/CatOnly/DataGPT-SQL-7B](https://github.com/CatOnly/DataGPT-SQL-7B) | 待核 |
| RFT（Recursive Fine-Tuning，检索式教师示例） | [arXiv 2502.06759](https://arxiv.org/abs/2502.06759) | 待核 |
| SAC-SQL（难度断层动机数据：JOIN 错 30%、嵌套错 23.3%） | 链接待核 | 待核 |
| Self-Verified Distillation（+16.7 AIME，仅取 AND 门思想） | arXiv 2605.26132（编号来自批判综合） | 待核 |
| DeepEye-SQL（零训练 BIRD-Test 75.07% / Spider-Test 89.8%） | 链接待核 | 待核 |
| LocalSQLAgent（A3B 实测 88%） | 链接待核 | 待核 |
| BREAD（专家锚点分支 rollout，fallback 方案） | 链接待核（源自调研笔记，与 SLM-SQL 相关） | 待核 |
| jenishk20/text2sql-grpo（GRPO 奖励饱和 std≈0.02、自蒸馏 −20 实证） | [github.com/jenishk20/text2sql-grpo](https://github.com/jenishk20/text2sql-grpo) | 待核 |

### 5.2 投票 / 测试时扩展（T4/T5 轨道）

| 名称 | 链接 | 状态 |
|------|------|:---:|
| Query and Conquer（执行引导 MBR 解码） | [arXiv 2503.24364](https://arxiv.org/abs/2503.24364) | 已核 |
| SIRIUS-SQL（Spider-test 91.20%；失败类型感知修复 + 置信度门控选择） | [arXiv 2606.01246](https://arxiv.org/abs/2606.01246)（代码未确认开源） | 已核 |
| R³-SQL（Ranking Reward & Resampling，ACL 2026 Findings；BIRD-dev 75.03%） | [arXiv 2604.25325](https://arxiv.org/abs/2604.25325)（**代码未开源**，缺口检测思想需自实现） | 已核 |
| GradeSQL（ORM 验证器，Spider +2.10 / BIRD +4.33 vs 执行 BoN） | [arXiv 2509.01308](https://arxiv.org/abs/2509.01308)；[github.com/sisinflab/GradeSQL](https://github.com/sisinflab/GradeSQL) | 已核 |
| MCS-SQL（置信度过滤 / 多模型投票） | [arXiv 2305.13172](https://arxiv.org/abs/2305.13172)；[github.com/BeachWang/MCS-SQL](https://github.com/BeachWang/MCS-SQL) | 已核 |
| CodeT（WMA 加权多数投票） | [arXiv 2202.13150](https://arxiv.org/abs/2202.13150)；[github.com/microsoft/CodeT](https://github.com/microsoft/CodeT) | 已核 |
| CHASE-SQL（多策略组合） | [arXiv 2410.01907](https://arxiv.org/abs/2410.01907)；[github.com/Redislabs-Solution-Architects/CHASE-SQL](https://github.com/Redislabs-Solution-Architects/CHASE-SQL) | 已核 |
| DAIL-SQL（示例选择 + 分解式提示，M3 用） | [arXiv 2403.15806](https://arxiv.org/abs/2403.15806)；[github.com/BeachWang/DAIL-SQL](https://github.com/BeachWang/DAIL-SQL) | 已核 |
| CSC-SQL（GRPO + 投票；"输出一致时投票收益消失" caveat） | [github.com/CycloneBoy/CSC-SQL](https://github.com/CycloneBoy/CSC-SQL)；[HF 模型](https://huggingface.co/cycloneboy/CscSQL-Grpo-Qwen2.5-Coder-7B-Instruct) | 待核 |
| Self-Refine（小模型自批评系统性失败） | [arXiv 2303.17651](https://arxiv.org/abs/2303.17651) | 已核 |
| SHARE（GPT-4o 自纠错也退化） | 链接待核 | 待核 |
| SSEV（非空执行结果 ≥2 次停止条件） | 链接待核 | 待核 |
| vav 投票（项目自定义投票口径，与论文 85.88% 吻合） | [github.com/vav-ai/vav_rank](https://github.com/vav-ai/vav_rank) | 待核 |
| Agentar-Scale-SQL（BIRD-dev 81.67%，投票放大本质） | 本地克隆：`C:\Users\13389\Desktop\女朋友\Agentar-Scale-SQL\` | 待核 |

### 5.3 RL 算法与框架（T2 轨道）

| 名称 | 链接 | 状态 |
|------|------|:---:|
| GRPO（DeepSeekMath 原始公式） | [arXiv 2402.03300](https://arxiv.org/abs/2402.03300) | 已核 |
| DAPO（Clip-Higher / Dynamic Sampling / Token 级损失 / Overlong；AIME 50%→52%） | [arXiv 2503.14476](https://arxiv.org/abs/2503.14476)；[verl-project/verl](https://github.com/verl-project/verl)（recipe/dapo） | 已核 |
| Dr.GRPO（去长度归一、RLVR 泛化） | [arXiv 2506.13608](https://arxiv.org/abs/2506.13608)；[github.com/sail-sg/Dr.GRPO](https://github.com/sail-sg/Dr.GRPO) | 已核 |
| TRL（pin 0.15.2；Liger 融合核缺失 vLLM IS 修正 → grad_norm 放大 ~100× 已知坑） | [github.com/huggingface/trl](https://github.com/huggingface/trl)（issue #1082 相关） | 已核 |
| PaVeRL（列级部分验证） | [github.com/Agent-RL/PaVeRL](https://github.com/Agent-RL/PaVeRL) | 待核 |
| MAD-GRPO（批评 Dr.GRPO 的 2026 不可能定理） / BNRM / P-GRPO / MARS-SQL / Sparks of Tabular Reasoning / Reward-SQL | 链接待核（批判综合 T2/T7 轨道记录） | 待核 |
| DeepSeek-R1（主教师） | [github.com/deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) | 已核 |
| gpt-oss-120b（双教师池之一） | [github.com/openai/gpt-oss](https://github.com/openai/gpt-oss) | 已核 |
| Qwen3（Qwen3-4B-Base / Qwen3-Coder-30B-A3B） | [github.com/QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) | 已核 |
| rlprobe（14 种失败签名自动诊断） | 链接待核 | 待核 |

### 5.4 数据 / 基准 / 评估（T3/T7 轨道）

| 名称 | 链接 | 状态 |
|------|------|:---:|
| Spider（Spider 1.0，dev 1034 / 榜单天花板 ~91%） | [yale-lily.github.io/spider](https://yale-lily.github.io/spider) | 已核 |
| BIRD | [bird-bench.github.io](https://bird-bench.github.io) | 已核 |
| test-suite-sql-eval（官方评估器，已集成） | [github.com/taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) | 已核 |
| Mini-Dev（BIRD 标注错误率 52.8%） | [github.com/HUSTAI-lab/Mini-Dev](https://github.com/HUSTAI-lab/Mini-Dev) | 待核 |
| SAR-Agent（标注错误筛查） | [github.com/tshu-w/SAR-Agent](https://github.com/tshu-w/SAR-Agent) | 待核 |
| SynSQL-2.5M（只取域内高质量子集） | [github.com/mikelma/synsql](https://github.com/mikelma/synsql) | 待核 |
| Mnemosyne-3B（混合 40k 离域数据 Spider −7.8pp 反例） | [arXiv 2505.09992](https://arxiv.org/abs/2505.09992)；[github.com/huchao-Lv/Mnemosyne](https://github.com/huchao-Lv/Mnemosyne) | 待核 |
| GBV-SQL（Spider gold 标注错误证据） | 链接待核 | 待核 |
| OmniSQL / Spider-DK / Spider-Syn / Spider-Realistic | 链接待核 | 待核 |

### 5.5 工程工具（T8 轨道）

| 名称 | 链接 | 状态 |
|------|------|:---:|
| LogSage（日志监控三要素：去重→归因→动作） | [github.com/ai-forever/LogSage](https://github.com/ai-forever/LogSage) | 已核 |
| vLLM V1（前缀缓存） | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) | 已核 |
| StatGuard（No-Invent 报告纪律） | 链接待核 | 待核 |

### 5.6 项目内部文件（锚点与既有资产）

`FINAL_SUMMARY.md`（发现 7：蒸馏前置缺口）、`RESEARCH_NOTES.md`（14 agent 调研）、`NEXT_STEPS.md`（旧计划）、`docs/FINER_REPLICATION_PLAN.md`（27KB 复刻路线）、`OFFICIAL_EVAL.md`、`EXPERIMENT_MATRIX.md`、`GRPO_CHECKLIST.md`、`.claude/skills/grpo-preflight/SKILL.md`、`.research_tmp/finer-sql/`（本地克隆 + reasoning_distilation/）、`finer_port/`（vav 移植，28 项自测）、`src/generate_api_data.py` / `scripts/generate_api_data.py`。

---

## 6. 与现有 NEXT_STEPS.md 的衔接

### 6.1 已有计划的三件事 → 本计划的升级

| NEXT_STEPS.md 旧计划 | 本计划的对应与强化 |
|----------------------|--------------------|
| ① 推理蒸馏（FINER Step 1，~1 天） | → **#1**：强化为完整纪律——执行过滤加"空结果/超时剔除"（Arctic 教训）、SFT 混合 10-20% gold、与 GRPO 不相交划分（#3）、G2 单次门槛 gate（<55% 停跑查数据）；API 成本先 50 题实测（L10）；"500-2000 条"放宽为全量 8659 不相交划分后 |
| ② 官方评分差异排查（~半天） | → **#2**：新增 **M1 官方权重校准探针**（用 FINER-SQL-3B-Spider 权重跑项目 harness，区分评估端/训练端）；新增对外口径纪律（G1）；旧清单（评估器版本/--plug_value/多实例库/空预测）保留为排查项 |
| ③ 大规模 GRPO 复刻（2-3 天） | → **#4**：升级为"不相交 5.5k + 四分量稠密奖励 + 梯度修复（DAPO 静默组过滤/去 std）+ β sweep + 每 100 步 checkpoint 早停"；新增"训练倒退 → 配方审计（不相交+非负稠密+组内方差），BREAD 为 fallback"；两阶段 44%<47% 根因（奖励饱和）交由 **#3** 处理，先于 #4 |

### 6.2 本计划新增（NEXT_STEPS.md 未覆盖）

- **#5 投票管线升级**（失败类型感知修复 + 加权投票 + 置信度过滤 + 采样温度）：NEXT_STEPS 只有"多视角投票 +1-3pp"一句（已并入 #5 作为子项），未含执行反馈修复与置信度机制；纠错信号只来自执行结果（3B 不当 critic）。
- **#6 工程可靠性包**：全新（监控已停摆 2026-08-07；vLLM V1 / rlprobe / latest.json 确定性报告 / slurm --requeue / grpo-preflight 提交前检查）。
- **#7 执行验证合成数据增强**：全新（NEXT_STEPS 提到 RFT 多模型合并与迭代蒸馏，但未提 SQL-PaLM 式"每条 gold 生成替代 SQL + 双验证"）。
- **#8 教师检索增强 + 难题 rationalization**：全新（RFT 式 cosine 检索 few-shot + gold-hint 反向推理治 tail narrowing）。
- **#9 Qwen3 基座 A/B**：全新（NEXT_STEPS 的"7B 基座管线"保留为本计划姊妹线——若预算允许 7B 线仍可跑，但换基座必须是 Qwen3-Coder-30B-A3B / Qwen3-4B-Base 的零训练 A/B 决策，**严禁通用 Qwen3-8B**）。
- **#10 ORM 验证器**：全新（7B 验证器 + 3B 生成器双模型分工）。

### 6.3 与旧计划冲突处（以本计划为准）

1. **"50→85 分的唯一已知路径 / 1.5 天追平 85%"的预期过乐观**：85% 降级为"待验证上限锚点"（C2/L1），一切以 G2 门槛 + 官方口径数字为准。
2. **"投票蒸馏"（学生自蒸馏）从主链降为对照**（C1）：NEXT_STEPS 将其列为参考路径之一，本计划明确主链用强教师（DeepSeek-R1 + 双教师池）。
3. **API 成本 ¥20-50 低估**（L10）：NEXT_STEPS 沿用该数字，本计划要求先 50 题实测再定预算（数百元级更可能）。
4. **7B 基座 vs Qwen3 基座**：NEXT_STEPS 的 7B 线（基线 81% → 88-90%）保留为备选，但优先级低于 #1-5 主战役；Qwen3-4B 线须先过 #9a A/B gate。

### 6.4 本计划之外仍成立、但排期靠后的旧计划项

- 迭代蒸馏（2 轮封顶，STaR-SQL 思想）→ 与 C1 约束合并，只做对照。
- BIRD 迁移 → 排下月，前置 SAR-Agent 标注筛查（Mini-Dev 错误率 52.8%）。
- 多视角投票（5 问法 × 6 采样）→ 并入 #5 的采样多样性子项（M4）。
