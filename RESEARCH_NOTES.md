# 调研笔记：训练提升方法（2026-08-06）

> 5 路并行调研 agent 的结论存档。来源均为 arXiv/开源项目深读。

## 核心结论（与我们的实验直接相关）

### 1. STaR-SQL（arXiv 2502.13550）— 与"投票蒸馏"最对齐
- **8B 模型自训练单次生成 +20 点（55%→75%）**，无需教师模型
- 做法：3 few-shot 采样 → 执行匹配过滤 → SFT 只训正确的 → 每轮从基座重初始化
- 错题处理（rationalization）：按错误次数 L 重采样 L 次，用 gold SQL 做 hint 引导重生成
- 数据量：7000 题 × 8 采样；叠加 ORM@16 到 86.6
- **对我们的预期锚点：3B 单次 50% → 蒸馏后瞄准 60-65%**

### 2. RFT / Rejection Sampling（arXiv 2308.01825）
- 执行校验过滤 + 按方案去重，弱模型稳定 +5-6 点
- **k=3 就稳定 +2 点**；收益来自"不同的推理路径条数"而非样本数
- 33B 失败教训：过拟合训练集 → 生成多样性崩 → 后续迭代失效
- 参数：3 epoch、batch 128、peak lr 2e-5、3% warmup

### 3. Self-Consistency（arXiv 2203.11171）
- 投票有效性的理论内核：正确路径收敛同一答案，错误路径分散
- **图 8：采样路径与投票结果一致的比例 = 置信度信号**
- 采样 T=0.7~0.8；一致度 ≥3/5 可做数据过滤信号

### 4. STaR（arXiv 2203.14465）
- rationalization（gold hint 反向推理）专治 tail narrowing（难题进不了训练集）
- 每轮从预训练基座重新初始化（防过拟合）；2-3 轮即够

### 5. CSC-SQL（arXiv 2505.13271）— 3B + BIRD + GRPO 同构项目
- 3B SC@64=61.30 → CSC(merge-revision)@64=65.28
- 投票细节：T=0.8、按执行结果分组、只取前两大投票组修正
- caveat：模型输出高度一致时收益消失——投票全靠采样多样性

### 6. 自进化综述（arXiv 2404.14387）
- 双过滤：metric-based（执行正确性）+ metric-free（内部一致度）
- **迭代 >3 轮普遍无提升**；防 model collapse：蒸馏数据混合 10-20% 真实 seed 数据

## 对我们的行动项

1. **实验 A（投票自蒸馏）方向确认** ✅ 已跑。优化：加正确性过滤（投票胜出须与 gold 执行一致）+ 混合 10-20% gold 数据（防 collapse）
2. 蒸馏迭代 1-2 轮，每轮从基座重初始化
3. 采样用 T=0.7~0.8（我们的投票是 greedy——多样性来自 prompt 视角，可考虑加采样温度）
4. 监控一致度分布：下降 = 过拟合前兆
5. GRPO 作为蒸馏后的下一步（增益与投票部分重叠）

## 待整合
- 其他 3 个 agent 报告到达后交叉验证，形成实验队列 v2

---

# 追加：Spider 训练侧方法调研（agent 2）

## 关键基准（同一赛道，可复现）

| 模型 | 方法 | Spider 官方 EX | 来源 |
|---|---|---|---|
| **FINER-SQL-3B**（Qwen2.5-Coder-3B，同基座！） | GRPO 四分量奖励(format+exec+atomic+memory) | **dev 85.0%** | arXiv:2605.03465, ICDE 2026, [finer-sql](https://github.com/thanhdath/finer-sql) |
| Arctic-Text2SQL-R1-7B | GRPO 简单三值奖励(1/0.1/0，同我们)+强数据过滤 | test 88.8%（dev混训有争议） | arXiv:2505.20315 |
| DB-Explore-7B | **纯 SFT**（结构化合成+验证数据） | 87.8% | arXiv:2503.04959 |

## 核心洞察

1. **FINER-3B 85% vs 我们 3B 50%——配方差距**：(a) 数据 100/500 → 全量 8659 条；(b) G=4 → 16-32；(c) 奖励加 **atomic（操作级 Jaccard，非行重叠）+ memory**；(d) 步数 25 → 1000+（lr 8e-6，2000 步）
   - ⚠️ 我们 c3_atomic 失败是因为配方太小（100条/G4/25步），不是 atomic 奖励本身无效（FINER 消融：去掉 atomic −3.26%）
2. **数据过滤是胜负手**：Arctic 过滤（空结果/超5s/合成集只留验证正确的）；未过滤合成数据必受伤（我们的 E10 同现象）
3. **DB-Explore 反例**：高质量 SFT 数据 → 7B 无需 RL 就 87.8%——我们的 SFT 无效是"数据无效"不是"方法无效"
4. **评估口径**：官方 dev = 1034 全量；test-suite EX（多实例模糊化）> 我们单实例行匹配；100 条子集不可比任何公开数字
5. **开源仓库**：finer-sql（奖励代码可移植）、ArcticTraining（数据过滤）、csc_sql、test-suite-sql-eval（官方评估器）

## 行动项 v2

- **P0: 集成官方评估器 test-suite-sql-eval + 全量 dev 1034 复验**（先做，半天）
- **P0: 3B 复刻 FINER 配方**（全量 8659 + G8~16 + atomic+exec+format 奖励 + 200+ 步 checkpoint 早停，A40 8 小时档）
- **P1: 数据过滤**（执行验证过滤进 build_dataset —— 已实现 --filter-gold，扩展为 Arctic 式：空结果/超时剔除）
- **P1: 投票升级**：执行引导（剔除执行失败的候选再投票）

---

# 追加：self-training/蒸馏专项（agent 3）

## 关键证据（LMSI/STaR-SQL/CSC-SQL）
- **LMSI**（arXiv 2210.11610）：无标签自改进——采样→投票选高置信→蒸馏 SFT，GSM8K 74.4→82.1（无 ground truth）
- **STaR-SQL**：执行匹配过滤 + gold-hint 重采样难题 + 每轮从 base 重启，Spider 86.6%（8B）
- **CSC-SQL**：同款 Qwen2.5-Coder-3B + GRPO，BIRD 65.28%

## 数据选择结论（执行验证 > 一致性投票 > 全量）
1. 全量使用（不过滤）是下策——所有方法都只训正确子集
2. **执行匹配过滤是第一优先级**（train 有 gold SQL = 完美验证器）
3. 投票胜出 SQL 仍有 ~29% 是错的——必须执行过滤后才进训练集
4. 无 gold 场景退化为一致度阈值 ≥3/5
5. 混合 gold:distilled ≈ 1:1~1:2 防 model collapse（ReST/R1 实证）
6. 投票应为"执行结果分组"而非字符串多数

## 蒸馏最佳实践
- 7000 题 × 5-8 候选 → 执行过滤后 3-6K 条 + 全量 gold 7K ≈ 1-1.4 万条 SFT 数据
- 3B LoRA rank 16-64、lr 1-2e-5、3-5 epochs、dev 早停
- 迭代最多 2 轮，每轮从 base 重启
- 之后接 GRPO（执行匹配做规则奖励）

---

# 追加：GRPO 工程细节（agent 4）—— 解释了我们所有负结果！

## 为什么 partial 无效 / 数据量无差别 / 7B 训练负收益
1. **组内扁平化**：GRPO advantage = (r−mean)/std。partial 的 0.1 档让组内全员同值 → std→0 → **梯度归零**
2. **静默组**：全对/全错的组 std=0，无学习信号——我们 100-7000 条无差别 = 数据不是瓶颈，**梯度信号退化是瓶颈**
3. **正负混合奖励不稳定**（PaVeRL 实证）：非负奖励更稳——我们的 0.1 档"可执行但错"在 7B 上伤害训练
4. **无 SFT 冷启动直接 RL 低效**（IRAC 定理/Think2SQL G1 实证）：SFT 起步 +9.7pp

## 3B GRPO 配置清单（按优先级）
1. 修梯度信号退化：静默组过滤（DAPO 式）+ pass-rate 过滤（保留 0.05-0.6 判别性样本）
2. **lr 1e-6~5e-6**（我们 3e-6 ✅）、max_grad_norm=0.1、warmup 0.1、cosine
3. **β 下探 0.01-0.02**（可验证奖励域；我们 0.04 可用但偏保守）
4. reward 重构：R = 0.7·R_exec + 0.3·R_component（组件分取 (0,1) 开放区间做 tie-break）；执行失败不发 0.1
5. temperature ≥0.9（防同质化）、G=8（4 以下基线噪声大）
6. 30-40 步上限 + 每 5 步 dev 评估早停；reward_std 触零即停
7. 从 SFT 检查点起步（B 实验正在做）
8. 升级 TRL v0.16+ 拿 scale_rewards（或 patch _compute_advantages 去 std）

## 投票升级（Query and Conquer，arXiv 2503.24364）
- 字符串多数投票会选错"语义等价不同写法"——执行结果相似度 MBR 选择
- 7B 上 +10 点；3B 预计 +2-5%

---

# 综合行动项 v3（5 agent 共识）

| 优先级 | 实验 | 依据 |
|:---:|------|------|
| P0 | 官方评估器集成 + 全量 1034 复验 | 所有数字可比的先决条件 |
| P0 | 投票升级：执行结果分组投票 + 淘汰失败候选 | Query and Conquer +7B +10pp |
| P0 | **E: FINER 式奖励**（exec 2/1/0 + atomic 操作级 + format + 非负）× 全量/2000条 × G8 × β0.02 × 100步早停 | FINER-3B 85% 同基座 |
| P0 | **F: pass-rate 过滤数据**（保留 0.05-0.6 判别性样本）+ GRPO | Open-R1/DAPO：修静默组 |
| P1 | 蒸馏 SFT（执行过滤版，混合 gold）→ GRPO 衔接 | STaR-SQL/LMSI/R1 路线 |
| P1 | 7B 重试：SFT 冷启动 + 非负奖励 + 困难样本定向 | Think2SQL 7B +2.8% |
