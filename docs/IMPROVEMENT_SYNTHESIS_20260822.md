# 改进方向总汇（2026-08-22，四路文献+代码侦察合成）

> 四份源报告：tmp_idea_research/ 下 verifier_adapt_scan_report.md、bird_gen_scan_report.md、
> training_2026_scan_report.md + docs/BIRD_JUDGE_ADAPTATION_PLAN.md。
> 材料：54+23+28 篇论文 PDF、13+9 个代码仓库，全部本地落盘。

## 一、三个跨报告的共识结论

1. **判卷跨基准失效是文献实证的普遍现象，且廉价适配可行**——瓶颈是**标签质量**而非数据量（2607.06799：未见 schema AUROC 跌到 0.66，加模型/加多样性/蒸馏强 judge 都补不上；2% 数据+DPO 段即可适配；纯 SFT 是"记而不泛"，RL/DPO 段才是泛化来源）。
2. **BIRD 金标准本身有严重标注错误**：Mini-Dev 52.8%（PVLDB 2601.08778）、BIRD train 61%（ReViSQL）——我们的自动打标会传播这些错误；清洗 gold 后 RLVR 可再 +11~16%。**先审计数据再谈配方**是铁律。
3. **BIRD 生成侧最大的免费杠杆是 schema 质量与自动 evidence**：LitE-SQL 实测 schema linking 值 11.5pp；SEED 自动 evidence 最高 +17.7pp（离线成本≈0）；FINER 67.80 = 不带 evidence + 全 schema。

## 二、改进菜单（按性价比排序）

### P0 档：零/近零成本（1-2 天内）

| # | 动作 | 依据 | 成本 |
|---|---|---|---|
| P0-1 | **BIRD 金标准审计**：dev.json 的 gold SQL 抽样执行核验 + 假负排查（TinyV：38% 假负最伤 RL） | 52.8%/61% 错误率证据 | CPU，1-2h |
| P0-2 | **SEED 式自动 evidence 生成**：库摘要+样例执行+关键词 → 给生成 prompt 加证据（Spider 域同样适用） | SEED +17.7pp 上界 | 离线生成，~0 |
| P0-3 | string-SC 天花板复现（BIRD 上） | 2607.06799 | 离线重算 |
| P0-4 | FINER 配方核对收尾（已做过大半，归档） | prior | ~0 |

### P1 档：低成本高把握（一周内）

| # | 动作 | 依据 | 成本 |
|---|---|---|---|
| P1-1 | **BIRD 判卷适配**（方案已冻结 docs/BIRD_JUDGE_ADAPTATION_PLAN.md）+ 按侦察加固：先测 AUROC 基线、gold 清洗先行、SFT+DPO 两段式、1:1 硬负保留 | GradeSQL 同构配方 +4.33pp；2607.06799 协议 | A40 ~4-6h |
| P1-2 | **0.6B schema retriever**（HN-SupCon 难负对比，代码已 clone）→ BIRD 生成时只喂检索出的相关表 | LitE-SQL：schema linking = 11.5pp | 单卡 3-5h + 推理接入 |
| P1-3 | **EXPO-SQL 执行定位**（不训练，定位错误子句→生成修复提示） | ACL Findings'26，+2.4 BIRD/+5.6 hard | 离线接入，~1 天 |

### P2 档：中成本（按 P0/P1 结果决定）

| # | 动作 | 依据 | 成本 |
|---|---|---|---|
| P2-1 | EvoSQL SDPO 执行感知自蒸馏（5k 数据，Coder-3B 边际 +9.65 最诱人） | 2607.20489 | A40 ~0.5-1 天 |
| P2-2 | SERL-SQL 教师后见重加权 GRPO / CSC-SQL GRPO（3B 58.87→64.41） | 2608.00485 | A40 ~1 天 |
| P2-3 | 判卷联合训练（verifier co-training）+ judge 集成（两 provider LLM judge 0.82 AUROC 及格线） | 2607.06799 | 视情况 |

### 已否决/砍掉

- few-shot 选择（≤7B 训练系都不依赖 ICL）
- LINQ 论文（多渠道检索证实不存在，功能由 DIVER/GATE 覆盖）
- token 级监督（CAPER 证明有害）

## 三、推荐执行顺序

1. 今天：P0-1 金标准审计 + P0-2 自动 evidence 生成（离线并行）
2. P1-1 判卷适配开跑（需用户批准 GPU 训练；方案已带预注册判定）
3. P1-1 出数后：P1-2 retriever → P1-3 EXPO-SQL → 依结果再议 P2
4. 全程：每个实验结果写入 records/experiments.jsonl；论文素材持续沉淀

## 四、待用户决策项

- P1-1（判卷适配，A40 ~4-6h）是否批准开跑
- HPC 账号申请邮件发送（可加 2a40 申请）
- GitHub token 吊销重发
