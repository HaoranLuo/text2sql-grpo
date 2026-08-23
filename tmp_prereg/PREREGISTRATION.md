# RFT 池集成（BIRD 5 源大池裁决）— 实验预注册

**状态**：预注册（数据收集前）
**日期**：2026-08-23
**项目**：reasoning_generator_3b（Text-to-SQL，BIRD dev 官方 EX 线）
**Git Commit**：8c254a4133430c7e0012b651753fa44596f3f9d6（HPC repo，数据收集前 HEAD）
**SHA-256（本文档冻结版）**：见 records/experiments.jsonl 封存记录（BIRD_RFT_pool5src，2026-08-23）。本文档自数据收集（RFT 池生成作业启动）起只读，任何修改可由哈希比对识别。

---

## 1. 实验目的

训练线决定性一战：验证离线 RFT 模型（checkpoints/rft_bird_v3/checkpoint-50，
eval100 单次官方 EX 49.0 vs sft_v3 基线 38.0）作为**第 5 个候选源**并入
4 源主池后，能否把整条裁决管线（执行分组 + ORM 组头选择）的官方 EX 推过
4 源基线 60.37。

核心假设：**RFT 候选携带主池 4 源未覆盖的正确答案/更优多数结构，5 源池的
arm_orm_grouphead 官方 EX total > 60.37（4 源基线）。**

如果该假设被证伪（5 源 EX ≤ 60.37），则：RFT 并入主池不成立，回到"RFT
作为独立生成器"的定位（eval100 49.0 已是独立线证据），不把 RFT 写进主池
配置；后续 RFT 投入转向改善 RFT 自身覆盖（提升独立 EX），而非池融合。

## 2. 假设

### H1（主假设：RFT 并入提升 ORM 裁决）

> 5 源池（4 源 + rft_bird_v3，80 候选/题）arm_orm_grouphead 官方 EX total
> 60.37（4 源池同臂基线，outputs/bird_select_ormbird_bird_bal2）。

**操作化定义**：5 源 summary.json 中 `official_exec_accuracy.arm_orm_grouphead.total`
（FINER evaluation_bird_ex.py，1534 题）。
- total > 60.37 → 正向（GO）；
- total ≥ 62.00 → 强（strong GO）；
- total ≤ 60.37 → 证伪（NO-GO，如实记录，不重新叙述）。

**检验**：官方评估器全量数字（唯一权威口径）；另做 McNemar 配对检验
（5 源 arm_orm_grouphead vs 4 源 arm_orm_grouphead 的 1534 题逐题正确性，
双尾精确 McNemar，α=0.05）。EX 主判据与 McNemar 均如实报告；
**若 total > 60.37 但 McNemar p ≥ 0.05：仍记"正向"但显式标注
"弱证据（未达配对显著）"，不得宣称统计显著。**

### H2（天花板诊断：RFT 顶高池子）

> pool_pass_oracle(5 源) > pool_pass_oracle(4 源)。

**操作化定义**：pool_pass_oracle = 池内存在与 gold 执行签名一致的候选的题目
占比（signature = AP.outcome_signature，sig == gold_sig 且 ≠ ERROR_SIG，
与 src/bird_coevolve.py 既有口径一致）。4 源池 oracle 从 4 源 prep.json 的
sigs_per_entry 计算，5 源池从 5 源 prep.json 计算。

**检验**：纯确定性计算（无抽样）。oracle 上升 → RFT 贡献了池子天花板；
oracle 持平/下降但 EX 仍上升 → 提升来自选择侧（RFT 候选改变组结构/票数）
而非天花板，报告时显式区分，不得混称为"天花板提升"。

### H3（次要对标：多数投票臂）

> arm_vav(5 源) 官方 EX vs 56.26（4 源基线）。

**操作化定义**：5 源 summary.json 中 `official_exec_accuracy.arm_vav.total`。
本假设**只作描述性报告**，不设 GO/NO-GO 判据（主判据只有 H1）。
RFT 使 arm_vav 下降而 arm_orm_grouphead 上升 → 如实报告"RFT 稀释多数票、
但强化 ORM 选择"，不掩饰。

## 3. GO/NO-GO 判据

| 条件 | 标准 | 类型 |
|------|------|------|
| C1（主判据） | 5 源 arm_orm_grouphead 官方 EX total > 60.37 | 必要且充分（GO）；≥62.0 记"强" |
| C2（显著性） | McNemar 双尾 p < 0.05（同一对比） | 非 gate，只决定"正向/强证据"措辞 |
| C3（天花板） | pool_pass_oracle(5) > pool_pass_oracle(4) | 非 gate，解释性证据 |
| C4（口径完整） | 5 源 prep/score/final 全程无缺分、1534 题全裁决、空胜者 SELECT 1 | 必要条件（完整性） |

**判定规则**：
- C1 满足且 C4 满足 → **GO**（RFT 并入主池成立）；total ≥ 62.0 → GO-strong。
- C1 不满足 → **NO-GO**（如实记录；不做事后 relabel）。

### 半通过场景的预先解释（数据收集前写定）

- **total 恰好等于 60.37**：不满足 ">"，记证伪。不允许事后说"基本打平也算融合成功"。
- **60.37 < total < 62.0**：记"正向（弱）"——并入成立但未到强线，报告与后续
  训练资源分配按弱正向处理（并入但继续找更强的源）。
- **total > 60.37 但 McNemar p ≥ 0.05**：记"正向（弱证据）"，显式声明未达配对显著。
- **C1 满足但 pool oracle 未升**：EX 提升归因于选择侧，报告中显式写清，不宣称"RFT 顶高了天花板"。
- **arm_vav 下降**：见 H3 预设表述；不构成对 H1 的否定，也不允许被事后叙述为"额外收益"。
- **C4 失败（如 ORM 缺分/部分题未裁决）**：结果作废，修复管线后重跑完整裁决（重跑前不修改本文档判据）。

**NO-GO 后的具体行动**：统计 RFT 候选在各难度档的 pass@16 与主池 4 源对比、
RFT 组代表被 arm_orm_grouphead 选中的题数及其中答错题数（诊断脚本只读，
不改判据）；结论写入报告"证伪"节，不转入叙述性解释。

## 4. 方法

### 4.1 设计概览

**1534 题 × 5 模型源 × 16 采样/源 = 122,720 候选**（每源每题 16；合并后 80 候选/题）

- **案例**：BIRD dev 全量 1534 题（dev_20240627，11 库单实例 sqlite；
  难度 simple 925 / moderate 464 / challenging 145）。
- **模型源**：sft_phase1 / sft_v2 / sft_v3 / p2a_500（4 源主池，已生成、
  字节级原样复用）+ **rft_bird_v3/checkpoint-50（全参 plain 模型，新增）**。
- **RFT 生成口径（与主池逐字一致）**：T=1.0, top_p=1.0, seed=0, n=16 单请求
  共享 prefill, max_new_tokens=2048；prompt = sqlite_master DDL + question +
  evidence（有值才加，ReasoningGeneratorAgent.build_prompt canonical 模板,
  dialect=sqlite）+ chat template（add_generation_prompt=True），截断 3072
  token；evidence 仅 dev.json 官方（与主池同口径，无 auto evidence、无
  retriever 裁剪）；解析 = VavSampler.extract_sql。
  RFT checkpoint 的 tokenizer 与 base 词表完全一致、chat_template 一致
  （已核验 md5 差异仅 truncation 默认值/extra_special_tokens 空列表，
  不影响 token 化）→ 同 prompt 文本产生同 token 序列。
- **裁决**：src/bird_select.py 原封不动（prep → score → final 三阶段；
  ORM checkpoint = checkpoints/orm_bird_bird_bal2；arm_orm_grouphead 与
  arm_vav 两臂；官方评估器 evaluation_bird_ex.py 参数与基线一致）。

### 4.2 评估维度

| 维度 | 名称 | 口径 | 说明 |
|------|------|------|------|
| D1 | 官方 EX total | FINER evaluation_bird_ex.py，1534 题 | 唯一权威数字（主判据） |
| D2 | 官方 EX 分难度 | simple/moderate/challenging 分列 | 报告用 |
| D3 | pool_pass_oracle | 签名相等口径（coevolve 同款） | 天花板诊断（H2） |
| D4 | McNemar | 双尾精确，两臂逐题 res 对比 | 配对显著性（H1 检验） |

### 4.3 执行流程（数据收集顺序）

1. RFT 生成（gpudebug 3090，50min 切片 + checkpoint 续跑，直到 fully_done）；
2. 拼池 src/merge_pool_rft.py（主池 4 源原样在前 + RFT 追加在后；
   model 字段 "rft_bird_v3"）；
3. 5 源 prep（cpu6348，bird_select.py --phase prep，新 out-dir）；
4. 5 源 ORM 打分（gpudebug）：4 源组代表分按 **(qi, prompt) 键** 从
   outputs/bird_select_ormbird_bird_bal2/work/orm_scores.json 复用
   （build_orm_prompt 确定性 + ORM 打分确定性 → 同 prompt 同分）；新组代表
   （含 RFT 引入的）用 VllmScorer 同参数新打分；合并为完整 orm_scores.json；
5. 5 源 final（cpu6348，bird_select.py --phase final → 官方 EX × 2 臂）；
6. 对照分析 + McNemar + oracle；报告。

### 4.4 案例/样本选择标准

- 全量 1534 题，无筛选、无排除（与主池同口径）。任何一题失败均如实计入
  （空预测写 SELECT 1 铁律）。

## 5. 数据记录格式

| 字段 | 类型 | 说明 |
|------|------|------|
| items.json.candidates[].model | str | "rft_bird_v3"（新源标识） |
| items.json.candidates[].sample_idx | int | 0..15 |
| items.json.candidates[].sql / parse_success | str / bool | 抽取结果（VavSampler） |
| orm_scores.json.entries[].{qi,ei,score} | int,int,float | P(Yes)，按 payload 顺序对齐 |
| summary.json.official_exec_accuracy | dict | 两臂官方 EX（唯一权威） |
| eval_result_dev.json[].res | int | 官方逐题正确性（McNemar 输入） |

**存储**：HPC outputs/（生成 + 裁决中间产物）、tmp_idea_research/rft_pool_integration/
（预注册 + 最终报告）；主键 = question_id（dataset_index）。生成物 checkpoint.json
每 25 题落盘，续跑由 run_config 逐字段比对防混配。备份：HPC 本地 + 本机镜像
（关键结论文件 scp 回本地）。

## 6. 预注册分析计划

**所有分析代码在数据收集前完成并（用既有 4 源产物）测试。**

### 6.1 主要分析（用于 GO/NO-GO 判定）

**A. 官方 EX 对比（检验 H1）**
- 统计方法：直接读取 5 源 summary.json 官方数字与基线 60.37 比较（确定性，
  无统计推断）。
- 成功标准：total > 60.37（GO）；≥ 62.0（strong GO）。

**B. McNemar 配对检验（检验 H1 的显著性）**
- 方法：双尾精确 McNemar（binom），对象 = 5 源 vs 4 源 arm_orm_grouphead 的
  eval_result_dev.json 逐题 res（question_id 对齐，n=1534）。
- 报告：fixed/broken/discordant/p_two，α=0.05（不校正；单一主对比）。

**C. pool oracle（检验 H2）**：确定性签名对比（见 H2 操作化定义）。

### 6.2 探索性分析（不用于 GO/NO-GO）

- RFT 候选 per-question pass@16 与 4 源对比（分难度）；
- arm_orm_grouphead 胜者来源分布（RFT 组被选中题数与其中正确率）；
- RFT 独有候选被 dedupe 后的组数/组大小分布。

### 6.3 多重比较

主判据为单一数字（H1 total），H2/H3 为解释性对照，无多重检验校正需求
（McNemar 仅一次配对检验）。

## 7. 假设声明

1. **口径一致性**：RFT 生成 prompt/采样参数与主池逐字一致；已核验 tokenizer/
   chat_template 等价。若核验有误（token 序列不同），结论仍有效但需在报告
   中显式标注口径差异。
2. **ORM 分复用等价性**：同 (qi, prompt) → 同分（确定性）。若发现打分引擎
   非确定（版本差异），弃复用改为全量重打分并标注 `[偏离预注册]`。
3. **官方评估器为唯一权威**：任何本地复算（local EX / 签名口径）只作诊断，
   不进入主判据。

## 8. 已知未知

- RFT 全参 3B 在 3090 上的生成吞吐（预估 ~4h，切片续跑吸收不确定性）；
- RFT 候选的解析失败率（可能高于/低于 4 源，影响池结构）；
- ORM（orm_bird_bird_bal2）对 RFT 风格 SQL 的打分校准是否成立（RFT 训练
  分布与 ORM 训练分布的重叠未知）——这是本次实验的核心未知，H1 直接检验。

## 9. 变更日志

| 日期 | 变更 | 原因 |
|------|------|------|
| 2026-08-23 | 初始版本 — 数据收集前预注册 | — |

> 此表在数据收集开始后只读。如需修改实验设计，在新的预注册文档中重新预注册。
