# 改进设计深度思考（四路深度调研合成 · 2026-08-19）

> 来源：近邻精读（DPC/SISelection/R³-SQL，含代码）/ 验证器路线（GradeSQL/JudgeSQL/MARS-SQL/SIRIUS，含代码）/ 2026 前沿选择机制 / 训练配方（FINER 源码级核对）。
> 四份原始报告在 tmp_idea_research/（frontier_2026_report.md、verifier_upgrade_report.md、near_neighbors_report.md、training_side_report.md），论文全文与代码在 tmp_idea_research/papers/、repos/、DPC/、SISelection/。

---

## 一、三个改变认知的决策级发现

1. **FINER 的 memory 奖励在官方最终配方里被禁用了**（train_3b.sh 明确传 `--no-memory`，作者注释："atomic + exec reward alone gives a strong, stable signal"）。→ **我们的"三份量"恰好就是官方最终配方，不存在缺口**。memory 补齐降为可选（代码三文件已读透：Qwen3-Embedding-0.6B + ChromaDB + cos 质心，只对 exec<2 轨迹加分）。
2. **orm_b1 的 pointwise 生成式路线被四篇验证器论文一致背书**——最大增量在**融合层**（0.9~3.3pp），不是换 loss/换模型。
3. **没有任何 2026 证据支持"强 SFT 起点 + 新奖励组件还能榨出显著提升"**（外部负结果仓库：BIRD 二值奖励 GRPO=+0.0、奖励饱和 0.98±0.02）→ 训练侧最后一搏应押**配方保真**，不是堆新组件。

---

## 二、融合后的优先级路线图

### P0：零成本（纯离线重算/改参数，本周可全部做完）

| # | 改进 | 来源 | 接入点 |
|---|---|---|---|
| P0-1 | **FINER 配方保真核对**：原子奖励参数对齐 S3（e=0.05/β=0.79/γ=0.20，而非默认 0.30/0.71/5.0）、格式错=总分 0、atomic 只在"可执行但错"叠加、GT 行缓存、每 100 步存档早停（官方 best 在 200-600 步） | FINER 官方代码精读 | **FINER 还在排队，趁现在改参数重投**（比跑 44h 后才发现参数不对强得多） |
| P0-2 | **行列序双不敏感等值判定**替换包语义分组 | SISelection `result_to_normal_form` | MI-VAV 分组函数 |
| P0-3 | **SIRIUS 式门控 AST 跨源裁决 + 平票兜底链**（小实例证据→组大小先验→保守保留；三篇近邻都没做兜底，我们做） | SIRIUS + 近邻三篇的弱点 | ORM 终裁层（多源池天然带来源标记） |
| P0-4 | **执行分层先置**（能跑通 > 空结果 > 报错）+ P(Yes) 温度缩放 | GradeSQL 代码实测 | ORM 打分语义 |
| P0-5 | **DPC BS-F1 软比较器（列名感知修正版）**——他们没修列位置敏感的洞，我们修=超越点 | DPC 代码 | ORM 终裁度量 |
| P0-6 | **R³ 组级打分 (r_list, r_point) + τ=0.05**，手工字典序换学习融合 | R³-SQL | ORM 组级聚合 |

### P1：低成本（一次重训 / 小改管线）

| # | 改进 | 来源 | 说明 |
|---|---|---|---|
| P1-1 | 正负比均衡对照重训（现 34% vs GradeSQL 58-69%）+ M 次采样平均打分 | GradeSQL/MARS-SQL | 一次 LoRA 重训（2h） |
| P1-2 | CAPS 级联判卷预算（部分证据先筛、top-k 全文判） | 前沿扫描 | 判卷 token 减半还反超 |
| P1-3 | RetrySQL 合入（`{错步}[BACK]{正确步}` 全参短 CPT）——**带 GO/NO-GO**：100 步内 BIRD 单次掉 >0.5pp 即停 | RetrySQL 代码 | 训练侧，LoRA 无效必须全参 |
| P1-4 | QATCH 结果集级部分奖励（Cell P/R + 行数比）——与 atomic 二选一勿叠加 | SQLight | 比 atomic_ops 轻 |

### P2：立项级（需设计/较重）

| # | 改进 | 来源 |
|---|---|---|
| P2-1 | 池质量门控重采样（无训练版：ORM 分数分布/分组熵触发） | R³-SQL + 前沿 |
| P2-2 | 双范式第二判官（SQL 执行 vs Python 执行）——跨范式补 ORM 泛化短板（未见 schema AUROC 0.66→0.82 实证） | DPC + 选择性预测研究 |
| P2-3 | memory 奖励可选补齐（复用 1000 教师轨迹 + 0.6B 嵌入） | FINER 代码 |
| P2-4 | 伪 pairwise 上界评估（用现有 ORM 的 P 差做）→ 决定是否重训 pairwise judge/GRPO | JudgeSQL |

### 不推荐（与否定结果冲突或证据不足）

ModeX 质心过滤、SIRIUS 修复链部分、多温度采样、同系第三判官、BAP-SQL（agentic 多步+无代码+预算大增益小）、Surprisal 尾部选择（SQL 奖励太稀疏）。

---

## 三、对论文写作的直接馈赠

- **超越点已明确**：DPC 的 BS-F1 列位置敏感（我们修）、SISelection 平局弃权（我们做兜底链）、R³ 手工字典序（我们学习融合）、三篇都丢 top-2 外信息（我们全组打分）。
- related-work 对比句草稿在 near_neighbors_report.md。
- 外部负证据链补强 L5：jenishk20（GRPO +0.0）、SLM-SQL/SQLight 增益全来自弱基座。

## 四、立即执行顺序（我的建议）

```
现在（FINER 排队中）: P0-1 配方保真核对 → 修正参数重投 FINER（趁还没开跑）
随后（3090/CPU）:    P0-2 ~ P0-6 全部零成本离线重算（一个 agent 或分两个并行）
出数后:              融合层增益若 ≥1pp → 并入主方法；P1-1 重训排队
FINER 出数后:        按预注册判据判定 → 决定 P1-3/P1-4
```
