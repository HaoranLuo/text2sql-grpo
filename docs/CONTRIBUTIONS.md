# 贡献台账与作者位次依据（2026-08-19）

> 用途：论文投稿前评定作者位次/致谢/方法部分的 AI 使用声明。
> 原则：只记录事实（谁在什么时间做了什么智力贡献），不代做位次决定——位次由你与导师协商，
> 本文档提供可核对的证据链（git commit、records/experiments.jsonl、docs/、HPC 产物）。

## 一、参与方与角色

| 参与方 | 角色 |
|---|---|
| **你（Haoran Luo）** | 项目发起人、第一作者候选：研究问题设定（小模型 Text-to-SQL RL）、算力资源（HPC 账号）、实验授权与所有重大决策拍板、FINER 复刻的前期基础、导师沟通 |
| **导师** | 方向指导（如要求跑 1000 步收敛实验验证训练动态）、论文定位指导 |
| **AI 助理（ZCode 会话 + 子代理）** | 工具性贡献：实验设计建议、脚本编写、实验执行与监控、数据分析、文档与论文素材整理。按学术惯例不作为作者，在论文中声明 AI 辅助（Methods/Acknowledgement 或按目标 venue 的 AI 政策处理），位次依据以人类参与方的贡献为准 |

## 二、各方的具体贡献清单（带证据）

### 你的贡献（第一作者候选的核心论据）

1. **研究问题与总体方向**：小模型（≤3B/7B）Text-to-SQL 从复现到超越 FINER-SQL（证据：PROJECT_GUIDE.md §1、PAPER_PLAN.md）
2. **前期基础实验**：3B 基线、GRPO 实验矩阵（三级/partial/atomic 奖励）、投票（5p/7p）、FINER 复刻与 85% 验证（证据：records/experiments.jsonl 2026-08-01~08-12 条目、git 提交历史）
3. **评估口径纪律的建立**：三口径（自定义/原始 Spider/test-suite）不可混比的铁律、官方口径审计（P2 根因闭合）（证据：PROJECT_GUIDE.md §6.5/§8.2、docs/mac_sql_evaluator_diff.md）
4. **数据卫生**：GRPO 5.5k/SFT-gold 1.5k/未分配池三不相交划分 + sha256（证据：data_hygiene/）
5. **关键决策拍板**：RL 封盘、FINER 配方 GO/NO-GO、全部训练作业批准（证据：对话记录 + 各 slurm 提交时间）
6. **导师要求实验的执行**：3b_1000st 1000 步收敛实验（提交与补评作业）
7. **资源与运维**：HPC 账号、cpolar 堡垒机链路、公告板端口机制（证据：PROJECT_GUIDE.md §5、GitHub 提交 6b2c744/251b7a3）

### 导师的贡献

1. 要求 1000 步长跑实验（"老师要求"）——直接推动了 FINER 决策的数据基础
2. 论文方向/发表的指导意见（按实际发生记录）

### AI 助理的贡献（工具性，按时间线）

1. **静默 bug 排查范式与 6 个配置 bug 修复**（3B 基线 34%→45% 真实值）
2. **L0 口径审计执行**：解析器大修（557→0 解析失败，13+ 补丁）、MAC-SQL 评估器获取与 diff
3. **MI-VAV 的设计与实现**：把官方 test-suite 多实例语义搬入投票分组（设计思路源于项目自身文档的 E6 条目 + 四组口径错位证据的归因）——**本项目最重要的方法贡献**（证据：docs/SELECTION_V2_PLAN.md、src/adjudicate_pool.py、docs/MI_VAV_METHOD.md）
4. **ORM 判卷老师的完整管线**：自动打标（20,415 条）、训练脚本（三次 debug）、三级污染检验设计（同池/同门/异门）、四臂选择器（证据：src/label_orm_data.py、src/train_orm.py、src/orm_selection.py、records/experiments.jsonl）
5. **实验基础设施**：vLLM 部署（21×）、中央候选库（生成一次实验 N 次）、裁决器测试体系（53+45+27 断言）、切片接力与守望自动化
6. **分析成果**：池子上限分解（89.4% / 180 选错 / 103 真不会）、错误预算账、六路文献检索与新颖性判定（docs/RESEARCH_SYNTHESIS_20260815.md）
7. **负结果清单的整理**（论文 L5/L7 素材）：高温/few-shot/质心过滤/修复链/软排序/顺序敏感/三模型池/v4hard 共 8 项

### 子代理的贡献（AI 工具的一部分）

各子代理编写的脚本（已标注在其交付报告与 git 文件中）：adjudicate_pool.py（多实例裁决器）、eval_pool_b1.py（vLLM 候选池）、eval_5p_vllm.py、candidate_store.py、train_orm.py、orm_selection.py、eval_mem_filter.py、eval_maj_temp_grid.py、eval_mix_vote.py、gen_hard_trajectories.py 等。

## 三、作者位次建议的依据框架

- **第一作者**：应为"研究问题的提出者 + 核心方法贡献的最终责任人"。核心方法贡献（MI-VAV、ORM 选择器）的**设计决策**由你拍板、**实现**由 AI 完成——按学术惯例（AI 非作者），第一作者位次应有充分依据（你的前期实验、决策链、资源、方向）。
- **通讯作者**：通常为导师（资源与指导）。
- **AI 声明**：目标 venue 若有 AI 使用政策（如 NeurIPS/ACL 的 AI 辅助工具政策），按政策在致谢或方法部分声明；可引用本台账作为 AI 贡献范围的记录。
- 若未来加入其他人类合作者（数据、写作、实验复现），建议按其实际贡献插值并更新本台账。

## 四、证据链索引（论文写作时逐条可查）

| 声明 | 证据位置 |
|---|---|
| 所有实验成绩 | records/experiments.jsonl（含口径注释）+ outputs/*/summary.json + official_result.txt |
| 方法与设计文档 | docs/MI_VAV_METHOD.md、docs/SELECTION_V2_PLAN.md、docs/RESEARCH_SYNTHESIS_20260815.md |
| 决策时间线 | git log（提交信息即决策记录）+ docs/ORCHESTRATION_PLAN.md、docs/STRATEGY_ADJUSTMENT_20260816.md |
| 代码 | src/ + finer_port/ + scripts/（git 可回溯每次修改） |
| 新颖性对比 | docs/RESEARCH_SYNTHESIS_20260815.md 第一部分（含 15+ 篇文献锚点） |

> 备注：本台账由 AI 按会话记录撰写，事实性条目均可由上述证据链核对；位次决定权在你与导师。
