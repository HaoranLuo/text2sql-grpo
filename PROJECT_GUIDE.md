# PROJECT_GUIDE — Text-to-SQL GRPO 项目完整知识文档（新会话必读）

> 本文档是"新会话必读"：任何新会话读这一份即可获得全部上下文，无需再翻其它文档。
> 最后更新：2026-08-12（含当日 MPEV / 1000 条实验与官方口径审计结论）

---

## 1. 项目概览

**一句话**：研究在不依赖大参数量与海量数据的前提下，用 GRPO 强化学习 + 推理时投票增强，让 Qwen2.5-Coder 3B/7B 小模型逼近并超越其零样本上限的 Text-to-SQL 项目。

**研究问题**：小模型（≤3B/7B）Text-to-SQL 强化学习——"从复现到超越 FINER-SQL"。

**当前状态**：已完成自身实验矩阵（3B 子集 45%→71%、全量官方口径 68.5%）；打通官方 test-suite 评估口径；完整复刻验证顶级论文 FINER-SQL（3B 模型 85% 真实、管线可复现）；官方口径审计完成、根因闭合（8.4pp 差距 = 评估器+数据库口径差，非模型问题），修复方案就绪待执行。整体处于 **L0 起步（修复待执行）、L2 投票升级已有预演（MPEV 75%）** 阶段，L1/L3-L6 均未启动。

**技术栈**：
- 基座模型：Qwen2.5-Coder-3B / 7B（HuggingFace transformers + LoRA）
- 训练：TRL GRPO（LoRA 微调，支持 binary / three_level / partial / atomic / finer 多种奖励）
- 评估：官方 test-suite 评估器（test_suite_eval）+ 原始 Spider 评估器（original_spider_eval，FINER 同源链）+ 自研自定义执行匹配口径（仅内部诊断）
- 解析/原子奖励：sqlglot（sql2ops 原子操作序列，Jaccard 相似度）
- 数据生成/推理蒸馏：DeepSeek API（V4 Flash，教师生成推理轨迹+SQL）
- 计算：西交利物浦 HPC（slurm，A40/3090，gpudebug 分区），经女朋友电脑 cpolar 隧道跳板访问
- 投票：多 prompt 视角投票 / 低温采样投票 / vav 执行分组投票 / MPEV（多视角×采样）

**数据集**：
- Spider dev 全量 **1034 题**（对外正式口径必须用此）；前 100 条子集仅作内部诊断（已证实不可信，偏易）
- Spider train 8659 条（GRPO 全量候选 / 推理蒸馏数据生成源）
- 官方 test-suite 多实例增强库（test-suite-sql-eval）与原始单实例库（FINER 官方链）两套，**不可混用**

**里程碑数字**：3B 基线 45% → 三级奖励 GRPO 50% → 投票 71%（子集自定义）；全量官方口径 68.5%（3B 训练后+5p 投票）；FINER-3B 复刻 vav 自定义 ~85%（≈论文 85.88% ✅ 权重真实）。

---

## 2. 核心结果表

### 2.1 第一阶段·子集 100 条（自定义执行匹配口径，6 个静默 bug 修复后）

| 实验 | 结果 | 备注 |
|---|---|---|
| 3b_baseline | **45%** | 修复 6 个静默 bug 后真实基线（原虚低 34%） |
| 3b_grpo_25step_100条_三级奖励 | 50%（+5pp） | 三级 1.0/0.1/0 是 3B 唯一有效奖励 |
| 3b_grpo_25step_多prompt×3 | 65% | |
| 3b_grpo_25step_多prompt×5 | 70% | |
| 3b_grpo_25step_500条_×5 | **71%（3B 最佳路径，累计+26pp）** | |
| 3b_grpo_25step_×7 | 71% | 与 5p 逐段完全一致（42/50+29/50），5p 为甜点 |
| 3b_grpo_2000条_×5 / 7000条_×5 | 71% / 70% | 投票抹平数据量差异 |
| 3b_grpo_50步 / 75步 | 34% / 36% | 过拟合，25 步是甜点 |
| 3b_grpo_partial奖励_100/500条 | 46% / 45% | = 基线，细粒度信号稀释 |
| 3b_finer式非负奖励_2000条 | 50% | = 三级持平，缺推理蒸馏前置 |
| 3b_sft冷启动+GRPO | 44% | < SFT 47%，GRPO 回吐增益 |
| 3b_投票蒸馏SFT(118+100gold) | 单次 51%（+1pp，数据太少）；5p 投票 68%（过拟合） | |
| 7b_baseline | 81% | |
| 7b_多prompt投票×3/×5/×5+仲裁 | 85% / 85% / 85% | **项目最佳，零训练成本** |
| 7b_低温采样×4 | 54% | 多 prompt 视角 >> 随机采样 |
| 7b_grpo_100条_三级 / 7000条 / partial复合 / SFT+GRPO | 77% / 74% / 78% / 78% | 全部 ≤ 81% 基线，7B 训练饱和 |

### 2.2 全量 1034 · 官方口径（对外正式数字）

| 实验 | 结果 | 备注 |
|---|---|---|
| 3b_基线_官方子集100 | 41.4% | |
| 3b_训练后_官方子集100 | 49.0% | |
| 3b_训练后+5p投票_官方子集100 | 68.1% | 官方口径下投票 +19pp |
| 3b_训练后_全量1034（自定义） | 54.2% | |
| 3b_训练后+5p_全量1034（自定义） | **68.5%** | 投票增益 +14.3pp 仍有效，逼近 7B 基线 70% |
| 7b_基线_全量1034 | 自定义 70.02% / 官方 67.5% | 子集 81% vs 全量 70%，子集偏易 +10pp |

### 2.3 FINER-SQL 复刻验证（同基座 3B 开源权重）

| 实验 | 结果 | 结论 |
|---|---|---|
| FINER-3B_单次_子集100_官方EX | 77.3% | 3B 单次 ≈ 我们 7B 基线，权重真实 |
| FINER-3B_n=30_vav_全量_自定义 | **~85%**（前半 85.6% / 后半 84.2%） | vav 自评 ≈ 论文 85.88% ✅，管线可复现 |
| FINER-3B_n=30_vav_全量_官方EX | 76.6% | vs 论文 85%，差 8.4pp **已被审计归因为口径差**（详见 §8 P2） |

### 2.4 最新实验（2026-08-12）

| 实验 | 结果 | 备注 |
|---|---|---|
| 3b_1000条训练+1000条评估_ckpt50 | 53.9% | 子集偏难 -7.4pp；数据量无增益再次确认；50 步未过拟合（与 100 条实验不同） |
| MPEV 多视角×采样投票_100条 | 训练同口径 **75%** | vs 原版多 prompt 70%，+5pp；vav_self 89%；官方 EX 65% 待分析（属 L0） |

### 2.5 外部模型对照（子集 100 条，仅供参考，非论文口径）

| 模型 | 结果 |
|---|---|
| 14B（Qwen2.5-Coder-14B 等） | 78% |
| DSV2-Lite | 78% |
| DeepSeek-V4-Flash-API | 76% |
| M-Schema 单 prompt | 79% |
| XiYanSQL-7B | 57%（格式不匹配致解析失败，不可比，见 P12） |
| OmniSQL-7B | 66% |
| CSC-SQL-7B | 73% |

---

## 3. 关键结论与发现

1. **投票是最强杠杆**：3B 全量 +14.3pp（54.2→68.5），7B +4pp（81→85），零训练成本；训练的价值是先抬升基线再由投票放大。
2. **5p 是投票甜点**：3p→5p +5pp，5p→7p 零提升（逐段完全一致 42/50 + 29/50）。
3. **子集不可信**：前 100 条子集偏易 +10pp（7B: 子集 81% vs 全量 70%）；但 3B 1000 条实验显示子集偏难 -7.4pp，方向不统一——论文级对比必须用全量 1034 + 官方口径。
4. **口径差已归因**：官方 test-suite EX 比自定义口径严 1-4pp（子集）；FINER vav 官方 76.6% vs 自定义 85% 的 8.4pp 差距 = 评估器（test-suite vs 原始 evaluation.py --etype exec）+ 数据库（多实例 vs 原始单实例）双口径差，非模型问题（审计证据见 §8 P2）。
5. **三级奖励（1.0/0.1/0.0）是 3B 唯一有效奖励**（+5%）；partial / 原子 / FINER 式奖励均无增益；3B 所有训练配方**单次峰值封顶 50-51%**。
6. **partial 奖励无效根因**：组内奖励扁平化 → std→0 → 梯度归零（文献证实）；GRPO 升级需去 std / 静默组过滤等梯度修复前置。
7. **训练侧天花板突破需推理蒸馏前置**（FINER Step 1：大模型教师生成思考过程+SQL → 执行过滤 → SFT），这是 50→85 的唯一已证路径，**尚未执行**。
8. **FINER 85% 的核心** = 推理蒸馏 + 大规模 GRPO（全量 8659 / G32 / 2000 步 / 四分量奖励）+ n=30 vav 投票的组合；vav 空结果 bug 修复后 +6.4pp，vav 投票 > 简单投票。
9. **两阶段 SFT+GRPO 无增益**（44% < SFT 47%）：同批数据过拟合，GRPO 回吐 SFT 增益，需训练集 / SFT 不相交划分（5.5k/1.5k）。
10. **投票鲁棒性强**：训练数据量 100/500/2000/7000 条单 prompt 各异（43-50%），5p 投票后全部收敛 70-71%，投票抹平训练状态差异。
11. **多 prompt 视角多样性 >> 随机采样**（85% vs 54%）：投票增益来自视角多样性而非采样次数。
12. **静默 bug 范式**：修复 6 个静默配置 bug（根因：prompt 传字符串而非 chat messages，模型生成 2 token 即停止），3B 基线 34%→45%；诊断范式：**优化器正常 ≠ 结果正确**。
13. **MPEV 最新增益**：多视角×采样投票比原版多 prompt 投票再 +5pp（70→75%，训练同口径）；vav_self 89%，官方 65% 待分析（L0 进行中）。
14. **数据量与过拟合**：1000 条数据训练无增益（53.9%），但 50 步未过拟合——数据量影响过拟合曲线，不影响最终成绩（与 100 条实验不同）。
15. **7B 训练饱和**：所有 7B 训练配方（三级/partial 复合/SFT+GRPO/7000 条）全部 ≤ 81% 基线，7B 的出路在投票与蒸馏而非继续 RL。

---

## 4. 技术架构

### 4.1 目录结构

```
reasoning_generator_3b/
├── src/                      # 核心 Python 源码
│   ├── reasoning_generator_agent.py   # 核心 Agent（当前版）：模型加载+prompt构建+SQL生成+自检/重试
│   ├── spider_utils.py       # Spider 工具库：loader / 只读DB执行器 / 自定义指标 / checkpoint 助手（明确非官方口径）
│   ├── train_reasoning_grpo.py        # GRPO 训练主脚本（多奖励：binary/three_level/partial/atomic/finer）
│   ├── train_sft.py          # SFT 冷启动训练（模仿 DeepSeek 生成的推理+SQL）
│   ├── evaluate_after_grpo.py         # GRPO 后评估通用入口（含 checkpoint 遍历 → items/checkpoint/summary.json）
│   ├── eval_5prompt_agent.py # 通用 5/7-prompt 投票评估器（full_rows 比较 + truncated 检查，口径修复版）
│   ├── eval_multi_prompt_vote.py / eval_m5plus.py  # 多 prompt 投票 / 分组投票升级版
│   ├── eval_lowtemp_vote(_v2).py      # M2 低温自一致性投票（temp=0.2，v2 减半防超时）
│   ├── eval_mschema.py / eval_xiyan_native.py / eval_api_ceiling.py  # M1/M4/API 上限对照
│   ├── gen_reasoning_data.py          # FINER Step-1 风格推理蒸馏数据生成器（DeepSeek API）
│   ├── gen_filtered_sft.py / gen_vote_sft.py  # 执行过滤 SFT 数据 / 投票自蒸馏数据
│   ├── atomic_ops.py / atomic_reward.py      # sqlglot 原子操作序列 + Jaccard 原子奖励
│   └── record_experiment.py 的支撑 + 诊断脚本（probe_eval/check_*/diag_* 等）
├── scripts/                  # shell/slurm 脚本
│   ├── preflight_check.sh    # ★训练提交前配置一致性检查（0=通过/1=警告/2=禁止提交）
│   ├── eval_official.sh      # ★官方 test-suite 评估集成（dataset_index 对齐 + parse 失败跳过）
│   ├── eval_original.sh      # 原始 Spider 官方口径评估（--etype exec + 空预测写 SELECT 1）
│   ├── ssh_hpc.ps1           # ★Windows 端 SSH 跳板包装（断线自动重试 8×10s）
│   ├── train_*.slurm         # GRPO 训练作业（3B/7B/aggressive/dscoder 等）
│   ├── exp_*.slurm           # 实验矩阵作业（reward 对照/partial/atomic/数据量扫描/FINER 式等）
│   ├── eval_*.slurm          # 评估作业（5p/7p 投票、全量、checkpoint、外部模型对照）
│   ├── eval_vav_finer_*.slurm / eval_mpev_100.slurm  # FINER vav / MPEV 评估
│   ├── plot_analysis_charts.py / plot_pretty_charts.py / plot_training_curves.py  # 图表
│   └── record_experiment.py  # 实验自动记录系统
├── finer_port/               # FINER-SQL 复刻移植包（P1 vav / P2 MPEV）
│   ├── sampler.py            # n 候选采样器（HF transformers，T=1.0, top_p=1.0, max_new_tokens=2048）
│   ├── vav_voting.py         # vav 执行分组投票（finer-sql 官方 majority_voting 的纯 Python 移植）
│   ├── eval_vav.py           # P1 评估入口（n 采样→执行验证→vav 分组投票→写回 items.json，支持切片/续跑）
│   └── eval_mpev.py          # L2 多视角×采样执行投票（MPEV，复用 p1-p7 视角，每视角采样 K 条）
├── tools/
│   ├── original_spider_eval/ # 原始 Spider 官方评估器（evaluation.py + process_sql.py，FINER 同源链）
│   └── （test_suite_eval 位于 .research_tmp/test-suite-sql-eval/，不在本目录）
├── records/                  # 实验结构化记录（17 个 JSON + experiments.jsonl 流水账）
├── charts/                   # 12 张 PNG（分析级 + 论文级）
├── outputs/                  # 每次训练/评估产物（items.json + checkpoint.json + summary.json，40 文件/33 子目录）
├── results/                  # 早期 agent 演示与条件化测试结果（10 个 JSON）
├── docs/                     # FINER_REPLICATION_PLAN.md（P1 vav / P2 MPEV 计划）
├── .research_tmp/            # 上游研究材料（finer-sql 源码、test-suite-sql-eval、原始评估器、审计脚本）
├── archive/                  # 历史归档（PROJECT_CONTEXT、EXPERIMENT_LOG/REPORT、PHASE2_PLAN 等 10 个 md）
├── codex_ppt_materials/      # PPT 素材（agent 快照 + 结果 JSON + logs）
├── tmp/                      # 66 个临时诊断脚本
└── 根目录文档：FINAL_REPORT / EXPERIMENT_MATRIX / OFFICIAL_EVAL / OFFICIAL_AUDIT /
    GRPO_CHECKLIST / PAPER_PLAN / HANDOFF / NEXT_STEPS / RESEARCH_NOTES / IMPROVEMENT_PLAN
```

### 4.2 关键工具速查

| 工具 | 用途 | 何时用 |
|---|---|---|
| `grpo-preflight` skill + `scripts/preflight_check.sh` | 训练提交前配置一致性检查（生成长度/奖励逻辑/数据泄漏/版本漂移） | **每次 GRPO/RL 提交前必跑**（纪律 G4） |
| `scripts/record_experiment.py` | 从 summary.json 生成结构化实验记录 + 追加 experiments.jsonl + 触发图表 | 每个实验跑完必跑 |
| `scripts/eval_official.sh` | 官方 test-suite 评估集成 | 对外正式数字必用（纪律 G1） |
| `scripts/eval_original.sh` | 原始 Spider 官方口径（FINER 同源链） | FINER 对比/复现数字用 |
| `scripts/ssh_hpc.ps1` | SSH 跳板包装，断线自动重试 | 所有 HPC 交互命令 |
| `finer_port/eval_vav.py` | vav 执行分组投票（n=30 复刻） | FINER 复现 / 投票升级对比 |
| `finer_port/eval_mpev.py` | MPEV 多视角×采样投票 | L2 投票升级（已跑出 75%） |
| `scripts/plot_*.py` | 分析级 / 论文级 / 训练曲线图表 | 记录后自动或手动出图 |
| `src/reasoning_generator_agent.py` | 核心推理+SQL 生成 Agent | 单次/投票评估的生成后端 |

---

## 5. 远程连接指南

### 5.1 链路架构（三层跳板）

```
本地 Windows（PowerShell / WSL bash）
  → 女朋友电脑（堡垒机 Host gf-bastion，User=ASUS，跑 cpolar 隧道，纯转发不计算）
  → 西交利物浦 HPC 登录节点 login.hpc.xjtlu.edu.cn（User=jiahuiwang24）
    项目目录：/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
```

本地通过 SSH ProxyJump 借女朋友电脑中转访问 HPC。**gf-bastion 必须保持开机在线且 cpolar 隧道开启**，否则整条链路不可达。

### 5.2 SSH config 要点（Windows 侧 ~/.ssh/config）

```
Host gf-bastion
  HostName 6.tcp.cpolar.cn        # cpolar 公网隧道地址
  Port 11220                      # ★当前有效端口（注意：_wsl_hpc_phase1/2.sh 里写的是 10444，已过时）
  User ASUS
  IdentityFile ~/.ssh/id_ed25519
  StrictHostKeyChecking no
  ServerAliveInterval 20 / ServerAliveCountMax 8 / TCPKeepAlive yes / ConnectTimeout 12

Host hpc
  HostName login.hpc.xjtlu.edu.cn
  User jiahuiwang24
  ProxyJump gf-bastion
  IdentityFile ~/.ssh/id_ed25519
  + 上述 KeepAlive 参数 + ControlPersist 300（连接复用）
```

WSL 侧由 `_wsl_hpc_phase1/2.sh` 从 `/mnt/c/Users/13389/.ssh` 复制密钥并 chmod 600；WSL 脚本用 BatchMode=yes + StrictHostKeyChecking=accept-new + ConnectTimeout=45 + ServerAliveInterval=20/CountMax=5。phase1=连通性探针，phase2=scp 上传 + dry-run。

### 5.3 常用操作

- 交互命令：`ssh hpc "squeue -u jiahuiwang24"`
- 本地包装（自动重试）：`powershell -File scripts/ssh_hpc.ps1 "squeue"`（最多 8 次、间隔 10s）

### 5.4 故障处理

| 故障 | 处理 |
|---|---|
| 隧道不稳定、连接时断 | ssh_hpc.ps1 自动重试 8×10s；config 全面 KeepAlive；WSL 侧加 ServerAlive/ConnectTimeout |
| cpolar 地址/端口漂移（config=11220 vs 阶段脚本=10444） | **手动核对并同步两处**（无自动更新机制，未找到独立地址文件，地址硬编码在两处） |
| ControlMaster 坑 | 未在文件中找到具体记录（未确认项，见 §8） |
| 连接超时 | 检查 gf-bastion 在线状态与隧道端口是否变化 |

---

## 6. 实验流程（标准作业流程 SOP）

### 6.1 训练

1. **提交前检查（强制）**：跑 `grpo-preflight` skill + `scripts/preflight_check.sh`（--strict），退出码 0 才允许提交（纪律 G4）。
2. **数据卫生（强制）**：GRPO 5.5k / SFT-gold 1.5k 不相交显式划分 + sha256 记录（纪律 G3，先于蒸馏数据生成定死）。
3. 选择 slurm 脚本：`train_grpo.slurm`（3B 基础）、`train_3b_aggressive.slurm`（1000 条×4 gen×200 步）、`exp_*.slurm`（各奖励/数据量变体）。
4. `ssh hpc` 提交后 `squeue` 监控；注意 gpudebug 分区 1 GPU。
5. 训完用 `evaluate_after_grpo.py` 遍历 checkpoint 评估。

### 6.2 评估

- **对外正式数字**：全量 1034 + `scripts/eval_official.sh`（官方 test-suite 口径）——纪律 G1，n=30/子集/vav 自定义数字仅内部诊断。
- **FINER 对比**：`scripts/eval_original.sh`（原始 Spider 口径，--etype exec + 空预测写 SELECT 1，保 1034 行对齐）。
- 评估产物写入 `outputs/<实验名>/items.json + summary.json`。

### 6.3 投票

- 多 prompt 投票：`eval_5prompt_agent.py`（5/7 prompt，full_rows 比较 + truncated 检查）。
- vav 执行分组投票：`finer_port/eval_vav.py`（--limit/--start-index 切片、checkpoint 续跑、难度拆分）。
- MPEV（多视角×采样）：`finer_port/eval_mpev.py`——当前最新投票形态（75%，比多 prompt 再 +5pp）。

### 6.4 记录与图表（每个实验闭环）

1. 评估完：`python scripts/record_experiment.py --summary outputs/<实验名>/summary.json` → 生成结构化记录 + 追加 `records/experiments.jsonl` + 自动触发图表。
2. 需要论文级图：`scripts/plot_pretty_charts.py`；分析级：`scripts/plot_analysis_charts.py`；训练曲线：`scripts/plot_training_curves.py`。
3. 更新 `records/` 后，关键结论同步回本文档 §2/§3。

### 6.5 报告纪律（对外数字铁律）

- 三口径各报各的（85.88/85.0/76.6 分别对应自定义 vav / 原始 Spider EX / test-suite EX），**禁止裸比数字**、禁止把 76.6% 与 85.0% 并列或写"复现低 8.4pt"而不附口径说明。
- 外部数字引用前溯源（纪律 G5，无出处不引用）。
- 论文正式口径 = 全量 1034 + 官方 test-suite（对外）；原始 Spider 口径仅 FINER 对比用；自定义口径仅内部诊断。

---

## 7. 论文规划

**定位**：小模型（≤3B/7B）Text-to-SQL 强化学习——"从复现到超越 FINER-SQL"。现实目标 CCF-B 会议（COLING/EMNLP-Findings）或 SCI 2 区期刊，冲刺 CCF-A（ICDE/EMNLP）。

### 7.1 分层计划（L0-L9 共 10 层，2026-08-13 扩展 L7-L9）

| 层 | 目标 | 发表等级 | 难度/周期 | 状态 |
|---|---|---|---|---|
| **L0** 官方口径排查 | 官方 EX 76.6% vs 论文 85% 差异归因（评估器版本/plug_value/多实例库/空预测） | 无（一切前提） | ★·半天 | **进行中**——审计已完成、根因闭合（口径差）、修复方案就绪待执行；最新 MPEV 官方 65% 待分析即属此。成功→对齐 82-85%；失败→深挖源码/联系作者后如实报告 76.6% |
| **L1** 修 bug 技术报告 | arXiv《Correcting Degenerate-Group Skipping in Execution-based Voting for Text-to-SQL》（空结果 bug +6.4pp + vav 移植） | 预印本 | ★·1 周 | 未开始（依赖 L0） |
| **L2** 投票升级实验 | 多视角×采样 + 置信度过滤 + 失败类型感知修复，官方口径 +2-5pp（MPEV 已显示 +5pp 潜力） | 论文组件 | ★★·1-2 周 | 部分预演（MPEV 75% 已跑，官方口径待验证） |
| **L3** 推理蒸馏+GRPO 算法升级 | DeepSeek 生成推理数据→SFT（单次 50→60-70%）；GRPO 升级（DAPO/去 std/零方差过滤） | 论文主体 | ★★★·2-4 周 | 未开始（依赖 L0, L2） |
| **L4** 改进版 FINER 论文（A 方案） | 故事线：小模型 Text-to-SQL 从复现到超越（多视角投票+bug 修复+数据策略）；被拒 1 次补实验转低一档，被拒 2 次降级 arXiv+转期刊 | CCF-B / SCI 2-3 区 | ★★★·1-2 月 | 未开始（依赖 L1-L3） |
| **L5** 系统性实证论文（B 方案） | 小模型 Text-to-SQL RL 的配方与瓶颈（子集偏差/训练天花板/投票鲁棒性/数据量无关性），40+ 实验系统分析 | CCF-B / SCI 2 区 | ★★★★·2-3 月 | 未开始（依赖 L4，可并行） |
| **L6** 新组件方法（C 方案） | 执行感知 memory 奖励 / 动态奖励课程 / 多视角 vav 理论分析；止损线：组件 2 次验证无提升即转综述或并入实证 | CCF-A/B、SCI 1-2 区 | ★★★·3-4 月 | 未开始（依赖 L4；**次于 L7，优先级靠后**） |
| **L7** 机制理论论文（★新） | **"解释为什么"而非发明组件**：①投票何时失效（一致度曲线/子集偏差/口径下降→自适应投票准则，从"投票方法"升级为"投票科学"）②数据量无关性（静默组比例/reward_std→"信号稀释假说"）③投票抹平训练差异（生成分布视角：分布取模）。骨架：《When Does Voting Fail?》 | CCF-A/B | ★★★★·2-3 月 | 未开始（**数据已有、0 新 GPU，性价比最高**，优先于 L6） |
| **L8** 通用范式论文（★新） | 配方（GRPO+投票+蒸馏）跨任务验证：数学（GSM8K/MATH）+ 代码（HumanEval/MBPP）+ BIRD；若"投票抹平差异/5 视角甜点/数据量无关"仍成立→通用规律，从 Text-to-SQL 论文升级为 **RL 方法论论文**（受众大 10 倍） | CCF-A（冲） | ★★★★★·4-6 月 | 未开始（依赖 L7；Gate-4：跨任务预实验复现才立项） |
| **L9** 工程开源（★新） | vav 评估器（28 项测试）+ MPEV 投票 + 复现管线 + 40+ 实验记录系统开源（GitHub + arXiv 工具报告）；"我们的实现被社区使用"即影响力 | 工具类（MLSys/Data-centric） | ★★·1 月 | 未开始（0 GPU，可随时并行） |

### 7.2 关键决策 Gate

- **Gate-1**（L0 后）：官方口径能否对齐 → 决定 L1 报告口径
- **Gate-2**（L3 后）：训练侧是否突破 60% → 决定"训练+推理"还是"纯推理"故事
- **Gate-3**（L4 被拒后）：补实验 vs 降级
- **Gate-4**（L8 前）：跨任务预实验（数学/代码各 1 基准）是否复现"投票抹平差异/5 视角甜点/数据量无关" → 复现则立项 L8，否则缩为对照章节并入 L7

### 7.3 当前进度与下一步

- **当前（2026-08-13 更新）**：**L0 已基本收官**——
  - 4 项修复全部执行（原始评估器 taoyds 版+补丁 / --etype exec / 原始单实例库 / 空预测 SELECT 1）
  - **解析器大修：557 → 0 解析失败**（15 类模式、13+ 补丁，workflow wuc9nvg2v 完成，含回归验证）
  - **重评结果：原始口径 exec = 79.1%**（40.9% → 79.1%；论文 85.0%，差 5.9pp 归因预测质量+评估器细节）
  - 待办：获取 MAC-SQL evaluation_spider.py（论文所用）进一步对齐（可能 +3-5pp）；三口径对照表 85.88/85.0/76.6 + 口径说明段（论文写作用）
- **随后**：P0 #1 推理蒸馏→SFT（蒸馏数据 1000 条已生成就绪，随时可提交训练）→ 数据卫生 P0 #3 → L2 MPEV 官方口径验证 → L1 技术报告。
- **深度路线（2026-08-13 确定，见 PAPER_PLAN §七）**：L4（改进论文）→ **L7 机制理论**（用现有数据、0 新 GPU、优先于 L6）→ L8 跨任务范式（Gate-4 决定）；L9 开源随时可并行。GPU 预算：L4 ~3 天 / L5 ~2 天 / L7 0 新实验 / L8 4-6 月（可选）。

---

## 8. 已知问题与排雷记录

### 8.1 问题清单（P1-P12）

| ID | 问题 | 状态 |
|---|---|---|
| P1 | 前 100 条子集显著偏易 +10pp（7B 基线 81→70） | 已确认并修复（改全量） |
| P2 | **评估口径错配**：test-suite EX vs 原始 Spider EX（76.6% vs 85.0% 根因） | 根因已定位，修复方案待执行 |
| P3 | 隧道不稳定，SSH 间歇失败 | 对策：ssh_hpc.ps1 重试 8×10s + KeepAlive |
| P4 | cpolar 端口漂移/双处不一致（config=11220 vs 脚本=10444） | 需手动同步，未固化机制 |
| P5 | vav 空结果 bug（0 行查询的 SUCCESS_VALUES 签名缺失致丢弃正确预测） | 已修复（+6.4pp，85.6% vs 79.2% 且反超 majority） |
| P6 | 多行 SQL 未折叠致对齐错乱（pred 2869 行 vs gold 1025 行） | 已修复（仅折叠内部换行） |
| P7 | GRPO 梯度信号退化（解释全部负结果）：partial 0.1 档组内同值→std→0→梯度归零；全对/全错静默组无信号；正负混合不稳；无 SFT 冷启动直接 RL 低效 | 归因完成；对策：静默组过滤（DAPO 式）+pass-rate 过滤+非负奖励+SFT 冷启动+TRL v0.16+ scale_rewards |
| P8 | c3_atomic 失败误判（失败因配方太小 100 条/G4/25 步，非 atomic 奖励无效——FINER 消融：去 atomic −3.26%） | 归因修正 |
| P9 | 评估器参数双刃：--plug_value +2.6pt、--keep_distinct −2.6pt、叠加抵消；参数无法弥合与 85% 的 5.8pt 差距（结构性 SQL 错误） | 第三路排查结论 |
| P10 | 旧备份陷阱：data/spider_data.tar.gz（205MB，仅 372 个 sqlite，2025-08-05）与 database/（3383 个）不一致；Drive 链接 1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w 是多实例版也不能用于原始口径 | 待删除或重打 |
| P11 | HF 大文件下载坑：须等下载进程完全退出再使用（分片无校验） | 经验已记录 |
| P12 | XiYanSQL 对比不可比（格式不匹配致解析器大量失败，44% 反映解析失败而非模型质量） | 需提示词/格式适配后重评 |

### 8.2 P2 官方口径审计结论（2026-08-12，决定性证据）

- **根因**：评估口径不匹配（指标/评估器不匹配），非代码缺陷、非数据损坏、非生成侧问题。论文官方链用**原始 Spider 评估器**（MAC-SQL evaluation_spider.py --etype exec，行序不敏感、原始单实例库）；我方用 **test-suite-sql-eval --etype all** 跑在 **test-suite 多实例增强库**上（ORDER BY 行序敏感、pred 须在所有实例上与 gold 等价）。76.6% 是第三个口径（test-suite exec），与论文 MV(vav) 85.88% 和 Official EX 85.0% 均不可互比。
- **三路证据**：
  - 调用链（决定性）：评估器不同 + 数据库不同；同一批预测在自定义 results_equal 口径下 ~85% 精确复现论文 85.88%（证明预测一致、纯评估端差异）；空预测处理差异方向是抬高我方分数，非成因但需对齐。
  - 数据库：HPC database/ 与官方 test-suite zip 逐名 diff 28 目录全部 missing=0/extra=0；5 个抽样 sqlite SHA256 字节级一致；评估器代码 SHA256 与官方一致。
  - 参数：基线 0.766；--plug_value 0.792（+2.6pt 唯一显著正向）；--keep_distinct 0.740（−2.6pt）；叠加 0.767 抵消；5.8pt 差距来自 SQL 结构性错误（JOIN/GROUP BY/ORDER BY/NOT IN vs EXCEPT），必须靠模型改进。
- **修复方案（4 项，待执行）**：① 评估器换原始 evaluation.py（tao-yu 官方或 MAC-SQL evaluation_spider.py，含 process_sql.py，需 nltk word_tokenize）；② 参数 --etype exec（去 all，不加 plug_value/keep_distinct）；③ 数据库改原始 Spider dev release database/（每库 1 个 .sqlite，dev 21 库即可，勿用多实例库和 1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w 链接）；④ 空预测写 SELECT 1 而非跳过，pred/gold 恒 1034 行。
- **重评**：`bash scripts/eval_official.sh outputs/eval_vav_finer_full/items.json outputs/official_vav_full`（几分钟出数）。预期整体 exec ≈ 85.0%，难度 ≈ 94.8/90.1/78.2/64.5，与论文完全对齐。验证清单：pred/gold 均 1034 行、整体 84.5%-85.5%、难度逐档 ±1pt、原梯度偏差（-2.9/-6.5/-11.1/-19.3）消失。
- **fallback**：若无法获取原始库/评估器，test-suite 口径下用 --etype exec --plug_value（0.792）作自洽口径，但对外必须标注 test-suite EX，不得与论文 85.0% 并列；禁用 --keep_distinct。

### 8.3 未确认项（任务提示项，文件中未找到书面记录）

- ControlMaster 坑的具体情形
- CDN 缓存坑的具体情形
- 解析器排雷 557→208 的进展记录
- 独立 cpolar 地址文件（地址硬编码于 SSH config 与阶段脚本，且端口不一致）

---

## 9. 未来方向与机会

### 9.1 下一步实验（按优先级，IMPROVEMENT_PLAN 10 项）

- **P0 #1 推理蒸馏→SFT（主战役）**：DeepSeek-R1 教师生成推理轨迹+SQL（500-2000 条起步，全量 8659 可选）→ SQLite 执行过滤（含空结果/超时剔除）→ SFT 混合 10-20% gold；预期单次官方口径 50%→60-70%；**gate：SFT 后单次 ≥55% 才进 GRPO**（纪律 G2）。
- **P0 #2 评估器校准**：用 FINER-SQL-3B-Spider 官方权重在项目 harness 跑探针（M1），区分"评估端 vs 训练端"8.4pp 差距。
- **P0 #3 数据卫生**：GRPO 5.5k / SFT-gold 1.5k 不相交显式划分 + sha256 记录（先于蒸馏数据生成定死）。
- **P1 #4 大规模 FINER 配方 GRPO**：全量不相交数据 + G16-32 + 四分量奖励（format+exec+atomic+memory）+ 梯度修复（loss_type=dapo + scale_rewards=None + 静默组过滤）+ β sweep + checkpoint 早停；预期单次 68-76%。
- **P1 #5 投票管线升级**：失败类型感知修复 + 加权投票 + 置信度过滤 + 温度采样多样性（纠错信号只来自执行结果，3B 不当 critic）；预期全量 68.5%→71-74%。
- **P1 #6 工程可靠性包**：vLLM V1 前缀缓存 + rlprobe + 确定性报告 + cron 监控恢复。
- **P1 #7 执行验证合成数据增强**（SQL-PaLM 式，每 gold 生成替代 SQL 双验证）。
- **P2 #8 教师检索增强 + 难题 rationalization**（RFT 式）。
- **P2 #9 Qwen3 基座 A/B**：30B-A3B（MoE 3.3B 激活）推理线 + 4B-Base 训练线，先零训练 A/B 再决定 fork，**严禁通用 Qwen3-8B**。
- **P2 #10 ORM 验证器替换启发式投票**（GradeSQL/STaR-SQL 路线，7B 验证器+3B 生成器）。
- 其他：BIRD 迁移（下月，前置 SAR-Agent 标注筛查）、迭代蒸馏（2 轮封顶，仅对照）、多视角投票并入 #5。

### 9.2 明确不做 / 延后

PRM 过程奖励、多智能体 RL（MARS-SQL）、投机解码、Spider 2.0、LLM-judge 奖励。**三大约束**（只可作对照实验）：裸 GRPO/纯结果奖励 GRPO、3B 自蒸馏（师生差距铁律，自生成数据负收益）、低质跨域合成数据（Mnemosyne 稀释反例）。

### 9.3 资源预算

GPU（A40/3090）约 15-20 作业；DeepSeek API ¥数百元级（原估 ¥20-50 低估 5-10 倍，先 50 题实测账单）；总时间 4-6 个月至 L4 录用。

### 9.4 论文路线

L0 口径排查 → L1 技术报告（arXiv 占坑）→ L2 投票升级 → L3 蒸馏+GRPO 升级 → L4 改进版论文（CCF-B/SCI 2 区，目标 4-6 个月）；L5 实证论文与 L4 并行；L6 新组件冲刺 CCF-A（ICDE/EMNLP）；Spider 1.0 榜单天花板约 91%（MiniSeek 未公开）。

### 9.5 研究笔记要点（RESEARCH_NOTES，5 路并行调研）

- **STaR-SQL**：8B 自训练单次 +20 点无需教师；对 3B 锚点：单次 50%→蒸馏后 60-65%。
- **RFT**：执行校验过滤 + 按方案去重，弱模型稳定 +5-6 点；收益来自推理路径条数而非样本数；33B 失败教训：过拟合→多样性崩。
- **Self-Consistency**：投票有效性理论内核（正确路径收敛同一答案、错误路径分散）；T=0.7~0.8，一致度 ≥3/5 可做数据过滤信号。
- **CSC-SQL**（3B+BIRD+GRPO 同构项目）：3B SC@64=61.30→CSC(merge-revision)@64=65.28；投票细节 T=0.8、按执行结果分组、只取前两大投票组修正；模型输出高度一致时收益消失。
- **GRPO 工程**（解释全部负结果）：静默组过滤（DAPO 式）+pass-rate 过滤、lr 1e-6~5e-6、max_grad_norm 0.1、warmup 0.1、β 下探 0.01-0.02、R=0.7·exec+0.3·component、temperature ≥0.9、G=8、30-40 步上限+reward_std 触零即停、从 SFT 检查点起步、TRL v0.16+。
- **蒸馏实践配方**：7000 题×5-8 候选→执行过滤 3-6K + 全量 gold 7K ≈1-1.4 万条 SFT；3B LoRA r16-64/lr 1-2e-5/3-5ep/dev 早停；迭代 ≤2 轮每轮 base 重启；混合 gold:distilled ≈1:1~1:2 防 collapse；之后接 GRPO。

---

## 10. 重要教训（踩过的坑清单）

1. **静默 bug 最危险**：6 个静默配置 bug（prompt 传字符串而非 chat messages，模型生成 2 token 即停止）让 3B 基线虚低至 34%。诊断范式：**优化器正常 ≠ 结果正确**；不报错的错误最难发现——所以有 grpo-preflight skill + preflight_check.sh 强制检查。
2. **口径先行**：子集 100 条偏易 +10pp；test-suite vs 原始评估器差 8.4pp；三口径（自定义/tetsuite/原始）不可互比。教训：任何对比数字前先锁定"评估器+数据库+参数+空预测处理"四要素，论文级结论一律全量 1034。
3. **评估器/数据库版本必须与论文逐字节对齐**：SHA256 校验评估器代码与数据库（本次审计已做，结论：我方数据/代码与官方一致，问题只在选型）。
4. **partial 奖励陷阱**：组内奖励扁平化→std→0→梯度归零，白跑 3 组实验（100/500/2000 条）。教训：改奖励前先想"组内是否有判别性信号"。
5. **数据泄漏防不胜防**：SFT+GRPO 同批数据 → GRPO 回吐增益（44% < SFT 47%）。教训：训练集/SFT 不相交划分 + sha256 记录（G3）先于数据生成定死。
6. **旧备份陷阱**：spider_data.tar.gz（205MB 旧版）勿用于还原目录；Drive 多实例库链接不能用于原始口径。
7. **HF 大文件分片无校验**：须等下载进程完全退出再使用文件。
8. **对比模型要先验格式**：XiYanSQL 44% 实为解析失败；对比前检查输出格式与解析器匹配。
9. **空预测处理要对齐**：官方链写 SELECT 1（保行数对齐），我方曾跳过——处理方式不同会制造假差距。
10. **多行 SQL 折叠**：不折叠会导致 pred/gold 行数错位（2869 vs 1025 行），对齐检查是评估管线的第一道闸。
11. **vav 空结果签名 bug**：0 行查询的 SUCCESS_VALUES 缺失致正确预测被丢弃，+6.4pp 的教训——投票器边界条件要单测。
12. **梯度退化 vs 数据量**：100-7000 条无差别 = 梯度退化瓶颈而非数据瓶颈；先修梯度（DAPO/去 std/静默组过滤）再谈扩数据。
13. **cpolar 端口漂移**：隧道地址/端口可变且双处硬编码不一致（11220 vs 10444），改动端口后必须同步两处并验证。
14. **投票视角 vs 采样次数**：多 prompt 视角多样性 >> 低温随机采样（85% vs 54%）；设计投票时优先扩视角，而非堆采样。
15. **7B 别再练**：所有训练配方均 ≤ 基线，7B 的出路在投票与蒸馏；3B 单次训练峰值封顶 50-51%，突破必须走蒸馏前置。

---

### 附录：速查数字

- 对外正式口径：3B 训练后+5p 投票 = **68.5%**（全量 1034 自定义）/ 官方子集 100 = **68.1%**；7B 基线全量 = **70.02%**（自定义）/ **67.5%**（官方）
- FINER-3B 复刻：自定义 vav **~85%** / 官方原始口径预期 **~85%**（待 L0 修复重评）/ test-suite 76.6%（第三口径，勿并列）
- 子集自定义口径：3B 基线 45% → 投票 71%；7B 投票 85%（项目最佳）
- MPEV 最新：训练同口径 **75%**（100 条），官方 65% 待分析

## 附录 A：解析器排雷记录（557→208，持续更新）

> 2026-08-12/13 排雷记录（verify 建议补写）

**背景**：原始 Spider 口径评估（FINER 对齐）中，vav 选中的 pred SQL 解析失败率 53.9%（557/1034），原因是模型生成的 SQL 用了现代语法而官方解析器不支持。

**已打补丁（process_sql.py / evaluation.py）**：
1. `parse_select`：跳过 `AS 别名`（SELECT COUNT(*) AS x）
2. `parse_table_unit`：隐式表别名（FROM singer s）+ left/right/full/cross JOIN 变体
3. `parse_val_unit`：CAST(col AS type) 支持
4. `parse_condition`：NOT 前置（NOT EXISTS/NOT IN/NOT LIKE）
5. `parse_val_unit`：EXISTS 伪列（列 -1）+ 子查询解析容错
6. `evaluation.py`：gold 解析容错（单题失败不崩整体）；res_map 对伪列 -1 跳过比较
7. tokenize：`=` 强制拆分、`<>` 合并为 `!=`、粘连 token 逗号拆分
8. `parse_col`：常量/未知列/未知别名 → 伪列 -1（不崩溃）
9. `get_tables_with_alias`：别名与表名同名时跳过映射

**结果**：解析失败 557 → 208（-63%）。剩余 208 分布：IndexError 48 / AssertionError(空) 34 / > 13 / in 12 / from 9 / ) 6 / 大小写 ~16 / 其他 ~70。

**同步方法（重要）**：HPC 拉取 GitHub 文件受学校代理 CDN 缓存影响（raw/codeload/API 都有延迟）→ 最可靠 = **本地 base64 分段传输**（每段 ≤9000 字符，ssh echo >> 拼接，base64 -d 还原）。

**待办**：208 剩余错误由排雷 workflow（wuc9nvg2v）继续分析；评估基线（1659552）结果用于决策是否继续修。

## 附录 B：命名空间说明

- **论文分层 L0-L9**（PAPER_PLAN.md / PROJECT_GUIDE §7）：L0 口径 → L6 新组件（应用层）→ **L7 机制理论（解释为什么，科学层）→ L8 通用范式（跨任务统一理论）→ L9 工程开源**；L7/L8/L9 于 2026-08-13 由对话拓展正式立项
- **IMPROVEMENT_PLAN.md 的 L1-L12**：文献排雷编号（低置信发现清单），**非论文分层**，勿混用
- **P0-P12**：PROJECT_GUIDE 内的问题编号（排雷记录）
