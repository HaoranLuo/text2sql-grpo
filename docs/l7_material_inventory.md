# L7 机制论文素材盘点（《When Does Voting Fail?》）

> 2026-08-13 · 本地只读盘点（未跑 GPU、未改任何产物、未触 HPC）
> 范围：`records/`（17 JSON + experiments.jsonl）、`outputs/*/summary.json`、`docs/mpev_65_analysis.md`、`docs/research_track_20260813.md`、`PROJECT_GUIDE.md`、`PAPER_PLAN.md`、`.research_tmp/mpev_*` 本地副本
> 用途：L7 机制论文三核心问题的数据支撑度评估与缺口实验清单

---

## 一、三核心问题支撑度结论（摘要）

| # | 问题 | 结论 | 判定 |
|---|---|---|---|
| ① | 投票何时失效（一致度-正确率曲线 / 5p-7p / 子集偏差 / 官方口径下降） | 曲线数据**本地已有且逐题**：5p 一致度-正确率（100 题逐题，`outputs/eval_5p_p2a500/items.json`）+ MPEV 30 候选置信度逐题（`.research_tmp/mpev_items.json`）+ 官方 EX 逐题与 4 评估器变体（`.research_tmp/mpev_off_per_item.json` / `mpev_variants.json`）+ 子集偏差数字齐全。缺：其他 checkpoint 的逐题一致度（在 HPC 未拉回）、全量 1034 的一致性曲线、maj@K/温度扫掠。 | **部分充分** |
| ② | 为什么训练数据量无关（100/500/2000/7000 无单调趋势） | 结果层面充分：单 prompt 47/50/43/45（+基线 45、1000 条 53.9）均在 records。机制层面**不充分**：reward_std 只有 7B-100 条一份 20 步日志（`tmp/metrics_7b_100.json`，18/20 步 std=0）；3B 各数据量训练日志、静默组比例、熵日志本地均不存在（训练脚本仅 RewardStdGuard 打印 reward_std，熵从未记录）。"信号稀释假说"缺直接证据。 | **部分充分（结果）／不充分（机制）** |
| ③ | 为什么投票抹平训练差异（任何训练状态投票后 70-71%） | 子集 100 题上数字**齐全且全在本地**：单 prompt {45,47,50,43,45,46,44,50,51} → 5p 投票 {70,71,71,70,70,68}，7B 81→85；已出图 `charts/analysis_data_volume.png`。缺：全量 1034 只有 1 个训练状态测过 5p（54.2→68.5），其余训练状态全量投票未跑——"抹平"结论在正式口径上未闭合。 | **部分充分（子集充分，全量缺 4-5 个点）** |

---

## 二、论文图/表 → 数据来源 → 完整度 → 缺口与补救

> 路径均为项目内绝对路径的简写：`records/`、`outputs/`、`.research_tmp/`、`docs/` 均指 `C:\Users\13389\Desktop\女朋友\reasoning_generator_3b\` 下对应目录。HPC 侧同名文件见 `outputs/` 各目录（本地缺 items.json 的以「HPC」标注）。

| 论文图/表 | 数据来源文件 | 关键数字 | 完整度 | 缺口与补救实验 |
|---|---|---|---|---|
| **F1 一致度-正确率曲线（5p）** | `outputs/eval_5p_p2a500/items.json`（本地，逐题 `{di, match, votes}`）；已出图 `charts/analysis_vote_agreement.png`（脚本 `scripts/plot_analysis_charts.py:41`） | 0票 8题 0% / 1票 3题 67% / 2票 14题 64% / 3票 24题 62.5% / 4票 38题 84% / 5票 13题 100% | **基本**（仅 p2a500 单模型 × 100 题） | ① HPC 回捞 `eval_5p_p2a100/2000/7000`、`eval_5prompt_3b_trained`、`eval_5p_7b_base`、`eval_5p_sft_vote`、`eval_5p_grpo_sft_gold500`、`eval_7p_p2a500` 的 items.json（零 GPU，分钟级）→ 曲线×7 模型；② 全量 1034 曲线（回捞 `eval_5p_full_1034_3b/items.json`） |
| **F2 MPEV 执行分组一致性-正确率（30 候选）** | `.research_tmp/mpev_items.json`（本地 2.5MB，100 题逐题：`confidence`、`result_groups{size,is_correct}`、`vav_self_match`、`majority_self_match`、`degenerate_skip_applied`、`num_unique_sql`）+ `outputs/eval_mpev_100_3b/summary.json` | vav_self 89% vs majority 86%；平均置信度 0.59；low_conf 12 题；degenerate_skip 12 题；avg_unique_sql 21.76 | **充分（100 题逐题）** | 全量 1034 MPEV 未跑（E5，约 1-2 GPU 天，见下） |
| **F3 视角数 k 阶梯（3p/5p/7p）** | `records/eval_m5_3b_trained.json`（65%）、`outputs/eval_5p_p2a500/summary.json`（71%）、`outputs/eval_7p_p2a500/summary.json`（71%，与 5p 逐段一致 42/50+29/50，见 `eval_7p_p2a500_a/b`）；7B：`records/eval_multi_prompt.json` 85% / `records/eval_m5plus.json` 85% | 3B: 3p=65 → 5p=70/71 → 7p=71（饱和）；7B: 3p=5p=85（饱和） | **部分**（仅 {3,5,7} 三档、单 checkpoint、子集 100） | k∈{1,3,5,7,9} 扫掠 + 全量（并入 E2/E3） |
| **F4 官方口径下降（三口径 + 评估器变体）** | `docs/mpev_65_analysis.md`（完整交叉表与 24 题归因）；`.research_tmp/mpev_off_per_item.json`（逐题官方 EX + `n_instances/n_exec_fail/n_denot_diff/order_matters`）；`.research_tmp/mpev_variants.json`（逐题 4 配置）；`outputs/eval_mpev_100_3b/summary.json` | 89(vav) / 75(custom) / 65(官方)；变体 65/62/68/65；FINER vav 85.6+84.2 vs 官方 76.6（`records/experiments.jsonl` 注释） | **充分（100 题，执行级证据）** | 全量 1034 官方逐题（FINER vav a/b 的 HPC official_result 回捞，零 GPU）；MPEV 全量官方（E5） |
| **F5 子集偏差** | `records/eval_rev_7b_base.json`（81%）vs `outputs/eval_full_1034_7b/summary.json`（70.02%）；`outputs/eval_full_1034_3b_trained/summary.json`（54.2%）vs 子集 50%；`outputs/3b_1000_checkpoint-50/summary.json`（53.9%，子集偏难）；段内证据 `outputs/eval_7p_p2a500_a/b`（84% vs 58%，前 50 易后 50 难） | 子集偏易 +10pp（7B）／偏难 −7.4pp（1000 条），方向不统一 | **充分** | 无（写作时用全量 1034 + 官方口径即可） |
| **T1 训练数据量 × 评估方式双线** | `records/p2a_100.json`(47) / `p2a_500.json`(50) / `p2a_2000.json`(43) / `p2a_7000.json`(45)；投票侧 `outputs/eval_5p_p2a500/2000/7000/summary.json` + `records/experiments.jsonl`（eval_5p_p2a500 71 等）；已出图 `charts/analysis_data_volume.png` | 单 prompt 47/50/43/45（无单调）→ 5p 全 70-71 | **充分（子集 100）** | 全量版见 T2（E3）；每数据量仅 1 个训练 run，无种子重复（论文需注明或补 2-3 seeds） |
| **T2 全量 1034 训练状态矩阵** | `outputs/eval_full_1034_3b_trained/summary.json`（54.2%）、`outputs/eval_5p_full_1034_3b/summary.json`（68.5%）、`outputs/eval_sft_phase1_full/summary.json`（63.9% 自定义 / 官方 60.8%）、`outputs/eval_full_1034_7b/summary.json`（70.02%） | 3B: 54.2→68.5（+14.3pp）；7B: 70.02 | **部分**（仅 grpo_3b_3lvl/ckpt-25 一个状态有全量投票点） | E3：p2a100/500/2000/7000 + sft_phase1 的全量 5p 投票（纯推理 ≈1 GPU 天）+ 官方口径重评 |
| **T3 训练配方全景（三级/partial/FINER式/SFT+GRPO/投票蒸馏/1000条/t21）** | `records/p2c2_partial.json`(46)、`records/3b_c25_v2.json`(50)、`records/3b_baseline_v2.json`(45)、`outputs/e_finer_checkpoint-50`(50)、`grpo_sft_gold500`(44)、`eval_sft_vote`(51)、`3b_1000_checkpoint-50`(53.9)、`experiments.jsonl`（sft_phase1 63.9、grpo_t21 53.8/官方 49.4） | 3B 单次峰值封顶 50-51%（训练侧天花板）；t21 回吐 −10pp | **充分** | 无 |
| **T4 训练动态机制（reward_std / 熵 / 静默组比例）——Q2 直接证据** | 本地仅 `tmp/metrics_7b_100.json` + `tmp/metrics_7b_500d.json`（7B，20 步：loss 恒 0、reward_std 18/20 步 =0、kl、grad_norm）；`src/train_reasoning_grpo.py` RewardStdGuard（只打印不落盘）；P7 定性归因见 PROJECT_GUIDE §8.1 | reward_std=0 主导（梯度退化佐证，仅 7B） | **不充分**（3B 各数据量训练日志本地无；熵从未记录；静默组比例未记录） | E0 回捞 HPC 训练日志（若有）；否则 E4 补跑 3B ×{100,500,2000} 并插桩熵/静默组/reward_std（1-2 GPU 天） |
| **T5 外部模型对照（子集，非论文口径）** | `records/eval_api_ceiling.json`(76)、`eval_dsv2_100.json`(78)、`eval_mschema.json`(79)、`eval_xiyan_native.json`(57)、`experiments.jsonl`（14B 78、OmniSQL 66、CSC 73，PROJECT_GUIDE §2.5） | — | **充分** | 无（XiYanSQL 不可比，注意 P12 备注） |
| **T6 vav vs majority（退化组跳过贡献）** | `outputs/eval_vav_finer_a/summary.json`（vav 85.6 vs maj 83.7，degenerate 33/520）+ `_b`（84.2 vs 82.9）+ `eval_mpev_100_3b/summary.json`（89 vs 86）+ `eval_vav_finer_smoke/summary.json` | 全量 +2pp 级；P5 bug 修复 +6.4pp | **充分** | 无 |

**已存在的成品图（`charts/`，12 张）**：`analysis_vote_agreement.png`（F1）、`analysis_data_volume.png`（T1）、`voting_curve_v2.png`、`analysis_overfit.png`、`training_reward_compare*.png`、`experiment_comparison.png` 等——论文图可从这些底稿升级。

---

## 三、缺口实验清单（按成本升序）

| 优先级 | 实验 | 内容 | 成本 | 支撑问题 |
|---|---|---|---|---|
| **E0** | HPC 数据回捞（零 GPU，分钟级） | scp 以下 items.json/日志到本地：`outputs/eval_5p_p2a100|2000|7000`、`eval_5prompt_3b_trained`、`eval_5p_7b_base`、`eval_5p_sft_vote`、`eval_5p_grpo_sft_gold500`、`eval_7p_p2a500`、`eval_5p_full_1034_3b`、`eval_vav_finer_a/b`（各含 1034 题 `chosen_group_size`）、`official_result*.txt`、p2a_*/t21 训练日志（trainer 输出中的 reward_std 打印） | 0 GPU，SSH 10 分钟 | ①②③（一致性曲线×多模型、全量抹平、reward_std 机制） |
| **E1** | 离线重分析（零 GPU，~0.5-1 人天） | 基于现有逐题文件产出：各 checkpoint 一致度-正确率曲线、MPEV 置信度分桶正确率、low_confidence 12 题 vs 正常组、vav vs majority 逐题差值、段内偏差表 | 0 GPU | ①②③（论文图最终数据） |
| **E2** | 投票失败诊断曲线（纯推理，research_track Phase 3 已规划） | 最佳 checkpoint：maj@{8,16,32,64} × 温度{0.7,1.0} 网格，记录 Pass@K vs Average@K、Self-BLEU、执行分组数、绝对多数命中率 → 定位本项目饱和点并给出 maj@K 曲线 | ~0.5-1 GPU 天（gpudebug 推理插队） | ① |
| **E3** | 全量 5p 投票补跑（纯推理，零训练） | 对 p2a100/500/2000/7000 + sft_phase1 五个 checkpoint 跑全量 1034 的 5p 投票 + `eval_official.sh` 官方重评（每套投票 ~6.5h，实测 `eval_5p_full_1034_3b` 23620s；官方重评每套几分钟） | ~1-1.5 GPU 天 | ③（全量口径闭合）+ ① |
| **E4** | 训练机制复测（Q2 直接证据） | 3B GRPO ×{100,500,2000}（或 QAE 分位数 baseline A/B，research_track Phase 2）重训，日志插桩：逐步 reward_std、组内熵、静默组比例、正优势占比、执行分组数 | ~1-2 GPU 天（3 次小规模 3B 训练） | ② |
| **E5** | MPEV 全量官方（可选） | 1034 题 × 5 视角 × 6 采样 MPEV + 官方多实例 EX（`eval_official.sh`），把 F2/F4 扩到正式口径 | ~1-2 GPU 天（生成 31k 候选 + 多实例执行，有执行缓存） | ① |
| **E6** | 多实例投票变体（可选，属 L2 非 L7 必需） | 投票执行验证改用多实例库（P1#4，mpev_65_analysis.md §六），预期把官方 65% 提回 70+ | ~2-4 GPU 天 | ①（官方口径失效的"治本"证据） |

**成本合计**：L7 必需（E0+E1+E2+E3+E4）≈ **0 训练 + 3-4 GPU 天 + 1 人天**；PAPER_PLAN 原估 L7 "0 新 GPU" 仅对 E0/E1 成立，E2-E4 需少量推理/训练算力（可并入 L2/L3 主线顺带完成）。

---

## 四、关键文件路径索引

**逐题数据（本地已有）**
- `C:\Users\13389\Desktop\女朋友\reasoning_generator_3b\outputs\eval_5p_p2a500\items.json` — 唯一本地 5p 逐题 {di, match, votes}
- `C:\Users\13389\Desktop\女朋友\reasoning_generator_3b\.research_tmp\mpev_items.json` — MPEV 100 题逐题（confidence/分组/正确性）
- `C:\Users\13389\Desktop\女朋友\reasoning_generator_3b\.research_tmp\mpev_off_per_item.json` — 官方 EX 逐题 + 实例级明细
- `C:\Users\13389\Desktop\女朋友\reasoning_generator_3b\.research_tmp\mpev_variants.json` — 评估器四变体逐题
- `C:\Users\13389\Desktop\女朋友\reasoning_generator_3b\records\experiments.jsonl` — 全量实验流水账（含官方口径注释）
- `C:\Users\13389\Desktop\女朋友\reasoning_generator_3b\tmp\metrics_7b_100.json` — 唯一本地训练动态日志（7B）

**叙事/结论文档**
- `C:\Users\13389\Desktop\女朋友\reasoning_generator_3b\docs\mpev_65_analysis.md` — 三口径差距归因（Q1 官方口径下降的执行级证据）
- `C:\Users\13389\Desktop\女朋友\reasoning_generator_3b\docs\research_track_20260813.md` — 文献弹药（投票失败曲线/E2 规划、QAE/E4 规划）
- `C:\Users\13389\Desktop\女朋友\reasoning_generator_3b\PAPER_PLAN.md`（L7 节，124-135 行）— 论文骨架与三个候选问题定义
- `C:\Users\13389\Desktop\女朋友\reasoning_generator_3b\PROJECT_GUIDE.md` — §2 核心结果表 / §3 结论 1-15 / §8 问题 P1-P12

**在 HPC、需回捞（E0）**：`outputs/eval_5p_p2a100|2000|7000`、`eval_5prompt_3b_trained`、`eval_5p_7b_base`、`eval_5p_sft_vote`、`eval_5p_grpo_sft_gold500`、`eval_7p_p2a500`、`eval_5p_full_1034_3b`、`eval_vav_finer_a/b`、`eval_vav_finer_full`、`official_original_vav` 的 items.json / official_result；p2a_*/t21 训练日志。
