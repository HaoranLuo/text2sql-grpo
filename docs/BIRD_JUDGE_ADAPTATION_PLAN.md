# BIRD 判卷老师便宜适配（Judge Adaptation）实验方案

> 状态：**已预注册**（本文档即预注册文本，先冻结判定口径，后看结果）
> 作者：实验设计 agent ｜ 日期：2026-08-21 ｜ 目标 HPC：`/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/`（`ssh hpc`）
> 配套草稿文件（均与本文档一起落盘，执行前先读 §3 接口核对）：
> `src/label_orm_bird.py`、`scripts/label_orm_bird_cpu.slurm`、`scripts/train_orm_bird_bal.slurm`、
> `scripts/bird_orm_score_adapted.slurm`、`scripts/bird_judge_adapt_chain.sh`
>
> 本方案**只写方案与草稿、不提交任何 sbatch、不改动任何既有脚本**。所有既有产物
> （`outputs/bird_select/*`、`checkpoints/orm_b1`、`checkpoints/orm_bal2` 等）保持原样。

---

## 1. 目标与基线

**目标**：用 BIRD dev 自身候选池给判卷老师（GradeSQL 式 Yes/No ORM，Qwen2.5-Coder-3B-Instruct + LoRA）
打 BIRD 域标签并微调，替换 `src/bird_select.py` 打分阶段当前使用的 Spider 域 `orm_b1`，
使 `arm_orm_grouphead`（组级 ORM 选择）在 **BIRD 官方执行准确率（FINER `evaluation_bird_ex.py`）**
上超过结构裁决基线 `arm_vav`。

**既成基线（outputs/bird_select/summary.json，同一候选池 `outputs/eval_pool_bird/items.json`）**：

| arm | 官方 EX total | simple | moderate | challenging | 说明 |
|---|---|---|---|---|---|
| `arm_vav` | **56.26** | 64.65 | 45.26 | 37.93 | 结构裁决（MI-VAV 移植），无判卷老师 |
| `arm_orm_grouphead` | 52.54 | 61.95 | 38.58 | 37.24 | 判卷 = Spider 域 `orm_b1`（未适配） |

判卷适配的净收益问题就是：**BIRD 域判卷能否把 52.54 抬到 56.26 以上**。

**候选池事实**：1534 题 × 4 模型（`sft_phase1`/`sft_v2`/`sft_v3`/`p2a_500`）× 16 采样 = 98,176
候选；按题去重后 **80,216 唯一候选**（每题均值 52.3，min 5 / max 64）；8 个 parse_fail 空 SQL；
`dataset_index` = `question_id` = 0..1533，与 `dev.json`/`dev.sql` 行序一致（`bird_select --phase final`
有断言）。11 个库，每库单实例 `dev_databases/<db_id>/<db_id>.sqlite`。

---

## 2. 预注册判定（冻结，结果出来后不得回改）

### 2.1 判定域

唯一有效数字 = FINER 官方评估器执行准确率（`evaluation_bird_ex.py`，`--meta_time_out 30.0`，
dev 全量 1534 题，simple/moderate/challenging/total 四列）。dev 侧 `eval_metrics.json` 只作软门。

### 2.2 成功 / 强成功阈值

| 判定 | 条件 | 备注 |
|---|---|---|
| **适配成功** | `arm_orm_grouphead` total **> 56.26**（严格大于），且配对 McNemar 单侧 p < 0.05（b=vav 错→orm 对，c=vav 对→orm 错） | 任何 +0.01pp 也算成功，但统计上必须显著 |
| **强成功** | total **≥ 58.26**（+2.0pp ≈ 1.6×单臂 SE，单臂 SE=√(0.56×0.44/1534)≈1.27pp），且 McNemar p < 0.01 | — |
| 难度列保护（软） | simple ≥ 62.65、moderate ≥ 43.26、challenging ≥ 35.93（各列相对 arm_vav 回退 < 2.0pp） | 违反时 total 达标仍记成功，但标记「tradeoff 成功」，需人工复核 |
| 进步但未达标 | 56.26 ≥ total > 52.54 | 仅说明 BIRD 域 > Spider 域，不构成「适配成功」 |
| 失败 | total ≤ 52.54 | 按 §7 升级路径处理（升级必须重新登记） |

### 2.3 dev 侧软门（花官方 EX 之前看）

训练完成后、提交 score/final 之前检查**主配方 checkpoint**（`checkpoints/orm_bird_bird_bal2` 或
`orm_bird_bird_raw`，按 §5.1 决策表）的 `eval_metrics.json`：

- `rank_acc ≥ 0.70` 且 `rank_acc_maj_wrong ≥ 0.35`（Spider 域参考：b1=0.7843/0.30，bal2=0.8235/0.40）。
  `rank_acc_maj_wrong` = dev 上「arm_vav 败局题」的救回率（`orm_questions_bird.json` 的 `maj_correct`
  字段取自 `outputs/bird_select/items_arm_vav.json` 的 `is_correct`）。
- 不满足 → **先诊断**（看 dev 题 label 分布、prompt 截断分布），不得直接进入官方 EX 比较环节。

### 2.4 过程纪律（一次提交原则）

1. 每个 checkpoint 只允许跑**一次** score + final 官方 EX；不允许看完结果后回改打分参数再跑。
2. 全部超参与数据切分固定（§4/§6）；任何改动（epoch 数、cap、seed、配比、prompt）都算新实验，
   需在本文档追加登记后才能跑。
3. 对照臂（`bird_raw` 自动 pos_weight）与主臂平等报告，不得挑结果。
4. 结果落盘到 `outputs/bird_select_ormbird_<DATA>/summary.json`，日志在 `logs/`，不得覆盖
   `outputs/bird_select/*` 基线。

---

## 3. 接口核对结论（写实依据，全部读过源码）

| 组件 | 关键接口事实 |
|---|---|
| `src/train_orm.py` | `--data` 接受样本列表 JSON：每样本需 `question_id`(int)、`label`(0/1)、`messages[0].content`（user prompt）、`messages[1]`（Yes/No）；`--questions` 接受每题元数据列表（`question_id` 必填，`maj_correct` 可选→开启 `rank_acc_maj_wrong`）。dev 切分 = 按题：`qids` 洗牌（`--seed 42`）后取前 `k = int(round(N×--dev-frac))` 题（`--dev-questions` 可显式覆盖）。tokenize：训练 = chat 模板 + 答案 token，**左截断** `--max-length 2048`；打分 = 同截断。`--pos-weight <=0` 时自动取 `neg/pos`。`--eval-only` 只加载 `--output` 适配器算 dev 指标写 `eval_metrics.json`。当前代码已修复「datasets 迭代切片返回列名」bug（按索引取行）。 |
| `src/label_orm_data.py`（Spider 版，被本方案克隆改造成 BIRD 版） | 整题剔除：gold 执行失败 / 无实例；样本字段：`question_id/db_id/difficulty/messages/label/candidate_sql/dup_count/models`；prompt = `build_orm_prompt`（canonical 生成端 prompt + Candidate SQL 块 + Yes/No 指令，**evidence=None**）；交叉核对用裁决器 items。 |
| `src/prep_orm_balanced.py` | `--in <train.json> --out-dir <dir> --stats-out <json>`；输出文件名固定 `orm_train_bal1/2/3.json`（**与输入名无关** → BIRD 用 `--out-dir data/bird_orm` 防覆盖 Spider 的 `data/orm_train_bal*.json`）；上采样=整题复制、降采样=每题保底 1 负、题目集合不变 → `train_orm.py --seed 42` 下各变体 dev 划分**完全相同**。 |
| `src/orm_selection.py` | `VllmScorer`（vLLM 0.11.2，top-k logprobs=20，`max_tokens=1`，prompt_token_ids 直送，LoRA→merged 回退）；`build_orm_prompt(question, ddl, sql)`：`ReasoningGeneratorAgent.build_prompt(question, ddl, schema_links=None, evidence=None, dialect="sqlite")` + 候选 SQL 块；`--max-length 2048` 左截断。`bird_select.py` 直接 import 这两个。 |
| `src/bird_select.py` | 三阶段：`--phase prep`（`AP._dedupe` → 单实例 `AP.ExecutionEngine` 签名分组 → `arm_vav` → 只对 rankable 组代表生成 ORM payload 写 `work/orm_payloads.json`）；`--phase score`（`VllmScorer`，`--orm-checkpoint` 可换，写 `work/orm_scores.json`）；`--phase final`（两臂胜者 → 官方格式 `predict_dev.json` → 调官方评估器 → `summary.json.official_exec_accuracy`）。**打分器是唯一要换的部件**，prep/final 无需改动，只需换 `--out-dir` 与 `--orm-checkpoint`。 |
| 官方评估器 `evaluation_bird_ex.py` | 判对逻辑（第 31-43 行）：`cursor.execute(pred).fetchall()` 与 `cursor.execute(gold).fetchall()` 后 **`set(predicted_res) == set(ground_truth_res)`**；`func_timeout(meta_time_out=30.0)` 包住整对执行，超时/异常 → res=0。无任何 SQL 文本变换（无 remove_distinct/replace_cur_year/列置换）。 |
| 实测校准 | ORM payload 10,711 条，prompt token 均值 **1205**（左截 2048 后 1180），919 条 ≥2048；打分 12.6M token / 3090 上 **180s**（70k tok/s）。prep 阶段 80,009 次执行 577s（37,918 失败 / 769 VM-interrupt / 72 timeout）。A40 训练吞吐实测（orm_b1 日志 1996104）：19,375 样本 ×3 epoch = **20,637s（≈1800 tok/s）**。gold 探针：1534 题中 **63 题 gold 执行失败**（25 个 VM-interrupt + 其余 sqlite 错误）→ 预计可打标 ≈ **1471 题**。 |

---

## 4. 自动打标方案（`src/label_orm_bird.py`）

### 4.1 判定语义：逐字复刻官方 `execute_sql`

对每个去重候选 `c`（`AP._dedupe` 同口径）与该题 gold：

```
执行 c  → rows_c；执行 gold → rows_g
label(c) = 1  ⟺  执行都成功 且  set(rows_c) == set(rows_g)
```

具体对比函数（草稿内 `bird_result_set_eq`）：

```python
def bird_result_set_eq(pred_rows, gold_rows):
    # 官方: set(predicted_res) == set(ground_truth_res)
    # 行序无关；重复行折叠（set 语义，官方如此，不可改 multiset）；
    # 列序敏感（tuple 内顺序 = SELECT 列序，无列置换容忍）；类型敏感。
    try:
        return set(map(tuple, pred_rows)) == set(map(tuple, gold_rows))
    except TypeError:  # sqlite fetchall 行值均可哈希，纯防御
        return sorted(map(str, pred_rows)) == sorted(map(str, gold_rows))
```

标签规则（与官方 res=0 对齐）：

- 候选执行失败 / 超时 / 空 SQL / 结果被行上限截断 → **label 0**；
- gold 执行失败 / 超时 / 截断 → **整题剔除**（标签不可靠，统计显式计数并留 id 清单）；
- 无实例文件 → 整题剔除（BIRD dev 实际无此情形）。

**执行工具选型**：复用 `AP.ExecutionEngine`（16 线程、每查询独立只读连接、30s 墙钟 +
`conn.interrupt()` 杀查询、5M VM 步上限、`(sql, db)` 全局缓存）。与官方 `func_timeout` 的
已知偏差（均写入 stats 并量化）：(a) VM 步上限会把极少数「30s 内能跑完但步数超限」的查询
判失败（prep 实测 769/80,009 ≈ 0.96%）；(b) 官方 30s 预算是 (pred+gold) 一对接力共用，本方案
每查询独立 30s（更宽松，只有 pred、gold 各自 10-30s 的病态组合才会产生标签漂移，BIRD dev
查询绝大多数 <1s）；(c) 行上限 500k（官方无上限；BIRD dev 结果集极小，此设置纯防内存）。
偏差用 §8.3 的 V1/V2 官方评估器交叉验证兜底。

### 4.2 防泄漏 dev 划分（确切方案）

- **按题切、同题同侧**：`train_orm.py` 的 `split_by_question`（题目洗牌 seed 42，取前 5%）。
- 预计可打标 N≈1471 题 → dev = `int(round(1471×0.05))` = **74 题**（确切值以打标统计为准，
  规则固定：`--dev-frac 0.05 --seed 42`；Spider 惯例 51 题正是同一公式：round(1026×0.05)=51）。
- **各变体 dev 集合完全一致的结构保证**：cap12 分层截断保留每题至少 1 样本（正样本全保留 +
  负样本随机补），`prep_orm_balanced` 降采样保底每题 1 负 → 全量 / cap12 / bal1 / bal2 / bal3
  的题目集合全部一致 → seed 42 洗牌后的前 74 题 dev 完全相同 → `eval_metrics.json` 横向可比。
- 同一 11 个库的跨题 schema 泄漏（同库不同题）与 Spider 惯例一致，接受并记录。

### 4.3 训练 prompt 口径（train/infer 逐字一致）

`build_orm_prompt(question, read_ddl(db), sql)`，其中 `read_ddl` = sqlite_master CREATE TABLE 按
表名排序拼接（与 `bird_select.phase_prep` 打分时的 DDL 表示**逐字一致**）；`evidence=None`——
`bird_select` 打分侧不带 evidence（生成侧带，但判卷推理侧没有），训练侧必须同口径，否则
train/infer 漂移。脚本内含与 `label_orm_data.build_orm_prompt` 的逐字一致性自检（可跳过）。

### 4.4 cap12 分层截断（便宜配方入口）

`--max-per-question 12 --cap-out data/orm_train_bird_cap12.json`（seed 42）：

- 每题**正样本全保留**；不足 12 用负样本随机补齐；正样本超 12（全对/近全对题）随机取 12 正。
- 预计 1397 训练题 × ≤12 ≈ **1.6 万样本**（vs 全量 ≈7.3 万）。理由见 §6.2 成本推导。
- 全量文件 `data/orm_train_bird.json` 同时落盘（升级路径用，零额外执行）。

### 4.5 输出格式（与 `data/orm_train.json` 完全同构 → `train_orm.py` 零改动）

- `data/orm_train_bird.json` / `orm_train_bird_cap12.json`：样本（chat + label + 元数据）。
- `data/orm_questions_bird.json`：每题 `question_id/db_id/difficulty(simple|moderate|challenging)/
  num_instances/num_unique_candidates/num_total_votes/num_correct/num_incorrect/maj_correct`
  （`maj_correct` = `outputs/bird_select/items_arm_vav.json` 的 `is_correct`）。
- `data/orm_label_stats_bird.json`：正负比、按难度/按模型正率、每题正确数直方图、
  剔除 id 清单、执行统计、prompt 长度统计。

---

## 5. 数据配比：建议与理由

**主配方 = bal2（60:40）**，对照 = cap12 原版分布 + `--pos-weight 0`（自动 `neg/pos`，原版 34.3% 口径）。

理由：

1. GradeSQL（arXiv:2509.01308）实测 ORM 最优正负比在 **58-69%**，60:40 恰在区间中位；
   本项目 Spider 域已复现收益（bal2 rank_acc 0.8235 / b1 0.7843，败局救回 0.40 / 0.30）。
2. BIRD 池正率未知（prep 失败率 47% → 正率大概率 25-45%）。bal2 公式
   `负样本降采样到 pos×2×0.667` 在正率 ≤43% 时自动落 60:40；正率 >43% 时负样本全保留
   （正率略超 60%，仍在 GradeSQL 区间上沿）——**对正率鲁棒**，而固定 pos_weight 需要先看
   打标结果才能定。
3. 原版自动 `pos_weight` 在 34.3% 时取 ≈1.92 有效；若 BIRD 正率偏离 34.3% 较多则漂出
   GradeSQL 区间——恰是它作为**对照**而非主配方的理由（同时也是消融：均衡 vs 不加权）。
4. bal1（1:1）低于 GradeSQL 区间下界、bal3（纯上采样）作为备用数据一并由 prep 产出，
   零成本；只有主配方失败时才升级评估（需重新登记）。

### 5.1 预注册决策表（按打标实测正率 p 选主配方，**打标前冻结**）

冒烟 5 题实测正率仅 6.2%（样本 305，多来自 california_schools——Spider 域模型在该库列名
对不上的失败率 68%），提示全量正率可能显著低于 Spider 的 34.3%。为避免事后挑配方，冻结
如下决策表；两个配方都在 `scripts/train_orm_bird_bal.slurm` 中（`bird_bal2` / `bird_raw`），
只决定「谁是主、谁是对照」，不改任何脚本参数：

| 打标实测正率 p（全量 orm_label_stats_bird.json） | 主配方（1 epoch, seed 42） | 对照 | 理由 |
|---|---|---|---|
| p ≥ 0.20 | `bird_bal2`（cap12 + 60:40，pos_weight 1.0） | `bird_raw`（cap12 原版 + pos_weight 0） | 与 GradeSQL 区间吻合；bal2 负样本降采样数据量充足（≈16-23k 样本） |
| p < 0.20 | `bird_raw`（cap12 原版 + pos_weight 0 自动 neg/pos） | `bird_bal2` | 正样本过稀时 bal2 会把负样本砍到 ~每题 1 条（总量 ≈3-4k，欠拟合风险）；此时 CE 类权重（auto neg/pos）是对稀有正类不丢数据、不造假的正解，也是任务书给定的备选口径 |

两个配方的训练成本接近（cap12 样本量对 p 不敏感：p<0.20 时 bal2 样本反而少、raw 恒 ≈16.4k
→ 1 epoch 均 ≤3.5h），决策表不改变 §6.8 的时长预算。

---

## 6. 训练与评测流程（每步命令写实）

### 6.0 前置冒烟（登录节点，可选但建议）

```bash
ssh hpc
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
envs/reasoning3b/bin/python src/label_orm_bird.py --limit 30 --threads 16 \
    --out /tmp/orm_train_bird_smoke.json \
    --questions-out /tmp/orm_questions_bird_smoke.json \
    --stats-out /tmp/orm_label_stats_bird_smoke.json
```

### 6.1 作业 1：全量打标（cpu6348，~15-25 min）

```bash
sbatch scripts/label_orm_bird_cpu.slurm          # 冒烟: sbatch --export=ALL,LIMIT=30 scripts/label_orm_bird_cpu.slurm
```

成本依据：同规模执行集（prep 实测 80,009 次 = 577s）+ gold 1534 + 80k 样本 JSON 落盘 ≈
15-25 分钟（slurm 上限 1h）。完成后人工核验 `data/orm_label_stats_bird.json`：
正率、剔除题数（预计 ≈63）、每题正确数直方图、prompt 长度。**任何异常（正率 <3% 或
>60%）停下诊断，不进入训练**；正率在 3-60% 内按 §5.1 决策表执行。

> 已预演：登录节点 `--limit 5` 冒烟跑通（305 样本 / 61 每题 / 正率 6.2% / 候选失败率
> 68%——首 5 题多为 california_schools，Spider 域模型在该库列名失配严重；全量正率以
> 打标统计为准）。

### 6.2 作业 2：均衡数据准备（登录节点 CPU，几秒）

```bash
envs/reasoning3b/bin/python src/prep_orm_balanced.py \
    --in data/orm_train_bird_cap12.json --out-dir data/bird_orm \
    --stats-out data/bird_orm/orm_bal_stats.json
# 产物: data/bird_orm/orm_train_bal1/bal2/bal3.json（输出目录与 Spider 数据隔离，防覆盖）
```

### 6.3 作业 3：训练（aiaca40，1 epoch，预计 2.5-4.5h，slurm 6h）

**先按 §5.1 决策表选主配方**（p 取 `data/orm_label_stats_bird.json` 的 `dataset.pos_ratio`）：

```bash
sbatch --export=ALL,DATA=bird_bal2 scripts/train_orm_bird_bal.slurm   # p>=0.20 时的主配方
sbatch --export=ALL,DATA=bird_raw  scripts/train_orm_bird_bal.slurm   # 对照（p<0.20 时两者互换身份）
```

配方（相对 `train_orm_bal.slurm` 的差异全部注明）：

| 项 | 值 | 理由 |
|---|---|---|
| 数据 | cap12 → bal2（主）/ cap12 原版（对照） | §5 |
| `--epochs` | **1**（主配方；原 3） | 成本推导：cap12 ≈16.4k 样本 × prompt 均值 1180 token ≈19.4M token；A40 实测 1800 tok/s → 1 epoch ≈3.0h（正率高时 bal2 样本增至 ~23k → ≈4.3h）；3 epoch ≈10h+ 超出「便宜」定位 |
| `--pos-weight` | 1.0（bal2 已均衡）/ 0（对照自动） | 与 train_orm_bal.slurm 同口径 |
| 其余 | `--batch-size 2 --grad-accum 4 --lr 1e-5 --lora-r 32 --lora-alpha 64 --max-length 2048 --dev-frac 0.05 --seed 42` | 完全克隆 Spider 配方 |
| 输出 | `checkpoints/orm_bird_bird_bal2`（对照 `orm_bird_bird_raw`）+ `eval_metrics.json` | — |

训练完成后：`cat checkpoints/orm_bird_bird_bal2/eval_metrics.json` → 过 §2.3 软门。
（若训练末段 `evaluate_dev` 意外报错——历史 bug 1996104 已修复——用
`envs/reasoning3b/bin/python src/train_orm.py --data ... --questions data/orm_questions_bird.json --output checkpoints/orm_bird_bird_bal2 --eval-only` 重算指标。）

### 6.4 作业 4：prep 分组（cpu6348，~14 min）—— 新 out-dir，不碰基线

```bash
sbatch --export=ALL,PHASE=prep,OUT_DIR=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/outputs/bird_select_ormbird_bird_bal2 \
    scripts/bird_select_cpu.slurm
```

产物 `outputs/bird_select_ormbird_bird_bal2/work/{prep.json, orm_payloads.json}`
（打分对象 = 10,711 个 rankable 组代表，与基线同口径）。

### 6.5 作业 5：新判卷打分（gpudebug 3090，~10 min，slurm 50min）

```bash
sbatch --export=ALL,ORM_CKPT=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/checkpoints/orm_bird_bird_bal2,OUT_DIR=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/outputs/bird_select_ormbird_bird_bal2 \
    scripts/bird_orm_score_adapted.slurm
```

实测依据：同规模打分（10,711 payload × 12.6M token）3090 上 180s + 引擎启动 ≈4min。

### 6.6 作业 6：final 裁决 + 官方 EX（cpu6348，~2 min）

```bash
sbatch --export=ALL,PHASE=final,OUT_DIR=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/outputs/bird_select_ormbird_bird_bal2 \
    scripts/bird_select_cpu.slurm
```

产物 `outputs/bird_select_ormbird_bird_bal2/summary.json` → `official_exec_accuracy.arm_orm_grouphead.total`。

### 6.7 收割与预注册判定

```bash
python3 - <<'EOF'
import json
d = json.load(open("outputs/bird_select_ormbird_bird_bal2/summary.json"))
o = d["official_exec_accuracy"]
print("arm_vav            :", o["arm_vav"]["total"])
print("arm_orm_grouphead  :", o["arm_orm_grouphead"]["total"], "-> PASS" if o["arm_orm_grouphead"]["total"] > 56.26 else "-> NOT PASS")
EOF

# McNemar 配对检验（成功/强成功的显著性要件）
python3 - <<'EOF'
import json, math
def load(p):
    d = json.load(open(p, encoding="utf-8"))
    if isinstance(d, dict) and "items" in d: d = d["items"]
    return {r["dataset_index"]: bool(r["is_correct"]) for r in d}
vav = load("outputs/bird_select_ormbird_bird_bal2/items_arm_vav.json")
orm = load("outputs/bird_select_ormbird_bird_bal2/items_arm_orm_grouphead.json")
b = sum(1 for q in vav if not vav[q] and orm[q])   # fixed（救回）
c = sum(1 for q in vav if vav[q] and not orm[q])   # broken（弄坏）
p = sum(math.comb(b + c, k) for k in range(b, b + c + 1)) / 2 ** (b + c)
print(f"fixed={b} broken={c} McNemar 单侧 p={p:.4f} -> {'显著' if p < 0.05 else '不显著'}")
EOF
```

### 6.8 时长汇总

| # | 作业 | 脚本 | 分区/QOS | 预计时长 | slurm 上限 |
|---|---|---|---|---|---|
| 1 | 打标 | `scripts/label_orm_bird_cpu.slurm` | cpu6348/32cores | 15-25 min | 01:00:00 |
| 2 | 均衡数据 | 登录节点直接跑 | — | <1 min | — |
| 3 | 训练 | `scripts/train_orm_bird_bal.slurm` | aiaca40/1a40 | **2.5-4.5h** | 06:00:00 |
| 4 | prep | `scripts/bird_select_cpu.slurm`（PHASE=prep） | cpu6348 | ~14 min | 04:00:00 |
| 5 | 打分 | `scripts/bird_orm_score_adapted.slurm` | gpudebug | ~10 min | 00:50:00 |
| 6 | final+官方EX | `scripts/bird_select_cpu.slurm`（PHASE=final） | cpu6348 | ~2 min | 04:00:00 |

串行墙钟 ≈ 4-5h；GPU 合计 ≈ 3h（A40）+ 10min（3090），与「A40 ~2-3h」的便宜定位一致。

---

## 7. 风险与回滚

| # | 风险 | 影响 | 对策 |
|---|---|---|---|
| R1 | 打标执行语义与官方存在小偏差（VM 步上限 0.96% 候选、行上限、独立超时预算） | 少量标签与官方 res 不一致 | §8.3 V1/V2 交叉验证量化；偏差样本量级 <1%，对 74 题 dev 排名指标影响可忽略 |
| R2 | 判卷 prompt 不带 evidence（推理侧没有），BIRD 题天然歧义 | 判卷老师盲区，可能封顶低于 vav | 属 train/infer 一致性硬约束；后续「evidence 版判卷」需改 `bird_select`（新实验，重新登记） |
| R3 | DDL 左截断（919/10,711 prompt ≥2048 token，大库 DDL 头部被截） | 大库题目判卷缺 schema 上下文 | 与既有口径一致（orm_b1 同款）；观察 dev 大库题指标，必要时缩短 DDL 表示（新实验） |
| R4 | 63 题 gold 失败被剔除 | 训练集缺这些题；官方 EX 两臂同享 95.9% 天花板 | 比较仍在同一分母上，有效；V1 验证剔除集合与官方一致 |
| R5 | 正率未知（冒烟 5 题仅 6.2%）；p>0.43 时 bal2 负样本全保留、正率超 60% | 配比漂出 GradeSQL 区间 / bal2 数据量骤减 | §5.1 预注册决策表已覆盖 p<0.20 与 p>0.43 两端；打标 stats 先看，按表执行，不静默调参 |
| R6 | 1 epoch 欠拟合 | rank_acc 软门不过 | 升级路径（需登记）：3 epoch cap12（~10h）/ 全量 1 epoch（~13h，slurm 16h）/ bal3 |
| R7 | dev 仅 74 题，rank 指标噪声大（rank_acc SE ≈±5pp） | 软门可能误判 | 软门只作「是否花官方 EX」的参考，不做成功判定；以官方 EX 为准 |
| R8 | 多组均正确题：grouphead 只打各组代表，其他正确组不被选中 | 判卷上限受裁决结构约束 | 既有管线限制，本适配只换打分器；超出本方案范围 |
| R9 | gpudebug MaxSubmitJobs=1 / 队列波动 | 打分作业排队 | 链脚本草稿未做串行等待（注释已注明）；首次执行建议按 §6 手工逐步提交 |
| R10 | 训练末段 evaluate_dev 崩溃（历史 bug） | 缺 eval_metrics.json | 已确认当前代码修复；兜底 `--eval-only` 重算 |
| 回滚 | 任何失败 | — | 基线产物 `outputs/bird_select/*`、`checkpoints/orm_b1` 全程只读；新产物全部在 `outputs/bird_select_ormbird_*`、`checkpoints/orm_bird_*`、`data/bird_orm/`、`data/orm_*_bird*.json`，删除即回滚 |

---

## 8. 附录

### 8.1 新增文件清单（本方案产物，均为新文件）

```
src/label_orm_bird.py                  BIRD 打标脚本（纯 CPU）
scripts/label_orm_bird_cpu.slurm       作业 1
scripts/train_orm_bird_bal.slurm       作业 3（DATA ∈ bird_bal1/2/3/raw）
scripts/bird_orm_score_adapted.slurm   作业 5
scripts/bird_judge_adapt_chain.sh      全链路编排草稿（可选，未处理 R9 串行）
docs/BIRD_JUDGE_ADAPTATION_PLAN.md     本文档
```

### 8.2 运行时产物

```
data/orm_train_bird.json / orm_train_bird_cap12.json / orm_questions_bird.json / orm_label_stats_bird.json
data/bird_orm/orm_train_bal1|bal2|bal3.json + orm_bal_stats.json
checkpoints/orm_bird_bird_bal2 / orm_bird_bird_raw（+ eval_metrics.json）
outputs/bird_select_ormbird_bird_bal2 / work/{prep,orm_payloads,orm_scores}.json / summary.json / items_arm_*.json
logs/orm_bird_*.out|err、bird_score_orm_*.out|err、bird_judge_adapt.log
```

### 8.3 打标语义交叉验证（打标完成后、训练前执行）

**V1 gold 自检**（打标器 gold 执行语义 vs 官方评估器）：

```bash
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
python3 - <<'EOF'
import json, subprocess, sys
base = "data/bird/bird_dev/dev_20240627"
items = json.load(open("outputs/eval_pool_bird/items.json", encoding="utf-8"))
pred = [[it["question"], f"{it['gold_sql']}\t----- bird -----\t{it['db_id']}"] for it in items]
json.dump(pred, open("/tmp/bird_gold_self_predict.json", "w"), ensure_ascii=False)
r = subprocess.run([sys.executable,
    "tmp_idea_research/finer-sql/evaluation/official_bird_evaluation/evaluation_bird_ex.py",
    "--db_root_path", base + "/dev_databases/", "--predicted_sql_json_path",
    "/tmp/bird_gold_self_predict.json", "--data_mode", "dev",
    "--ground_truth_sql_path", base + "/dev.sql", "--num_cpus", "12",
    "--mode_predict", "gpt", "--diff_json_path", base + "/dev.json",
    "--meta_time_out", "30.0"], capture_output=True, text=True)
print(r.stdout[-800:])
# 预期: total accuracy == (1534 - n_gold_fail)/1534*100（≈95.9%），
# 且 res=0 的 question_id 集合 == orm_label_stats_bird.json 的 questions_excluded_gold_fail_ids
EOF
```

**V2 胜者判定一致性**（每次 final 后必做；官方评估器同批产出 `res`，与管线内判定逐题比对）：

```bash
python3 - <<'EOF'
import json
def load(p):
    d = json.load(open(p, encoding="utf-8"))
    if isinstance(d, dict) and "items" in d: d = d["items"]
    return {r["dataset_index"]: bool(r["is_correct"]) for r in d}
base = "outputs/bird_select_ormbird_bird_bal2"
for arm in ("arm_vav", "arm_orm_grouphead"):
    items = load(f"{base}/items_{arm}.json")
    off = json.load(open(f"{base}/{arm}/eval_result_dev.json", encoding="utf-8"))
    bad = [i for i, r in enumerate(off) if items.get(i) is not None and items[i] != bool(r["res"])]
    print(arm, "mismatch:", len(bad), bad[:10])
# 预期 0 不一致；非 0 → 触发 R1 诊断（执行语义偏差定位）
EOF
```

V1 必做（几分钟，训练前）；V2 由每次 `bird_select --phase final` 天然产出，一致率必须 100%。

### 8.4 升级登记区（留白；任何改动在此登记后再跑）

| 日期 | 变更 | 理由 | 是否重新预注册 |
|---|---|---|---|
| — | — | — | — |
