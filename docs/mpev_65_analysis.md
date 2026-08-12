# MPEV 官方 EX 65% 差距归因分析（T0.4）

> 2026-08-13 · 只读分析（未跑 GPU、未改任何产物）
> 对象：`outputs/eval_mpev_100_3b`（5 视角 × 6 采样 = 30 候选/题，GRPO-3B checkpoint-25，dev 前 100 条）

## 一、三个数字的来源文件（已逐一对上）

| 数字 | 值 | 来源文件 | 口径 |
|---|---|---|---|
| 训练同口径 | **75%** (75/100) | `outputs/eval_mpev_100_3b/summary.json` → `selected_custom_exec_match_rate`（本地 & HPC 同名文件） | `compare_execution_results`：单实例原库、ORDER BY 感知、multiset（保留重复行）、列序敏感 |
| vav_self | **89%** (89/100) | 同上 → `vav_self_match_rate` | FINER majority_voting 口径：单实例原库、header-agnostic 排序值集合（行内值排序、集合去重 → **忽略重复行、忽略列序**） |
| 官方 EX | **65%** (65/100) | HPC `outputs/official_mpev_100/official_result.txt` → `execution ... 0.650`（由 `scripts/eval_official.sh` 对 `eval_mpev_100_3b/items.json` 重评产生） | test-suite-sql-eval：**多实例库**（每题 27/32/60 个 `.sqlite` 全跑）、bag 语义 + 列置换、行序敏感（gold 含 ORDER BY）、`keep_distinct=False`、`plug_value=False` |

复现验证：本次在 HPC 逐题重跑 `eval_exec_match`（CPU 只读），得到 **65/100**，与 `official_result.txt` 完全一致；分难度 12/50/21/17 与 easy 0.917 / medium 0.660 / hard 0.619 / extra 0.471 亦完全一致。

## 二、差距的算术分解（100 题交叉表）

三口径判定交叉计数（官方 × vav_self × custom）：

| official | vav_self | custom | 题数 | 含义 |
|---|---|---|---|---|
| 1 | 1 | 1 | 58 | 三口径全对 |
| 1 | 1 | 0 | **7** | 官方对、训练同口径判错（自评**过严**） |
| 0 | 1 | 1 | **17** | 官方错、两个自评都对（自评**过松**，主因多实例） |
| 0 | 1 | 0 | **7** | 官方错、仅 vav 判对（vav 更松一档） |
| 0 | 0 | 0 | 11 | 三口径全错（真错，投票救不回） |

- **89% → 75% (−14pp)**：14 题 vav=T 且 custom=F。其中 7 题官方也判对（列序/DISTINCT 使 custom 过严），7 题官方判错。
- **75% → 65% (−10pp)**：净 = 17（custom=T 官方=0）− 7（custom=F 官方=1）。
- **89% → 65% (−24pp)**：24 题 vav=T 官方=0，全部归因见第三节。

结论：**官方 65% 与 75%/89% 的差距不是随机噪声，而是三种口径机制差异的系统性叠加**；其中「多实例执行验证」贡献最大。

## 三、24 个差距题的失败模式归因（逐题执行级证据）

24 个 gap 题（vav=T、官方=0）按模式归类：

| 模式 | 题数 | 机制 |
|---|---|---|
| **A1 多实例分歧**（原库匹配、test-suite 实例不匹配，custom=T） | 13 | 预测在唯一原库上恰好等价，在合成实例上暴露语义漏洞 |
| **B 列序+多实例**（custom=F、官方也错） | 4 | 预测列序与 gold 不同（custom 判 F），且冗余 JOIN 在实例上拉出重复行/抬高聚合值 |
| **D vav 集合语义盲区**（custom=F） | 3 | vav 用集合去重比较 → 预测多返回重复行也判对；custom/官方保留重复计数 |
| **E remove_distinct 破坏 DISTINCT 语义** | 3 | 官方 `keep_distinct=False` 把 `SELECT DISTINCT` / `COUNT(DISTINCT …)` 里的 DISTINCT 词元删掉，改写了 gold 或预测的语义 |
| **C3 注释被折叠毁掉** | 1 | 预测含 `--` 行注释，`eval_official.sh` 折叠空白后注释吞掉整条 SQL |
| 合计 | 24 | |

### A1 多实例分歧的典型子模式（13 题）

- **LIMIT 1 并列 tie**（idx 7, 18, 67）：`WHERE age = (SELECT MIN(age) …)` / `ORDER BY … LIMIT 1` 在 tie 或 JOIN 重复下于实例上多行/错行。原库无 tie → 自评对。
- **冗余/INNER JOIN 依赖"实例人人有宠物/有演唱会"**（idx 22, 23, 36, 69, 70）：gold 直接单表/LEFT 语义，预测 INNER JOIN 把零匹配行过滤掉；原库恰好无零匹配，实例上出现 → 分歧。
- **NOT IN 遇 NULL**（idx 31, 32）：`NOT IN (子查询)` 在实例上子查询含 NULL 时整体返回空，等价改写失效。
- **GROUP BY/去重与 gold 不去重的差**（idx 25, 41）：预测多写 GROUP BY 或 DISTINCT 去重，原库无重复，实例有重复 → 行数分歧。
- **COUNT 语义**（idx 65）：IN 子查询 vs JOIN 在实例上边界行不同。

这 13 题里 8 题 medium、5 题 hard/extra（2 hard + 3 extra）—— 与直觉一致：越复杂的题越依赖数据巧合。

### 其余模式要点

- **B（idx 49, 71, 72, 74）**：`SELECT PetType, AVG(pet_age) …` 类预测列序与 gold 相反（custom 列序敏感判 F；官方列置换允许），同时 `JOIN Has_Pet` 冗余导致实例上 avg 被重复行抬高 → 官方也 0。
- **D（idx 39, 40, 62）**：预测经 JOIN 返回 3 行相同 (Justin Brown, France) 而 gold 1 行；vav 集合比较把重复折叠 → 误判对。
- **E（idx 57, 76, 87）**：官方 `remove_distinct` 用 sqlparse 删除**所有** distinct 词元：57/76 把 gold 顶层 DISTINCT 删掉（gold 出现重复行，预测去重后行数反而"不对"）；87 把预测 `COUNT(DISTINCT Continent)` 改成 `COUNT(Continent)`（5 → 15，与 gold `count(*) FROM CONTINENTS`=5 不符）。**其中 57/76 在 `keep_distinct=True` 下官方判对（实测翻盘为 1）**；87 不翻盘（countries 与 CONTINENTS 两张表在实例上数据本就不同，属 A1 类硬伤）。
- **C3（idx 83）**：预测为合法多行 SQL（`--` 行注释 + 换行），eval 阶段（DatabaseExecutor 原样执行）判对；`eval_official.sh` 把 `\n` 折叠成空格后 `--` 吞掉 FROM 以下全部 → 官方全部实例 `no such column` 执行失败。

## 四、抽样证据表（25 题，覆盖全部模式）

难度列 = 官方 `eval_hardness(gold)`。三口径判定：O=官方 EX，C=训练同口径，V=vav_self。

| idx | 难度 | 问题（节选） | O | C | V | 失败模式 |
|---|---|---|---|---|---|---|
| 7 | medium | youngest singer 的歌名与年份 | 0 | 1 | 1 | A1：`WHERE age=MIN(age)` vs `ORDER BY age LIMIT 1`，实例上 min 年龄并列 → 多行 |
| 18 | medium | 平均上座率最高体育馆 | 0 | 1 | 1 | A1：ORDER BY average DESC LIMIT 1，实例上 tie 选出不同 stadium |
| 22 | medium | 每个体育馆的演唱会数 | 0 | 1 | 1 | A1：INNER JOIN 过滤零演唱会场馆，原库无零匹配、实例有 |
| 25 | medium | 2013 年后演唱会最多的体育馆 | 0 | 1 | 1 | A1：多余 GROUP BY 去重，实例有重复行 |
| 31 | medium | 除 2014 年外所有体育馆名 | 0 | 1 | 1 | A1：`NOT IN` 遇实例子查询 NULL → 空集 |
| 36 | medium | 每位歌手演唱会数 | 0 | 1 | 1 | A1：INNER JOIN 丢无演唱会歌手 |
| 41 | medium | 2014 年有演唱会的体育馆 | 0 | 1 | 1 | A1：`SELECT DISTINCT` 去重 vs gold 不去重 |
| 65 | medium | 养狗不养猫的学生姓名年龄 | 0 | 1 | 1 | A1：IN 子查询改写，实例上边界行分歧 |
| 67 | medium | 最年轻宠物的种类体重 | 0 | 1 | 1 | A1：ORDER BY pet_age LIMIT 1 + JOIN，实例 tie/重复分歧 |
| 69 | medium | 年龄>1 的宠物 id 体重 | 0 | 1 | 1 | A1：冗余 INNER JOIN 丢无主宠物 |
| 49 | medium | 每种宠物最大体重 | 0 | 0 | 1 | B：列序（PetType, MAX）互换 + JOIN 重复行拉高聚合 |
| 71 | medium | 每种宠物平均/最大年龄 | 0 | 0 | 1 | B：列序互换 + JOIN 重复行抬高 avg |
| 39 | medium | 有 'Hey' 歌的歌手姓名国籍 | 0 | 0 | 1 | D：预测返回 3 行相同行（JOIN 重复），vav 集合去重误判 |
| 62 | hard | 不养猫学生的专业年龄 | 0 | 0 | 1 | D：行数 34 vs 33（重复行差异） |
| 57 | medium | 养猫或狗的学生名 | 0 | 1 | 1 | E：官方删 gold 顶层 DISTINCT → gold 出现重复行 |
| 76 | medium | 养宠物学生的不同姓名年龄 | 0 | 1 | 1 | E：同上（keep_distinct=True 下翻盘） |
| 87 | easy | 有多少个大洲 | 0 | 1 | 1 | E：`COUNT(DISTINCT Continent)` 被删 DISTINCT → 15 vs 5 |
| 83 | medium | 有 3 岁猫的学生姓氏 | 0 | 1 | 1 | C3：`--` 行注释被空白折叠吞掉 SQL |
| 37 | medium | 2014 年演唱会的歌手名 | 1 | 0 | 1 | 自评过严：官方删预测 DISTINCT → 与 gold 重复行一致；custom 判行数 5 vs 6 |
| 50 | medium | 每种宠物最大体重与种类 | 1 | 0 | 1 | 自评过严：官方列置换容忍列序互换 |
| 75 | medium | 养宠物学生姓名年龄 | 1 | 0 | 1 | 自评过严：官方删 gold DISTINCT 后与预测行数一致 |
| 12 | hard | 高于平均年龄歌手的所有歌 | 0 | 0 | 0 | 真错：CTE 假 JOIN 写法 |
| 43 | medium | 容量最大体育馆的演唱会数 | 0 | 0 | 0 | 真错：低置信（conf=0.17） |
| 59 | medium | 同时养猫狗的学生 | 0 | 0 | 0 | 真错：低置信（conf=0.10） |
| 99 | extra | 1970 年造车的厂商 | 0 | 0 | 0 | 真错：conf=0.03，投票近乎无效 |

（18 行 A1/B/D/E/C3 为 gap 题证据，另含 3 行"官方对自评错"、4 行"三口径全错"对照。）

## 五、评估器变体量化（喂给 Phase 3 的判断依据）

在 HPC 对 100 题逐题重跑官方评估器四种配置：

| 配置 | EX |
|---|---|
| keep_distinct=F, plug_value=F（**当前官方 65%**） | 65 |
| keep_distinct=T, plug_value=F | 62 |
| keep_distinct=F, plug_value=T | 68 |
| keep_distinct=T, plug_value=T | 65 |

- `plug_value=True` 仅 +3（68）：**值预测不是主要差距源**——模型确实在"预测值"上没有大问题，差距在语义等价（翻盘题 62/63/64，均为 NOT IN/EXCEPT 类含值条件的改写）。
- `keep_distinct=True` 反而 −3（62）：57/76 翻盘 +2，但 37/38/75/84/88 五题（原本靠"双方 DISTINCT 都被删"而凑巧匹配）掉下去 −5。
- **换评估器配置救不了 65%**：与论文 85% 的差距（历史 L0 结论）及 MPEV 65% vs 自评 89% 的差距主体都是**预测本身的语义鲁棒性**（多实例下 17 题暴露）+ **管线小 bug**（C3/E 共 4 题）。

## 六、修复建议（喂给 Phase 3 官方验证）

**P0 管线修复（4 题，立即生效，零训练）：**
1. `eval_official.sh` 折叠空白前先按行还原 SQL：不要用 `' '.join(sql.split())` 折叠预测 SQL 中的换行（把 `\n` 折叠会把 `--` 行注释变成整句注释）。改为折叠时把 `--` 注释截断（`split('--')[0]`）或保留换行语义（`\n` → 空格但先摘除 `--` 到行尾）。→ 修 idx 83。
2. 官方评估加 `--keep_distinct`：至少并报一个 keep_distinct=True 口径（实测 57/76 翻盘 +2、37/38/75/84/88 掉 −5，净 −3，但语义上更公平，且与 MAC-SQL 评估器对齐）。建议默认改成 `--keep_distinct`（MAC-SQL evaluation_spider.py 即该口径），双口径并报。
3. 训练同口径 `compare_execution_results` 增加列置换匹配（对齐官方 bag 语义），消除 7 个"自评过严"（idx 37/50/73/75/79/80 类）。vav 投票的 `results_equal` 保持 FINER 原语义不动（它驱动投票，不动）。

**P1 数据侧（对 17 个多实例分歧题治本）：**
4. 投票/训练执行验证改用**多实例库**（`data/spider_data/database/{db}/*.sqlite` 全部实例）而不是只跑原库——这是官方 65% 与自评 89% 差距的 70%（17/24）。代价：执行量 ×~40（有 `(db_id, normalize_sql)` 缓存和 30 候选高重复兜底，实测可行）。预期：多实例投票会把 A1 类候选的"实例上分歧"记入失败组，投票选到实例鲁棒的候选。
5. 候选生成侧抑制三类脆模式（作 reward 负信号或 prompt 约束）：冗余 INNER JOIN、`NOT IN`（→ `NOT EXISTS`）、无 ORDER BY 的 LIMIT 1。

**P2 决策参考：**
6. Phase 3 gate（官方 ≥5p+2pp）在管线修复（P0）后复测才有意义；P0 仅约 +4~7（65 → 69-72），达标需要 P1#4 多实例投票为主。难度分层预期收益：medium 档（33/50）是最大可捞池。

## 七、附：口径判定逻辑速查

| 维度 | vav_self (89%) | 训练同口径 (75%) | 官方 EX (65%) |
|---|---|---|---|
| 执行库 | 原库单实例 | 原库单实例 | **全部实例（27/32/60）** |
| 行匹配 | 排序值**集合**（去重） | multiset（保留重复） | bag 语义 + 列置换 |
| 列序 | 不敏感 | **敏感** | 不敏感（置换） |
| 行序 | 不敏感 | gold 有 ORDER BY 时敏感 | gold 有 ORDER BY 时敏感 |
| DISTINCT | 原样执行 | 原样执行 | **双向删除 distinct 词元** |
| 值 | 全等比较 | 全等比较 | 全等（plug_value=F） |
| 空预测 | 判错 | 判错 | **被 eval_official.sh 跳过**（本题集 0 条） |

产物（本次分析生成，未改任何既有文件）：
- 本地 `.research_tmp/mpev_items.json`（2.5MB，HPC items.json 副本）
- 本地 `.research_tmp/mpev_off_per_item.json`（逐题官方 EX + 实例级明细）
- 本地 `.research_tmp/mpev_variants.json`（四配置逐题结果）
- HPC `/tmp/mpev_off_per_item.py`、`/tmp/mpev_eval_variants.py`（只读脚本）
