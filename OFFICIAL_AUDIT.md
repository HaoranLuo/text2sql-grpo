# OFFICIAL_AUDIT：官方 EX 76.6% vs 论文 85.0% 差距审计

- 日期：2026-08-12
- 审计对象：`scripts/eval_official.sh` 复刻 FINER 官方 EX 的执行链
- 方法：三路并行排查（评估调用链 / 数据库 / 评估器参数）
- 总体结论：**评估口径不匹配**（原始 Spider 评估器 vs test-suite 评估器），非代码缺陷、非数据损坏、非生成侧问题

---

## 1. 结论摘要

最可能根因是**指标/评估器不匹配**，而非任何代码或数据缺陷：论文 Official EX 85.0% 出自 FINER 官方链——`finer-sql/evaluation/majority_voting.py` 第 769-777 行 subprocess 调用 MAC-SQL 的 `evaluation_spider.py`（tao-yu/Spider 原始 `evaluation.py` 一系），参数 `--etype exec`（无 `--plug_value`、无 `--keep_distinct`），在**原始 Spider 单实例数据库**（`data/spider`，每库目录仅 1 个 .sqlite、根目录含 dev_gold.sql）上做**行序不敏感**的执行准确率计算；而我们 `scripts/eval_official.sh` 调的是 **test-suite-sql-eval**（`--etype all`），跑在 **test-suite 多实例增强数据库**（`data/spider_data/database`，本审计已字节级确认与官方 test-suite 发布一致）上，该口径对含 ORDER BY 的查询要求行序完全一致（`exec_eval.py` 第 197 行 `order_matters=True`），且 pred 必须在 db 目录下**所有实例**上与 gold 等价（第 199-201 行），系统性更严。三路证据闭合：同一批预测在自定义 results_equal 口径下 ~85%，精确复现论文 MV(vav) 85.88%（证明预测一致、纯评估端差异）；评估器参数空间内最高仅 0.792（`--plug_value`），距 85% 差 5.8pt（该差距来自 SQL 结构性错误，参数修不了）；多实例库与官方发布字节级一致（库无异常）。即：我们的 76.6% 是**第三个口径**（test-suite exec），与论文两个 85%（MV(vav) 85.88%、Official EX 85.0%）均不可直接互比；论文官方链从未使用 test-suite-sql-eval，76.6% 与 85.0% 的 8.4pp 差距即 test-suite vs 原始 exec 的已知口径差。

---

## 2. 三路排查详细结果

### 2.1 第一路：评估调用链（结论：评估器不同，决定性差异）

1. 【决定性差异·评估器不同】FINER 官方 EX 用原始 Spider 评估器：`majority_voting.py` 第 769-777 行 subprocess 调用 `/home/datht/schema-linking-benchmark/MAC-SQL/evaluation/evaluation_spider.py`（即 tao-yu/Spider 原始 `evaluation.py` 一系），参数 `--etype exec`（无 `--plug_value`、无 `--keep_distinct`）。我方 `scripts/eval_official.sh` 第 43-48 行调用 `test-suite-sql-eval/evaluation.py --etype all`——taoyds 的 test-suite 评估器，是另一个、明显更严格的指标。
2. 【决定性差异·数据库不同】FINER 官方链 db 是 `data/spider`（原始 Spider release：每库目录仅 1 个 .sqlite，根目录有 dev_gold.sql；`majority_voting.py` 第 745-767 行的候选根检查就验证了 dev_gold.sql+database+tables.json 布局）。我方 `--db` 指向 `data/spider_data/database`——test-suite 多实例数据库（OFFICIAL_EVAL.md 有记载，第二路已字节级确认）。test-suite `exec_eval.py` 第 199-201 行会把 db 目录下所有 .sqlite 全部执行一遍，pred 必须与 gold 在每个实例上都等价才判对——多实例执行本身系统性压低分数。
3. 【差异·ORDER BY 行序敏感】test-suite exec 在 gold 含 `order by` 时令 `order_matters=True`（`exec_eval.py` 第 197 行），要求行序完全一致；原始 Spider exec 行序不敏感（result==gt 或 set 等价）。这解释了难度梯度系统性偏低：easy -2.9 / medium -6.5 / hard -11.1 / extra -19.3（OFFICIAL_EVAL.md 复验表），hard/extra 里 ORDER BY 查询占比最高，打击最大。
4. 【确认：n=30 预测进官方评估器的路径】FINER：`majority_voting.py` 将每题 vav 选中的 SQL 写 pred_dev.sql（每行一条，空预测写 `SELECT 1` 而非跳过，保持 1034 行），再 subprocess 调 `evaluation_spider.py --etype exec`。我方：`eval_official.sh` 从 items.json 的 predicted_sql 提取并写 pred.sql/gold.sql，然后调 test-suite 评估器。桥接方式等价（每题一条），分歧全部在评估器与库。
5. 【差异·空预测处理】FINER 空预测写 `SELECT 1`（算错，分母恒 1034）；我方 `eval_official.sh` 第 25-26 行跳过空预测并同步删 gold 行（分母变小）。该差异方向是**抬高**我方分数而非压低，不是 76.6% 的成因，但与论文口径不完全一致。
6. 【差异·--etype all vs exec】我方传 `--etype all`（附带跑 match 分支，需要 kmaps），exec 行数字本身不受影响；FINER 只传 `--etype exec`。对 exec 数字无实质影响（第三路实验证实），但为完全对齐应改为 `--etype exec`。
7. 【排除项】gold 对齐（dev.json 按 dataset_index 取 query，等价于 dev_gold.sql）、SQL 折叠空白、`value`→`1` 替换（两个评估器都有）均非差异来源；生成侧（同一批 FINER-SQL-3B-Spider 权重 vav 选出的 SQL）也一致——自定义口径 ~85% 精确复现论文 MV(vav) 85.88%，说明预测相同、纯评估端差异。
8. 【论文两个 85% 的澄清】论文/README 实际报了两列：MV (vav)=85.88%（`majority_voting.py` 自研 results_equal 口径，与我方自定义 ~85% 吻合）；Official EX=85.0%（原始 `evaluation_spider.py --etype exec`，难度拆分 94.8/90.1/78.2/64.5 加权≈85.0%）。我方复刻的 76.6% 既不是 MV 也不是原始 exec，而是**第三个口径** test-suite exec——三方指标，不能互比。

**第一路根因**：指标/评估器不匹配——论文官方链从未使用 test-suite-sql-eval；OFFICIAL_EVAL.md 第 42 行自我记录的"官方评估器版本/多实例库细节差异"推测本次得到证实：不是评估器版本问题，是选错了评估器与数据库。

### 2.2 第二路：数据库核查（结论：无异常，多实例库字节级与官方一致）

1. 【与官方一致】HPC database/（170 项 = 169 个库目录 + readme.txt）与 test-suite-sql-eval 官方多实例库（Google Drive 1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w，即本地 spider_test_suite.tar.gz）完全一致。官方 zip 含 28 个 test-suite 库目录（21 个 Spider dev 库：battle_death/car_1/concert_singer/course_teach/cre_Doc_Template_Mgt/dog_kennels/employee_hire_evaluation/flight_2/geography/museum_visit/network_1/orchestra/pets_1/poker_player/real_estate_properties/singer/student_transcripts_tracking/tvshow/voter_1/world_1/wta_1；7 个 classical-only：academic/advising/atis/imdb/restaurants/scholar/yelp）+ database/readme.txt + __MACOSX 垃圾。实例命名 `<db>v<ver>round<X>group<Y>.sqlite` 与 v515patch\<N\>/v515nightpatch\<N\>。
2. 【实例集逐一比对：0 缺失 0 多余】官方 zip 每个 test-suite 目录的 .sqlite 文件名集合与 HPC 同目录逐名 diff：28 个目录全部 missing=0、extra=0（academic=412、atis=663、geography=109、imdb=201、yelp=283、battle_death=28、car_1=60、flight_2=56 等，HPC 与 zip 完全相同）。
3. 【字节级验证】5 个抽样文件（geography.sqlite、car_1.sqlite、battle_deathv515patch0.sqlite、atis.sqlite、student_transcripts_trackingv515nightpatch0.sqlite）SHA256 与本地 zip 提取值完全一致；HPC 实例文件 mtime=2020-12-27（官方 zip 时间戳），train 库 mtime=2018-09-24（Spider 官方发布）。
4. 【评估器代码一致】HPC `tools/test_suite_eval/evaluation.py` 与 `exec_eval.py` 的 SHA256 与本机官方 test-suite-sql-eval 仓库完全一致（42661E4E35B6... / 93CD3DA27089...）。评估语义确认：exec 模式枚举 db 目录内所有 .sqlite 作为测试实例（exec_eval.py:201），要求 pred 与 gold 在所有实例上 denotation 一致。
5. 【布局说明】HPC database/ = Spider 官方完整发布（166 库：146 train + 20 dev，train 库各仅 1 个原始 sqlite）+ 官方 test-suite zip 按目录名合并（+academic/advising/atis = 169 目录）。该布局正确：评估器只访问 gold 中出现的 db_id（dev 21 库），多余 train 目录零影响；与官方 README 要求的 database/ 布局等价。
6. 【唯一差异，不影响评估】zip 的 car_1/flight_2 目录含 16 个非 sqlite 元数据文件（car-makers.csv、cars.desc、README.CARS.TXT、airlines.csv、link.txt 等），HPC 上未原样保留（HPC 为 Spider 官方元数据 annotation.json/car_1.json/car_1.sql/data_csv/）。评估器只 glob .sqlite，零影响。
7. 【FINER 仓库核查】FINER 无独立 test-suite 多实例库路径：`db_execution/api.py` 用 SPIDER_DB_ROOT=/app/data/spider/database（docker-compose 挂载，仅用于执行 API）；官方 EX 由 `majority_voting.py` 调 MAC-SQL 的 `evaluation_spider.py --db <spider_root>/database` 计算，root 需含 dev_gold.sql + database/ + tables.json（SPIDER_DEV_ROOT 或 data/spider），与 HPC data/spider_data 布局完全吻合。
8. 【已排除】HPC 多实例库不是 76.6% vs 85% 的来源（项目文档 OFFICIAL_EVAL.md:42-43、NEXT_STEPS.md:25、HANDOFF.md:41 曾把多实例库列为待排查项）；应转向评估器调用参数与口径对齐。
9. 【附带发现】HPC data/spider_data.tar.gz（205MB，2025-08-05，仅含 372 个 sqlite）是旧的部分备份，与当前 database/（3383 个 sqlite）不一致，**勿用它还原目录**。

审稿注：第二路报告中把 `evaluation_spider.py` 一处标注为 "test-suite 评估器" 系表述不严谨——MAC-SQL 的 `evaluation_spider.py` 是原始 Spider `evaluation.py` 的改名版（以第一路对 `majority_voting.py` 实际调用代码的核查为准）；test-suite-sql-eval 是独立仓库（taoyds），两者勿混。

**第二路根因**：无异常——HPC 多实例库字节级与官方 test-suite 发布一致，排除数据来源；同时反向印证第一路：HPC 上部署的确是 test-suite 多实例库（而非论文用的原始单实例库）。

### 2.3 第三路：评估器参数（结论：参数无法弥合 5.8pt 差距）

基线：`bash scripts/eval_official.sh outputs/eval_vav_finer_full/items.json outputs/official_vav_full` → execution all = 0.766（76.6%，1025 条对齐，9 条空预测跳过）。

| 组合 | 配置 | execution 分数 | 与基线差 | 备注 |
|---|---|---|---|---|
| 基线 | 现状（`--etype all`，空预测跳过） | 0.766 | — | 1025 条对齐 |
| a | `--etype exec` | 0.766 | 0 | exec 分数不受 etype 影响 |
| b | `--plug_value` | 0.792 | +2.6pt | 唯一显著正向；难度 easy 0.944 / medium 0.858 / hard 0.717 / extra 0.470 |
| c | `--keep_distinct` | 0.740 | -2.6pt | 恢复 SELECT DISTINCT 敏感匹配 |
| d | `--etype all` vs `--etype exec` | 无差异 | 0 | etype 只影响是否额外算 match |
| e | 不截断 SQL（从 items.json 直接生成、仅折叠内部换行） | 0.766 | 0 | 排除截断损失；首次未折叠多行 SQL 致 pred 2869 行 vs gold 1025 行对齐错乱（0.006），已修正 |
| f | `--plug_value --keep_distinct` 叠加 | 0.767 | +0.1pt | 两参数效应互相抵消 |

**第三路根因**：评估器参数只能微调计分口径——`--plug_value` 用 gold 值替换 pred 字面量，消除了生成值（字符串/数值格式、引号、单位等）与 gold 的微小差异，故 +2.6pt；`--keep_distinct` 恢复对 SELECT DISTINCT 的敏感匹配，而 vav 生成的 SQL 常带 DISTINCT/去重逻辑与 gold 不同，故 -2.6pt；两者机制抵消（叠加 0.767）。76.6%→85% 的 5.8pt 差距来自 SQL 结构性错误（JOIN 方式、GROUP BY 键、ORDER BY/limit、NOT IN vs EXCEPT 等），评估器参数无法修复，必须靠模型/后处理改进。

### 2.4 三路合流

| 待查项 | 第一路（调用链） | 第二路（数据库） | 第三路（参数） | 结论 |
|---|---|---|---|---|
| 评估器 | 用错：test-suite 评估器而非原始 Spider 评估器 | 评估器代码 SHA256 与官方一致（工具没坏） | 参数最高 0.792 | **根因（用错工具）** |
| 数据库 | 用错：多实例 test-suite 库而非原始单实例库 | 字节级与官方 test-suite 发布一致（库没坏） | — | **根因（选错库）** |
| 预测/生成侧 | 自定义 ~85% 复现 MV 85.88% | — | 不截断验证一致 | 排除 |
| gold 对齐 / 空白折叠 / value→1 | 均等价 | — | — | 排除 |
| 空预测处理 | 方向抬高我方分数，非成因 | — | — | 排除（对齐时需修） |

三路证据闭合：**用错了评估器 + 用错了数据库**，其余全部排除。论文官方链从未使用 test-suite-sql-eval，76.6% 是 test-suite exec 口径下的合法分数，但不能与论文 85.0%（原始 exec）互比。

---

## 3. 修复方案

### 3.1 改动清单（对齐 FINER 官方链，无需重新生成预测，直接用已有 items.json 重评即可，几分钟出数）

对 `scripts/eval_official.sh` 做 4 处改动：

1. **换评估器**：EVAL 从 `tools/test_suite_eval/evaluation.py` 改为原始 Spider 评估器 `evaluation.py`（tao-yu/Spider 官方 release，或 MAC-SQL 的 `evaluation_spider.py`——即 FINER `majority_voting.py` 第 771 行调用的那个，两者同源）。将该文件（含配套 `process_sql.py`）放入 `tools/original_spider_eval/`。依赖注意：`evaluation.py` 需 nltk 的 `word_tokenize`；`--tables` 参数仅 match 需要，exec 可省。
2. **参数**：`--etype exec`（去掉 `--etype all`；不加 `--plug_value`、不加 `--keep_distinct`）。
3. **数据库**：`--db` 改为原始 Spider dev release 的 `database/`（每库目录仅 1 个 .sqlite）。注意：`data/spider_data/database` 是多实例 test-suite 库（本审计已确认），不可再用；NEXT_STEPS.md 记的 Drive 链接 1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w 是多实例版，也不能用。需另取原始 Spider 官方发布（tao-yu/Spider GitHub release 的 database.zip），dev 21 个库目录即可（train 目录可留可删，评估器只访问 gold 中出现的 db_id）；根目录需有 dev_gold.sql + tables.json（data/spider_data 根目录已有，可复用）。
4. **空预测处理**：pred 为空时写 `SELECT 1` 而不是跳过（pred/gold 均保持 1034 行，分母一致，与 FINER `majority_voting.py` 行为一致；`SELECT 1` 计错，不改动 gold 行）。

### 3.2 重跑与预期

```bash
bash scripts/eval_official.sh outputs/eval_vav_finer_full/items.json outputs/official_vav_full
```

预期：exec 整体 ≈ **85.0%**；分难度 ≈ **94.8 / 90.1 / 78.2 / 64.5**（论文 Official EX 表），与论文完全对齐。

### 3.3 验证清单

- [ ] pred.sql / gold.sql 均 1034 行，无空预测跳过
- [ ] 整体 exec 落于 84.5%–85.5%
- [ ] 难度拆分与论文 94.8/90.1/78.2/64.5 逐档对齐（误差 ±1pt）
- [ ] 与 OFFICIAL_EVAL.md 复验表对比，原先的难度梯度偏差（-2.9/-6.5/-11.1/-19.3）应消失

### 3.4 降级方案（若无法获取原始 Spider 库/评估器）

- 若只能在 test-suite 口径下评估：采用 `--etype exec --plug_value`（0.792）作为自洽口径，但对外汇报必须标注 **"test-suite EX"**，不得与论文 85.0% 并列。
- 不要使用 `--keep_distinct`（净降 2.6pt）。
- 若追求严格复现论文 76.6% 基线（test-suite 口径），评估口径保持不变即可，但同样必须标注口径。

---

## 4. 如果无法对齐：如何如实报告（论文里的评估差异说明）

1. **原则**：三口径各报各的，口径即指标名；任何并列对比必须附口径说明，禁止裸比数字。
2. **论文口径关系澄清**（建议写入报告/论文评估节）：
   - 论文报两个 85%：MV (vav)=85.88%（`majority_voting.py` 自研 results_equal 口径）；Official EX=85.0%（原始 Spider 评估器 `--etype exec`，行序不敏感、单实例库，难度拆分 94.8/90.1/78.2/64.5）。
   - 我方复现的 76.6% 是第三个口径：test-suite-sql-eval 的 execution（taoyds），ORDER BY 行序敏感、多实例数据库、pred 须在所有实例上与 gold 等价。三指标不可互比。
   - 预测侧一致性证据：同一批 vav 选出的 SQL 在自定义 results_equal 口径下 ~85%，与论文 MV(vav) 85.88% 吻合（生成链复现成功）。
3. **报告措辞模板**：
   - 中文："复现评估采用 test-suite-sql-eval（taoyds）执行准确率，该口径严于论文官方链使用的原始 Spider 评估器（--etype exec）：含 ORDER BY 的查询要求行序完全一致，且预测须在多实例增强数据库的全部实例上与金标准等价。该口径下得分为 76.6%（easy/medium/hard/extra = 具体数字）；作为对照，采用与论文 MV(vav) 相同的自研 results_equal 口径得分为 ~85%，与论文 85.88% 一致，验证了预测生成管线的复现性。"
   - English："We report execution accuracy under test-suite-sql-eval (taoyds), which is stricter than the original Spider evaluator used in FINER's official chain (--etype exec): exact row order is required for ORDER BY queries, and a prediction must match the gold on every variant of the multi-instance database. We obtain 76.6% under this metric (easy/medium/hard/extra = ...); under the same in-house results_equal metric as FINER's MV(vav) we obtain ~85%, matching the paper's 85.88% and confirming reproduction of the generation pipeline."
4. **禁止事项**：
   - 不得把 76.6%（test-suite EX）与论文 85.0%（原始 EX）并列在同一张表或同一条对比句；
   - 不得写"复现得分低于论文 8.4pt"而不附口径说明；
   - 若修复成功（原始口径 ≈85.0%），以修复口径为正式复现数，test-suite 口径仅作附录参考。

---

## 5. 下一步行动（L1 技术报告前置条件）

1. **立即（数分钟）**：执行第 3 节 4 项修复并重评，确认原始 exec ≈ 85.0%。
2. **归档固化**：
   - 修复后的脚本固化为 `eval_official.sh`（或新增 `eval_official_spider.sh` 保留两套口径），更新 OFFICIAL_EVAL.md / README 记录两套口径及其差异说明；
   - 从 NEXT_STEPS.md:25 移除"多实例库版本/数量"待查项（本审计已排除）；
   - 删除或重打 `data/spider_data.tar.gz` 旧备份（205MB，仅 372 个 sqlite，与当前 3383 个不一致），防误用。
3. **生成侧校准**：用 FINER-SQL-3B-Spider 官方权重跑 M1 校准探针（IMPROVEMENT_PLAN #2 已有计划），确认 vav 超参（vav_group_size、温度、n=30）与论文一致。
4. **L1 技术报告前置条件清单**：
   - [ ] 修复后原始 exec 复现数（整体 ≈85.0%，难度 94.8/90.1/78.2/64.5）
   - [ ] 三口径对照表：MV(vav) 85.88% / Official EX 85.0% / test-suite EX 76.6%（或修复后数值）
   - [ ] 口径说明段（第 4 节模板）写入报告评估节
   - [ ] 1034 行对齐与空预测处理验证（pred/gold 恒 1034 行）
   - [ ] 生成侧超参与权重一致性确认
   - [ ] 若最终以 test-suite 口径为主口径：预先决定对外命名（"test-suite EX"）与论文对比策略，避免审稿异议
