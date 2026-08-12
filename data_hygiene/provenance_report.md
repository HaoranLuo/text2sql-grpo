# Provenance Report — DeepSeek 蒸馏 1000 条来源核查

- 执行时间:2026-08-13
- 蒸馏文件(唯一):HPC `data/reasoning_data/deepseek-chat_spider_train_think.jsonl`,JSONL 1000 行,sha256 `27ac01f716938ffda5ff1ca7115e59f8c8fbe88c608b6a79f75ca8ad64c24c2b`。
- 每条字段:`index`(train_spider.json 位置)、`question`、`db_id`、`gold_sql`、`success`、`model`(deepseek-chat)、`response`(含 `<think>` 推理轨迹 + SQL)、`messages`、`ddl`、token 统计等。

## 1. 来源核查(与 train_spider.json 逐条比对)

| 检查项 | 结果 |
|---|---|
| 总条数 | 1000 |
| `index` 缺失 | 0 |
| `index` 范围 | 0..999(**恰为 train_spider.json 前 1000 条,连续块**) |
| `index` 唯一性 | 1000/1000 唯一 |
| `index` 是否均在 0..6999 内 | 是 |
| 该 index 处 `db_id` 与 `question` 是否与 train_spider 完全一致 | 1000/1000 一致(0 处不一致) |
| `success` 字段 | 1000/1000 = true |
| 覆盖 db_id | 19 个:allergy_1, bike_1, book_2, chinook_1, coffee_shop, customers_card_transactions, department_management, farm, flight_1, insurance_fnol, journal_committee, medicine_enzyme_interaction, musical, product_catalog, race_track, store_1, student_assessment, twitter_1, university_basketball |

## 2. 落在 GRPO / SFT-gold / 未分配池的分布(基于 split_record.md 的 seed=42 划分)

| 集合 | 蒸馏来源条数 | 占蒸馏总量 |
|---|---|---|
| GRPO 5.5k | **0** | 0% |
| SFT-gold 1.5k | **1000** | 100% |
| 未分配池 1.7k | 0 | 0% |

**与 GRPO 集重叠的条目清单:空**(`overlap_with_grpo = []`)。

**结论:理想目标达成** —— 蒸馏来源 ⊆ SFT-gold + 未分配池,与 GRPO 训练集零重叠。该结果由划分规则构造性保证(SFT-gold 1.5k 显式包含 train_spider 索引 0..999),并已在 HPC 上独立核查验证。

## 3. 发现的问题

1. **蒸馏来源是连续块**:1000 条全部来自 train_spider 前 1000 行(0..999),只覆盖 19 个 db_id。SFT-gold 中的 `distilled_source` 条目数据库多样性受限;已用 500 条 `sampled_extra`(来自其余 db_id)补充,若未来继续蒸馏,建议优先从 19 个 db_id 之外取样。
2. **train_spider 源数据本身有 9 对重复 (db_id, question)**(索引对:1348/1420, 1698/1716, 2211/2212, 2340/2342, 2438/2439, 3414/3484, 3459/3485, 3463/3525, 4398/4399)。蒸馏 1000 条(0..999)不涉及这些重复对,无影响;但其中 hr_1 一对分别落在 GRPO(3463)与 SFT-gold(3525),若下游按 question 文本去重会造成"假重叠",务必以 `idx` 为准。
3. **格式提示**:蒸馏文件 `reasoning_content` 字段为 null,推理轨迹在 `response` 的 `<think>...</think>` 中,SQL 在 markdown 代码块内——下游解析时应以 `response` 为准而非 `reasoning_content`。
4. 未发现数据格式异常(1000 行全部可解析、全部 success、与源完全吻合)。

## 4. 红线遵守情况

- 未修改/删除任何现有数据文件(HPC 零写入,仅 sha256sum 与 python 读)。
- 未提交任何 GPU 作业;未解压/删除 205MB 备份(`data/spider_data.tar.gz`、`spider_data.zip`,仅记录路径与 sha256)。
- 未 git commit/push;未向本地下载 HPC 大文件(仅清单与报告落在本地 `data_hygiene/`)。
