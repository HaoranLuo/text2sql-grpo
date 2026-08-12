# Split Record — Spider train 8659 不相交划分(G3 纪律)

- 执行时间:2026-08-13
- 执行 agent:data-hygiene(Text-to-SQL GRPO 项目)
- 原则:HPC 全程只读(仅 python 读文件 + sha256sum),未在 HPC 上新建/修改任何文件;本地只在 `data_hygiene/` 下新建记录。

## 1. 数据资产定位(均位于 HPC)

| 文件 | 格式 | 规模 | 条目数 |
|---|---|---|---|
| `data/spider_data/train_spider.json` | JSON list | 24,928,884 B (~25 MB) | 7000(全部含 gold `sql`) |
| `data/spider_data/train_others.json` | JSON list | 8,514,621 B (~8.5 MB) | 1659(全部含 gold `sql`) |
| `data/spider_data/train_gold.sql` | SQL | 1,176,039 B | 对应 train 的 gold SQL |
| `data/spider_data/tables.json` | JSON | 810,971 B | schema 表定义 |
| `data/reasoning_data/deepseek-chat_spider_train_think.jsonl` | JSONL | 5,153,417 B (~5.2 MB) | 1000(DeepSeek 蒸馏:推理轨迹+SQL,字段含 `index`/`db_id`/`question`/`gold_sql`/`response`/`messages` 等) |

- HPC 根路径:`/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b`
- train_spider + train_others = 7000 + 1659 = **8659**,与纪律 G3 一致。
- 本地无 Spider 原始数据,相关资产仅存在于 HPC `data/`。
- 旧备份(红线 ④,仅引用路径,未解压/未删除/未修改):
  - `data/spider_data.tar.gz`(205,204,261 B)
  - `spider_data.zip`(205,800,266 B)
- 已有的下游产物(`data/sft_gold500.json`、`data/sft_spider.json`、`data/sft_filtered.json`、`data/api_sft_data.json`、`data/vote_sft_*.json`)均为 prompt 格式化产物,不带 Spider 索引,**不属于本划分、未被改动**;本划分是权威的不相交基准。

## 2. 划分规则(确定性、可复现)

- **规范索引(canonical index)**:`0..6999` = train_spider.json 中的位置;`7000..8658` = train_others.json 中的位置 + 7000。划分原子单位是规范索引(每行一条),question 文本只作为冗余标识写入清单。
- **随机种子:`seed = 42`**(`random.Random(42).sample`,仅用于抽取 SFT-gold 的 500 条补充)。
- 规则:
  1. **SFT-gold 1500** = train_spider 索引 `0..999`(恰为 1000 条蒸馏来源,见 provenance_report.md)+ 从 train_spider 索引 `1000..6999` 中按 seed=42 随机抽 500 条(`origin=sample_extra`)。
  2. **GRPO 5500** = train_spider 索引 `1000..6999` 中未被抽入 SFT-gold 的全部条目(恰好 5500 条)。
  3. **未分配池(unassigned pool)1659** = train_others 全部条目(规范索引 7000..8658)。
- 由该规则**构造性保证**:GRPO ∩ SFT-gold = ∅、GRPO ∩ 未分配池 = ∅、SFT-gold ∩ 未分配池 = ∅,三集并集 = 8659(脚本内已 assert 验证)。
- 复现脚本:本地 `tmp/partition_hygiene.py`(通过 stdin 在 HPC 上执行,输出 blob → 本地拆分)。同一批原始文件 + seed=42 可完整复现本划分。

## 3. 清单文件(本地 data_hygiene/)

| 文件 | 行数 | 字段 |
|---|---|---|
| `data_hygiene/grpo_5500_manifest.jsonl` | 5500 | `idx`(规范索引), `db_id`, `question`, `origin` |
| `data_hygiene/sft_gold_1500_manifest.jsonl` | 1500 | 同上(`origin` = `distilled_source` ×1000 / `sampled_extra` ×500) |
| `data_hygiene/unassigned_pool_1659_manifest.jsonl` | 1659 | 同上 |

集统计:GRPO 5500 条覆盖 122 个 db_id;SFT-gold 1500 条覆盖 133 个 db_id;未分配池 1659 条覆盖 6 个 db_id。

## 4. sha256 指纹表

### HPC 原始数据(只读计算)

| 文件(HPC 相对路径) | sha256 |
|---|---|
| `data/spider_data/train_spider.json` | `c43d0d72e59e1a9e1a60837da9bf70d5a6277226bdb7f634d544f380646f527a` |
| `data/spider_data/train_others.json` | `7adb04af470b3c9be653504e03c9a36c1b963a861f308ecf25d436472284e10f` |
| `data/spider_data/train_gold.sql` | `20b623d4873dad57f5f66cbeaaee5d400def52d77c5a136bba879605660a6613` |
| `data/spider_data/tables.json` | `61bb20aa401f03164e2d7f3b16509b7b5f79cc9c943ca7bd159046df1159e2ed` |
| `data/reasoning_data/deepseek-chat_spider_train_think.jsonl` | `27ac01f716938ffda5ff1ca7115e59f8c8fbe88c608b6a79f75ca8ad64c24c2b` |
| `data/spider_data.tar.gz`(备份,仅引用) | `3e0a95e4541768cfd89af9f5c4ff41e877c9ac09431d87954d641d86ca810940` |
| `spider_data.zip`(备份,仅引用) | `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b` |

### 本地划分清单

| 文件 | sha256 |
|---|---|
| `data_hygiene/grpo_5500_manifest.jsonl` | `fbf7e07dc4d3e8d60de881e0c55e51fd429f1912e04833c5bf08961f3b430b3c` |
| `data_hygiene/sft_gold_1500_manifest.jsonl` | `1b4bdc848635d9d930b450a688494ad728ef08f07450a37f1eb70f5beee773da` |
| `data_hygiene/unassigned_pool_1659_manifest.jsonl` | `ba49c9ccf4a0802ea715be38f7c811feb47d3eb221eab3d09cc0add5b6917701` |

## 5. 已知注意事项

1. **train_spider 内存在 9 对重复 (db_id, question)**(共 18 条),其中 8 对两两同落在 GRPO,1 对(hr_1 "display the department name and number of employees...")分别落在 GRPO(idx 3463)与 SFT-gold(idx 3525)。**按索引划分两集仍严格不相交**,但若下游按 question 文本去重,会把这 2 条当作同一题——请始终以 `idx` 为原子标识(见 provenance_report.md)。
2. 清单中 `origin=pool` 表示该条目非蒸馏来源、非 SFT-gold 补充抽取(即 GRPO 常规条目或未分配池条目)。
3. 未分配池(train_others)1659 条全部含 gold SQL,后续如需扩 SFT 或 GRPO,可从池中按 seed 续取,不会破坏已记录的两集。
