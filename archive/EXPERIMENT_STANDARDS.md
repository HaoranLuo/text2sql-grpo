# Text-to-SQL GRPO 项目实验规范标准（EXPERIMENT_STANDARDS）

> **版本**: v1.0 ｜ **生效日期**: 2026-08-04 ｜ **维护人**: jiahuiwang24
> **适用范围**: 本项目（reasoning_generator_3b）全部 GRPO / SFT / 零样本评估实验，含 3B 与 7B 两条线
> **运行环境**: XJTLU HPC `/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b`（本地编辑镜像 `C:\Users\13389\Desktop\女朋友\reasoning_generator_3b`）
> **关联文档**: [GRPO_CHECKLIST.md](./GRPO_CHECKLIST.md)（检查清单来源）· [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md)（结果登记）· [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md) · [RESEARCH_PLAN.md](./RESEARCH_PLAN.md) · [3B_PROGRESS_REPORT.md](./3B_PROGRESS_REPORT.md) · [7B_PROGRESS_REPORT.md](./7B_PROGRESS_REPORT.md)

**制定背景**: 截至 2026-08-03，项目已在 HPC 上运行 20+ 次实验。其中多次 GPU 小时被静默配置不匹配浪费：`max_completion_length=256` vs 推理 `max_new_tokens=512` 导致 SQL 截断、批大小与 `num_generations` 不整除、TRL API 参数改名（`reward_functions` vs `reward_funcs`、`tokenizer` vs `processing_class`）等。本规范将教训固化为强制流程：**任何不满足本规范的实验视为无效，不计入结论，不写入 EXPERIMENT_LOG.md 汇总表。**

## 六条铁律

| # | 铁律 | 违反后果 |
|---|------|---------|
| 1 | 没有 E-NN 实验 ID 不得开跑 | 结果无法登记，直接作废 |
| 2 | Pre-flight 检查未全部通过不得提交 SLURM 作业 | 烧掉的 GPU 小时自担，不补做 |
| 3 | 训练启动后未看前 20 步仪表盘不得离开 | 静默失败可能整晚空转 |
| 4 | 无 `metadata.json` 的实验视为无效（与铁律 1 连带） | 结果不可复现，不得引用 |
| 5 | 未对照同协议基线（E1 / E4）的指标不做任何结论 | 差值无意义 |
| 6 | 任何评估协议改动必须同时：更新本规范 + 重跑基线 | 新旧结果不可比（E13 教训） |

---

## 1. 实验命名规范

### 1.1 实验 ID（E-NN）

- 每个**独立运行**（一次训练、或一次零样本/外部模型评估）分配一个唯一 ID，格式 `E-NN`（NN 从 01 递增），在 `EXPERIMENT_LOG.md` 的实验登记表中**领取**，先登记后开跑。
- 基线、SFT、外部模型零样本评估同样是实验，同样占 ID（既有 E1/E4 即为基线）。
- 同一训练的变体（如 checkpoint-25 与 final）用字母后缀：`E5b`（既有先例）；不允许出现 `E5bb` 之类的多级后缀。
- ID 一旦分配不得复用、不得跳号；中途放弃的实验保留 ID 并在状态列标注「放弃/无效」及原因。
- 既有实验 E1–E15（含 E5b）保持原 ID 不变，已完成的无需回填，但**后续任何复跑/延展实验必须领新 ID**。

### 1.2 目录与文件命名约定

统一采用 `grpo_<model>_<config>` 模式，并在最前面带实验 ID，保证按 ID 可追溯、按名字可读：

| 对象 | 命名模式 | 示例（对应 7B + 三级奖励 50 步） |
|------|---------|--------------------------------|
| LoRA 检查点 | `checkpoints/<E-ID>_grpo_<model>_<config>` | `checkpoints/E07_grpo_7b_threelevel_50` |
| 评估输出 | `outputs/<E-ID>_grpo_<model>_<config>` | `outputs/E07_grpo_7b_threelevel_50` |
| 零样本基线 | `outputs/<E-ID>_baseline_<model>_<n>` | `outputs/E04_baseline_7b_100` |
| 纯评估（外部模型） | `outputs/<E-ID>_eval_<model>_<n>` | `outputs/E13_eval_xianysql_100` |
| SLURM 脚本 | `scripts/<E-ID>_<任务>.slurm` | `scripts/E07_grpo_7b_threelevel_50.slurm` |
| 运行日志 | `logs/<E-ID>_<任务>_%j.out` / `.err` | `logs/E07_grpo_%j.out` |
| 元数据 | 每个输出目录内 `metadata.json` | `outputs/E07_grpo_7b_threelevel_50/metadata.json` |
| 评估结果 | 每个输出目录内 `items.json` + `summary.json` + `checkpoint.json` | —（既有格式沿用） |

命名中 `<model>` 取 `3b` / `7b`，`<config>` 用 `<奖励类型>[_<步数>][_<tuned>]` 等短标识（参照既有：`grpo_7b_threelevel_50`、`grpo_7b_tuned_25`）。目录名、slurm 脚本名、日志名三者必须使用同一个实验 ID。

> 注：既有 E2–E15 的目录沿用旧名（如 `outputs/grpo_7b_threelevel_50`），以 `EXPERIMENT_LOG.md` 的 ID↔目录映射表为准，不强制改名；新实验一律按本节执行。

### 1.3 metadata.json（强制，每个实验一份）

保存到输出目录根下（与 `summary.json` 同级），由训练/评估脚本自动生成，人工核对补充。必填字段：

| 字段 | 说明 | 自动来源 |
|------|------|---------|
| `exp_id` | E-NN | 脚本入参（新加 `--exp-id`） |
| `date` / `time` | 运行日期与时刻 | `datetime.now()` |
| `base_model` / `model_path` | 基座模型名与路径 | `--model-path` |
| `train_method` | GRPO / SFT / Zero-shot | 脚本类型 |
| `train_data` | 文件、切片（如 `train_spider[:100]`）、条数 | `--num-train` 等 |
| `eval_data` | `dev.json[:100]`（start_index=0, limit=100） | `--limit` / `--start-index` |
| `generation` | `num_generations`、`temperature`、`max_prompt_length`、`max_completion_length`、`do_sample` | 配置打印 |
| `optimizer` | `learning_rate`、`beta`（KL）、`max_steps`、`per_device_train_batch_size`、`gradient_accumulation_steps` | 配置打印 |
| `reward` | 类型（binary / three_level / partial）与函数来源 | `--reward-type` |
| `lora` | r、alpha、dropout、target_modules | `LoraConfig` 实参 |
| `seed` | 随机种子（见 1.4） | `--seed` |
| `environment` | GPU 名、分区、QoS、SLURM 作业号、Python 版本、关键库版本、训练脚本 hash | `nvidia-smi` + `SLURM_JOB_ID` + `SLURM_JOB_PARTITION` + 版本查询 |
| `run_stats` | `steps_per_second`、墙钟时长 | 运行时统计 |
| `notes` | 异常情况、偏离规范之处 | 人工填写 |

JSON 模板（GRPO 训练示例）：

```json
{
  "exp_id": "E16",
  "date": "2026-08-04",
  "time": "14:30:00",
  "base_model": "Qwen2.5-Coder-7B-Instruct",
  "model_path": "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/models/qwen2.5-coder-7b-instruct",
  "train_method": "GRPO",
  "train_data": {"file": "data/spider_data/train_spider.json", "split": "train_spider[:100]", "num_examples": 100},
  "eval_data": {"file": "data/spider_data/dev.json", "split": "dev[:100]", "start_index": 0, "limit": 100},
  "generation": {"num_generations": 2, "temperature": 0.7, "do_sample": true,
                 "max_prompt_length": 1536, "max_completion_length": 512},
  "optimizer": {"learning_rate": 5e-6, "beta_kl": 0.04, "max_steps": 50,
                "per_device_train_batch_size": 2, "gradient_accumulation_steps": 1},
  "reward": {"type": "three_level", "source": "src/train_reasoning_grpo.py::create_reward_function"},
  "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
           "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]},
  "seed": 42,
  "environment": {"gpu": "NVIDIA A40 (48GB)", "partition": "aiaca40", "qos": "1a40",
                  "slurm_job_id": 12345678, "python": "3.10.20",
                  "versions": {"torch": "2.5.1+cu124", "transformers": "4.48.3", "trl": "0.15.2",
                               "peft": "0.14.0", "datasets": "3.2.0", "accelerate": "1.3.0"},
                  "train_script_sha": "<git 提交号或 sha256>"},
  "run_stats": {"steps_per_second": 0.5, "wall_clock_seconds": 7200},
  "notes": ""
}
```

**落地动作（待办代码改动）**: `src/train_reasoning_grpo.py` 现有 `training_metadata.json`（字段不全）升级为上述 `metadata.json`：补 `exp_id`、`seed`、`date/time`、GPU/分区/作业号（读环境变量 `SLURM_JOB_ID`、`SLURM_JOB_PARTITION`）、版本清单与脚本 hash；评估脚本 `src/evaluate_after_grpo.py` 同样在 `summary.json` 旁输出一份 `metadata.json`（不含训练字段，标注 `train_method: Zero-shot` 或引用训练实验 ID）。

### 1.4 随机种子

- 训练与评估都必须是**种子可复现**的：GRPO 采样、LoRA 初始化、数据打乱均受 seed 控制。
- 默认 `--seed 42`；脚本当前未实现 seed 参数，**新实验前先补上**（`torch.manual_seed` + `transformers` 的 `set_seed`），并在 `metadata.json` 记录。
- 冒烟测试与正式运行使用相同 seed，保证冒烟通过后正式复跑行为一致。

---

## 2. 训练前检查清单（Pre-flight）

来源：`GRPO_CHECKLIST.md`（已含本项目两次实测教训）。**每一项通过后在打卡表打勾，随 metadata.json 存档；任一关键项失败即不得提交作业。**

### 2.1 训练 / 推理配置一致性（最高优先级）

| # | 检查项 | 预期 | 检查方法 | 本项目实测坑 |
|---|--------|------|---------|-------------|
| P1 | `max_completion_length` == 推理 `max_new_tokens` | 两者均为 **512** | `grep -n "max_completion_length" src/train_reasoning_grpo.py` 与 `grep -n "max_new_tokens" src/evaluate_after_grpo.py src/reasoning_generator_agent.py` | ❌ 训练 256 vs 推理 512，SQL 截断，整轮实验作废 |
| P2 | 训练采样温度 vs 评估温度 | 训练 `temperature=0.7` 采样；评估 `do_sample=False` 贪婪（设计选择，需记录） | 读脚本与 metadata | ✅ 已固定 |
| P3 | Prompt 模板训练/推理字节级一致 | 相同 | 代码审查：训练 `build_dataset` 与评估 `agent.generate` 均调用 `ReasoningGeneratorAgent.build_prompt` | ✅ 共用函数 |
| P4 | Chat 模板一致 | 相同 | 打印一次 prompt 对比 | ✅ |
| P5 | 金标准 SQL 不进入 prompt | 训练/评估 prompt 中均无 `query` 字段内容 | 打印 `dataset[0]["prompt"]` 人工检查 | ⚠️ 高危，每次必查 |
| P6 | SQL 提取路径一致 | 训练 `extract_sql` 与评估 `ReasoningGeneratorAgent.extract_sql` 同一实现（前者已包装后者） | 代码审查 | ✅ |

### 2.2 Tokenizer 设置

| # | 检查项 | 预期 | 方法 |
|---|--------|------|------|
| T1 | `pad_token` 非 None | 已设置（脚本默认 `pad_token = eos_token`） | 打印 `tokenizer.pad_token` |
| T2 | `model.config.pad_token_id` == `tokenizer.pad_token_id` | 相等 | 打印对比 |
| T3 | 左 padding + attention mask 正确 | 生成与训练正常 | 冒烟测试覆盖 |
| T4 | 训练 / 推理用同一 tokenizer | 同一个本地 checkpoint | 路径核对 |

### 2.3 奖励函数单元测试（关键，每次必做）

任何新奖励函数（或对 `create_reward_function` 的任何修改）先做单元测试，**断言「好答案 > 坏答案」**，不通过不得训练。

```python
# 在 HPC 项目根目录下运行（复用训练脚本的奖励工厂，避免测了 A 用了 B）
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
python - <<'EOF'
import sys; sys.path.insert(0, "src")
from train_reasoning_grpo import create_reward_function

r = create_reward_function("data/spider_data", reward_type="three_level")
db, gold = "car_1", "SELECT count(*) FROM cars_data WHERE year > 2000"

assert r([gold],            [gold], [db])[0] == 1.0, "金标准必须得 1.0"
assert r(["SELECT 1;"],     [gold], [db])[0] == 0.1, "可执行但错应得 0.1"
assert r(["not sql at all"],[gold], [db])[0] == 0.0, "不可执行应得 0.0"
assert r(["SELECT * FROM cars_data;"], [gold], [db])[0] < 1.0, "行数不匹配不得满分"
print("reward unit test PASSED")
EOF
```

| 检查项 | 预期 | 说明 |
|--------|------|------|
| R1 签名 | `(completions, query, db_id, **kwargs) -> List[float]` | TRL 会额外传 `prompts` 等 kwargs，必须有 `**kwargs` |
| R2 好坏排序 | `r_good > r_bad` 对所有手写样本成立 | 3–5 个样本覆盖：精确命中 / 可执行错 / 语法错 / 空输出 / 危险语句 |
| R3 `remove_unused_columns=False` | 保持 False | 否则 `query`/`db_id` 列被剥掉，奖励函数运行时炸或拿不到列 |
| R4 组内区分度 | 前 20 步 `reward_std > 0` | 全 0 = 无学习信号（见 §3） |

### 2.4 批大小与 num_generations 整除性

| # | 检查项 | 预期 | 方法 |
|---|--------|------|------|
| B1 | `per_device_train_batch_size % num_generations == 0` | 整除 | 脚本当前 `per_device_train_batch_size = num_generations`，改批大小时重查 |
| B2 | 训练数据条数 ≥ 批大小 | `--num-train >= per_device_train_batch_size` | 参数核对 |
| B3 | OOM 不会静默缩批 | 日志中批大小恒定 | 前 20 步日志核对（缩批 → 组内候选不足） |

### 2.5 冒烟测试（正式运行前强制，5–10 分钟）

冒烟在 `gpudebug`（3090）或 `aiaca40`（A40）上跑，**与正式实验同一套代码、同一 seed**：

```bash
# 3B 冒烟（gpudebug）：
python src/train_reasoning_grpo.py --num-train 8 --num-generations 2 --max-steps 5 \
    --reward-type three_level --model-path models/Qwen2.5-Coder-3B-Instruct \
    --output-dir checkpoints/E16_smoke
# 7B 冒烟（aiaca40）：
python src/train_reasoning_grpo.py --num-train 8 --num-generations 2 --max-steps 5 \
    --reward-type three_level --model-path models/qwen2.5-coder-7b-instruct \
    --output-dir checkpoints/E16_smoke
```

> 提示：`GRPOConfig(logging_steps=5)` 是硬编码，冒烟时可临时调为 1 以便每步出日志；冒烟通过后恢复。也可以用完整小跑 `--num-train 2 --max-steps 16` 做全链路验证（GRPO_CHECKLIST 第 6 节）。

**冒烟通过标准（全部满足才算过）**:

| 项 | 标准 |
|----|------|
| 进程无异常退出 | 0 错误 |
| 前 20 步仪表盘全部落在健康区间（§3.1 表） | 逐项核对 |
| 奖励函数单元测试通过（§2.3） | PASSED |
| 生成探针：用训练配置生成 10 条，人工比对与推理输出一致 | 一致 |
| `completion_length` 未顶格 512 | 中位数显著小于 512 |
| 记录 `steps_per_second`（用于 §7.2 预算） | 打印 |

### 2.6 Pre-flight 打卡表（随 metadata.json 存档）

```markdown
实验 E-NN  Pre-flight 检查记录（日期 / 检查人）
- [x] P1 长度一致（max_completion_length = max_new_tokens = 512）
- [x] P2 温度策略已记录（训练 0.7 采样 / 评估贪婪）
- [x] P3/P4 Prompt 与 chat 模板一致
- [x] P5 金标准未进 prompt
- [x] T1–T4 tokenizer 设置
- [x] R1–R4 奖励函数单元测试（含输出截图/日志行）
- [x] B1–B3 批大小整除性
- [x] 冒烟测试通过（含 steps/s 记录）
- [x] 版本锁定核对（§5）与 metadata.json 模板就绪
- [x] 实验 ID 已登记（EXPERIMENT_LOG.md）
```

---

## 3. 训练中监控规范

### 3.1 前 20 步仪表盘（启动后必须看完再离开）

`logging_steps=5` 时前 20 步 = 4 条日志行；日志键名以 TRL 0.15.2 实际输出为准，如有出入按原始日志字段核对。

| 指标 | 健康范围 | 异常含义 | 处置 |
|------|---------|---------|------|
| `rewards/*/mean` | > 0.15（**注**：依赖奖励类型与题目难度，若 mean 低但 std>0 且单元测试通过，属正常；mean=0 且 std=0 才是异常） | 恒 0 = 奖励或解码坏了 | 查奖励函数、查输出是否被剥格式 token |
| `reward_std`（`rewards/std`） | > 0 | = 0 = 无学习信号（采样温度过低 / 奖励函数退化为常数 / 所有 completion 相同） | 查温度、查奖励 |
| `frac_reward_zero_std` | < 0.8 | 趋近 1.0 = 绝大多数组内奖励全相同 | 同上，严重时停跑 |
| `grad_norm` | 0.001–1.0（允许前 5 步偏高并快速回落，如 E2 实测 27.1→0.013） | 恒 >10 不稳定；NaN 反向传播已坏 | NaN 立即停跑；恒高查参考模型（§8） |
| `entropy` | 0.05–0.5 | <0.01 模式崩塌 | 停跑，查奖励与温度 |
| `kl` | 健康 0–0.5；>1.0 警戒 | >2.0 发散 | 停跑，调 β / lr |
| `completions/clipped_ratio` | < 0.3 | >0.8 = 生成顶格被截断 → 加长 `max_completion_length` | 与 P1 联动（256/512 教训）；顶格即停 |
| `completion_length` | 中位数显著小于上限 | 恒 = 上限 512 = 截断信号 | 同上 |
| loss | 从 0 附近**上升是正常**的（KL 惩罚所致） | — | 不要误读为回归 |

### 3.2 处置规则

| 级别 | 条件 | 动作 |
|------|------|------|
| 🟢 继续 | 所有指标在健康范围，或仅轻微偏离且单调恢复 | 安心离开，定时回来 |
| 🟡 观察 | 单项触及警戒线（kl>1.0、clipped_ratio 0.3–0.8、grad_norm 5–10） | 再观察 20 步；准备备选参数 |
| 🔴 停跑 | NaN、kl>2.0、clipped_ratio>0.8 顶格、rewards 恒 0 且 std=0、entropy<0.01、grad_norm>10 持续 | 立即 `scancel`，诊断（§8）后重来，**GPU 小时留着比烧着强** |

### 3.3 日志规范

- `report_to="none"`（HPC 无 wandb/tensorboard），指标只能从 stdout 日志读：`logs/<E-ID>_%j.out`。
- 新配置首次运行时把 `logging_steps` 临时调为 1，跑满 20 步后恢复默认。
- 训练结束后将前 20 步原始日志片段（含指标数值）贴进 EXPERIMENT_LOG.md 对应条目，作为「训练动态健康」的证据（E2 已有先例：KL 0.0014→0.0004、std 0.42→0.14、grad 27.1→0.013）。

---

## 4. 训练后评估规范

### 4.1 统一评估协议（全部实验强制一致）

| 项 | 固定值 | 备注 |
|----|--------|------|
| 评估集 | Spider `dev.json` **前 100 条**（`--limit 100 --start-index 0`） | 与 E1 基线同一切片 |
| 提示词 | 与训练完全相同的 11 条指令模板（`ReasoningGeneratorAgent.build_prompt`，仅 schema、无金标准、sqlite 方言） | 禁止为个别模型改模板（E13 教训：改模板=不可比） |
| 生成参数 | `max_new_tokens=512`、`do_sample=False`（贪婪） | 与训练解码路径一致 |
| SQL 提取 | 与训练同一 `extract_sql` 实现（```sql 代码块解析） | — |
| 执行环境 | SQLite + `DatabaseExecutor`（只读、安全检查），ORDER BY 感知的行比较 | — |
| 指标 | **Parse**（SQL 提取成功率）· **Exec**（预测 SQL 执行成功率）· **Match**（执行结果匹配率，**主指标**）；字符串精确匹配率只跟踪不作主指标 | 与 `spider_utils.compute_summary` 一致 |
| 输出 | `summary.json`（含 `evaluator_type`、`is_official_spider_metric=False`）+ `items.json` + `checkpoint.json` | 既有格式 |
| 评估脚本 | `src/evaluate_after_grpo.py`，`--batch-size 8–16`（A40） | 不可修改协议细节后改名继续用 |

**评估器版本化**：`spider_utils.py` 中 `EVALUATOR_TYPE` 标识评估实现版本；任何评估代码改动必须自增该标识，并在本规范修订记录中登记——`evaluator_type` 不同的 `summary.json` **禁止横向比较**。

### 4.2 基线对比（强制）

- 每个训练实验评估后**必须**与同规模零样本基线对比：3B → E1（37%，`outputs/baseline_pretrain_100/summary.json`）；7B → E4（81%，`outputs/baseline_7b_100/summary.json`）。`evaluate_after_grpo.py` 会自动打印 Δ，人工核对。
- 基线缺失时先跑基线（零样本评估 <30 分钟），不得以「大概记得基线」代替。
- 与既有实验的对比统一用「对比自 XX」表述（参照 E11/E12 的「+7.0（对比 E10）」写法），指明参照实验 ID。

### 4.3 统计显著性

同一 100 条样本的配对对比，用 **McNemar 检验**（前后不一致的配对 = 一条 b、一条 c；精确双侧 p = 2·Σ_{k=0}^{min(b,c)} C(n,k)·0.5^n，n=b+c）。

| 判别标准 | 结论 |
|---------|------|
| 不一致配对总数 < 10，或 b:c 接近 1:1 | **不显著**：不能下正/负结论 |
| 不一致配对 ≥ 10 且单方向占 ≥ 9 成（McNemar p < 0.05） | 显著 |
| 纯比例差 | 100 条样本下 |Δ|=1–2 点（1–2 条）**在噪声内**，不做结论；|Δ| ≥ 5 点通常达到显著，但仍建议用 McNemar 确认 |
| 论文级结论 | 同一配置跑 **3 个 seed**，报告 mean ± std，提升需 mean Δ ≥ 5 点且大于跨 seed 波动 | 计入资源预算（§7） |

**实际案例**：E5（81%→79%，−2 点）与 E7（81%→79%）在统计上均属于「与基线无差异」，结论应写「持平/无显著差异」，不写「下降 2 个百分点导致性能退化」。

### 4.4 结果归属

- 一个实验（一个 E-ID）的最终结果 = **最佳 checkpoint**（或最终 checkpoint，二选一并在日志注明；e.g. E6 取 checkpoint-25、E7 取 final）。
- 汇总表只允许出现已完成且协议一致的实验；「待定」行不得长期保留（E15 先例：结果补回后立即更新）。

---

## 5. 版本锁定

### 5.1 每个 run 必须记录的版本清单

| 库 | 本项目固定版本 | 说明 |
|----|--------------|------|
| Python | 3.10.20 | 环境 `envs/reasoning3b` |
| torch | 2.5.1+cu124 | bf16、A40/3090 均支持 |
| transformers | 4.48.3 | — |
| trl | **0.15.2** | API 变动高发区（见 5.3） |
| peft | 0.14.0 | LoRA |
| datasets | 3.2.0 | — |
| accelerate | 1.3.0 | — |

- 每条实验的 `metadata.json` 记录上述 7 项 + `tokenizers` 版本 + 训练脚本 sha256（`sha256sum src/train_reasoning_grpo.py`）。
- **升级任何库 = 新实验**：升级后必须重跑冒烟 + 至少一个对照实验（如 7B 三级奖励 50 步复刻 E7），确认指标在噪声范围内再继续。

### 5.2 环境快照

```bash
# 每次实验前生成快照，与 metadata.json 一同归档到输出目录
$PYTHON -m pip freeze > outputs/<E-ID>_grpo_7b_threelevel_50/pip_freeze_<E-ID>.txt
```

### 5.3 TRL API 变动警示（已踩坑 × 2）

| 参数名（旧） | 参数名（TRL 0.15.2） | 症状 |
|-------------|---------------------|------|
| `reward_functions` | `reward_funcs` | 构造 `GRPOTrainer` 时 TypeError |
| `tokenizer` | `processing_class` | 同上 |
| `num_generations`（位置/语义变动） | 属于 `GRPOConfig` 参数 | 传错位置或版本间改名 → TypeError 或静默分组错误 |

**防御命令（每次新环境必跑）**:

```bash
python -c "
import trl, inspect
print('trl', trl.__version__)
print(list(inspect.signature(trl.GRPOTrainer.__init__).parameters))
print(list(inspect.signature(trl.GRPOConfig.__init__).parameters))
"
# 确认出现: reward_funcs / processing_class / num_generations，与源码调用一致后再开跑
```

### 5.4 版本与复现声明

- 论文/报告引用任何结果时附：环境版本 7 件套 + `metadata.json` 路径 + 训练脚本 hash。
- 复现实验 = 同版本 + 同 seed + 同配置（§1.4）：只满足「同配置」不算复现。

---

## 6. 结果记录模板

### 6.1 EXPERIMENT_LOG.md 单实验条目模板（新实验一律按此填写）

```markdown
### E16 — 7B + GRPO 1000 条（三级奖励，50 步）

| 项目 | 值 |
|------|-----|
| 日期 | 2026-08-04 |
| 模型 | Qwen2.5-Coder-7B-Instruct + LoRA |
| 基座 | Qwen2.5-Coder-7B-Instruct |
| 训练方式 | GRPO |
| 训练步数 | 50 steps |
| 训练数据量 | 1000 examples from train_spider.json |
| 每步候选数 | 4 candidates/group |
| 奖励类型 | three_level |
| 学习率 | 5e-6 |
| Beta (KL) | 0.04 |
| 温度 | 0.7（训练采样）/ 贪婪（评估） |
| Seed | 42 |
| GPU | NVIDIA A40（48GB） |
| 分区 | aiaca40（QoS 1a40） |
| 版本 | torch 2.5.1+cu124 · transformers 4.48.3 · trl 0.15.2 · peft 0.14.0 · datasets 3.2.0 |
| 训练动态（前20步） | rewards mean 0.22 / std 0.31 / kl 0.05→0.01 / grad_norm 1.2→0.08 / clipped_ratio 0.05 |

| 指标 | 训练前 | 训练后 | 变化 |
|------|:---:|:---:|:---:|
| SQL提取成功率 (Parse) | 100%（E4 基线） | xx% | ±x.x |
| SQL执行成功率 (Exec) | 97%（E4 基线） | xx% | ±x.x |
| 执行结果匹配率 (Match) | 81%（E4 基线） | **xx%** | ±x.x |

**统计显著性**: McNemar：不一致配对 b=xx, c=xx，p=xx → 显著 / 不显著
**分析**: ……（结论、与既有实验对照、下一步）
**证据文件**: `scripts/E16_grpo_7b_threelevel_1000.slurm`、`outputs/E16_grpo_7b_threelevel_1000/`（含 `metadata.json`、`summary.json`、`items.json`）、`logs/E16_grpo_%j.out`
```

### 6.2 汇总表模板（§1 结果总览）

| # | 实验 | Match | Δ | 对比参照 | 统计显著性 | 状态 |
|---|------|------|:---:|---------|:---:|:---:|
| E16 | 7B + GRPO 1000 条（三级, 50 步） | **xx%** | +x.x | E4 基线 | 显著（p=xx） | ✅ 已完成 |

### 6.3 归档要求（缺一即不算完成）

| 产物 | 必须存在 |
|------|---------|
| `metadata.json` | ✅ |
| `summary.json`（+ `items.json`） | ✅ |
| 训练前 20 步日志片段 | ✅（贴入条目） |
| SLURM 脚本 | ✅（`scripts/`） |
| 运行日志 | ✅（`logs/`） |
| 基线对照值 | ✅（引用 E1/E4） |
| 奖励单元测试输出 | ✅（新奖励函数时） |

---

## 7. 资源规范

### 7.1 任务类型 × GPU / 分区 / 预算

| 任务 | GPU / 分区 | 时间预算 | 实测依据 |
|------|-----------|---------|---------|
| 奖励单元测试 + 生成探针 | CPU / 交互节点 | ≤ 5 分钟 | — |
| 冒烟测试（3B，8 条 × 5 步） | RTX 3090 / `gpudebug` | ≤ 10 分钟 | — |
| 冒烟测试（7B，8 条 × 5 步） | A40 / `aiaca40` | ≤ 15 分钟 | — |
| 3B GRPO（100 条，2 候选，40 步） | RTX 3090 / `gpudebug` | ~1 小时 | E2 实测（`gpudebug` 上限 1 h，正好够） |
| 3B GRPO（1000 条，4 候选，100 步） | A40 / `aiaca40` | 2–4 小时 | E3 实测 |
| 7B GRPO（100 条，2 候选，25–50 步） | A40 / `aiaca40` | 1–2 小时 | E5–E8 实测 |
| 7B GRPO（500–1000 条） | A40 / `aiaca40` | 2–6 小时 | P0 计划估算 |
| 7B SFT（500 条 API 数据，2 epochs） | A40 / `aiaca40` | 1–2 小时 | E10 实测 |
| dev-100 评估（batch 8–16） | A40 / `aiaca40` | 10–30 分钟 | 实测；3090 上 30–60 分钟 |
| 零样本基线（3B / 7B） | 3090 / A40 | 20–30 分钟 | E1 / E4 实测 |
| 论文级 3-seed 复跑（7B，100 条） | A40 | 3–6 小时 | = 3 × 单次预算 |

### 7.2 A40 vs 3090 选择

| 判据 | RTX 3090（24 GB） | A40（48 GB） |
|------|------------------|--------------|
| 模型规模 | 仅 3B | 3B / 7B 均可 |
| 显存需求 | 7B LoRA + bf16 + 梯度检查点 ≈ 20–25 GB，放不下（无余量） | 从容（E4–E15 全部实测） |
| 分区时长 | `gpudebug` **上限 1 小时** | `aiaca40` 数小时级 |
| 适用场景 | 冒烟、3B 小跑、≤1h 的零样本评估 | 7B 一切、>1h 的 3B、评估 batch 8–16 |
| 决策规则 | 时长 >1h 或模型 ≥7B → 一律 A40，不要硬塞 3090 | — |

### 7.3 队列纪律

| 纪律 | 说明 |
|------|------|
| 交互节点 | 先 `salloc --partition=aiaca40 --qos=1a40 --gres=gpu:1` 再 `srun --pty bash`；`gpudebug` 交互同理 |
| QoS 确认 | 以 `sacctmgr show qos` 实查为准；本项目为 `1a40`（踩过非 `1a40` 无法排队的坑） |
| `--time` 设定 | 冒烟实测 `steps_per_second` × 目标步数 × **1.5 安全系数**；`gpudebug` 不得超过 1 h |
| 作业命名 | `--job-name` 取实验 ID（≤15 字符，如 `E16_grpo`）；全信息写在脚本 echo 里 |
| 单作业单实验 | 一个 slurm 脚本 = 一个 E-ID（训练 + 评估可串联，参照 `exp_grpo_tuned.slurm` 模式） |
| 不重复排队 | 提交前 `squeue -u $USER` 检查；同一实验不得排两份 |
| 提前止损 | 前 20 步仪表盘异常（§3.2 🔴）立即 `scancel`，不抱「再跑跑看」心态 |
| 日志路径 | 统一 `logs/<E-ID>_%j.out` / `.err`，禁止默认路径（`slurm-*.out` 找不回） |

---

## 8. 常见静默失败速查表

**「不报错但结果坏」比报错更贵**——报错 5 分钟能发现，静默失败要等整轮跑完（数 GPU 小时）才暴露。训练曲线停滞或指标异常时，按本表逐行排查。

| # | 症状 | 根因 | 检测方法 | 处置 |
|---|------|------|---------|------|
| 1 | `completion_length` 恒顶格上限；`clipped_ratio` > 0.8；SQL 尾部被截断；reward 平台期 | 训练 `max_completion_length`（256）< 推理 `max_new_tokens`（512）（**本项目真实踩坑**） | 前 20 步日志 + `grep` 两个配置 | 统一为 512；§2.1 P1 前置检查 |
| 2 | KL≈0 恒定、loss~1e-8、`grad_norm` 爆炸（1700–4000） | 参考模型被初始化为策略自身权重（`ref_model` 指向同一权重） | 前 20 步日志三连 | 确认参考模型独立加载；复跑冒烟 |
| 3 | `reward_std`=0 恒定 / `frac_reward_zero_std`=1.0 | 组内奖励全相同：采样温度过低、奖励函数退化为常数、所有 completion 雷同 | §3.1 仪表盘 | 查温度（GRPO 需 >0 采样）、重跑奖励单元测试 |
| 4 | rewards 全 0 但解码正常 | 奖励函数坏了，或解码时剥掉格式 token（如未闭合 `<think>` 被清空）→ 合法输出变空串 | 打印 3 条 completion 人工检查 + 单元测试 | 修 `extract_sql` / 奖励函数；§2.3 |
| 5 | 训练曲线正常但评估结果不变 / 与训练版本不符 | vLLM / 服务进程仍加载**旧权重**（LoRA 未挂上） | 对比服务加载的权重 hash 与训练输出 | 重启服务、重新挂载 LoRA |
| 6 | vLLM 生成时 `temperature≠1` 但 logprobs 未缩放 | vLLM 对非 1 温度返回的 logprobs 不做缩放 → importance-sampling 比错误 | 审查解码路径 | 用原生 HF 生成（本项目路径）或修正缩放 |
| 7 | 训练照常、指标逐步下降 | 小数据 + 稀疏二值奖励过拟合（E3/E5/E5b 教训） | 对比 train/eval 指标；看是否记忆训练样本 | 换 dense 奖励（three_level/partial）、加大数据（P0） |
| 8 | 构造 `GRPOTrainer` 报 TypeError | TRL 版本 API 改名：`reward_functions`→`reward_funcs`、`tokenizer`→`processing_class`（**本项目真实踩坑**） | `inspect.signature` 校验（§5.3） | 按安装版本签名调用；锁定 trl==0.15.2 |
| 9 | 批大小不可被 `num_generations` 整除 | 分组静默错乱或运行时炸 | §2.4 前置检查 | `per_device_train_batch_size = num_generations × k` |
| 10 | 显存 OOM 后自动缩批 | 组内候选数 < `num_generations`，advantage 计算错 | 日志中 batch 数变化 | 开梯度检查点、减 `--num-train` 批、换 A40 |
| 11 | Iterable dataset 训练 | GRPO 从不重复 prompt，advantage 全部错误且无提示 | 检查 `Dataset` 类型（应为普通 `datasets.Dataset`，`build_dataset` 输出即此） | 用 `Dataset.from_list`，不用 Iterable |
| 12 | 生成/训练报 padding 异常或 mask 错误 | `pad_token=None` 或左右 padding 不一致 | §2.2 T1–T3 | `pad_token = eos_token`；左 padding + mask |
| 13 | 评估结果与基线「离奇」偏高/偏低 | 评估协议漂移：`start_index`≠0、`limit`≠100、改过提示词、`evaluator_type` 不同（E13 格式失配教训） | 核对 `metadata.json` 的 `eval_data` + `summary.json` 的 `evaluator_type` | 统一协议（§4.1）；改协议必须重跑基线（铁律 6） |
| 14 | reward 虚高、训练「神速」收敛 | 金标准 SQL 泄漏进 prompt（训练或评估） | 打印 prompt 人工检查（§2.1 P5） | 移除；金标准只进 reward |
| 15 | 训练/推理结果不一致（同一模型同一 prompt） | 温度策略不同（训练 0.7 采样 vs 评估贪婪）被误认为 bug | 核对 §2.1 P2 记录 | 属设计选择，记录即可；如追求一致则统一 |
| 16 | entropy→0、输出退化成单一模板 | 模式崩塌：奖励过密或温度过低、β 过小 | 仪表盘 entropy <0.01 | 停跑，调温度/β，查奖励尺度 |
| 17 | 3B 训练「正常」但 40 步后收益微小（+2 点） | 容量与数据上限（E2 先例），非故障 | 对照 §7.1 预算与既有结论 | 按研究计划转向 7B / 更大数据，不要在 3B 上反复横跳 |
| 18 | 不同 seed 结果忽高忽低 | 100 条样本噪声大，未做 seed 均值 | 多次 seed 对比 | §4.3：3 seed 取 mean±std 再下结论 |

**速查卡（一页纸，贴在前 20 步检查处）**:

```
rewards mean >0.15 | std >0 | frac_zero_std <0.8 | grad_norm 0.001–1.0(降)
entropy 0.05–0.5 | kl <1.0(>2.0 停) | clipped_ratio <0.3 | length 不顶格
mean=0 且 std=0 → 奖励坏     KL≈0+grad 爆炸 → 参考模型用错
顶格 → 长度不匹配(512)       NaN / entropy<0.01 → 立即 scancel
```

---

## 修订记录

| 版本 | 日期 | 修订内容 | 作者 |
|------|------|---------|------|
| v1.0 | 2026-08-04 | 初稿：固化 E-NN 命名、pre-flight、监控仪表盘、统一评估协议、版本锁定、记录模板、资源规范、静默失败速查表 | jiahuiwang24 |
