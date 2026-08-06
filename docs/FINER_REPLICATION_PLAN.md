# FINER-SQL 复刻方案（可执行版）

> 目标：把 FINER-SQL 的「推理蒸馏 + 四分量奖励 GRPO + n=30 vav 投票评估」完整移植到本项目 TRL 管线（`src/train_reasoning_grpo.py` 等），在 Qwen2.5-Coder-3B-Instruct 上逼近 FINER-SQL-3B-Spider 官方 Spider dev EX 85.0%。
> 依据：5 个学习 agent 对 `.research_tmp/finer-sql` 的分析结论 + 本项目实际代码核实。
> 参考锚点：FINER-SQL-3B-Spider 官方 EX **85.0%**（vav MV 85.88%，±0.3pp）；BIRD dev 67.80%（--no-memory joint 配置）。本项目现状：3B 零样本 5-prompt 官方 EX 约 41.4%–49.0%（贪心），E 实验（2000 条 finer 奖励 G8 100 步）已完成。

---

## 0. 现状盘点（已核实）

| 维度 | 本项目现状 | FINER-SQL | 差距 |
|---|---|---|---|
| 训练入口 | `src/train_reasoning_grpo.py`（TRL GRPOTrainer + LoRA r16/α32 全线性层，bf16 单卡） | `grpo_writer.py` 全参 bf16 + flash-attn + grad-ckpt，双卡（GPU0 训练 + GPU1 vLLM server） | LoRA vs 全参；G4 vs G16/32；max_completion 512 vs 2048；lr 5e-6 vs 8e-6（3B joint 5e-6）；beta 0.04（E 用 0.02）vs 0.04 |
| 奖励 | `reward_type∈{binary, three_level, partial, atomic, finer}`，全部非负 [0,1]：finer = 1.0 / 0.5+0.5×atomic / 0；无格式门、无 memory | format 硬门（无合法 `<think>` 结构 total=0）×（exec 2.0 / 1.0+atomic / 0）+ thought_reward（memory 余弦） | 无格式门、无 memory、无 GT 缓存 |
| atomic | `src/atomic_ops.py`（自研解析器）+ `src/atomic_reward.py`，e=0.05/β=0.79/γ=0.20，与 FINER 训练参数一致 | sqlglot 解析，训练参数同左 | 解析器语义等价性未验证（自研 vs sqlglot） |
| 执行 | `spider_utils.DatabaseExecutor` 本地 SQLite（只读、白名单），训练时 pred 与 gold 都现场执行 | 外部 HTTP API（8001），gold 行 pickle 缓存 `gt_rows_cache.pkl`，训练只执行 pred | 无 GT 缓存 → 训练执行耗时翻倍 |
| 评估 | 5-prompt 贪心 + `eval_official.sh`（test-suite EX）；`evaluate_after_grpo.py` checkpoint.json 断点续跑 | vLLM n=30 @T=1.0 采样 → vav 执行分组投票 → test-suite EX | 无 n=30 采样、无 vav 投票（我们投票用 Counter 原始行 tuple，对列序/重复行敏感且不过滤退化组） |
| 推理蒸馏 | 无 think 管线（`generate_api_data.py` 是 DeepSeek V4 Flash 无 think 的 SFT 冷启动）；agent 完成格式 = 推理文本 + ```sql 代码块（无 `<think>` 标签） | 四段管线：教师池生成 → 执行验证 → SFT 蒸馏 → ChromaDB Reasoning Bank | 整个 Step 1 缺失 |
| 权重 | `download_finer3b.slurm` 正在下载 `griffith-bigdata/FINER-SQL-3B-Spider` → `models/FINER-SQL-3B-Spider` | — | P0 前置依赖 |

**关键事实**：我们 agent 的完成格式是「自由推理文本 + ```sql 代码块」；`extract_sql` 能容错提取。`<think>` 标签格式必须通过 P2 蒸馏引入（SFT 先决），否则格式硬门会把全部样本归零。

---

## 1. 分阶段路线总览

```
P0  权重验证评估（下载已在进行）        — 验证 85% 真实性 + 打通官方评估
   ↓
P1  vav 投票评估移植                    — 复刻 n=30 采样 + vav 分组投票（评估基建，先于训练）
   ↓
P2  推理蒸馏数据管线                    — 教师生成 → 执行验证 → think-SFT → Reasoning Bank 种子
   ↓
P3  大规模 GRPO 复刻                    — 四分量奖励 + GT 缓存 + 配置对齐 + 双卡模式
```

| 阶段 | 改动文件 | 预期结果 | 时间估算 |
|---|---|---|---|
| P0 | 无（仅跑已有 `scripts/eval_official.sh` 链路） | FINER 权重贪心/小规模官方 EX ≈ 70–80%（n=30+vav 才是 85.0%）；确认评估链路与权重可用 | 0.5–1 天 |
| P1 | 新建 `src/eval_vav_1034.py`；复用 `evaluate_after_grpo.py` 的 checkpoint.json 协议、`eval_official.sh` | FINER-SQL-3B-Spider 复现官方 EX ≈ 85.0%±0.3；我们 3B 基线获得 vav 版本数字（预期 > 5-prompt 贪心） | 实现 1–2 天 + 全量跑 0.5–1 天 |
| P2 | 新建 `src/gen_reasoning_traces.py`、`src/verify_traces.py`、`src/build_sft_reasoning.py`、`src/build_reasoning_bank.py`；改造 `src/train_sft.py` | 2k 条先跑通全链路；think-SFT 基座（`<think>` 格式稳定）；Reasoning Bank 种子库 | 数据生成 1–2 天（API）+ 验证 0.5 天 + SFT 0.5–1 天 |
| P3 | `src/train_reasoning_grpo.py`（四分量奖励 + GT 缓存 + 配置参数化）、新建 `src/memory_reward.py`、`src/build_gt_cache.py`；新 slurm | 2000–4000 条缩样先验证信号单调上升；全量 500–1000 步后官方 EX 超越 E 实验基线（目标 ≥ 60%+，向 85% 逼近） | 缩样 1–2 天；全量 1000 步双 A40 约 2–4 天 |

---

## 2. P0 — 权重验证评估（进行中）

**做法**：
1. 等 `scripts/download_finer3b.slurm` 完成（`models/FINER-SQL-3B-Spider`，约 7GB）。
2. 先做轻量 sanity：复用 `evaluate_after_grpo.py`（--model-path 指向 FINER 权重，不加 LoRA）或 `eval_5prompt_3b_trained.py`，`--limit 100`，拿贪心执行匹配率与官方 EX 子集数字，确认权重加载、DDL/提取链路兼容（FINER 权重是 Qwen2.5-3B 系，tokenizer 兼容）。
3. 若轻量链路通畅，直接进入 P1 全量 n=30 复现（P1 的 1 号验收目标就是 FINER 权重）。

**预期结果**：贪心 100 条自定义执行匹配率应显著高于我们 3B（FINER 权重已有 BIRD/Spider RL 训练史）；官方 EX 全量需 vav 才能到 85%，贪心单候选通常 70% 档，属正常，不 panic。

**注意**：FINER 权重评估用其自带系统提示模板（`You are a meticulous SQL expert…`）比我们 agent 的长 prompt 更贴近其训练分布，P1 里做「FINER 模板 vs 我们 build_prompt」对照臂。

---

## 3. P1 — vav 投票评估移植

**目标**：评估基建与 FINER 同口径（n=30 采样 → 执行分组投票 → 官方 test-suite EX），训练前后的公平比较基准。

### 3.1 新建 `src/eval_vav_1034.py`

结构完全复刻 `evaluate_after_grpo.py` 的 checkpoint.json 协议（`completed_indices` + `items` + dataset_index 映射，与 `eval_official.sh` 兼容）：

1. **加载**：`SpiderLoader.load_dev(limit=1034, start_index=0)`；prompt 用单一 canonical prompt——默认用 `ReasoningGeneratorAgent.build_prompt`，加 `--prompt-style finer` 对照臂（FINER 系统提示模板）。
2. **采样**（替代现有贪心 `candidate_count=1`）：
   - HF transformers 路径：`model.generate(do_sample=True, temperature=1.0, top_p=1.0, num_return_sequences=30, max_new_tokens=2048)`（batch forward 一次产 30 条）；必须给每条候选独立 `extract_sql`（我们无 `<think>`，走现有提取逻辑；FINER 权重则用 `rfind('</think>')` 后内容）。
   - 可选 vLLM n=30 路径（吞吐），但保持与我们 transformers 环境的版本兼容性，首版不做。
3. **执行分组**（移植自 `finer-sql/evaluation/majority_voting.py`，本地化）：
   - `normalize_execution_result(result, gt_sql)`：失败 → `ERROR: {err}`；成功空行 → `SUCCESS_ROWS_COUNT:{n}`；成功有行 → 每行值转 str 排序后 `|` 连接、行集合去重排序后 `;` 连接、截断 200 字符 → `SUCCESS_VALUES:{sig}`（header-agnostic：忽略列名列序）。
   - `is_syntax_error(result)`：失败且错误信息不含 timeout/network/http/request/api/server/connection 等基础设施关键词。
   - `choose_group_vav(groups)`：仅 `SUCCESS_VALUES:` 组；`is_empty` 或 `is_all_zero`（全部值可解析为数值且 |x|<1e-12）硬跳过；取 `max(groups, key=(int(size), key字符串))`；全被过滤 fallback 最大 SUCCESS_VALUES 组；再无 → `NO_RESULTS`；组内取第一条 SQL。
   - 执行后端：`DatabaseExecutor.execute`（本地 SQLite，无需起 FINER 的 FastAPI 沙箱）；加 `(db_id, normalize_sql)` 内存 dict 缓存（30 条候选高度重复，可省 60–70% 执行）。
4. **输出**：每题 `selected_sql` 写回 items.json 的 `predicted_sql`（保留 dataset_index 结构），`candidates` 数组留作分析；同时记录 vav 自评 MV accuracy 与 majority（不过滤）对照，量化「跳过退化组」启发式的贡献。
5. **断点续跑**：按 dataset_index checkpoint/resume（1034×30=31020 次生成，A40 上约 1–2 小时，必须可续）。

### 3.2 跑法与验收

- 验收 1：FINER-SQL-3B-Spider → 官方 EX ≈ 85.0%±0.3（`bash scripts/eval_official.sh <items.json> <out_dir>`）。
- 验收 2：我们 3B（当前 LoRA/基座）100–200 条子集先看 vav 增益方向（对照现有官方 EX 41.4%/49.0%），再全量。
- 难度拆分 easy/medium/hard/extra 记录（对照 FINER 94.8/90.1/78.2/64.5）。

**预期**：官方 EX 与 vav 自评差 ~0.9pp 属正常（官方 test-suite 多实例执行 + ORDER BY 语义更严格）。

---

## 4. P2 — 推理蒸馏数据管线（FINER Step 1 移植）

**目标**：让 3B 学会 `<think>` 推理格式 + 生成正确 SQL 的轨迹，产出 think-SFT 基座（P3 格式门的先决条件）+ Reasoning Bank 种子（memory reward 的库）。

### 4.1 教师模型选择与成本

| 教师 | 角色 | 说明 | 成本 |
|---|---|---|---|
| `deepseek-reasoner`（DeepSeek API） | 主力 | `reasoning_content` 字段抽取 + `<think>` 包裹（FINER `call_deepseek_chat` 逻辑直接抄）；系统提示不提 think | ~$230（18k 题 × n=3，输入≈3k tok/输出≈1.2k tok；官方缓存命中可再降 50–80%） |
| `deepseek-chat`（V3） | 次力 | 系统提示显式要求「think step by step, 输出在 `<think>` 段，`</think>` 后只出 SQL」，模型自吐 think 段；约半价 | ~$115 |
| 已有 DeepSeek V4 Flash 通道（`generate_api_data.py` 用过） | 兜底/低成本 | 无 reasoning_content，走显式 think 提示 | 最低 |

n=1–3（重点 question 多采），temperature 0.2–1.0。**建议先用 2k 条（Spider train 前 2k）小规模跑通全链路再放大**。

### 4.2 Prompt 模板（FINER 一致）

```text
system: You are a meticulous SQL expert. Generate a single, correct SQL query
        for the user question and the provided database schema. Rules:
        - Output exactly one SQL statement.
        - The SQL must be executable on SQLite.
        - Do not include any explanatory text.
        - Output one SQL statement only. Do not include any extra text, tags,
          or code fences.
user:   Database Schema:
        {ddl}
        Question: {question}
```
（reasoning 模型 system 用 `SYSTEM_PROMPT_FOR_REASONING_MODEL`——不提 think；非 reasoning 模型用 `SYSTEM_PROMPT_FOR_NON_REASONING_MODEL`——明确要求 think 段。）

### 4.3 新文件与流程

1. **`src/gen_reasoning_traces.py`**：批量生成，输入 Spider train（`data/spider_data/train_spider.json`，8659 条）+ 可选 BIRD train；`multiprocessing` 16 进程；429 指数退避（base 1s cap 90s + jitter）；输出 JSONL（不用 MongoDB）；`(model_name, sample_id)` 幂等去重断点续跑；`--limit 2000` 先小跑。记录字段：`model_name / sample_id / response / meta(dataset_name, db_id, question, ground_truth_sql)`。
2. **`src/verify_traces.py`**：`rfind('</think>')` 提取 SQL（无标签则走 `ReasoningGeneratorAgent.extract_sql`）；本地 `DatabaseExecutor` 执行 pred 与 gold，`compare_execution_results` 行集相等 → 打 `is_execution_correct` 标记（复用训练同款比较口径）。gold 执行失败样本也保留标记（SFT 过滤用）。
3. **`src/build_sft_reasoning.py`**：**只取执行正确的轨迹**（与 FINER「正确错误混入」不同，我们推理生成项目要纯正轨迹），构造 messages：
   - `[{system, user}, {role: "assistant", content: "<think>\n{reasoning}\n</think>\n```sql\n{sql}\n```"}]`
   - 完成格式双兼容设计：`<think>` 标签（FINER 格式门 + memory 提取用）与 ```sql 代码块（我们 `extract_sql` 用）并存。
   - 存 HF Dataset 本地（train + 10 条 test）。
4. **SFT**：改造/复用 `src/train_sft.py`（LoRA r=16，2–4 epoch，lr 2e-5，损失只算 assistant 部分——TRL SFTTrainer 默认即如此）；产物就是 P3 的 `--lora-init` 冷启动（现有 merge_and_unload 机制已支持）。
5. **`src/build_reasoning_bank.py`**：把执行正确的轨迹（thought 长度 ≥ 50 字符）嵌入进 ChromaDB `reasoning_paths`（metadata：dataset_name/db_id/model_name/reasoning_path），作为 memory reward 的种子库（实现见 §6，默认异库检索防泄漏）。

**预期**：think-SFT 后 3B 贪心执行匹配率提升（think 让模型在采样前先推理），且 `<think>` 格式稳定率 > 90%（P3 格式门的 preflight 判据）。

---

## 5. P3 — 大规模 GRPO 复刻（四分量奖励移植）

### 5.1 目标配置（对齐 FINER train_3b.sh，A40 约束下）

| 参数 | 本项目现状（E） | P3 目标 | 说明 |
|---|---|---|---|
| 数据 | 2000 条 filter-gold | Spider train 全量 8659（可选 + BIRD 9428 联合，GROUP BY 3x 过采样） | GT 行缓存（§5.3）过滤 timeout/空结果 |
| G | 8 | 16（单卡 OOM 则 8 起步） | 每优化步 = G×batch rollouts |
| 步数 | 100 | 500–1000，save_steps=100 早停 | 每 100 步 checkpoint 用 P1 评估粗筛（n=10）再对候选跑 n=30 全量 |
| lr / beta | 3e-6 / 0.02 | 5e-6 / 0.04 | |
| max_completion | 512 | 2048 | 思考空间（配合 P2 think 格式） |
| max_prompt | 1536 | 4096 | |
| temperature | 1.0 | 1.0 | 已一致 |
| 训练方式 | LoRA r16 | 先 LoRA（单卡 24GB 硬约束），双 A40 时切全参 + flash-attn + grad-ckpt（照搬 train_3b.sh 双卡：GPU0 训练 + GPU1 vLLM server，NCCL_P2P_DISABLE） | 全参是 FINER 原配方；LoRA 是过渡 |
| memory | 无 | 默认关（--no-memory，同 FINER 最终脚本），可选开（§6） | |

### 5.2 四分量奖励代码设计（`src/train_reasoning_grpo.py`）

新增 `--reward-type finer2`（FINER 原版语义），保留 `finer`（E 的非负变体）作对比臂。逐样本组装：

```python
# 在 create_reward_function 内（伪代码，含真实函数名）
def reward_func(completions, query, db_id, **kwargs):
    rewards = []
    for comp, gold_sql, db in zip(completions, query, db_id):
        # ── 分量 1：format（硬门，FINER has_format_bonus 的适配版）──
        thought, format_ok = check_format(comp)      # 新函数，见下
        format_reward = 1.0 if format_ok else 0.0
        if format_reward == 0:
            rewards.append(0.0)                      # 硬门：total=0
            continue

        # ── 分量 2：exec（2.0 / 1.0+atomic / 0）──
        pred_sql = extract_sql(comp)                 # 现有 extract_sql
        if not pred_sql:
            rewards.append(0.0); continue
        if not _references_schema_table(pred_sql, gold_sql, db, executor):
            rewards.append(0.0); continue            # 保留我们的 anti-hacking（FINER 无，我们靠它兜底 stub）

        pred_outcome = executor.execute(db, pred_sql)
        gold_outcome = get_gold_rows(db, gold_sql)   # 只查 GT 缓存，不执行（§5.3）
        if not pred_outcome["success"]:
            exec_score = 0.0
        elif pred_outcome["full_rows_truncated"]:
            exec_score = 0.0
        elif gold_outcome is not None and compare_execution_results(
                pred_outcome["full_rows"], gold_outcome["rows"],
                gold_sql=gold_sql)["match"]:
            exec_score = 2.0
        elif gold_outcome is not None:
            exec_score = 1.0
            if not args.no_atomic:                   # 新增开关，默认 False（用 atomic）
                exec_score += _get_atomic_reward().score_against_list(
                    pred_sql, [gold_sql])            # e=0.05/β=0.79/γ=0.20 已一致
        else:
            exec_score = 0.0                         # gold 缓存缺失/超时：不给分

        # ── 分量 3：memory（可选，默认关）──
        thought_reward = 0.0
        if args.use_memory and thought:
            if exec_score < 2.0 and len(thought) > 30:
                thought_reward = _get_memory().compute_thought_reward(
                    thought, db, top_k=args.memory_top_k)   # 空库/异常一律 0.0
            elif exec_score == 2.0:
                thought_reward = 1.0
            # 在线写库：exec==2.0 → 质量门 + save_thought；
            # exec<2.0 → 质量门 + save_failed_thought（均在 reward 循环外抽样执行，
            # 避免每个 completion 都做嵌入拖慢训练——见 5.4 性能注）

        total = exec_score + format_reward + thought_reward
        rewards.append(float(total))
    return rewards
```

**`check_format` 设计（关键决策）**：
- 若 P2 think-SFT 已完成（`<think>` 格式稳定率 > 90%）：**照抄 FINER 门**——恰好 1 个 `<think>` + 1 个 `</think>`、`<think>` 前无 token、thought 长度 > 100 字符、`</think>` 后能提取 SQL（`rfind('</think>')` 后 strip 非空）。
- 过渡期（无 think 训练史，不建议直接上硬门）：软门 `--format-gate soft`——格式不合格 total 不归零，改为 `total *= 0.1`（或减 1.0），防止训练崩溃。**preflight 判据：format_reward==0 占比必须 < 50%**，否则模型学不出格式。
- 模型原生完成格式（推理文本 + ```sql 块）在新 SFT 基座上被 think 格式取代后，`extract_sql` 的 rfind('</think>') 路径与 ```sql 块路径都保留（双兼容）。

**采样监控（移植 FINER L683-693）**：每 100 次 reward 调用打印首样本 thought 前 500 字符 + FORMAT_OK / EXEC_SCORE / THOUGHT_REWARD / TOTAL，用于 collapse 退化检测。

### 5.3 GT 行缓存（`src/build_gt_cache.py` 移植）

- 离线用 `DatabaseExecutor` 对训练集全部 (db_id, gold_sql) 执行，存 pickle `data/gt_rows_cache.pkl`：`{(db_id, gold_sql): {"rows": [...], "truncated": bool}}` + `_TIMEOUT_ENTRIES` 集合。
- `build_dataset` 增加 `--gt-cache` 参数：过滤 timeout/空结果/超长 gold 样本（对齐 FINER `filter_out_timeouts` + `keep_if_short`），行字段增加 `groundtruth_sqls` 兼容未来 BIRD 多 GT。
- 训练循环只执行 pred，gold 只查缓存 → 执行耗时减半。静态 Spider 命中率 100%。
- 缓存构建时间：8659 条 × ~0.1–0.5s ≈ 0.5–2 小时（可复用 E 实验 filter-gold 已执行过的部分）。

### 5.4 性能与显存注

- A40 24GB 单卡 G16 × max_completion 2048 全参必 OOM。路径选择（按优先级）：
  1. 双 A40：GPU1 起 vLLM server（`--use-vllm` 若 TRL 版本支持，否则自写 rollout 桥接）做生成，GPU0 纯训练；**这是 FINER 原配方，最稳**。
  2. 单卡过渡：G8 + LoRA + gradient_checkpointing + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + 必要时 liger kernel。
- memory 嵌入调用（每 completion 一次 0.6B 嵌入）会成为瓶颈：reward 侧嵌入用 OpenAI-SDK 批处理（一次请求多文本），写库侧抽样执行（每 50 个 exec==2.0 的 thought 才做一次质量门 + 写库）。
- 我们的 trl 是 0.15.2，FINER 用 0.25.1：`use_vllm` / `num_generations` 语义基本一致，但 `GRPOConfig` 字段有差异（如 `processing_class` vs `tokenizer`），P3 前先做版本核对（§8 风险 6）。

---

## 6. Memory Reward 新依赖与实现方案

### 6.1 新依赖（仅 3 个 + 模型权重）

| 依赖 | 用途 | 安装/获取 |
|---|---|---|
| `chromadb` | 持久化向量库（`reasoning_paths` / `fail_reasoning_traces`） | `pip install chromadb` |
| `openai`（SDK） | OpenAI 兼容接口打 vLLM embed server | `pip install openai`（FINER 也是这么用的） |
| `Qwen/Qwen3-Embedding-0.6B`（~1.2GB） | 嵌入模型 | HF 下载；经 vLLM `task=embed`（max_model_len=8192）或 `sentence-transformers`（`truncate_dim=2048`）加载 |
| （可选）`vllm`、`sentence-transformers` | 嵌入服务端/降级路径 | 二选一 |

**不需要** MongoDB/pymongo：种子库由我们自己的执行验证轨迹（P2 产物或既有 eval 正确集）直接灌入。

### 6.2 新建 `src/memory_reward.py`（约 250 行，移植 `finer-sql/memory/compute_intrinsic_reward.py`）

```python
class IntrinsicRewardComputer:
    # ChromaDB PersistentClient(path=args.chroma_path, default "./chroma_db")
    # 两个 collection: "reasoning_paths"(正) + "fail_reasoning_traces"(负)
    # 嵌入后端二选一：
    #   A) OpenAI(base_url=args.embedding_api_base, api_key="dummy") → vLLM embed server
    #      （独立 vLLM 进程 + GPU，避开 FINER 踩过的 colocate 问题）
    #   B) SentenceTransformer("Qwen/Qwen3-Embedding-0.6B") 进程内推理（无 GPU 时降级）
    def compute_thought_reward(thought, this_db_id, top_k=20) -> float:
        # 1) 嵌入 thought（8192 token 截断）
        # 2) Chroma query top_k，默认 where={"db_id": {"$ne": this_db_id}}（防泄漏）
        #    --same-db-retrieval / --only-same-db-retrieval 可选
        # 3) R = cos(a_emb, centroid(检索路径嵌入均值))；空库/异常/nan → 0.0
    def save_thought(dataset_name, db_id, thought, model_name) -> str|None:
        # 质量门 calculate_reasoning_quality（30–2000 词、schema 密度≥0.05 且命中≥2、
        # bigram TTR≥0.6，composite=0.33*len+0.33*schema+0.34*TTR）→ should_insert
        # sha1(dataset_name+db_id+thought) id；最近邻余弦≥0.9 去重跳过
    def save_failed_thought(...):  # 同上，写 fail_reasoning_traces（contrastive 可选）
```

### 6.3 接入训练脚本与权重建议

- CLI：`--use-memory / --memory-top-k 20 / --chroma-path ./chroma_db / --embedding-api-base http://localhost:9000/v1 / --no-atomic`。
- **默认禁用**（同 FINER 最终脚本 `--no-memory`——其 vLLM 0.10.2 服务 Qwen3-Embedding 有坑，且 joint 关 memory 仍 67.80%）。定位为「可选增益」。
- 权重：FINER 是等权加法（exec + format + thought，上限 4）；我们保持该风格但给 memory 项系数 `--memory-w 0.1–0.3` 起步消融，避免干扰已稳定的 atomic 信号。
- 上线顺序：**离线建库（P2 正确轨迹种子）→ 单机短跑验证 reward 分布非全零 → 正式训练**。

---

## 7. 与现有实验（A 投票蒸馏 / E FINER 奖励）的衔接与取舍

| 现有资产 | 衔接方式 | 取舍 |
|---|---|---|
| 实验 A（5-prompt 投票蒸馏：`gen_vote_sft.py` / `exp_a2_vote_sft.slurm` / `exp_a3_vote_5p.slurm`） | A 的投票正确集可直接作为 P2 Reasoning Bank 的种子来源之一（免 API 成本）；A 的 SFT 基座可作为 P3 起点对比臂 | A 是无 think 的 SQL 蒸馏，P2 是 think 推理蒸馏——P2 是升级路径，A 保留为轻量对照；若 P2 数据有限，先用 A 正确集灌 memory 种子、再蒸馏 think |
| 实验 E（`exp_e_finer.slurm`：2000 条 finer 奖励 G8 100 步 lr 3e-6 β 0.02 已跑） | E 的 checkpoint-100 与 E 的 filter-gold 数据集直接作为 P3 缩样起点（避免重跑数据过滤）；E 的每 25 步 checkpoint 评估协议升级为每 100 步 + P1 全口径 | E 的 `0.5+0.5×atomic` 非负变体与 FINER `1.0+atomic` 可做消融对比臂（`--reward-type finer` vs `finer2`）；E 的结论「finer 非负组合在 2000 条/G8/512 completion 下比 three_level 好」是 P3 全量化的基线理由 |
| `src/atomic_ops.py`（自研解析器） | 继续使用；P3 前补一轮与 sqlglot 的语义等价抽查（50 条 SQL，Jaccard 差异 > 0.1 的样本人工比对） | 若抽查发现系统性偏差，换用 FINER 的 sqlglot 版 `atomic_ops/atomic_ops.py`（本仓库已有，逐字节验证过） |
| 我们特有的 anti-hacking（`_references_schema_table`） | 保留（FINER 靠格式门兜底，我们双保险） | 与格式门并存不冲突 |
| `eval_official.sh` / `OFFICIAL_EVAL.md` / `spider_test_suite.tar.gz` | P1 直接复用 | 不引入 MAC-SQL 依赖 |

**总体取舍原则**：A/E 是「在当前小规模预算内验证信号」的存量资产，P1–P3 是「向 FINER 配方全量化」的增量路径；每一步都有与存量实验同口径的对照，不做重复建设。

---

## 8. 风险清单

| 风险 | 等级 | 缓解措施 |
|---|---|---|
| **显存**：G16 + max_completion 2048 全参在 24GB 单卡必 OOM | 高 | 双 A40（vLLM server 分离，FINER 原配方）优先；单卡过渡用 G8 + LoRA + grad-ckpt + expandable_segments；`train-batch-size` 已可调 |
| **生成长度不一致**：训练 max_completion 512→2048 时，评估侧 `max_new_tokens` 必须同步 2048（我们 agent 默认 512），否则评估与训练错位 | 高 | 提交前用 grpo-preflight 检查训练/评估长度一致性；P1 采样器显式 2048 |
| **格式门归零**：无 think 训练史直接上硬门，format_reward==0 占比可能 > 50%，模型学不出格式 | 高 | P2 think-SFT 先决；软门过渡（--format-gate soft）；preflight 统计 format==0 占比 |
| **memory 服务问题**：vLLM 0.10.2 服务 Qwen3-Embedding 有坑（FINER 亲历，最终关 memory）；嵌入请求拖慢训练 | 中 | 默认 --no-memory；独立 embed server + OpenAI SDK 批处理；SentenceTransformer 降级路径；先离线建库验证 reward 分布非全零 |
| **时间**：单 A40 1000 步 G16 全参可能 3–7 天（FINER 双卡 1–3 天） | 中 | 每 100 步 checkpoint + n=10 粗筛早停（FINER 经验最佳 200–600 步）；先用 2000–4000 条缩样验证信号单调再全量 |
| **trl 版本差**：我们 0.15.2 vs FINER 0.25.1（`GRPOConfig` 字段/`processing_class`/`use_vllm` 语义） | 中 | P3 前在独立环境装 trl 0.25.1 核对 API；必要时锁 0.25.1 单独环境（不动现网 0.15.2 环境） |
| **数据获取**：BIRD train 需授权申请；FINER 训练数据非公开 | 中 | Spider train 8659 全量即可开跑（公开）；BIRD 走申请通道或跳过（memory 检索跨库多样性略降）；`thanhdath/spider_dev_prompts` 公开可拉 |
| **数据泄漏**：memory 检索含同 db_id 路径、GT SQL 泄进 prompt | 中 | 默认 `where db_id != 当前`（防泄漏）；prompt 构造沿用现有协议（query 不进 prompt）；GT 缓存键含 db_id |
| **API 成本**：18k 题 × n=3 deepseek-reasoner ≈ $230（¥1600+） | 中 | n=1 起步（¥300–800）；V3 半价；官方缓存命中降 50–80%；先 2k 条跑通再放大 |
| **奖励/评估口径漂移**：训练用 compare_execution_results（ORDER BY 感知 multiset）vs vav 排序值集合 vs 官方 test-suite（多实例）三者不一致 | 低 | 明确三层口径并记录对照（差 ~0.9pp 正常）；训练 reward 与 P1 评估用同一 compare 函数 |
| **权重下载失败/兼容**：FINER-3B-Spider 是 Qwen2.5 系，tokenizer 与本地提取逻辑需验证 | 低 | P0 先 100 条 sanity；有问题回退 HF 镜像/ModelScope（`tmp/dl_modelscope.py` 已有通道） |

---

## 9. 建议执行顺序与里程碑

1. **P0（0.5–1 天）**：权重落地 → 100 条 sanity → 确认评估链路。
2. **P1（2–3 天）**：`src/eval_vav_1034.py` → FINER 权重复现 85.0% → 我们 3B 基线 vav 数字（子集先行）。
3. **P2（2–4 天，API 并行）**：2k 条 think 蒸馏 → 验证 → SFT → bank 种子；全量 8659 放大。
4. **P3（缩样 1–2 天 → 全量 2–4 天）**：四分量奖励 + GT 缓存 + 配置对齐 → 2000 条 G8 100–300 步信号验证（对照 E 的 checkpoint 数字）→ 全量 500–1000 步 + 每 100 步 P1 全口径评估早停。

**总周期估算**：约 1.5–2.5 周（单卡资源为主），双 A40 可压缩到 1–1.5 周。
