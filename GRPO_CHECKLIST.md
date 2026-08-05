# GRPO 训练前检查清单（Pre-Flight Checklist）

> 来源：TRL/verl GitHub issues、Axolotl 稳定性文档、skillsbench 诊断流程 + 本项目踩坑记录
> 目的：**在烧 GPU 小时之前**发现"不报错但结果坏"的静默失败

---

## 1. 训练/推理配置一致性（最高优先级）

| 检查项 | 预期 | 检查方法 | 本项目的坑 |
|--------|------|---------|-----------|
| `max_completion_length` == 推理 `max_new_tokens` | 相等 | grep 两个配置 | ❌ 256 vs 512，训练被截断 |
| `clipped_ratio` < 0.3 | <0.3 | 看前20步日志 | ❌ completion_length 恒=256 顶格 |
| Prompt 格式训练/推理字节级一致 | 相同 | 打印对比 | ✅ 共用 build_prompt |
| chat template 一致 | 相同 | 打印对比 | ✅ |
| 温度策略 | 训练采样/推理贪婪（或一致） | 记录 | ✅ |
| 解码→再分词是同一路径 | 不二次分词 | 代码审查 | ⚠️ 需确认 |

## 2. Tokenizer 设置

| 检查项 | 预期 |
|--------|------|
| `pad_token` 已设置 | 非 None（常需 = eos_token） |
| `model.config.pad_token_id` == `tokenizer.pad_token_id` | 相等 |
| 左 padding + attention mask 正确传递 | 生成正常 |
| 训练/推理用同一 tokenizer | 相同 |

## 3. 奖励函数

| 检查项 | 预期 | 方法 |
|--------|------|------|
| 签名 `(prompts, completions, **kwargs) -> list[float]` | 正确 | 审查 |
| `remove_unused_columns=False` | False | 配置检查 |
| **单元测试：好答案 > 坏答案** | 严格大于 | 3-5 个手写样本 |
| 奖励有区分度（组内 std > 0） | > 0 | 前20步观察 |
| 奖励尺度 | 均值≈0, std 1-3 | 观察 |

## 4. 训练启动后 20 步内检查（最实用）

| 指标 | 健康范围 | 异常含义 |
|------|---------|---------|
| `rewards/*/mean` | > 0.15 | 恒 0 = 奖励/解码坏了 |
| `reward_std` | > 0 | = 0 = 无学习信号 |
| `frac_reward_zero_std` | < 0.8 | = 1.0 = 组内奖励全相同，无梯度 |
| `grad_norm` | 0.001-1.0 | >10 不稳定, NaN 反向传播坏 |
| `entropy` | 0.05-0.5 | <0.01 模式崩塌 |
| `kl` | 0.0-0.5 | >2.0 发散 |
| `completions/clipped_ratio` | < 0.3 | >0.8 = 生成顶格被截断 → 加长 max_completion_length |
| loss 趋势 | 从 0 附近上升是**正常**的（KL 惩罚） | 不要误读为回归 |

## 5. 静默失败高危场景（来自社区）

- **参考模型用自己的权重** → KL≈0 恒成立，loss~1e-8，grad_norm 爆炸 1700-4000
- **vLLM temperature≠1 时 logprobs 不缩放** → importance-sampling 比错误
- **服务旧权重**（vLLM 没加载 LoRA）→ 奖励永远基于旧模型，曲线停滞
- **decode 时剥掉格式 token**（如未闭合的 `<think>`）→ 合法输出被清空，奖励全 0
- **Iterable dataset** → GRPO 从不重复 prompt，advantage 全错
- **OOM 自动缩 batch** → 低于 num_generations，分组崩溃
- **TRL 版本 API 变动** → `num_generations`/`processing_class` 等随时改名，先查版本

## 6. 正式跑之前的冒烟测试（5-10 分钟）

1. **奖励单元测试**：手写 3-5 个好坏样本，断言 `r_good > r_bad`
2. **生成探针**：用训练配置生成 10 条，人工检查是否和推理输出一致
3. **单样本过拟合测试**：1 个 prompt 训几步，模型应向高奖励答案偏移
4. **前 20 步仪表盘**：对照第 4 节表格检查所有指标
5. **完整小跑**：2 prompts × 16 步，全链路验证

---

## 本项目实测记录

| 日期 | 发现 | 修复 |
|------|------|------|
| 2026-08-04 | `max_completion_length=256` vs 推理 `512`，completion_length 恒顶格 256 | 改为 512 ✅ |
| 2026-08-03 | GRPO 失败全系列（reward 不上升）| 可能是长度截断导致，重跑验证中 |
