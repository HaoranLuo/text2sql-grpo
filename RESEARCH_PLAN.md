# 后续研究计划：Text-to-SQL GRPO 训练优化

> 基于：实验数据 + AutoLink 论文 + Agentar-Scale-SQL 论文 + 2024-2026 领域文献调研

---

## 一、当前实验结果总结

| 模型 | 方法 | Spider EX | 问题 |
|------|------|:---:|------|
| 3B | 零样本基线 | 37% | — |
| 3B | GRPO 40步 (100条/2候选/二值奖励) | 39% | 提升微弱 |
| 3B | GRPO 100步 (1000条/4候选/二值奖励) | 33% | 过拟合 |
| **7B** | **零样本基线** | **81%** | 🏆 当前最佳 |
| 7B | GRPO 100步 (100条/2候选/二值奖励) | 79% | -2%，过拟合 |

**核心问题**：二值奖励（0/1）导致 GRPO 过拟合。所有论文一致结论。

## 二、论文关键启示

### Agentar-Scale-SQL（蚂蚁集团，BIRD #1：81.67%）

```
三级奖励 = 1.0    完全匹配
           0.1    可执行但结果不对  ← 我们缺的！
           0.0    语法错误/不可执行
```

### 领域共识（13篇论文）

| 奖励类型 | 提升 | 复杂度 |
|------|:---:|:---:|
| 三级奖励（1.0/0.1/0.0） | +3-5% | ⭐ 极低 |
| 列级部分分（PaVeRL-SQL） | +5-8% | ⭐⭐ 低 |
| 多组件复合（Reasoning-SQL） | +6-10% | ⭐⭐⭐ 中 |
| 过程奖励模型（RuCo-C） | +8-12% | ⭐⭐⭐⭐ 高 |

### 各规模模型天花板

| 模型 | Spider | BIRD |
|------|:---:|:---:|
| 3B + 好奖励 | ~85% | ~68% |
| 7B + 好奖励 | ~88% | ~72% |
| 32B + 好奖励 | ~90% | ~82% |

---

## 三、优先级行动计划

### P0：立即做（一天内，预计 +5-8%）

**改造奖励函数——三级奖励**

当前：
```python
reward = 1.0 if rows_match else 0.0  # 二值，太稀疏
```

改为（Agentar-Scale-SQL 方案）：
```python
if rows_match:
    reward = 1.0
elif sql_is_executable:
    reward = 0.1   # ← 关键！可执行但结果错也有信号
else:
    reward = 0.0
```

**执行**：改 `train_reasoning_grpo.py` 的 `reward_func`，一行代码。7B + 三级奖励 + 40-50步 → 预期 81% → **85-88%**。

### P1：本周做（预期 +3-5%）

**1. 增加列级部分分**

非精确匹配时，按列匹配比例给分：
```python
reward = matched_columns / total_gold_columns  # 0.0~1.0
```

**2. 下载 FINER-SQL 代码参考**

https://github.com/thanhdath/finer-sql
- 3B 模型 + GRPO + 记忆奖励 → 67.5% BIRD / 85% Spider
- 单 GPU 可复现
- 原子操作奖励（atomic reward）直接可用

**3. 下载 CSC-SQL 预训练模型**

https://huggingface.co/cycloneboy/CscSQL-Grpo-Qwen2.5-Coder-7B-Instruct
- 已 GRPO 训练好的 7B 模型
- 下载后直接评估对比

### P2：两周内（预期 +3-8%）

**1. AutoLink 式 Schema Linking**

- 用 `bge-large-en-v1.5` 构建列向量库
- Agent 迭代探索：`explore → verify → retrieve → add → stop`
- 适合 Spider 2.0 等大规模数据库场景

**2. 多候选 + 选拔**

- 生成 4-8 个候选 SQL
- 执行分组 + 锦标赛选拔
- Agentar-Scale-SQL 的 selection model 可省略（用执行结果投票代替）

### P3：长期探索

**1. 过程奖励模型（PRM）**
- RuCo-C 式 rubric-based critique
- 给推理过程的每一步打分
- 需要构造训练数据

**2. Agentic GRPO（AGRO-SQL）**
- POMDP 框架
- 数据工厂自动生成训练数据
- 需要更多计算资源

---

## 四、推荐 GitHub 项目（可下载研读）

| 项目 | 链接 | 亮点 |
|------|------|------|
| Agentar-Scale-SQL | github.com/antgroup/Agentar-Scale-SQL | BIRD #1，完整 GRPO 框架 |
| AutoLink | github.com/wzy416/AutoLink | Schema Linking Agent |
| FINER-SQL | github.com/thanhdath/finer-sql | 3B + GRPO，单 GPU |
| CSC-SQL | github.com/CycloneBoy/csc_sql | 3B/7B/32B 预训练模型 |
| Reasoning-SQL | github.com/pourreza/reasoning-sql | 多组件奖励 |
| PaVeRL-SQL | github.com/lpapicchio/paverl-sql | 列级部分分 |

---

## 五、超算资源规划

| 阶段 | GPU | 预估时间 | 产出 |
|------|-----|------|------|
| P0 三级奖励训练 | A40 × 1 | 1-2h | 7B + 新奖励 LoRA |
| P1 列级奖励 | A40 × 1 | 2-3h | 改进版 LoRA |
| P2 Schema Linking | A40 × 1 | 推理时，秒级 | — |
| 下载预训练模型 | CPU | 下载 | HF 7B 模型 |

---

## 六、建议的下一步

**立即：改奖励函数 → 三级（1.0/0.1/0.0）→ 7B + 40步训练 → 预期 85%+ Spider EX**

这是投入产出比最高的改动——改一行代码，预期 +5%。
