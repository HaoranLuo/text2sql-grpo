---
name: grpo-preflight
description: 每次 GRPO/RL 训练提交前必做的配置一致性检查。防止"不报错但结果坏"的静默失败（生成长度不一致、奖励/评估逻辑错位、数据泄漏、版本漂移等）。在提交任何训练或评估作业到超算之前使用。
---

# GRPO Pre-flight 检查

## 何时使用

**每次提交训练/评估作业到超算之前**，必须运行此检查。特别是：
- 修改过任何 src/ 代码后
- 更换模型、数据集、奖励函数后
- 升级任何依赖（torch/transformers/trl/peft）后
- 长时间没有运行后

## 检查流程

### Step 1: 运行自动化检查脚本

```bash
ssh jiahuiwang24@login.hpc.xjtlu.edu.cn
bash ~/reasoning_generator_3b/scripts/preflight_check.sh
```

16 项自动检查，输出 [✅] [⚠️] [❌]：
- **❌ 任何错误** → 禁止提交，先修复
- **⚠️ 警告** → 确认是否可接受（如 checkpoint 覆盖）
- **全绿** → 可以提交

### Step 2: 人工确认检查清单

自动化覆盖不到的，人工确认：

| # | 检查项 | 确认方法 |
|---|--------|---------|
| 1 | 训练/推理生成长度一致 | preflight [1/16] 已自动检查 |
| 2 | prompt 格式训练推理一致 | preflight [2/16] 已自动检查 |
| 3 | 奖励函数与评估逻辑对齐 | preflight [3/16] + [11/16] |
| 4 | 数据泄漏 | preflight [12/16] |
| 5 | 本次实验 ID 已登记 | 在 EXPERIMENT_LOG.md 添加 E-NN 条目 |
| 6 | 输出目录不与其他实验冲突 | preflight [14/16] |
| 7 | 温度策略（训练采样/推理贪婪） | 有意识设计，非偶然 |
| 8 | 本次实验的 metadata.json 会写入 | 训练脚本已写 training_metadata.json |
| 9 | GPU 资源与作业时长匹配 | 对照 EXPERIMENT_STANDARDS.md §7 |

### Step 3: 训练启动后 20 步内检查

训练开始后，**前 20 步必须检查**以下指标（日志在 `logs/<jobname>_<jobid>.err`）：

```bash
# 实时查看训练指标
tail -f ~/reasoning_generator_3b/logs/*.err | grep -E "rewards|loss|kl|grad_norm"
```

| 指标 | 健康范围 | 异常处理 |
|------|---------|---------|
| rewards/*/mean | > 0.15 | 恒 0 = 奖励/解码坏了，立即停止 |
| reward_std | > 0 | = 0 = 无学习信号，停止 |
| grad_norm | 0.001-1.0 | >10 不稳定，观察 |
| kl | 0.0-0.5 | >2.0 发散，停止 |
| completions/clipped_ratio | < 0.3 | >0.8 = 生成被截断，加长 max_completion_length |
| completion_length | 明显 < 上限 | = 上限 = 被截断信号 |

### Step 4: 训练完成后

1. 评估必须用**同一协议**（100 条 dev，同 prompt，同 extract_sql）
2. 与基线（E4: 7B 零样本 81%）对比
3. 结果写入 EXPERIMENT_LOG.md（标准模板）
4. 记录环境版本到实验条目

## 常见静默失败速查

| 症状 | 根因 | 检测 |
|------|------|------|
| reward 不上升 | 生成长度截断 / 奖励无区分度 | clipped_ratio, 单元测试 |
| 训练比推理差很多 | 生成长度不一致 | preflight [1/16] |
| 结果全 0 分 | decode 剥掉格式 token | 检查 extract_sql |
| KL≈0, loss~1e-8 | 参考模型用了自己的权重 | grad_norm 爆炸 |
| 同一问题结果波动 | 数据泄漏 | preflight [12/16] |
| TRL 报未知参数 | 版本 API 改名 | preflight [15/16] |

## 参考文档

- `GRPO_CHECKLIST.md` — 详细检查清单（8 类）
- `EXPERIMENT_STANDARDS.md` — 完整实验规范（命名、监控、评估、版本锁定）
- `EXPERIMENT_LOG.md` — 实验结果登记
