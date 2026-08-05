# 项目交接文档（HANDOFF）

> 新对话开始时先读这个文件。最后更新：2026-08-05

## 一、项目一句话

用 GRPO 强化学习训练 3B/7B 模型做 Text-to-SQL（Spider 数据集），已修复 7 个静默配置 bug，当前验证"训练+多prompt投票"最优组合。

## 二、核心结果（修复后真实数字）

| 实验 | Match |
|------|:---:|
| 3B 零样本基线 | 45% |
| 3B 三级奖励 G=4 25步 | **50%** ✅ |
| 3B 训练后 3prompt 投票 | **65%** |
| 3B 训练后 5prompt 投票 | **70%** 🏆 3B最佳 |
| 7B 零样本 | 81% |
| 7B 3prompt 投票 | **85%** 🏆 项目最佳 |
| 7B 训练后投票 | 84% |
| P2A-A1 (三级 G=8) | 47% |
| C2 (partial G=8) | 46% |

**关键结论**：
- 推理增强（投票）> 训练（对已强模型）
- 3B 训练有效（+5%），7B 训练饱和
- **G=4 > G=8**（3B 甜点）
- 25 步是 3B 甜点，50 步起过拟合
- 多prompt视角 >> 随机采样（85% vs 54%）
- **投票数 3→5 再 +5%**（65%→70%），更多视角 = 更高上界
- **修复验证通过**：11 项代码审查修复后重跑基线/训练后 = 45%/50% 完全一致

## 三、HPC 登录

```bash
ssh jiahuiwang24@login.hpc.xjtlu.edu.cn
cd ~/reasoning_generator_3b
```

- 分区：aiaca40 (A40, qos=1a40) / gpudebug (3090, qos=gpudebug)
- QoS 限制：每个分区同时 1 个作业
- 环境：envs/reasoning3b/bin/python

## 四、运行中的作业（检查）

```bash
squeue -u jiahuiwang24
```

（2026-08-05 晚：全部完成，GPU 空闲。验证作业 rev_base/rev_trained 已完成，数字一致。）

## 五、关键文件

```
src/train_reasoning_grpo.py    # GRPO训练（--reward-type: binary/three_level/partial/atomic）
src/reasoning_generator_agent.py  # 推理Agent（chat格式+注释剥离）
src/evaluate_after_grpo.py     # 评估
src/atomic_ops.py + atomic_reward.py  # FINER原子奖励
src/gen_filtered_sft.py        # SFT数据生成（需API）
scripts/exp_c3_atomic.slurm    # C3 原子奖励
scripts/phase2_a_scan.slurm    # P2A 数据量扫描
scripts/record_experiment.py   # 实验记录系统
scripts/preflight_check.sh     # 16项训练前检查
FINAL_REPORT.md / EXPERIMENT_MATRIX.md / PHASE2_PLAN.md
```

## 六、下一步（GPU 空闲，按优先级）

1. **7B 训练后 × 5prompt 投票**（验证 3B 的 3→5 增益是否在 7B 重现；M5plus 显示 5prompt+仲裁=85% 饱和，预期收益低）
2. **3B 投票数据量扫描**：用 5prompt 投票重测 P2A 系列 checkpoint（100/500/2000/7000）
3. SFT 冷启动（gold SQL，本地构建，无需 API）
4. 官方 test-suite 评测器
5. 7B 关键实验修复验证（81%/85% 是修复前评估，可重跑确认）
6. GitHub 推送（网络不稳，手动推）

> 自动监控：Claude 会话内 cron（15,45 * * * *）每 30 分钟检查作业+记录+推送，会话结束即失效，需重新建立。

## 七、GitHub

- 仓库：https://github.com/HaoranLuo/text2sql-grpo（私有）
- 推送可能失败（网络），重试或开梯子

## 八、重要教训（别忘）

1. 训练/推理生成长度必须一致（512）
2. prompt 必须是 chat 格式（messages 列表）
3. 奖励/评估逻辑必须对齐
4. pad/eos + model.config 都要统一
5. 评估脚本要处理多语句 SQL
6. G=4 对 3B 最优，G=8 稀释信号
7. 提交前跑 preflight_check.sh
