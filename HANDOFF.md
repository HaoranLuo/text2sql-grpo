# 项目交接文档（HANDOFF）

> 新对话开始时先读这个文件。最后更新：2026-08-13

## 零、最新状态（2026-08-13 中午）

- **L0 已收官**:原始口径 exec 79.1%(40.9%→79.1%);MAC-SQL 评估器已获取待对齐(+3-5pp 潜力)
- **Phase 0(蒸馏 SFT 前准备)全部完成**:数据卫生三不相交划分(GRPO 5.5k/SFT-gold 1.5k,sha256+对抗验证 PASS)、SFT 混合数据 1176 条已生成(1000 蒸馏+176 gold,chat 格式与评估端一致)、preflight 全绿、MPEV 官方 65% 归因完成+eval_official.sh P0 已修
- **进行中**:Phase 1 蒸馏 SFT 提交前监管审查;提交后接 D5 账单实测(50 题)
- **执行计划**:见 ORCHESTRATION_PLAN.md(v1.2 授权实测版:2 卡并行+批量入队);文献弹药见 docs/research_track_20260813.md
- **GitHub**:HEAD=`0107590`(master,已推送)

## 一、项目一句话

Text-to-SQL GRPO（Qwen2.5-Coder 3B/7B + Spider）：完成自身实验矩阵（3B 45→71% 子集/68.5% 全量）、官方评估口径打通、完整复刻验证顶级论文 FINER-SQL（85% 真实）。

## 二、核心结果（最终口径）

| 实验 | 子集100 官方EX | 全量1034 |
|------|:---:|:---:|
| 3B 基线 | 41.4% | — |
| 3B 训练后 | 49.0% | 54.2%（自定义） |
| **3B 训练+5p投票** | **68.1%** | 68.5%（自定义） |
| 7B 基线 | 78.0% | **67.5%**（官方） |
| FINER-3B 单次 | 77.3% | — |
| FINER-3B n=30 vav | — | **76.6%**（官方）/ ~85%（自定义≈论文） |

## 三、关键结论（10 条，详见 FINAL_SUMMARY.md）

1. 投票是最强杠杆（全量 +14.3pp）；5p 是甜点
2. 前 100 条子集偏易 +10pp——必须全量
3. 三级奖励是 3B 唯一有效奖励（50% 天花板）
4. 训练侧突破需推理蒸馏前置（FINER Step 1）
5. vav 投票 > 简单投票；空结果 bug 已修复（+6.4pp）

## 四、HPC 登录与资源

```bash
ssh jiahuiwang24@login.hpc.xjtlu.edu.cn
cd ~/reasoning_generator_3b
# 分区: aiaca40 (qos=1a40, 8h) / gpudebug (qos=gpudebug, 50min)
# 环境: envs/reasoning3b/bin/python
# 权重: models/FINER-SQL-3B-Spider（完整，344+90 张量已验证）
```

## 五、后续计划（详见 NEXT_STEPS.md）

1. **① 推理蒸馏**（~1 天）：DeepSeek API 生成思考+SQL → 过滤 → SFT → 评估（50→60-70%）
2. **② 评分排查**（~半天）：官方 EX 76.6% vs 85% 的差异（评估器版本/plug_value/多实例库）
3. **③ 大规模 GRPO**（2-3 天，依赖①）：全量 8659 + G32 + 四分量奖励
4. 超过 FINER 的机会：多视角投票（5p×采样）、更强教师池、7B 基座、迭代蒸馏

## 六、关键文件

```
FINAL_SUMMARY.md            # 最终总结（10 条发现）
FINAL_REPORT.md             # 第一阶段完整报告
EXPERIMENT_MATRIX.md        # 40+ 实验总表
OFFICIAL_EVAL.md            # 官方口径全景 + 复刻终验
NEXT_STEPS.md               # 后续计划与机会分析
RESEARCH_NOTES.md           # 14 agent 调研结论
docs/FINER_REPLICATION_PLAN.md  # 复刻路线图（27KB）
finer_port/                 # vav 投票移植（28 项自测）
scripts/eval_official.sh    # 官方评估器
data/spider_data/database/  # test-suite 多实例库
```

## 七、GitHub

- 仓库：https://github.com/HaoranLuo/text2sql-grpo（私有）
- 最近提交：e3951d4（最终总结）
- 推送失败时重试或开代理

## 八、重要教训

1. prompt 必须是 chat 格式（messages 列表）
2. 生成长度训练/推理必须一致（512）
3. 评估必须全量 1034 + 官方口径（子集不可比）
4. HF 大文件下载要等进程完全退出再使用（分片无校验）
5. vav 投票的空结果（0 行）必须是 SUCCESS_VALUES 签名（否则丢题）
6. lora-init 用 merge_and_unload（直接传 PeftModel 给 TRL 会 requires_grad 报错）
7. 提交训练前跑 grpo-preflight skill

## 九、自动监控

- 定时检查已停止（2026-08-07 用户要求）
- 重建方法：新会话说"建立自动监控"，用 CronCreate `17,47 * * * *`
- 监控指令：查 squeue → tail 日志 → 记录 → 更新报告 → git push
