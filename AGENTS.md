# AGENTS.md — Text-to-SQL GRPO 项目常驻铁律

> 本文档常驻上下文（比 skills 触发更可靠）。详细背景见 `PROJECT_GUIDE.md`（新会话必读）。

## 数字铁律（违反即返工）

1. **三口径不可混比**：自定义执行匹配 / 原始 Spider（FINER 同源链）/ 官方 test-suite 是三个不同口径。对外正式数字 = 全量 1034 + 官方 test-suite；任何对比必须注明口径，禁止裸比。
2. **子集 100 条仅内部诊断**，不可进论文（偏易/偏难方向不统一）。
3. **训练提交前必跑** `scripts/preflight_check.sh --strict`（退出码 0 才可提交）+ grpo-preflight 检查。
4. **评估产物核对**：items 恒 1034 行、pred/gold 对齐、空预测写 SELECT 1（不跳过）。

## 实验纪律

- 数据卫生：GRPO 5.5k / SFT-gold 1.5k 不相交划分，sha256 记录（G3）。
- 花 GPU/API 的动作（提交 slurm 训练、DeepSeek 批量生成）先报方案等批准；只读动作（查 squeue、读日志、分析）可自主。
- 每实验闭环：出数 → 核对 → `record_experiment.py` → git commit+push → 更新 PROJECT_GUIDE §2/§3。

## 环境速记

- 计算：西交利物浦 HPC（`ssh hpc`，经 gf-bastion cpolar 隧道端口 11545；断线用 `scripts/ssh_hpc.ps1` 重试）。
- 关键权重：`models/FINER-SQL-3B-Spider`（复刻基准）、checkpoints/sft_v2（当前推理主力）。
- GitHub 需要本机 7897 代理（Vortex）在线；HPC 直连 GitHub 正常。
- 当前主线：训练侧 RL 已封盘（T2.1/2/3 三连负）；最优 = 5p 投票×SFT v1 官方 70.1%；重心在投票管线升级 + L7 机制论文数据（E0-E4）。

## Skills

用户级 skills 在 `~/.agents/skills/`（编排/研究/工程套件，2026-08-14 优化版）。编排任务用 `orchestrate`；实验分析用 `expt-analyst`；论文产出用 `paper-compiler` + `peer-review`。
