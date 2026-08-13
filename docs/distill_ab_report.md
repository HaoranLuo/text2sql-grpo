# 蒸馏 A/B 实测报告(2026-08-13,作业 1676735)

> 目的:DISTILLATION_PLAN Step 1——验证 4 个风险点并定稿放量配方。55 题(flash 非思考 25 / flash 思考 25 / pro 思考 5),总花费 ~$0.01。

## 验证清单结果

| 检查项 | 结果 |
|---|---|
| temperature 参数在 thinking 模式兼容 | ✅ 无 400,全成功 |
| 双 `<think>` 结构 | ❌ **thinking 模式 100% 双 think**(reasoning_content 包装 + 模型按系统提示自写);非思考 0 双 |
| max_tokens=4096 截断 | ⚠️ flash 思考 1/25 触顶(轨迹失控变长);非思考最远 224 token,余量巨大 |
| 真实单价 | ✅ 55 题 ~$0.01 → 5000 条 ≈ **$1 级别**(远低于 ¥30 预算) |
| 轨迹质量抽检 | ✅ flash 非思考最佳:思路清晰、SQL 正确、结构标准(与现有 1000 条同格式);thinking 轨迹机械重复(碎碎念) |

## 配方定稿(Step 2)

- **主力 = flash 非思考**(--thinking disabled):质量最好、最快、最便宜,与现有 1000 条同源
- **难题补强 = pro 非思考**(gold SQL 含 JOIN/子查询/UNION 的题,~10%):同参数,QC 过滤兜底
- **多样性 = 每题 2 条轨迹**(temperature 0.7 / 1.0):RFT 文献"收益来自推理路径条数"
- 系统提示不变(现有 SYSTEM_PROMPT_CHAT 已验证);thinking 模式数据全部弃用(双 think + 质量差)

## 教训

- 搜索结论"thinking 未必更好"被实测再次验证(4× 输出 token 换来更差的轨迹)
- 旧别名弃用后脚本价格表必须三处同步(两次 KeyError 漏网,第三次才干净)
