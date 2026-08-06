# 官方 test-suite 评估结果

> 评估器：官方 [test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval)（Spider 官方口径，多实例模糊化执行匹配）
> 子集：dev 前 100 条（与全部实验同一切片）；全量 1034 复验待 GPU

## 100 条子集对比（2026-08-06）

| 实验 | 自定义口径 | 官方 test-suite EX | 官方 EM | 差值 |
|------|:---:|:---:|:---:|:---:|
| 3B 基线 | 45.0% | **41.4%** | 11.1% | -3.6pp |
| 3B 训练后 (25步) | 50.0% | **49.0%** | 15.3% | -1.0pp |
| 7B 基线 | 81.0% | **78.0%** | — | -3.0pp |

## 结论

1. **官方口径比自定义严格约 1-4pp**（多实例模糊化 + 结果集匹配，验证了调研判断）
2. **训练增益在官方口径下依然成立且更明显**：41.4% → 49.0%（+7.6pp，自定义口径 +5pp）
3. 官方 EM（结构匹配）远低于执行匹配（11-15%），符合已知结论（EM 过于严格）

## 命令

```bash
bash scripts/eval_official.sh outputs/eval_rev_X/items.json outputs/official_X
# 需要: test-suite 多实例数据库 (data/spider_data/database)、nltk punkt、sqlparse
```
