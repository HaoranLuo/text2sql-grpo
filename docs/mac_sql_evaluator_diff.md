# MAC-SQL evaluation_spider.py 与本地 original_spider_eval 差异清单

> 目标:对齐 L0 待办「获取 MAC-SQL evaluation_spider.py(论文所用)进一步对齐,可能 +3-5pp」。
> 本文档只做差异记录,不改动 `tools/original_spider_eval/` 下任何文件。

## 0. 来源信息

- 官方仓库: https://github.com/wbbeyourself/MAC-SQL
- 分支 / commit: `main` @ `31a9df5e0d520be4769be57a4b9022e5e34a14f4`(2025-02-27,"add new flag")
- 评估相关文件(已下载到 `.research_tmp/mac_sql/`):

| 文件 | 大小 | 用途 |
|---|---|---|
| `evaluation/evaluation_spider.py` | 34,983 B | Spider 评估主入口(exec + exact match 融合版) |
| `evaluation/process_sql.py` | 16,828 B | 官方 Spider SQL 结构解析器(未修改的原版) |
| `evaluation/exec_eval.py` | 9,160 B | 测试套件式执行等价判断(denotation) |
| `evaluation/parse.py` | 8,186 B | value 提取 / plug_value 枚举 / remove_distinct |
| `evaluation/evaluation_bird_ex.py` | 7,882 B | BIRD-EX 评估(依赖 func_timeout,非 Spider 用) |
| `evaluation/evaluation_bird_ves.py` | 7,803 B | BIRD-VES 评估(同上) |

- 代码血统:`evaluation_spider.py` ≈ [taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) 的 `evaluation.py`(Spider 官方 exact-match 评估器 + 测试套件 exec 评估器融合),MAC-SQL 在其上加了少量调试输出与 `evaluation.json` 导出;`exec_eval.py`/`parse.py` 同样来自 test-suite-sql-eval;`process_sql.py` 是官方 Spider 解析器原版。
- 下载时 jsdelivr 镜像曾返回 Varnish 503 假页面,已用 raw.githubusercontent.com / gh-proxy.com 重下并通过内容校验(见 `.research_tmp/mac_sql/` 实际内容)。

## 1. 结论速览(影响打分的 5 条核心差异)

1. **exec 比较语义完全不同(最大差异)**:MAC-SQL 用测试套件式「denotation 等价」——行序不敏感(仅当 gold 含 ORDER BY 时行序敏感)、列序可任意置换、多重集比较、双方都空结果算对、`DISTINCT` 默认从两侧 SQL 中删除(除非 `--keep_distinct`)。本地 `evaluation.py` 内联的 `eval_exec_match` 按解析出的 select 列逐列对齐、**行序敏感**、DISTINCT 保留、只在单库执行一次。这是最可能造成本地 EX 系统性偏低(+3-5pp 假设的主要来源)。
2. **exec 数据库集合不同**:MAC-SQL 把 db 目录下**所有 .sqlite**(含 test-suite 变体库)都跑一遍,全部等价才判对;本地只跑原库。若项目里只有原始 spider db(没有 test-suite 变体),MAC-SQL 退化为单库评估但语义仍宽松得多。
3. **exact-match 核心逻辑逐行一致**:Evaluator / partial match / rebuild_* / kmaps / `DISABLE_VALUE=True` / `DISABLE_DISTINCT=True` 与本地完全相同,两侧的 set-match(EM) 理论数值一致。差异只在进程级(见 §4)。
4. **process_sql.py 健壮性天差地别**:MAC-SQL 带的是官方原版解析器,遇到非规范 SQL(隐式别名、SELECT AS、IN 多值无空格、CASE/CAST/EXISTS 等)直接 assert 崩溃 → 空 SQL 兜底 → EM 记 0;本地 process_sql.py 打了一整套「伪列 -1 + 语法分支」补丁,能解析成功并参与打分。解析成功率差异会直接影响 EM 绝对值(本地更宽松)。
5. **--etype 默认值**:MAC-SQL 默认 `exec`,本地必填且无默认;两者对 Spider 都支持 `all/exec/match` 三选。另 MAC-SQL 有 `--plug_value`、`--keep_distinct`、`--progress_bar_for_each_datapoint` 三个 exec 专用开关,本地没有。

## 2. 文件与依赖结构对比

| | 本地 tools/original_spider_eval/ | MAC-SQL evaluation/ |
|---|---|---|
| 主入口 | `evaluation.py`(30,733 B;`evaluation_spider.py` 为 14 B 的 404 占位文件,勿用) | `evaluation_spider.py` |
| SQL 解析 | `process_sql.py`(30,981 B,官方版 + 多处 FIX) | `process_sql.py`(16,828 B,官方原版) |
| exec 依赖 | 内联 `eval_exec_match` | 额外需 `exec_eval.py` + `parse.py` |
| 额外 pip 依赖 | 无新增 | `sqlparse`(parse.py)、`tqdm`(exec_eval.py);`nltk` 两侧都要 |

**集成注意**:MAC-SQL 的 `evaluation_spider.py` 里 `from exec_eval import eval_exec_match`、`from process_sql import get_schema, Schema, get_sql`,若引入本地需同时拷入 `exec_eval.py`/`parse.py`(或仅保留 match 模式时删掉 exec 分支)。

## 3. exec 路径详细对比(etype=exec / all)

### 3.1 MAC-SQL(exec_eval.py 语义)
- 对 gold 和 pred 先 `postprocess`(`> =`→`>=` 等),默认 `remove_distinct` 删两侧 DISTINCT。
- `order_matters = 'order by' in g_str.lower()`:仅当 gold 含 ORDER BY 时要求行序一致,否则行序、列序都做置换搜索(bag/multiset 语义,含重复行校验)。
- `replace_cur_year`:`YEAR(CURDATE())` 替换为 2020。
- 遍历 `db_dir` 下所有 `.sqlite` 文件执行双方查询,60s asyncio 超时;gold 执行失败直接 `assert` 崩;pred 失败记该库不过。
- `--plug_value` 时用 gold 的值枚举填入 pred 的 value 槽(`get_all_preds_for_execution`,最多试 50 个)。
- `p_str = p_str.replace("value", "1")`:pred 文本中字面量 `value` 一律被替换成 1(官方遗留行为,列/表名叫 value 的查询会被误伤,本地无此行为)。

### 3.2 本地(evaluation.py 内联 eval_exec_match)
- 单库执行一次,`p_res`/`q_res` 按解析出的 select val_units 逐列对齐:`rmap[key]=[r[idx] for r in res]`,**行序敏感**,列序由 key 对齐(列序不敏感,列身份敏感)。
- 无 DISTINCT 移除、无 value 插值、无 ORDER BY 特判、无超时、无多库。
- `val_unit[1] == -1`(EXISTS 等伪列,与本地 process_sql 的 -1 约定配套)跳过该列比较。

### 3.3 影响面
- 本地在以下情况把正确 SQL 记 0:MACS-SQL 记 1:行序不同(gold 无 ORDER BY 时)、SELECT 列序不同、pred 多/少 DISTINCT(数据有重复时)。
- 空结果处理两侧一致:双方都空 → 记对(MAC-SQL `result_eq` 首行;本地 `res_map` 两侧同为 `{col: []}` 相等);仅一方为空 → 记错。
- MAC-SQL 在有 test-suite 变体库时更严格(必须全部库等价);本地目录没有变体库时实际只有单库。
- 注意 MAC-SQL 的 exec 分在 gold 含 `order by` 时是行序敏感的,与本地一致。

## 4. exact-match 路径对比(etype=match / all)

- Evaluator / eval_sel / eval_where / eval_group / eval_having / eval_order / eval_and_or / eval_IUEN / eval_keywords / get_scores / rebuild_* / build_foreign_key_map 等**与本地逐行一致**;`DISABLE_VALUE`/`DISABLE_DISTINCT` 两侧均硬编码 True。
- 进程级差异:
  - **输入分组**:MAC-SQL 按空行把 gold/pred 拆成 session 组(SParC 式),`assert len(plist)==len(glist)`,Spider 平铺文件等价于本地逐行 zip;额外多 `joint_all` 与 turn 维度输出(Spider 单 session 时不影响数值)。
  - **gold 解析失败**:本地 `try/except` 跳过该题(不计数);MAC-SQL 无保护,直接崩。
  - **pred 解析失败**:两侧都用「空 SQL dict」兜底 → EM=0;本地额外打印 `eval_err_num` 计数。
  - **调试输出**:MAC-SQL 每 session 打印 `len(p)/len(g)`,exact 不匹配时打印 pred/gold 文本(本地也有后者)。
  - **产物**:MAC-SQL 额外导出 `evaluation.json`(逐条 pred/gold/db/exec_result,question 与 gold 字段对 Spider 未填充);本地只打印控制台表格。

## 5. CLI 参数对比

| 参数 | 本地 evaluation.py | MAC-SQL evaluation_spider.py |
|---|---|---|
| `--etype` | 必填,无默认;`assert in all/exec/match` | **默认 `exec`**,`choices=('all','exec','match')` |
| `--table` | 必填,始终 build kmaps | 仅 `etype in ['all','match']` 时必填 |
| `--plug_value` | 无 | store_true,默认 False |
| `--keep_distinct` | 无 | store_true,默认 False(默认删 DISTINCT) |
| `--progress_bar_for_each_datapoint` | 无 | store_true(需 tqdm) |
| `--gold/--pred/--db` | 同 | 同 |

## 6. process_sql.py 对比

- MAC-SQL 版 = 官方原版(567 行):大量 `assert`,`Error col: ...` 即崩;不支持隐式表别名、SELECT AS 别名、CASE/CAST/EXISTS/标量子查询、IN 多值列表、`<>`、无空格 `name=value` 粘连等。
- 本地版(821 行)= 官方版 + 以下 FIX(全部 `# FIX:` 标注):
  1. 未知列/常量字面量/`*` 返回**伪列 id -1**(eval 层配套跳过),替代断言崩溃;
  2. `=` 强制拆 token、`<>` 合并为 `!=`、逗号粘连 token 拆分(解决 nltk 分词问题);
  3. CASE WHEN...END、CAST(col AS type)、EXISTS (SELECT...)、标量子查询、LOWER/ROUND 等标量函数的专用解析分支;
  4. 隐式表别名 `FROM singer s`、`LEFT/RIGHT/FULL/CROSS/INNER/OUTER JOIN` 前缀、派生表 `) AS t`;
  5. `IS [NOT] NULL`、`NOT IN/LIKE/BETWEEN` 前置与后置两种写法、IN 多值列表收集;
  6. 越界保护(截断 SQL 缺右括号/缺 by 等)。
- 兼容性:本地 process_sql.py 的 `get_sql`/`get_schema`/`Schema` 接口与 MAC-SQL evaluation_spider.py 的 import 完全兼容,可 drop-in 替换(解析成功率更高 → EM 单调不减,但会偏离「纯 MAC-SQL 数值」;且 -1 伪列只被本地 evaluation.py 的 exec 分支识别,MAC-SQL exec 分支不看解析结果,无冲突)。

## 7. 对 FINER 复刻的启示

- 若目标是与论文/官方报告数对齐,应直接用 `.research_tmp/mac_sql/` 这套(默认 `--etype exec` 输出 exec 口径;`--etype all` 同时输出 EM)。
- 假设「+3-5pp」主要落在 exec 口径:行序/列序/DISTINCT 三类放宽对本地 3B 模型常见错误(多余 DISTINCT、列序颠倒、行序不同)收益最大。
- 风险与红线:
  - `evaluation_spider.py` 的 `replace("value","1")` 会误伤列名含 `value` 的 pred,复刻时需要知晓;
  - gold 解析/执行失败会让 MAC-SQL 版直接崩(本地版已加固),批量评估时应保留本地版或为 MAC-SQL 版补同样的兜底;
  - 若只有原始 spider db 目录(无 test-suite 变体),MAC-SQL exec 退化为单库但语义不变;
  - 换评估器前应固定 pred 文件,跑一次新旧对照(建议 dev 子集),确认 pp 变化与错误案例分布,再决定是否切换口径。

## 8. 文件清单

- 下载目录:`C:\Users\13389\Desktop\女朋友\reasoning_generator_3b\.research_tmp\mac_sql\`
  - `evaluation_spider.py`、`process_sql.py`、`exec_eval.py`、`parse.py`、`evaluation_bird_ex.py`、`evaluation_bird_ves.py`
- 本地对照(未改动):`C:\Users\13389\Desktop\女朋友\reasoning_generator_3b\tools\original_spider_eval\evaluation.py`、`process_sql.py`
