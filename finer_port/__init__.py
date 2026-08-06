"""
finer_port — FINER-SQL 复刻 P1：vav 投票评估移植包。

本包实现 FINER_REPLICATION_PLAN.md 的 P1 阶段（vav 投票评估移植）：
  - vav_voting.py   执行分组投票逻辑（移植自 finer-sql/evaluation/majority_voting.py，本地化）
  - sampler.py      n 候选采样器（HF transformers 路径，单次 forward 产 n 条）
  - eval_vav.py     评估入口（n 采样 → 执行验证 → vav 投票 → 输出 SQL + items.json）

约定：
  - 不修改项目现有 src/ 下的任何文件；所有新代码在本包内。
  - 复用项目 src/spider_utils.py 的 SpiderLoader / DatabaseExecutor /
    compare_execution_results / normalize_sql 与 checkpoint 协议。
  - 模块导入时自动把项目 src/ 加入 sys.path（本包与项目源码解耦部署，
    不要求 cwd 是 src/）。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (str(_SRC_DIR), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__version__ = "0.1.0"
__all__ = ["vav_voting", "sampler", "eval_vav"]
