#!/usr/bin/env python3
"""
scripts/test_adjudicate_order_aware.py — adjudicate_order_aware 合成 fixture 自测。

构造 2 实例 sqlite 库 + 5 道题，覆盖：
  Q0 order_matters 固定题：3 个行序错误候选（ORDER BY id DESC，行集相同）+ 4 个
     正确候选（ORDER BY id）→ 包语义合并成一组（size 7，胜者 min_idx 落到错误
     候选）基线错；顺序敏感签名分裂后正确组 size 4 胜出 → fixed
  Q1 order_matters 破损题：3 正确 + 4 行序错误 → 包语义合并组胜者=正确（基线
     对）；分裂后错误组 size 4 压正确组 size 3 → broken
  Q2 order_matters 分裂但胜者不变：4 正确 + 1 行序错误 → 两臂同选正确（same_right）
  Q3 无 'order by'（order_matters=False）：包语义与顺序敏感签名逐题一致 → 不分裂
  Q4 order_matters 但全 ERROR 候选 → 两臂 NO_RESULTS → fallback_maj
另跑 adjudicate_pool.main 同 fixture，验证基线臂 predicted_sql/is_correct 逐题
一致（复用函数产出与源裁决器完全同判）。

用法: python scripts/test_adjudicate_order_aware.py
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import adjudicate_pool  # noqa: E402
import adjudicate_order_aware as AOA  # noqa: E402

FAILS = []


def check(cond, msg):
    if cond:
        print(f"PASS  {msg}")
    else:
        FAILS.append(msg)
        print(f"FAIL  {msg}")


def cand(model, sql, sidx):
    return {"model": model, "sql": sql, "parse_success": True, "sample_idx": sidx}


def build_db(spider_dir: Path):
    db_dir = spider_dir / "database" / "test_db"
    db_dir.mkdir(parents=True)
    specs = {
        # 插入序 ≠ id 序：无 ORDER BY 的结果行序与 gold 不同（order_matters 判错）
        "test_db.sqlite": [(2, "b", 10), (1, "a", 20), (3, "c", 30)],
        "test_db1.sqlite": [(2, "x", 10), (1, "y", 20), (3, "z", 30)],
    }
    for fname, rows in specs.items():
        con = sqlite3.connect(str(db_dir / fname))
        con.executescript(
            "CREATE TABLE t1(id INTEGER, name TEXT, val INTEGER);"
        )
        con.executemany("INSERT INTO t1 VALUES (?,?,?)", rows)
        con.commit()
        con.close()


GOLD_ORDERED = "SELECT name FROM t1 ORDER BY id"
GOLD_BAG = "SELECT name FROM t1"
SQL_ASC = "select name from t1 order by id"          # 行序 [a,b,c] / [x,y,z]
SQL_DESC = "select name from t1 order by id desc"    # 行序 [c,b,a] / [z,y,x]，行集同
SQL_BAG_CORRECT = "select name from t1"              # 返回插入序，行集 {a,b,c}
SQL_BAG_WRONG = "select name from t1 where val >= 20"  # 行集 {a,c}，与 {a,b,c} 不同袋
SQL_ERROR = "select * from missing_table"


def build_items() -> list:
    items = []
    # Q0: 3 DESC 错序(idx 0-2) + 4 ASC 正确(idx 3-6) → 基线并组选错，order-aware 修正
    items.append({
        "dataset_index": 0, "di": 0, "db_id": "test_db", "question": "q0",
        "gold_sql": GOLD_ORDERED,
        "candidates": (
            [cand("sft_v2", SQL_DESC, i) for i in range(3)]
            + [cand("sft_phase1", SQL_ASC, i) for i in range(3, 6)]
            + [cand("sft_v2", SQL_ASC, 6)]
        ),
    })
    # Q1: 3 ASC 正确(idx 0-2) + 4 DESC 错序(idx 3-6) → 基线并组选对，order-aware 分裂后选错
    items.append({
        "dataset_index": 1, "di": 1, "db_id": "test_db", "question": "q1",
        "gold_sql": GOLD_ORDERED,
        "candidates": (
            [cand("sft_phase1", SQL_ASC, i) for i in range(3)]
            + [cand("sft_v2", SQL_DESC, i) for i in range(3, 7)]
        ),
    })
    # Q2: 4 ASC(idx 0-3) + 1 DESC(idx 4) → 分裂但两臂同选正确
    items.append({
        "dataset_index": 2, "di": 2, "db_id": "test_db", "question": "q2",
        "gold_sql": GOLD_ORDERED,
        "candidates": (
            [cand("sft_phase1", SQL_ASC, i) for i in range(4)]
            + [cand("sft_v2", SQL_DESC, 4)]
        ),
    })
    # Q3: gold 无 'order by' → order_matters=False，两臂逐题同判
    items.append({
        "dataset_index": 3, "di": 3, "db_id": "test_db", "question": "q3",
        "gold_sql": GOLD_BAG,
        "candidates": (
            [cand("sft_phase1", SQL_BAG_CORRECT, i) for i in range(3)]
            + [cand("sft_v2", SQL_BAG_WRONG, i) for i in range(3, 5)]
        ),
    })
    # Q4: order_matters 但全 ERROR → NO_RESULTS → fallback_maj（两臂同）
    items.append({
        "dataset_index": 4, "di": 4, "db_id": "test_db", "question": "q4",
        "gold_sql": GOLD_ORDERED,
        "candidates": (
            [cand("sft_phase1", SQL_ERROR, 0), cand("sft_v2", SQL_ERROR, 1)]
        ),
    })
    return items


def load_items(path: Path, arm: str) -> list:
    return json.loads((path / f"items_{arm}.json").read_text(encoding="utf-8"))


def main():
    # ---- 单元级：保序签名 vs 包语义签名 ----
    r1 = AOA.rows_to_group_signature_ordered([["a"], ["b"]])
    r2 = AOA.rows_to_group_signature_ordered([["b"], ["a"]])
    check(r1 != r2, "保序签名区分行序（[a,b] != [b,a]）")
    check(adjudicate_pool.rows_to_group_signature([["a"], ["b"]])
          == adjudicate_pool.rows_to_group_signature([["b"], ["a"]]),
          "包语义签名不区分行序（对照组）")
    check(AOA.gold_order_matters("SELECT name FROM t1 ORDER BY id", False)
          and not AOA.gold_order_matters("SELECT name FROM t1", False),
          "gold_order_matters 判定正确")

    with tempfile.TemporaryDirectory(prefix="adj_ordaw_test_") as td:
        tmp = Path(td)
        spider_dir = tmp / "spider_data"
        build_db(spider_dir)
        items_path = tmp / "items.json"
        items_path.write_text(json.dumps(build_items()), encoding="utf-8")

        out1 = tmp / "out_ordaw"
        rc = AOA.main([
            "--items", str(items_path), "--out-dir", str(out1),
            "--spider-dir", str(spider_dir), "--threads", "4"])
        check(rc == 0, "adjudicate_order_aware.main 返回 0")
        if rc != 0:
            sys.exit(1)

        base = load_items(out1, "arm_vav_multi_all")
        ordaw = load_items(out1, "arm_orderaware")
        summary = json.loads((out1 / "summary.json").read_text(encoding="utf-8"))
        splits = json.loads((out1 / "split_questions.json").read_text(encoding="utf-8"))

        check(len(base) == 5 and len(ordaw) == 5, "items 两臂各 5 行")
        check(all(r["predicted_sql"] for r in base + ordaw), "predicted 非空")

        # ---- Q0：分裂 → 基线并组选错，order-aware 修正（fixed）----
        check(base[0]["order_matters"] and base[0]["split_question"]
              and base[0]["is_correct"] is False,
              "Q0 基线并组选中行序错误候选（隐性彩票）判错")
        check(base[0]["winner_group_size"] == 7,
              f"Q0 基线包语义并组 size 7（实际 {base[0]['winner_group_size']}）")
        check(ordaw[0]["is_correct"] and ordaw[0]["winner_group_size"] == 4
              and ordaw[0]["winner_source"] == "vav_ordered"
              and ordaw[0]["winner_changed_vs_baseline"] is True,
              "Q0 order-aware 分裂后正确组 size 4 胜出（fixed）")

        # ---- Q1：分裂 → 基线并组选对，order-aware 选错（broken）----
        check(base[1]["is_correct"] and base[1]["winner_group_size"] == 7,
              "Q1 基线并组胜者=正确候选")
        check(ordaw[1]["is_correct"] is False and ordaw[1]["winner_group_size"] == 4,
              "Q1 order-aware 分裂后错误组 size 4 胜出（broken）")

        # ---- Q2：分裂但胜者不变 ----
        check(base[2]["split_question"] and ordaw[2]["split_question"]
              and ordaw[2]["winner_changed_vs_baseline"] is False,
              "Q2 分裂但两臂同胜者")
        check(base[2]["is_correct"] and ordaw[2]["is_correct"], "Q2 两臂同判对")

        # ---- Q3：非 order_matters → 不分裂，两臂同判 ----
        check(base[3]["order_matters"] is False and base[3]["split_question"] is False
              and ordaw[3]["split_question"] is False,
              "Q3 order_matters=False 不分裂")
        check(base[3]["predicted_sql"] == ordaw[3]["predicted_sql"]
              and base[3]["is_correct"] and ordaw[3]["is_correct"],
              "Q3 两臂 predicted/is_correct 一致")

        # ---- Q4：全 ERROR → fallback_maj ----
        check(base[4]["winner_source"] == "fallback_maj"
              and ordaw[4]["winner_source"] == "fallback_maj"
              and base[4]["is_correct"] is False and ordaw[4]["is_correct"] is False,
              "Q4 两臂 fallback_maj 判错")

        # ---- summary 统计（关键统计 ①②）----
        sa = summary["split_analysis"]
        check(sa["order_matters_questions"] == 4,
              f"order_matters 题数 4（实际 {sa['order_matters_questions']}）")
        check(sa["split_questions"] == 3 and sa["split_where_winner_changed"] == 2,
              f"分裂题 3、胜者变更 2（实际 split={sa['split_questions']} "
              f"changed={sa['split_where_winner_changed']}）")
        check(sa["migration_on_split"] == {"fixed": 1, "broken": 1,
                                           "same_right": 1, "same_wrong": 0},
              f"分裂题迁移 fixed/broken/same（实际 {sa['migration_on_split']}）")
        check(sa["baseline_on_split"]["n"] == 3
              and sa["baseline_on_split"]["correct"] == 2
              and sa["orderaware_on_split"]["correct"] == 2,
              "分裂题两臂各 2/3 对")
        check(sa["non_split_arms_disagree"] == 0, "非分裂题两臂无分歧（一致性自检）")

        vb = summary["vs_baseline"]
        check(vb["fixed"] == 1 and vb["broken"] == 1 and vb["net"] == 0,
              f"全量 vs_baseline fixed=1 broken=1（实际 {vb}）")
        check(summary["accuracy"]["arm_vav_multi_all"]["accuracy"] == 0.6
              and summary["accuracy"]["arm_vav_multi_all_orderaware"]["accuracy"] == 0.6,
              "两臂准确率 0.6（Q0/Q1 迁移抵消，Q2/Q3 对 Q4 错）")

        # ---- 分裂题明细 ----
        check(len(splits) == 3, "split_questions.json 3 条")
        q0 = splits[0]
        check(q0["migration"] == "fixed" and len(q0["splits"]) == 1,
              "Q0 明细 migration=fixed 且 1 个包语义组被分裂")
        check(q0["splits"][0]["bag_group_size"] == 7
              and q0["splits"][0]["bag_group_is_baseline_winner"] is True
              and sorted(p["order_group_size"] for p in q0["splits"][0]["pieces"]) == [3, 4],
              f"Q0 明细：size7 包语义组裂成 3+4（实际 {q0['splits'][0]}）")
        check(q0["orderaware_winner"]["group_id"] is not None
              and q0["baseline_winner"]["group_id"] is not None,
              "Q0 明细胜者组 id 已标注")

        # ---- 基线一致性：与 adjudicate_pool 同 fixture 逐题同判 ----
        out2 = tmp / "out_ap"
        rc2 = adjudicate_pool.main([
            "--items", str(items_path), "--out-dir", str(out2),
            "--spider-dir", str(spider_dir), "--threads", "4"])
        check(rc2 == 0, "adjudicate_pool.main 返回 0")
        theirs = json.loads(
            (out2 / "items_arm_vav_multi_all_both.json").read_text(encoding="utf-8"))
        same = all(a["predicted_sql"] == b["predicted_sql"]
                   and a["is_correct"] == b["is_correct"] for a, b in zip(base, theirs))
        check(same and len(theirs) == 5,
              "基线臂 predicted_sql/is_correct 与 adjudicate_pool 逐题一致")

    if FAILS:
        print(f"\n{len(FAILS)} 个断言失败")
        for m in FAILS:
            print("  -", m)
        sys.exit(1)
    print("\n全部通过")


if __name__ == "__main__":
    main()
