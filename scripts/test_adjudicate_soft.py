#!/usr/bin/env python3
"""
scripts/test_adjudicate_soft.py — adjudicate_soft 合成 fixture 自测。

构造 2 实例 sqlite 库 + 5 道题，覆盖：
  Q0 双模型大正确组（size 5, dual）压单模型错误组 → 全部臂正确
  Q1 单模型大错误组（size 3）压双模型小正确组（size 2）→ 基线错；ladder /
     cross_w2 / cross_w3 / gated 修正（cross_w1 评分打平走 str(key)）
  Q2 门控 JOIN 平票：两个 dual 组同 size，JOIN 少者胜（1-join 正确 vs 2-join 错）
  Q3 纯单模型题（winner_dual=False），门控触发但二次裁决不改胜者
  Q4 全 ERROR 候选 → NO_RESULTS → fallback_maj
另跑 adjudicate_pool.main 同 fixture，验证基线臂 predicted_sql 逐题一致
（复用函数产出与源裁决器完全同判）。

用法: python scripts/test_adjudicate_soft.py
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import adjudicate_pool  # noqa: E402
import adjudicate_soft  # noqa: E402

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
            "CREATE TABLE t2(id INTEGER, tid INTEGER, tag TEXT);"
        )
        con.executemany("INSERT INTO t1 VALUES (?,?,?)", rows)
        con.executemany("INSERT INTO t2 VALUES (?,?,?)", [(1, 1, "p"), (2, 2, "q")])
        con.commit()
        con.close()


GOLD_Q01 = "SELECT name FROM t1 ORDER BY id"
GOLD_Q2 = "SELECT a.name FROM t1 a JOIN t1 b ON a.id = b.id ORDER BY a.id"
SQL_CORRECT = "select name from t1 order by id"
# 注意：分组签名是包语义（行排序、去行序），order-only 错误 SQL 会与正确 SQL 同组，
# 故错误候选必须用「行集(bag)不同」的 SQL，不能用乱序 SQL 构造
SQL_WRONG_A = "select name from t1 where val >= 20"   # → {b,c}，与 {a,b,c} 不同袋
SQL_WRONG_ROWS = "select name from t1 where val >= 30"  # → {c}
SQL_ERROR = "select * from missing_table"
SQL_G_1JOIN = "SELECT a.name FROM t1 a JOIN t1 b ON a.id = b.id ORDER BY a.id"
SQL_H_2JOIN = ("SELECT a.name FROM t1 a JOIN t1 b ON a.id = b.id "
               "JOIN t1 c ON b.id = c.id WHERE c.val >= 20")  # → {b,c}，非空且错


def build_items() -> list:
    items = []
    # Q0: A dual 5 正确 / B single 3 错 / C single 2 错 / 1 个 ERROR 候选
    items.append({
        "dataset_index": 0, "di": 0, "db_id": "test_db", "question": "q0",
        "gold_sql": GOLD_Q01,
        "candidates": (
            [cand("sft_phase1", SQL_CORRECT, i) for i in range(3)]
            + [cand("sft_v2", SQL_CORRECT, i) for i in range(3, 5)]
            + [cand("sft_v2", SQL_WRONG_A, i) for i in range(5, 8)]
            + [cand("sft_phase1", SQL_WRONG_ROWS, i) for i in range(8, 10)]
            + [cand("sft_phase1", SQL_ERROR, 10)]
        ),
    })
    # Q1: D single 3 错 / E dual 2 正确（基线被大错误组压掉，软排序应修正）
    items.append({
        "dataset_index": 1, "di": 1, "db_id": "test_db", "question": "q1",
        "gold_sql": GOLD_Q01,
        "candidates": (
            [cand("sft_v2", SQL_WRONG_A, i) for i in range(3)]
            + [cand("sft_phase1", SQL_CORRECT, 3), cand("sft_v2", SQL_CORRECT, 4)]
        ),
    })
    # Q2: G dual 2 正确 1-join / H dual 2 错 2-join（门控二次裁决 → JOIN 少者胜）
    items.append({
        "dataset_index": 2, "di": 2, "db_id": "test_db", "question": "q2",
        "gold_sql": GOLD_Q2,
        "candidates": (
            [cand("sft_phase1", SQL_G_1JOIN, 0), cand("sft_v2", SQL_G_1JOIN, 1),
             cand("sft_phase1", SQL_H_2JOIN, 2), cand("sft_v2", SQL_H_2JOIN, 3)]
        ),
    })
    # Q3: 纯单模型（winner_dual=False），门控触发但二次裁决不改胜者
    items.append({
        "dataset_index": 3, "di": 3, "db_id": "test_db", "question": "q3",
        "gold_sql": GOLD_Q01,
        "candidates": (
            [cand("sft_phase1", SQL_CORRECT, i) for i in range(2)]
            + [cand("sft_phase1", SQL_WRONG_A, 2)]
        ),
    })
    # Q4: 全 ERROR → NO_RESULTS → fallback_maj
    items.append({
        "dataset_index": 4, "di": 4, "db_id": "test_db", "question": "q4",
        "gold_sql": GOLD_Q01,
        "candidates": (
            [cand("sft_phase1", SQL_ERROR, 0), cand("sft_v2", SQL_ERROR, 1)]
        ),
    })
    return items


def load_items(path: Path, arm: str) -> list:
    return json.loads((path / f"items_{arm}.json").read_text(encoding="utf-8"))


def main():
    with tempfile.TemporaryDirectory(prefix="adj_soft_test_") as td:
        tmp = Path(td)
        spider_dir = tmp / "spider_data"
        build_db(spider_dir)
        items_path = tmp / "items.json"
        items_path.write_text(json.dumps(build_items()), encoding="utf-8")

        out1 = tmp / "out_soft"
        rc = adjudicate_soft.main([
            "--items", str(items_path), "--out-dir", str(out1),
            "--spider-dir", str(spider_dir), "--threads", "4"])
        check(rc == 0, "adjudicate_soft.main 返回 0")
        if rc != 0:
            sys.exit(1)

        base = load_items(out1, "arm_vav_multi_all")
        ladder = load_items(out1, "arm_soft_ladder")
        cross1 = load_items(out1, "arm_soft_cross_w1")
        cross2 = load_items(out1, "arm_soft_cross_w2")
        cross3 = load_items(out1, "arm_soft_cross_w3")
        gated = load_items(out1, "arm_gated_structural")
        summary = json.loads((out1 / "summary.json").read_text(encoding="utf-8"))

        for arm, rows in [("baseline", base), ("ladder", ladder), ("cross_w1", cross1),
                          ("cross_w2", cross2), ("cross_w3", cross3), ("gated", gated)]:
            check(len(rows) == 5, f"items_{arm} 5 行")
            check(all(r["predicted_sql"] for r in rows), f"items_{arm} predicted 非空")

        # ---- Q0：大正确双模型组 ----
        check(base[0]["is_correct"] and base[0]["winner_group_size"] == 5
              and base[0]["winner_dual"] and base[0]["winner_source"] == "vav",
              "Q0 基线选 size5 双模型正确组")
        check(ladder[0]["is_correct"] and cross2[0]["is_correct"]
              and cross3[0]["is_correct"] and gated[0]["is_correct"],
              "Q0 全部新臂正确")
        check(gated[0]["gated_triggered"] is False, "Q0 门控不触发（差 2 且 top1>=2）")

        # ---- Q1：单模型大错误组压双模型小正确组 ----
        check(base[1]["is_correct"] is False and base[1]["winner_group_size"] == 3
              and base[1]["winner_dual"] is False, "Q1 基线被 size3 单模型错误组压掉")
        check(ladder[1]["is_correct"], "Q1 ladder 双模型组优先修正")
        check(cross1[1]["winner_group_size"] in (2, 3),
              f"Q1 cross_w1 评分打平走 str(key)（实际 size={cross1[1]['winner_group_size']}）")
        check(cross2[1]["is_correct"] and cross3[1]["is_correct"],
              "Q1 cross_w2/w3 修正")
        check(gated[1]["is_correct"] and gated[1]["gated_triggered"] is True
              and gated[1]["top1_size"] == 3 and gated[1]["top2_size"] == 2
              and gated[1]["winner_source"] == "gated",
              "Q1 门控触发且二次裁决修正")

        # ---- Q2：门控 JOIN 平票 ----
        check(gated[2]["gated_triggered"] is True and gated[2]["winner_dual"],
              "Q2 门控触发（size 2-2 平票）")
        check(" ".join(gated[2]["predicted_sql"].lower().split())
              == " ".join(SQL_G_1JOIN.lower().split()), "Q2 二次裁决选 1-join SQL")
        check(gated[2]["n_joins"] == 1 and gated[2]["is_correct"], "Q2 门控胜者正确")
        check(gated[2]["join_counter"] in ("sqlglot", "regex"),
              f"Q2 join_counter 合法（实际 {gated[2]['join_counter']}）")
        n1, m1 = adjudicate_soft.count_joins(SQL_G_1JOIN)
        n2, m2 = adjudicate_soft.count_joins(SQL_H_2JOIN)
        check(n1 == 1 and n2 == 2, f"count_joins 计数正确（{m1}:{n1}, {m2}:{n2}）")

        # ---- Q3：纯单模型 ----
        check(base[3]["is_correct"] and base[3]["winner_dual"] is False,
              "Q3 基线正确且 winner_dual=False")
        check(gated[3]["gated_triggered"] is True and gated[3]["gated_changed_winner"] is False
              and gated[3]["is_correct"], "Q3 门控触发但二次裁决不改胜者")

        # ---- Q4：全 ERROR → fallback_maj ----
        for arm, rows in [("baseline", base), ("ladder", ladder), ("gated", gated)]:
            check(rows[4]["winner_source"] == "fallback_maj"
                  and rows[4]["is_correct"] is False, f"Q4 {arm} fallback_maj 判错")
        check(gated[4]["gated_triggered"] is False, "Q4 门控不触发（无组）")

        # ---- summary 统计 ----
        cells = summary["accuracy"]
        check(cells["arm_vav_multi_all"]["accuracy"] in (0.4, 0.6),
              f"基线准确率 ∈ {{0.4, 0.6}}（Q2 平票走 str(key)，实际 "
              f"{cells['arm_vav_multi_all']['accuracy']}）")
        check(cells["arm_gated_structural"]["accuracy"] == 0.8,
              f"gated 准确率 0.8（实际 {cells['arm_gated_structural']['accuracy']}）")
        vb = summary["vs_baseline"]
        check(vb["arm_soft_ladder"]["fixed"] >= 1, "ladder fixed >= 1（Q1）")
        check(vb["arm_gated_structural"]["fixed"] in (1, 2)
              and vb["arm_gated_structural"]["broken"] == 0, "gated fixed 且无 broken")
        ga = summary["gated_analysis"]
        check(ga["triggered_questions"] == 3,
              f"门控触发题数 == 3（实际 {ga['triggered_questions']}）")
        check(ga["baseline_on_triggered"]["n"] == 3, "触发子集基线样本数 3")
        cvm = summary["cross_model_validation"]["group_level"]
        check(cvm["n_groups"] == 9 and cvm["n_dual"] == 4 and cvm["n_single"] == 5,
              f"组数 9（dual 4 / single 5）（实际 {cvm['n_groups']}）")
        check(cvm["dual"]["accuracy"] == 0.75 and cvm["single"]["accuracy"] == 0.2,
              f"双模型组正确率 0.75 > 单模型组 0.2（实际 dual={cvm['dual']['accuracy']} "
              f"single={cvm['single']['accuracy']}）")
        wl = summary["cross_model_validation"]["winner_level"]["arm_vav_multi_all"]
        check(wl["single"]["n"] == 2 and wl["single"]["correct"] == 1,
              "winner_level 基线 single 组统计正确")
        check(wl["dual"]["n"] == 2 and wl["dual"]["correct"] in (1, 2),
              "winner_level 基线 dual 组样本数正确")
        check(summary["execution_stats"]["group_reps_phase"] is not None,
              "group_reps 阶段已执行")

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
