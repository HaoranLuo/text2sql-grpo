#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/repair_chain_expo.py — P2A：EXPO 错误子句定位注入修复链（方案 + 冒烟准备）。

不修改 src/repair_chain.py（早期实现保持原样）；本脚本为独立入口，把
repair_chain 的"错误提示"从「失败 SQL + 首个报错实例名 + 错误文本」升级为
「+ EXPO 定位诊断（根因子句 + 报错点 + 修复建议，英文提示段 ≤260 字符）」。

预注册（PREREG，2026-08-23 封盘，写入方案并随本文件固化）：
  - 主判据：修复候选并入候选池（outputs/eval_pool_bird/items.json，模型标签
    "repair_expo"）后重跑 bird_select 判卷（prep → score → final），BIRD dev
    官方 EX（FINER evaluation_bird_ex.py 口径）相对「未并入修复候选」基线
    提升 ≥ +0.5pp 才算正向（判成功）；
    < +0.5pp → 判「组件有效但不足以上线」，转入组合消融。
  - 基线：同池同 arm（arm_vav / arm_orm_grouphead）在实施日以官方跑分封盘
    基线（items.json + bird_select 现口径）。
  - 次级判据：两 arm EX 均不下降；修复候选可执行率（全实例无报错）不低于
    repair_chain 基线口径；prompt 注入不使解析率显著下降（以 parse_success
    率量化）。
  - 纪律：修复候选只进判卷池，不进训练数据（dev 泄漏红线）。

方案（诊断注入口径）：
  - 定位器：src/expo_localize.py（EXPO 论文 Appendix A.3 Table 14 错误分类 +
    沿逻辑执行顺序回溯根因 + schema 探测辅助；离线诊断见
    outputs/expo_diag/localize_stats.json）。
  - 注入位置：build_repair_prompt 修复段「error message」之后插入
    expo_localize.render_repair_hint() 输出（空定位 → 与 repair_chain 现
    prompt 逐字等价，无损降级）。
  - canonical 部分 / 截断预算 / SQL 解析（VavSampler.extract_sql）/ 接受条件
    （全实例执行成功）全部与 repair_chain 同源同口径；唯一变量 = 定位提示。

四阶段（GPU 校准由主控排期；本脚本 CPU 部分现在可用）：
  Stage A（CPU，现在可跑）--stage scan  ：对执行失败候选批量定位 + 构建
            双变体修复 prompt（with/without 定位），落盘供 GPU/人工冒烟；
  Stage B（GPU，排队中）--stage generate：sft_v2 vLLM 贪心 T=0 生成修复 SQL
            （结构与 repair_chain Phase B/C 同源；本脚本不主动抢 GPU）；
  Stage C（CPU，现在可跑）--stage merge  ：把 Stage B 产出的修复候选并入
            候选池 → outputs/expo_diag/repair_expo_merged_items.json；
  Stage D（CPU，现在可跑）--stage prereg：打印预注册判定口径与执行命令
            （bird_select prep/score/final 重跑 + EX 对比）。

用法（HPC, CPU）：
  envs/reasoning3b/bin/python src/repair_chain_expo.py --stage scan \
      --items outputs/eval_pool_bird/items.json \
      --db-root data/bird/bird_dev/dev_20240627/dev_databases \
      --out-dir outputs/expo_diag
  envs/reasoning3b/bin/python src/repair_chain_expo.py --stage merge \
      --items outputs/eval_pool_bird/items.json \
      --repairs outputs/expo_diag/repairs_generated.jsonl \
      --out-dir outputs/expo_diag
  envs/reasoning3b/bin/python src/repair_chain_expo.py --stage prereg \
      --items outputs/eval_pool_bird/items.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(PROJECT_ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import adjudicate_pool as AP  # noqa: E402  执行引擎（纯 CPU，口径一致）
import expo_localize as EL  # noqa: E402  EXPO 错误子句定位

DEFAULT_ITEMS = PROJECT_ROOT / "outputs" / "eval_pool_bird" / "items.json"
DEFAULT_DB_ROOT = (PROJECT_ROOT / "data" / "bird" / "bird_dev" / "dev_20240627"
                   / "dev_databases")
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "expo_diag"
DEFAULT_EXEC_CACHE = DEFAULT_OUT_DIR / "exec_cache.jsonl"

REPAIR_MODEL_LABEL = "repair_expo"

# ---- 预注册常量（封盘：2026-08-23）----
PREREG_EX_GAIN_PP = 0.5
PREREG = {
    "frozen_at": "2026-08-23",
    "primary_metric": "BIRD dev 官方 EX（FINER evaluation_bird_ex.py，"
                      "bird_select --phase final 口径，dev 1534）",
    "baseline": "outputs/eval_pool_bird/items.json + bird_select 现口径 "
                "arm_vav / arm_orm_grouphead 官方 EX（实施日封盘重跑）",
    "treatment": "对执行失败候选生成修复 SQL（repair_chain_expo，注入 EXPO "
                 "定位诊断，sft_v2 贪心 T=0 n=1）→ 以模型标签 repair_expo 并入"
                 "候选池 → 重跑 bird_select 判卷",
    "success_criterion": f"并入池后任一 arm 官方 EX 增益 ≥ +{PREREG_EX_GAIN_PP}pp "
                         "（相对封盘基线，arm 级取 max）；< +0.5pp 判"
                         "「组件有效但不足以上线」",
    "secondary": ["两 arm EX 均不下降",
                  "修复候选全实例可执行率不低于 repair_chain 基线口径",
                  "修复生成 parse_success 率不显著下降"],
    "discipline": "修复候选只进判卷池，不进训练数据（dev 泄漏红线）；"
                  "超参改动 = 新实验 = 另行预注册",
}


def read_ddl(db_root: Path, db_id: str) -> str:
    """BIRD schema DDL（sqlite_master CREATE TABLE 按表名排序拼接；
    与 bird_select.read_ddl 同款，独立复制避免拖入其 import 链）。"""
    import sqlite3
    db_path = Path(db_root) / db_id / f"{db_id}.sqlite"
    if not db_path.is_file():
        return ""
    con = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY name").fetchall()
    finally:
        con.close()
    return "\n".join(r[0] for r in rows)


def build_repair_prompt(question: str, ddl: str, failed_sql: str,
                        instance_name: str, error_text: str,
                        loc: Optional[Dict[str, Any]] = None,
                        sql_cap: int = 1200, err_cap: int = 500,
                        hint_cap: int = 260) -> str:
    """canonical prompt + 修复段；loc 非空时注入 EXPO 定位提示。

    canonical 部分与 repair_chain 同源（ReasoningGeneratorAgent.build_prompt，
    dialect=sqlite，无 evidence/schema_links）；截断口径相同。唯一差异 = 定位
    提示段（--skip-localize / loc=None 时与 repair_chain prompt 逐字等价）。
    """
    from reasoning_generator_agent import ReasoningGeneratorAgent  # 延迟导入（torch 依赖）
    canonical = ReasoningGeneratorAgent.build_prompt(
        question=question, ddl_schema=ddl, schema_links=None, evidence=None,
        dialect="sqlite")
    fs = (failed_sql or "").strip()
    if len(fs) > sql_cap:
        fs = fs[:sql_cap].rstrip() + "\n... [truncated]"
    er = (error_text or "").strip()
    if len(er) > err_cap:
        er = er[:err_cap] + " ... [truncated]"

    hint = EL.render_repair_hint(loc, max_chars=hint_cap) if loc else ""
    loc_block = f"\n\n{hint}" if hint else ""

    repair_section = (
        "\n\n=== Repair Request (execution feedback) ===\n"
        f'The SQL written for the question above raised an execution error on '
        f'database instance "{instance_name}".\n'
        "The failing SQL:\n```sql\n" + fs + "\n```\n"
        "The error message:\n" + er + "\n"
        + loc_block +
        "\n\nPlease rewrite the SQL so that it (1) executes successfully on EVERY "
        "database instance of this database and (2) keeps the same meaning with "
        "respect to the original question (do not change what the query asks for; "
        "only fix the execution error). Output the corrected SQL inside a "
        "```sql code block."
    )
    return canonical + repair_section


def _load_items(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        data = data["items"]
    if not isinstance(data, list):
        raise ValueError(f"items 结构异常: {path}")
    return data


def _load_exec_cache(cache_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    cache[(rec["sql"], rec["db"])] = rec["outcome"]
                except Exception:
                    continue
    return cache


# ---------------------------------------------------------------------------
# Stage A：扫描失败候选 + 构建双变体修复 prompt（纯 CPU）
# ---------------------------------------------------------------------------

def stage_scan(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    items = _load_items(Path(args.items))
    if args.limit:
        items = items[: args.limit]

    cache = _load_exec_cache(Path(args.exec_cache))
    print(f"[repair_expo:scan] exec cache: {len(cache)} entries", file=sys.stderr)

    engine = AP.ExecutionEngine(args.threads, args.query_timeout,
                                args.max_vm_steps, args.row_cap)
    tasks: Dict[Tuple[str, str], None] = {}
    for it in items:
        db = Path(args.db_root) / it.get("db_id", "") / f"{it.get('db_id', '')}.sqlite"
        if not Path(db).is_file():
            continue
        for c in it.get("candidates") or []:
            sql = (c.get("sql") or "").strip()
            if sql:
                tasks[(sql, str(db))] = None
    todo = [(s, d) for (s, d) in tasks if (s, d) not in cache]
    print(f"[repair_expo:scan] unique (sql,db): {len(tasks)}; to execute: {len(todo)}",
          file=sys.stderr)
    if todo:
        engine.run(todo, phase="repair_expo_scan")
        with open(Path(args.exec_cache), "a", encoding="utf-8") as fh:
            for (s, d) in todo:
                fh.write(json.dumps({"sql": s, "db": d,
                                     "outcome": engine.get(s, d)},
                                    ensure_ascii=False) + "\n")

    ddl_cache: Dict[str, str] = {}
    rows: List[Dict[str, Any]] = []
    n_fail = n_localized = 0
    loc_clause: Counter = Counter()
    for it in items:
        db_id = it.get("db_id", "")
        db = Path(args.db_root) / db_id / f"{db_id}.sqlite"
        if not Path(db).is_file():
            continue
        if db_id not in ddl_cache:
            ddl_cache[db_id] = read_ddl(Path(args.db_root), db_id)
        seen = set()
        for c in it.get("candidates") or []:
            sql = (c.get("sql") or "").strip()
            key = AP.normalize_for_dedup(sql)
            if not sql or key in seen:
                continue
            seen.add(key)
            outcome = cache.get((sql, str(db))) or engine.get(sql, str(db))
            if outcome["ok"]:
                continue
            n_fail += 1
            loc = EL.localize(sql, str(db), outcome.get("error"),
                              outcome.get("error_type"))
            if loc.get("ok"):
                n_localized += 1
                for cl in loc["err_clauses"]:
                    loc_clause[cl] += 1
            p_with = build_repair_prompt(it.get("question", ""),
                                         ddl_cache[db_id], sql,
                                         f"{db_id}.sqlite",
                                         outcome.get("error") or "",
                                         loc)
            p_without = build_repair_prompt(it.get("question", ""),
                                            ddl_cache[db_id], sql,
                                            f"{db_id}.sqlite",
                                            outcome.get("error") or "",
                                            None)
            rows.append({
                "dataset_index": it.get("dataset_index", it.get("di")),
                "db_id": db_id,
                "question": (it.get("question") or "")[:200],
                "model": c.get("model"),
                "sql": sql,
                "error": outcome.get("error"),
                "localized": loc.get("ok"),
                "err_clauses": loc.get("err_clauses"),
                "root_cause": loc.get("root_cause"),
                "prompt_with_localization": p_with,
                "prompt_without_localization": p_without,
            })

    with open(out_dir / "repair_prompts_dryrun.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_prompts = len(rows)
    avg_len_with = (sum(len(r["prompt_with_localization"]) for r in rows)
                    / max(n_prompts, 1))
    avg_len_without = (sum(len(r["prompt_without_localization"]) for r in rows)
                       / max(n_prompts, 1))
    summary = {
        "stage": "scan",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items_file": str(Path(args.items).resolve()),
        "db_root": str(Path(args.db_root).resolve()),
        "n_items": len(items),
        "n_fail_candidates_unique": n_fail,
        "n_localized": n_localized,
        "localization_rate": round(n_localized / max(n_fail, 1), 4),
        "err_clause_dist": dict(loc_clause.most_common(15)),
        "avg_prompt_chars_with": round(avg_len_with, 0),
        "avg_prompt_chars_without": round(avg_len_without, 0),
        "avg_hint_chars": round(avg_len_with - avg_len_without, 0),
        "prompt_file": str(out_dir / "repair_prompts_dryrun.jsonl"),
        "note": "双变体 prompt（with/without EXPO 定位）已落盘，供 GPU 冒烟/"
                "人工抽检；GPU 生成 = --stage generate（等主控排期）。",
    }
    with open(out_dir / "repair_scan_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("\n=== repair_chain_expo Stage A (scan) ===")
    print(f"失败候选（去重）: {n_fail} | 可定位: {n_localized} "
          f"({summary['localization_rate']:.1%})")
    print(f"定位根因子句分布: {dict(loc_clause.most_common(8))}")
    print(f"prompt 平均长度: with={avg_len_with:.0f} 字符 / "
          f"without={avg_len_without:.0f} 字符（注入增量 "
          f"{avg_len_with - avg_len_without:.0f} 字符 ≈ "
          f"{(avg_len_with - avg_len_without) / 4:.0f} token）")
    print(f"产物: {out_dir / 'repair_prompts_dryrun.jsonl'} + "
          f"{out_dir / 'repair_scan_summary.json'}")
    print(f"墙钟 {time.perf_counter() - t0:.0f}s")


# ---------------------------------------------------------------------------
# Stage B：GPU 生成修复 SQL（排队中，不主动抢 GPU）
# ---------------------------------------------------------------------------

def stage_generate(args: argparse.Namespace) -> None:
    """对 Stage A 选出的失败候选生成修复 SQL。

    结构与 repair_chain.py Phase B/C 同源：sft_v2（LoRA 优先，失败回退合并
    权重）vLLM 贪心 T=0 n=1，VavSampler.extract_sql 解析，修复结果全实例执行
    接受（接受 = 全部实例无报错）。唯一差异 = prompt 注入 EXPO 定位诊断。
    """
    from sampler import VavSampler  # noqa: E402
    prompts_file = Path(args.prompts_file)
    if not prompts_file.exists():
        raise RuntimeError(f"缺少 {prompts_file}：先跑 --stage scan")
    if not args.confirm_gpu:
        raise RuntimeError(
            "GPU 生成需主控排期确认（gpudebug 排队中）。排期后运行：\n"
            "  envs/vllmenv/bin/python src/repair_chain_expo.py --stage generate "
            "--prompts-file <scan 产物> --confirm-gpu ...\n"
            "本脚本不主动抢 GPU。")
    # ---- vLLM 服务（同 repair_chain._serve：LoRA 优先，失败回退 merged）----
    from transformers import AutoTokenizer  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402
    from vllm.lora.request import LoRARequest  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                              local_files_only=True,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    rows = [json.loads(l) for l in prompts_file.read_text(encoding="utf-8").splitlines()]
    if args.max_repairs:
        rows = rows[: args.max_repairs]
    print(f"[repair_expo:generate] {len(rows)} prompts", file=sys.stderr)

    lora_dir = Path(args.checkpoints_dir) / args.lora
    llm_kwargs = dict(model=args.base_model, dtype="bfloat16",
                      trust_remote_code=True, seed=0,
                      max_model_len=args.max_prompt_tokens + args.max_new_tokens)
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    lr = None
    if lora_dir.is_dir():
        try:
            llm = LLM(enable_lora=True, max_loras=1, max_lora_rank=32, **llm_kwargs)
            lr = LoRARequest(args.lora, 1, str(lora_dir))
        except Exception as exc:
            print(f"[repair_expo:generate] LoRA serve failed ({exc})，"
                  f"回退 merged", file=sys.stderr)
            merged = Path(args.checkpoints_dir) / f"{args.lora}_merged"
            if not merged.is_dir():
                raise RuntimeError(f"LoRA 服务失败且无 merged 权重: {merged}")
            llm = LLM(model=str(merged), **llm_kwargs)
            lr = None
    else:
        merged = Path(args.checkpoints_dir) / f"{args.lora}_merged"
        if not merged.is_dir():
            raise RuntimeError(f"LoRA 目录与 merged 目录均不存在: {lora_dir}")
        llm = LLM(model=str(merged), **llm_kwargs)

    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, seed=0,
                        max_tokens=args.max_new_tokens)
    engine = AP.ExecutionEngine(args.threads, args.query_timeout,
                                args.max_vm_steps, args.row_cap)
    out_path = Path(args.out_dir) / "repairs_generated.jsonl"
    done = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": r["prompt_with_localization"]}],
                tokenize=False, add_generation_prompt=True)
            ids = tokenizer(chat, truncation=True,
                            max_length=args.max_prompt_tokens)["input_ids"]
            out = llm.generate([{"prompt_token_ids": ids}], sp, lora_request=lr)
            text = (tokenizer.decode(out[0].outputs[0].token_ids,
                                     skip_special_tokens=True)
                    if out and out[0].outputs else "")
            parsed = VavSampler.extract_sql(text)
            rep_sql = parsed["sql"] if parsed["parse_success"] else ""
            # 接受 = 全实例执行成功（BIRD 单实例 = 单次执行）
            db = Path(args.db_root) / r["db_id"] / f"{r['db_id']}.sqlite"
            ok, err = False, "parse failed"
            if rep_sql and Path(db).is_file():
                engine.run([(rep_sql, str(db))], phase="repair_exec")
                outcome = engine.get(rep_sql, str(db))
                ok, err = outcome["ok"], outcome.get("error")
            fh.write(json.dumps({
                "dataset_index": r["dataset_index"], "db_id": r["db_id"],
                "model": r["model"], "sql_orig": r["sql"],
                "error_orig": r["error"], "localized": r["localized"],
                "err_clauses": r["err_clauses"],
                "sql_repaired": rep_sql,
                "parse_success": bool(parsed["parse_success"]),
                "accepted": bool(ok), "error_after": err,
            }, ensure_ascii=False) + "\n")
            done += 1
            if done % 10 == 0:
                print(f"[repair_expo:generate] {done}/{len(rows)}", file=sys.stderr)
    print(f"[repair_expo:generate] done -> {out_path}")


# ---------------------------------------------------------------------------
# Stage C：修复候选并入候选池（纯 CPU）
# ---------------------------------------------------------------------------

def stage_merge(args: argparse.Namespace) -> None:
    items = _load_items(Path(args.items))
    repairs_file = Path(args.repairs)
    if not repairs_file.exists():
        raise RuntimeError(f"缺少 {repairs_file}：先跑 --stage generate（GPU 排期）")
    reps = [json.loads(l) for l in repairs_file.read_text(encoding="utf-8").splitlines()]
    by_di: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in reps:
        if r.get("accepted") and r.get("sql_repaired"):
            by_di[r.get("dataset_index", r.get("di"))].append(r)
    n_added = 0
    n_items_touched = 0
    for it in items:
        di = it.get("dataset_index", it.get("di"))
        cands = it.setdefault("candidates", [])
        existing = {AP.normalize_for_dedup(c.get("sql")) for c in cands}
        for r in by_di.get(di, []):
            key = AP.normalize_for_dedup(r["sql_repaired"])
            if key in existing:
                continue  # 与池内现有候选重复 → 不重复并入
            cands.append({"model": REPAIR_MODEL_LABEL, "sql": r["sql_repaired"],
                          "parse_success": True, "sample_idx": -1,
                          "repaired_from_error": r.get("error_orig"),
                          "localized_clauses": r.get("err_clauses")})
            existing.add(key)
            n_added += 1
        if by_di.get(di):
            n_items_touched += 1
    out_path = Path(args.out_dir) / "repair_expo_merged_items.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False)
    print(f"[repair_expo:merge] 并入 {n_added} 条修复候选（{n_items_touched} 题）"
          f" -> {out_path}")
    print("[repair_expo:merge] 下一步（预注册判定）：\n"
          f"  envs/reasoning3b/bin/python src/bird_select.py --phase prep "
          f"--items {out_path} --out-dir outputs/bird_select_expo\n"
          f"  envs/vllmenv/bin/python src/bird_select.py --phase score "
          f"--out-dir outputs/bird_select_expo\n"
          f"  envs/reasoning3b/bin/python src/bird_select.py --phase final "
          f"--out-dir outputs/bird_select_expo --num-cpus 12\n"
          f"  （基线 = outputs/bird_select 同臂官方 EX；判定见 --stage prereg）")


# ---------------------------------------------------------------------------
# Stage D：预注册判定口径
# ---------------------------------------------------------------------------

def stage_prereg(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("  repair_chain_expo 预注册（封盘 2026-08-23）")
    print("=" * 72)
    for k in ("primary_metric", "baseline", "treatment", "success_criterion",
              "discipline"):
        print(f"  {k:18s}: {PREREG[k]}")
    print("  次级判据:")
    for s in PREREG["secondary"]:
        print(f"    - {s}")
    print()
    print("  判定流程（全 CPU + 官方评估器，GPU 只在 generate 阶段）:")
    print("    1) 基线封盘：重跑 bird_select final on outputs/eval_pool_bird/")
    print("       items.json → baseline EX（arm_vav / arm_orm_grouphead）。")
    print("    2) Stage B 生成修复 → Stage C 并入池 → 同口径 bird_select 重跑")
    print("       → treatment EX。")
    print(f"    3) 正向判据：max(treatment - baseline) ≥ +{PREREG_EX_GAIN_PP}pp。")
    print("    4) 结果写入 tmp_idea_research/expo_localize_report.md 增补节 +")
    print("       record_experiment.py 闭环。")
    print(f"  EX_GAIN_THRESHOLD_PP = {PREREG_EX_GAIN_PP}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="P2A 修复链 EXPO 定位注入（方案+冒烟）")
    ap.add_argument("--stage", choices=["scan", "generate", "merge", "prereg"],
                    required=True)
    ap.add_argument("--items", default=str(DEFAULT_ITEMS))
    ap.add_argument("--db-root", default=str(DEFAULT_DB_ROOT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--exec-cache", default=str(DEFAULT_EXEC_CACHE),
                    help="复用 expo_diag 执行缓存（避免重复执行 98k 候选）")
    ap.add_argument("--limit", type=int, default=None, help="scan 阶段只处理前 N 题")
    # 执行引擎（与 adjudicate_pool 默认一致）
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--query-timeout", type=float, default=30.0)
    ap.add_argument("--max-vm-steps", type=int, default=5_000_000)
    ap.add_argument("--row-cap", type=int, default=100_000)
    # generate 阶段（GPU，排队中）
    ap.add_argument("--prompts-file", default=None)
    ap.add_argument("--confirm-gpu", action="store_true",
                    help="显式确认 GPU 排期后才允许 --stage generate")
    ap.add_argument("--base-model",
                    default=str(PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"))
    ap.add_argument("--checkpoints-dir", default=str(PROJECT_ROOT / "checkpoints"))
    ap.add_argument("--lora", default="sft_v2")
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-repairs", type=int, default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    # merge 阶段
    ap.add_argument("--repairs", default=None,
                    help="Stage B 产物 repairs_generated.jsonl")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.stage == "scan":
        stage_scan(args)
    elif args.stage == "generate":
        stage_generate(args)
    elif args.stage == "merge":
        stage_merge(args)
    else:
        stage_prereg(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
