#!/usr/bin/env python3
"""v1 + v2 跨 checkpoint 混合投票评估（10 票制, vav 执行裁决为主口径）。

背景:
  - sft_phase1 (v1): 单次官方 60.8%, 5p 投票官方 70.1% (增益 +9.3pp)
  - sft_v2     (v2): 单次官方 63.5%, 5p 投票官方 68.7% (增益 +5.2pp)
  - 假设: v2 投票增益缩水是"同一老师教的 5 个视角太像"; 把两个不同 checkpoint
    的答案混合投票(10 票制)可能恢复多样性, 超过 v1 5p 的 70.1%。
  - 离线分析发现简单多数存在平票风险: 29 题两模型各 5 票全票一致但都错; 30 题
    v2 独有一致且错而 v1 对。v1 5 对 + v2 5 错时, 10 票多数 = 5:5 平票无解。
    因此主裁决 = vav 执行结果分组投票(参考 finer_port/vav_voting.py), 简单
    多数只作对照臂输出。

四路逐题结果(items.json; 正确性判定统一 compare_execution_results + truncated
检查, 与 5p 基线完全同口径, 避免口径混比):
  a5p    = A 单独 5 票: 原版执行结果分组投票(full_rows 分组/最大组/平局最短
           SQL, 与 eval_5prompt_agent 逐行一致) → 直接对照官方 70.1%
  b5p    = B 单独 5 票(同上) → 对照官方 68.7%
  mixvav = A5+B5 共 10 票 vav 执行裁决【主口径】: 按 normalize_execution_result
           签名分组(header-agnostic, FINER 口径); 空结果组(SUCCESS_VALUES:)算
           一组不丢弃(P5 教训); 全零组跳过(与 FINER 一致); 最大 size 组胜出;
           size 平票 → 回退 v1(A) 的 5p 胜出答案。items.json 顶层
           match/predicted_sql 即此臂(喂 scripts/eval_official.sh 官方复评)
  maj10  = 简单多数对照臂: SQL 文本归一化(normalize_sql)计票, 平票取 A 的票

候选过滤(与 FINER 分组语义一致): 执行失败且非基础设施错误(语法错, 见
vav_voting.is_syntax_error)的候选跳过——只有执行成功的候选参与分组。

生成协议: 每 checkpoint 各生成 5 个视角(import build_prompt_variants 前 5
视角, 与 eval_5prompt_agent / eval_5p_t10 完全一致), 温度 0 贪心, 截断窗口
1536、max_new_tokens 256 与 eval_5p_fs.py 一致。

显存纪律(双 LoRA 交替加载, 绝不共存):
  - base 模型(Qwen2.5-Coder-3B-Instruct)只加载一次常驻; A 适配器跑完全部
    1034×5 票后 PeftModel.unload()(移除 A 的 LoRA 层, base 回到原始状态) +
    torch.cuda.empty_cache(), 再挂 B 适配器;
  - 峰值显存 = base(bf16 ≈ 6GB) + 单个 LoRA 适配器 + 激活, 与单模型 5p 作业
    一致, 不存在双适配器同时驻留;
  - 代价: A 的 5 票(执行结果行/截断标志/SQL/签名)需跨 B 阶段驻留内存,
    Spider 结果普遍很小预计 <1GB; 若个别查询返回接近 FULL_ROWS_HARD_LIMIT
    (100k 行)会显著膨胀, 见"风险"。

风险(另见 summary.note):
  - A 票跨阶段内存占用: 见上; 出现 OOM 时需把 B 阶段改为逐题即时投票+落盘,
    只保留 A 票(本版不做);
  - 两个 LoRA 同 base 顺序推理完全确定(温度 0 贪心), a5p/b5p 应精确复现
    5p 基线(官方复评 70.1%/68.7%); 若偏差 >0.5pp 说明环境/权重有漂移, 先查
    再信 mixvav;
  - 空票题(全部 parse/exec 失败)沿用原版语义: voted_rows=[] 与 gold 比较
    (gold 也为空时会被判 match, 原版已知边界 quirk, 口径保持一致);
  - vav 签名与全零组语义忠实移植 vav_voting(P5 已修: 空结果算一组; 全零组
    主选跳过与 FINER 一致, 多列全零不命中属已知局限);
  - PeftModel.unload() 依赖 peft>=0.5(本环境必满足); 若换更老环境需改为
    整模卸载重载。

用法(GPU):
    python src/eval_mix_vote.py \
        --lora-a checkpoints/sft_phase1 --lora-b checkpoints/sft_v2 \
        --output-dir outputs/eval_mix_vote --limit 1034 --temperature 0.0
    # 完成后官方 test-suite 复评(mixvav 胜者):
    bash scripts/eval_official.sh outputs/eval_mix_vote/items.json \
        outputs/official_eval_mix_vote
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))
_FINER_DIR = str(PROJECT / "finer_port")
if _FINER_DIR not in sys.path:
    sys.path.insert(0, _FINER_DIR)

# 复用原版 5 视角 prompt + 模型路径常量(不复制, 保证与原版完全一致)
from eval_5p_t10 import BASE_MODEL, SPIDER, build_prompt_variants  # noqa: E402
from reasoning_generator_agent import ReasoningGeneratorAgent  # noqa: E402
from spider_utils import (  # noqa: E402
    SpiderLoader,
    DatabaseExecutor,
    compare_execution_results,
    normalize_sql,
)
# vav 裁决原语(忠实移植, 不重写; 私有 _parse_vals/_is_all_zero_group 与
# choose_group_vav 同款过滤语义, 直接复用以避免复制漂移)
from vav_voting import (  # noqa: E402
    normalize_execution_result,
    is_syntax_error,
    _parse_vals,
    _is_all_zero_group,
)

WINDOW = 1536             # 与原版 eval_5p_t10 的截断窗口一致
N_VIEWS_PER_MODEL = 5     # 每模型视角数(复用 build_prompt_variants 的前 5 视角)
DEFAULT_LORA_A = str(PROJECT / "checkpoints" / "sft_phase1")
DEFAULT_LORA_B = str(PROJECT / "checkpoints" / "sft_v2")
DEFAULT_OUTPUT_DIR = str(PROJECT / "outputs" / "eval_mix_vote")
EMPTY_PRED = "SELECT 1"   # AGENTS.md: 空预测不跳过, 写 SELECT 1

# 每张票的结构(仅执行成功的候选才成票):
#   {"sql", "norm", "rows", "truncated", "source", "vav_key"}
#   norm    = normalize_sql(sql)                          → maj10 计票键
#   vav_key = normalize_execution_result(执行结果 dict)     → mixvav 分组键
#   rows    = tuple(tuple(row) ...)                       → 匹配判定用
Vote = Dict[str, Any]


# ===================================================================
# 投票判定
# ===================================================================

def vote_exec_results(exec_results: List[Tuple[Any, bool, str]]
                      ) -> Tuple[List[List[Any]], int, bool, str]:
    """执行结果分组投票(与 eval_5p_t10/eval_5p_fs 逐行一致): 最大组, 平局最短 SQL。"""
    voted_rows: List[List[Any]] = []
    vc, voted_truncated, selected_sql = 0, False, ""
    if exec_results:
        groups = {}
        for rows, truncated, sql in exec_results:
            g = groups.setdefault(rows, {"count": 0, "truncated": False,
                                         "shortest_sql": sql})
            g["count"] += 1
            g["truncated"] = g["truncated"] or truncated
            if len(sql) < len(g["shortest_sql"]):
                g["shortest_sql"] = sql
        best = max(groups.values(), key=lambda g: g["count"])
        voted_rows = [list(row) for row in
                      next(k for k, v in groups.items() if v is best)]
        vc = best["count"]
        voted_truncated = best["truncated"]
        selected_sql = best["shortest_sql"]
    return voted_rows, vc, voted_truncated, selected_sql


def vote_sql_text(votes: List[Vote], prefer_source: str = "A"
                  ) -> Tuple[List[List[Any]], int, bool, str, bool]:
    """SQL 文本归一化计票投票(maj10 对照臂): 最大组胜出, 平票取 prefer_source 的票。

    平票细规则: 优先 prefer_source(A) 投过的文本; 仍平 → 投票序列中最早出现的
    文本(确定性)。胜者 SQL = 组内最短原始 SQL(与原版组内最短惯例一致)。
    Returns (voted_rows, vc, voted_truncated, selected_sql, tie_broken)。
    """
    if not votes:
        return [], 0, False, "", False
    counts: Counter[str] = Counter(v["norm"] for v in votes)
    max_count = max(counts.values())
    tied = [norm for norm, c in counts.items() if c == max_count]
    tie_broken = len(tied) > 1
    if tie_broken:
        pref = {v["norm"] for v in votes if v["source"] == prefer_source}
        pref_tied = [norm for norm in tied if norm in pref]
        if pref_tied:
            tied = pref_tied
        first_idx = {norm: next(i for i, v in enumerate(votes) if v["norm"] == norm)
                     for norm in tied}
        tied.sort(key=lambda norm: first_idx[norm])
    winner_norm = tied[0]
    group = [v for v in votes if v["norm"] == winner_norm]
    voted_rows = [list(row) for row in group[0]["rows"]]
    voted_truncated = any(v["truncated"] for v in group)
    selected_sql = min((v["sql"] for v in group), key=len)
    return voted_rows, max_count, voted_truncated, selected_sql, tie_broken


def adjudicate_mixvav(votes: List[Vote],
                      v1_winner: Optional[Tuple[List[List[Any]], bool, str, int]]
                      ) -> Tuple[List[List[Any]], int, bool, str, bool, int]:
    """10 票 vav 执行裁决(主口径)。与 choose_group_vav 的三点差异:

    1) 空结果组(SUCCESS_VALUES:)算一组不跳过(P5 教训: 空结果可能是正确答案);
    2) 全零组主选跳过与 FINER 一致, 全被过滤时 fallback 回全部 SUCCESS_VALUES
       组(FINER fallback 语义);
    3) size 平票 → 回退 v1(A) 的 5p 胜出答案(v1_winner); v1 无胜出答案 →
       按 FINER 语义取 key 字符串最大(确定性)。

    v1_winner = (rows, truncated, sql, vc) = A 单独 5p 的胜出结果; 平票回退时
    整体采用 v1 的胜出答案(含其票数), 以 tie_fallback_v1=True 标记。
    Returns (voted_rows, vc, voted_truncated, selected_sql, tie_fallback_v1,
             n_groups)。
    """
    groups: Dict[str, List[Vote]] = {}
    for v in votes:
        groups.setdefault(v["vav_key"], []).append(v)
    sv_keys = [k for k in groups if k.startswith("SUCCESS_VALUES:")]
    if not sv_keys:
        return [], 0, False, "", False, 0
    pool = [k for k in sv_keys if not _is_all_zero_group(_parse_vals(k))]
    if not pool:
        pool = sv_keys  # 全被过滤 → fallback 全部 SUCCESS_VALUES 组(FINER 语义)
    max_size = max(len(groups[k]) for k in pool)
    top = [k for k in pool if len(groups[k]) == max_size]
    if len(top) == 1:
        key = top[0]
    elif v1_winner is not None:
        rows, trunc, sql, vc = v1_winner
        return list(rows), vc, trunc, sql, True, len(sv_keys)
    else:
        key = max(top)  # 无 v1 胜出答案可回退 → FINER 语义 key 字符串最大
    group = groups[key]
    voted_rows = [list(row) for row in group[0]["rows"]]
    voted_truncated = any(v["truncated"] for v in group)
    selected_sql = min((v["sql"] for v in group), key=len)
    return voted_rows, len(group), voted_truncated, selected_sql, False, len(sv_keys)


# ===================================================================
# 主流程
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-a", default=DEFAULT_LORA_A,
                        help="第一个模型的 LoRA 目录(v1, sft_phase1)")
    parser.add_argument("--lora-b", default=DEFAULT_LORA_B,
                        help="第二个模型的 LoRA 目录(v2, sft_v2)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--limit", type=int, default=1034,
                        help="评估条数(默认全量 1034)")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help="基础模型路径(默认 3B)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度(默认 0 = 贪心, 与基线 eval_5p_sft_v2 一致)")
    args = parser.parse_args()

    lora_a, lora_b = args.lora_a, args.lora_b
    out_dir = Path(args.output_dir)
    limit = args.limit
    start_index = args.start_index
    base_model = args.base_model
    temperature = args.temperature
    print(f"A={lora_a}(5 票) + B={lora_b}(5 票) = 10 票混合投票 | vav 执行裁决为主口径 | "
          f"base: {base_model} | {limit} 条 (start={start_index}) | temperature={temperature}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map={"": 0},
        local_files_only=True, trust_remote_code=True)
    base.config.pad_token_id = tokenizer.eos_token_id
    from peft import PeftModel

    loader = SpiderLoader(SPIDER)
    executor = DatabaseExecutor(SPIDER)
    items = loader.load_dev(limit=limit, start_index=start_index)
    n_items = len(items)

    # 每题的模型票: votes_by_item[i] = {"A": [...], "B": [...]}
    votes_by_item: List[Dict[str, List[Vote]]] = [
        {"A": [], "B": []} for _ in items]
    n_parse_fail = {"A": 0, "B": 0}
    n_syntax_err = {"A": 0, "B": 0}   # 执行失败且为语法错(不参与分组)
    n_infra_err = {"A": 0, "B": 0}    # 执行失败但为基础设施错(不参与分组)

    start_t = time.time()

    # ===== 阶段 1: 逐模型顺序推理(显存纪律: A 跑完 unload 再挂 B, 绝不共存) =====
    for source_label, lora in (("A", lora_a), ("B", lora_b)):
        if source_label == "B":
            # 显存纪律: 移除 A 的 LoRA 层(base 回到原始状态)并释放显存, 再挂 B
            model = model.unload()
            torch.cuda.empty_cache()
            print("[B] A 适配器已 unload")
        model = PeftModel.from_pretrained(base, lora)
        model.eval()
        print(f"[{source_label}] LoRA 已就绪: {lora}")

        for i, item in enumerate(items):
            db_id, question = item["db_id"], item["question"]
            ddl, _ = loader.get_ddl_with_source(db_id)
            prompts = build_prompt_variants(question, ddl)[:N_VIEWS_PER_MODEL]
            for p in prompts:
                chat = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False,
                    add_generation_prompt=True)
                enc = tokenizer(chat, return_tensors="pt", truncation=True,
                                max_length=WINDOW).to("cuda:0")
                in_len = enc["input_ids"].shape[1]
                with torch.inference_mode():
                    if temperature > 0:
                        out = model.generate(
                            **enc,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=None,
                            top_k=None,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    else:
                        # temperature == 0: 与原版 eval_5prompt_agent 一致的贪心解码
                        out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                             do_sample=False,
                                             pad_token_id=tokenizer.eos_token_id)
                text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
                parsed = ReasoningGeneratorAgent.extract_sql(text)
                if not parsed["parse_success"]:
                    n_parse_fail[source_label] += 1
                    continue
                r = executor.execute(db_id, parsed["sql"])
                if not r["success"]:
                    # 语法错跳过不参与分组(与 FINER 一致); 基础设施错仅计数
                    if is_syntax_error(r):
                        n_syntax_err[source_label] += 1
                    else:
                        n_infra_err[source_label] += 1
                    continue
                votes_by_item[i][source_label].append({
                    "sql": parsed["sql"],
                    "norm": normalize_sql(parsed["sql"]),
                    "vav_key": normalize_execution_result(r),
                    "rows": tuple(tuple(row) for row in r["full_rows"]),
                    "truncated": r["full_rows_truncated"],
                    "source": source_label,
                })
            if (i + 1) % 100 == 0:
                done = sum(len(v[source_label]) for v in votes_by_item[:i + 1])
                print(f"  [{source_label} {i+1}/{n_items}] 成票 {done} | "
                      f"parse 失败 {n_parse_fail[source_label]} | "
                      f"语法错 {n_syntax_err[source_label]} | "
                      f"基础设施错 {n_infra_err[source_label]}")

    del model
    torch.cuda.empty_cache()

    # ===== 阶段 2: 四路投票 + 匹配(不再推理, 纯 CPU 内存内) =====
    counts = {"a5p": 0, "b5p": 0, "mixvav": 0, "maj10": 0}
    n_mixvav_tie = 0
    n_maj10_tie = 0
    n_both_unanimous = 0
    results = []

    for i, item in enumerate(items):
        db_id, question, gold_sql = item["db_id"], item["question"], item["query"]
        a_votes = votes_by_item[i]["A"]
        b_votes = votes_by_item[i]["B"]
        a_unanimous = len(a_votes) == N_VIEWS_PER_MODEL and \
            len({v["vav_key"] for v in a_votes}) == 1
        b_unanimous = len(b_votes) == N_VIEWS_PER_MODEL and \
            len({v["vav_key"] for v in b_votes}) == 1
        if a_unanimous and b_unanimous:
            n_both_unanimous += 1

        # A/B 单独 5p(原版口径, 可直接对照官方 70.1% / 68.7%)
        a_rows, a_vc, a_trunc, a_sql = vote_exec_results(
            [(v["rows"], v["truncated"], v["sql"]) for v in a_votes])
        b_rows, b_vc, b_trunc, b_sql = vote_exec_results(
            [(v["rows"], v["truncated"], v["sql"]) for v in b_votes])

        # maj10 简单多数对照臂
        maj_rows, maj_vc, maj_trunc, maj_sql, maj_tie = vote_sql_text(
            a_votes + b_votes, prefer_source="A")

        # mixvav 主裁决臂; 平票回退 v1(A) 的 5p 胜出答案
        v1_winner = (a_rows, a_trunc, a_sql, a_vc) if a_sql else None
        mix_rows, mix_vc, mix_trunc, mix_sql, mix_tie, mix_n_groups = \
            adjudicate_mixvav(a_votes + b_votes, v1_winner=v1_winner)

        gold_r = executor.execute(db_id, gold_sql)
        gold_ok = gold_r["success"]
        gold_rows = gold_r["full_rows"] if gold_ok else []
        gold_trunc = (gold_r.get("full_rows_truncated", False)
                      if gold_ok else False)

        def judge(voted_rows: List[List[Any]], voted_truncated: bool) -> bool:
            if gold_ok and not voted_truncated and not gold_trunc:
                return compare_execution_results(
                    voted_rows, gold_rows, gold_sql=gold_sql)["match"]
            return False

        m_a = judge(a_rows, a_trunc)
        m_b = judge(b_rows, b_trunc)
        m_mix = judge(mix_rows, mix_trunc)
        m_maj = judge(maj_rows, maj_trunc)
        counts["a5p"] += m_a
        counts["b5p"] += m_b
        counts["mixvav"] += m_mix
        counts["maj10"] += m_maj
        n_mixvav_tie += int(mix_tie)
        n_maj10_tie += int(maj_tie)

        results.append({
            "di": item["dataset_index"],
            "db_id": db_id,
            "match": m_mix,                     # mixvav 主口径
            "predicted_sql": mix_sql or EMPTY_PRED,
            "a_ok": len(a_votes), "b_ok": len(b_votes),
            "a_unanimous": a_unanimous, "b_unanimous": b_unanimous,
            "a5p": {"match": m_a, "votes": a_vc, "truncated": a_trunc,
                    "predicted_sql": a_sql or EMPTY_PRED},
            "b5p": {"match": m_b, "votes": b_vc, "truncated": b_trunc,
                    "predicted_sql": b_sql or EMPTY_PRED},
            "mixvav": {"match": m_mix, "votes": mix_vc, "truncated": mix_trunc,
                       "predicted_sql": mix_sql or EMPTY_PRED,
                       "tie_fallback_v1": mix_tie, "n_groups": mix_n_groups},
            "maj10": {"match": m_maj, "votes": maj_vc, "truncated": maj_trunc,
                      "predicted_sql": maj_sql or EMPTY_PRED,
                      "tie_broken": maj_tie},
        })
        if (i + 1) % 100 == 0:
            print(f"  [vote {i+1}/{n_items}] a5p={counts['a5p']} "
                  f"b5p={counts['b5p']} maj10={counts['maj10']} "
                  f"mixvav={counts['mixvav']}")

    elapsed = time.time() - start_t
    rate = {k: (v / n_items if n_items else 0.0) for k, v in counts.items()}
    best_key = "a5p" if counts["a5p"] >= counts["b5p"] else "b5p"
    delta_mix = rate["mixvav"] - rate[best_key]
    delta_maj = rate["maj10"] - rate[best_key]
    print(f"\n=== v1+v2 混合投票 RESULT (T={temperature}) ===")
    print(f"A 单独 5p : {counts['a5p']}/{n_items} ({rate['a5p']:.1%})  [官方基线 70.1%]")
    print(f"B 单独 5p : {counts['b5p']}/{n_items} ({rate['b5p']:.1%})  [官方基线 68.7%]")
    print(f"maj10 对照: {counts['maj10']}/{n_items} ({rate['maj10']:.1%})  "
          f"[Δ vs {best_key} = {delta_maj:+.1%}]")
    print(f"mixvav 主 : {counts['mixvav']}/{n_items} ({rate['mixvav']:.1%})  "
          f"[Δ vs {best_key} = {delta_mix:+.1%}]")
    print(f"平票: mixvav 回退 v1 {n_mixvav_tie} 题 | maj10 平票 {n_maj10_tie} 题 | "
          f"两模型全票一致 {n_both_unanimous} 题")
    print(f"Time: {elapsed:.0f}s | parse 失败 A={n_parse_fail['A']} B={n_parse_fail['B']} | "
          f"语法错 A={n_syntax_err['A']} B={n_syntax_err['B']} | "
          f"基础设施错 A={n_infra_err['A']} B={n_infra_err['B']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": "mix_vote_10_vav",
            "lora_a": lora_a, "lora_b": lora_b,
            "base_model": base_model, "temperature": temperature,
            "n_views_per_model": N_VIEWS_PER_MODEL, "n_views_total": 10,
            "start_index": start_index, "limit": limit, "n_items": n_items,
            "arms": {
                "a5p": {"match_count": counts["a5p"],
                        "match_rate": round(rate["a5p"], 6)},
                "b5p": {"match_count": counts["b5p"],
                        "match_rate": round(rate["b5p"], 6)},
                "mixvav": {"match_count": counts["mixvav"],
                           "match_rate": round(rate["mixvav"], 6),
                           "primary": True},
                "maj10": {"match_count": counts["maj10"],
                          "match_rate": round(rate["maj10"], 6),
                          "primary": False},
            },
            "best_single": best_key,
            "best_single_match_rate": round(rate[best_key], 6),
            "mixvav_minus_best_single": round(delta_mix, 6),
            "mixvav_minus_best_single_pp": round(delta_mix * 100, 3),
            "maj10_minus_best_single": round(delta_maj, 6),
            "maj10_minus_best_single_pp": round(delta_maj * 100, 3),
            "n_mixvav_tie_fallback_v1": n_mixvav_tie,
            "n_maj10_tie": n_maj10_tie,
            "n_both_unanimous": n_both_unanimous,
            "parse_fail": n_parse_fail,
            "syntax_err": n_syntax_err,
            "infra_err": n_infra_err,
            "elapsed_seconds": round(elapsed, 1),
            "note": (
                "v1(sft_phase1) 与 v2(sft_v2) 各 5 视角共 10 票混合投票。主口径 = "
                "mixvav: vav 执行结果分组裁决(normalize_execution_result 签名, "
                "header-agnostic FINER 口径; 空结果组算一组不丢弃 P5 教训; 全零组"
                "主选跳过与 FINER 一致; 最大 size 组胜出; size 平票回退 v1 的 5p "
                "胜出答案)。maj10 = SQL 文本归一化简单多数(平票取 A 的票), 仅对照。"
                "a5p/b5p = 原版执行结果分组投票(与 eval_5prompt_agent 逐行一致), "
                "用于对照官方 70.1%/68.7%。四臂正确性判定统一 compare_execution_"
                "results + truncated 检查(与 5p 基线同口径); 语法错候选跳过不参与"
                "分组(与 FINER 一致)。items.json 顶层 match/predicted_sql = mixvav "
                "臂(喂 eval_official.sh 官方复评); 空预测按 AGENTS.md 写 SELECT 1 "
                "不跳过。空票题沿用原版语义(voted_rows=[] 与 gold 比较, gold 也空"
                "时判 match, 原版已知边界 quirk)。温度 0 贪心确定性推理, a5p/b5p "
                "应精确复现 5p 基线, 偏差 >0.5pp 需先查环境/权重再信 mixvav。"
            ),
        }, f, ensure_ascii=False, indent=2)
    with open(out_dir / "items.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
