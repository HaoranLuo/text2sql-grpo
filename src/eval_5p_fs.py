#!/usr/bin/env python3
"""few-shot 轨迹第 6 视角投票评估（视角多样性实验, 期望 vs 5p +1~2pp）。

背景: 5p 投票增益缩水(v1 +9.3pp → v2 +5.2pp), 高温采样实验证明采样噪声多样性
无效(66.1 < 68.7); 文献(Reasoning-SQL Table 4)与结论 11 指向正解 = 视角多样性。
本实验: 把 SFT v2 教案中 3 条"执行正确"的推理轨迹(<think>+SQL)作为 few-shot 示例
(Worked Examples), 拼进 build_prompt 的 Instructions 之前, 作为第 6 视角;
其余 5 视角与 eval_5p_t10 完全一致(直接 import build_prompt_variants 复用)。
6 视角 × 每视角 1 采样(默认温度 0 贪心, 与基线 eval_5p_sft_v2 一致) → 执行分组
投票(复用原版投票逻辑: 按 full_rows 分组, 最大组, 平局最短 SQL)。

用法:
    # 主流程(GPU):
    python src/eval_5p_fs.py --lora-path checkpoints/sft_v2 \
        --output-dir outputs/eval_5p_fs_sft_v2 \
        --fewshot-file outputs/fewshot_examples.json \
        --n-prompts 5 --limit 1034 --temperature 0.0

    # 可选第一步(CPU): 从 HPC 已有产物收集 3 条轨迹(优先 maj_diag/5p_sft_v2,
    # 读结构后确认两者不可用 → 回退 train 侧 sft_v2_mix.json; 找不到则
    # --fewshot-file 留空, 由主控提供):
    python src/eval_5p_fs.py --collect-mode

约束: 示例严禁 dev 轨迹(评估端再做一次 dev.json 问题重叠检查, 重叠即中止);
示例注入后若超 1536 截断窗口, 逐档压缩示例推理正文(保留 SQL 块), 仍超则丢弃
示例块(p6 退化为 p1, 记 p6_dropped 字段)。
"""
import argparse
import json
import re
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

# 复用原版 5 视角 prompt + 模型路径常量(不复制, 保证与原版完全一致)
from eval_5p_t10 import BASE_MODEL, SPIDER, build_prompt_variants  # noqa: E402
from reasoning_generator_agent import ReasoningGeneratorAgent  # noqa: E402
from spider_utils import SpiderLoader, DatabaseExecutor, compare_execution_results  # noqa: E402

WINDOW = 1536  # 与原版 eval_5p_t10 的 max_length 一致(截断窗口)
DEFAULT_FEWSHOT_FILE = str(PROJECT / "outputs" / "fewshot_examples.json")
DEFAULT_DEV_JSON = str(PROJECT / "data" / "spider_data" / "dev.json")
# 示例响应长度逐档压缩上限(保护 1536 窗口; 首档 None = 不压缩)
EXAMPLE_RESPONSE_CAPS = [1200, 600, 300, 150]

# --collect-mode 候选来源(按优先级; 结构调查结论见 collect_fs_examples.py docstring)
COLLECT_SOURCE_CANDIDATES = [
    str(PROJECT / "outputs" / "eval_maj_diag_100" / "items.json"),
    str(PROJECT / "outputs" / "eval_5p_sft_v2" / "items.json"),
    str(PROJECT / "data" / "sft_v2_mix.json"),
]


# ===================================================================
# few-shot prompt 构造与 1536 窗口适配
# ===================================================================

def load_fewshot(path: str) -> List[Dict[str, Any]]:
    """加载 fewshot 文件({examples:[{question, response, ...}]}); 缺失返回 []。"""
    p = Path(path)
    if not p.exists():
        print(f"[fewshot] 文件不存在({p}) → 退化为纯 5 视角投票(fewshot_used=[])")
        return []
    payload = json.loads(p.read_text(encoding="utf-8"))
    examples = payload.get("examples") or []
    for ex in examples:
        if not ex.get("question") or not ex.get("response"):
            raise ValueError(f"fewshot 文件条目缺 question/response: {p}")
    print(f"[fewshot] 加载 {len(examples)} 条示例: source={payload.get('source')}")
    return examples


def guard_no_dev_leak(examples: List[Dict[str, Any]], dev_json: str) -> None:
    """评估端泄漏守卫: 示例问题与 dev.json 任一问题重叠 → 直接中止(exit 4)。"""
    dev = json.loads(Path(dev_json).read_text(encoding="utf-8"))
    dev_qs = {" ".join((d.get("question") or "").strip().lower().split()) for d in dev}
    for ex in examples:
        q = " ".join((ex.get("question") or "").strip().lower().split())
        if q in dev_qs:
            print(f"[fewshot] FATAL: 示例问题出现在 dev.json, 禁止使用(泄漏): {q!r}",
                  file=sys.stderr)
            sys.exit(4)


def trim_response(response: str, cap: int) -> str:
    """压缩示例响应: 截断 <think> 推理正文, 完整保留 ```sql 块(投票只看 SQL)。"""
    if len(response) <= cap:
        return response
    m = re.match(r"\s*(<think>)(.*?)(</think>)(.*)", response, re.IGNORECASE | re.DOTALL)
    if not m:
        return response[:cap] + " ... (truncated)"
    head, body, tail, rest = m.group(1), m.group(2), m.group(3), m.group(4)
    budget = cap - len(head) - len(tail) - len(rest) - 20
    if budget <= 0:
        return head + tail + rest  # 结构保底: 只留 think 标记 + SQL 块
    return head + "\n" + body[:budget].rstrip() + " ... (truncated)\n" + tail + rest


def build_example_block(examples: List[Dict[str, Any]], cap: Optional[int] = None) -> str:
    parts = ["## Worked Examples"]
    for i, ex in enumerate(examples, 1):
        resp = ex["response"]
        if cap is not None and len(resp) > cap:
            resp = trim_response(resp, cap)
        parts.append(f"Example {i}:\nQuestion: {ex['question']}\n{resp}")
    return "\n\n".join(parts)


def build_fewshot_prompt(question: str, ddl: str, block: str) -> str:
    """p1(build_prompt) 的 Instructions 前插入 Worked Examples 块(第 6 视角)。"""
    p1 = ReasoningGeneratorAgent.build_prompt(question=question, ddl_schema=ddl,
                                              dialect="sqlite")
    anchor = "Instructions:"
    if anchor in p1:
        return p1.replace(anchor, block + "\n\n" + anchor, 1)
    return p1 + "\n\n" + block


def _chat_token_len(tokenizer, prompt: str) -> int:
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return len(tokenizer(chat, add_special_tokens=False)["input_ids"])


def fit_p6(tokenizer, question: str, ddl: str, examples: List[Dict[str, Any]]
           ) -> Tuple[str, int, bool, bool]:
    """
    构造第 6 视角 prompt 并适配 1536 窗口。

    Returns (prompt, token_len, trimmed, dropped):
      trimmed = 示例响应被压缩过; dropped = 压缩到底仍超窗, 丢弃示例块(p6 退化为 p1)。
    """
    for cap in [None] + EXAMPLE_RESPONSE_CAPS:
        block = build_example_block(examples, cap=cap)
        prompt = build_fewshot_prompt(question, ddl, block)
        n = _chat_token_len(tokenizer, prompt)
        if n <= WINDOW:
            return prompt, n, (cap is not None), False
    p1 = ReasoningGeneratorAgent.build_prompt(question=question, ddl_schema=ddl,
                                              dialect="sqlite")
    n = _chat_token_len(tokenizer, p1)
    print(f"[fewshot][warn] 示例压缩到底仍超 {WINDOW} 窗口 → 本题丢弃示例块 "
          f"(p6 退化为 p1, 记录 p6_dropped)")
    return p1, n, True, True


# ===================================================================
# 投票判定(与 eval_5p_t10 逐行一致: 执行结果分组, 最大组, 平局最短 SQL)
# ===================================================================

def vote_exec_results(exec_results: List[Tuple[Any, bool, str]]
                      ) -> Tuple[List[List[Any]], int, bool, str]:
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


# ===================================================================
# --collect-mode(可选第一步, CPU): 从 HPC 已有产物收集 3 条轨迹
# ===================================================================

def run_collect_mode(out_path: str, min_examples: int = 3) -> int:
    from collect_fs_examples import run_collection, write_output
    for src in COLLECT_SOURCE_CANDIDATES:
        if not Path(src).exists():
            print(f"[collect-mode] 跳过(不存在): {src}")
            continue
        try:
            result = run_collection(source=src, min_examples=min_examples)
        except (ValueError, FileNotFoundError) as exc:
            print(f"[collect-mode] 跳过(结构不可用): {src} → {exc}")
            continue
        if result["n"] >= min_examples:
            write_output(result, out_path)
            return 0
        print(f"[collect-mode] {src}: 只有 {result['n']} 条可用 (< {min_examples})")
    print("[collect-mode] 未找到 ≥3 条可用轨迹 → --fewshot-file 留空, 由主控提供")
    return 3


# ===================================================================
# 主流程
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-path", default=None, help="LoRA adapter 目录(省略=无 LoRA 基线)")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--n-prompts", type=int, default=5, choices=[5, 7],
                        help="原版视角数(5 或 7); few-shot 视角额外 +1")
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help="基础模型路径(默认 3B)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度(默认 0 = 贪心, 与基线 eval_5p_sft_v2 一致)")
    parser.add_argument("--fewshot-file", default=DEFAULT_FEWSHOT_FILE,
                        help="fewshot_examples.json(collect_fs_examples.py 产物); "
                             "留空/缺失 → 纯 5 视角退化运行")
    parser.add_argument("--collect-mode", action="store_true",
                        help="只做第一步: 从 HPC 已有产物收集 3 条轨迹并退出(CPU)")
    args = parser.parse_args()

    if args.collect_mode:
        sys.exit(run_collect_mode(args.fewshot_file))

    lora = args.lora_path
    out_dir = Path(args.output_dir)
    limit = args.limit
    start_index = args.start_index
    n_prompts = args.n_prompts
    base_model = args.base_model
    temperature = args.temperature

    examples = load_fewshot(args.fewshot_file)
    if examples:
        guard_no_dev_leak(examples, DEFAULT_DEV_JSON)
    total_prompts = n_prompts + (1 if examples else 0)
    print(f"{n_prompts}prompt + {len(examples)} few-shot 示例 = {total_prompts} 视角投票 | "
          f"LoRA: {lora} | base: {base_model} | {limit} 条 (start={start_index}) | "
          f"temperature={temperature}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map={"": 0},
        local_files_only=True, trust_remote_code=True)
    if lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, lora)
    else:
        model = base
    model.eval()
    model.config.pad_token_id = tokenizer.eos_token_id

    loader = SpiderLoader(SPIDER)
    executor = DatabaseExecutor(SPIDER)
    items = loader.load_dev(limit=limit, start_index=start_index)

    match_count = 0
    results = []
    n_trimmed = 0
    n_dropped = 0
    start_t = time.time()

    for i, item in enumerate(items):
        db_id, question, gold_sql = item["db_id"], item["question"], item["query"]
        ddl, _ = loader.get_ddl_with_source(db_id)
        prompts = build_prompt_variants(question, ddl)[:n_prompts]

        p6_token_len = None
        p6_trimmed = False
        p6_dropped = False
        if examples:
            p6, p6_token_len, p6_trimmed, p6_dropped = fit_p6(
                tokenizer, question, ddl, examples)
            prompts.append(p6)
            if p6_trimmed:
                n_trimmed += 1
            if p6_dropped:
                n_dropped += 1

        exec_results = []  # (full_rows_tuple, truncated_flag, sql)
        for p in prompts:
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
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
            if parsed["parse_success"]:
                r = executor.execute(db_id, parsed["sql"])
                if r["success"]:
                    exec_results.append((tuple(tuple(row) for row in r["full_rows"]),
                                         r["full_rows_truncated"], parsed["sql"]))

        voted_rows, vc, voted_truncated, selected_sql = vote_exec_results(exec_results)

        gold_r = executor.execute(db_id, gold_sql)
        gold_rows = gold_r["full_rows"] if gold_r["success"] else []
        gold_truncated = (gold_r.get("full_rows_truncated", False)
                          if gold_r["success"] else False)

        if gold_r["success"] and not voted_truncated and not gold_truncated:
            is_match = compare_execution_results(
                voted_rows, gold_rows, gold_sql=gold_sql)["match"]
        else:
            is_match = False
        if is_match:
            match_count += 1

        results.append({
            "di": item["dataset_index"], "match": is_match,
            "votes": vc, "truncated": voted_truncated,
            "db_id": db_id, "predicted_sql": selected_sql,
            "n_prompts": total_prompts,
            "fewshot_used": bool(examples) and not p6_dropped,
            "p6_token_len": p6_token_len,
            "p6_trimmed": p6_trimmed,
            "p6_dropped": p6_dropped,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{limit}] match={match_count}/{i+1} ({match_count/(i+1):.1%})")

    elapsed = time.time() - start_t
    rate = match_count / limit
    print(f"\n=== 5p + few-shot 轨迹第 6 视角投票 RESULT (T={temperature}) ===")
    print(f"Match: {match_count}/{limit} ({rate:.1%})")
    print(f"Time: {elapsed:.0f}s | LoRA: {lora}")
    print(f"few-shot: {len(examples)} 条示例 | 压缩过的题: {n_trimmed} | "
          f"丢弃示例块的题: {n_dropped}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": "5p_fewshot_vote",
            "lora": lora,
            "temperature": temperature,
            "n_prompts": total_prompts,
            "fewshot_used": [
                {
                    "question": ex["question"],
                    "db_id": ex.get("db_id"),
                    "difficulty": ex.get("difficulty"),
                    "join_count": ex.get("join_count"),
                    "response_len": ex.get("response_len", len(ex["response"])),
                }
                for ex in examples
            ],
            "fewshot_source": args.fewshot_file if examples else None,
            "fewshot_trimmed_items": n_trimmed,
            "fewshot_dropped_items": n_dropped,
            "match_rate": rate, "match_count": match_count,
            "start_index": start_index, "limit": limit,
            "elapsed_seconds": round(elapsed, 1),
            "note": (
                "第 6 视角 = build_prompt 的 Instructions 前插入 '## Worked Examples'"
                "(3 条 train 侧执行正确轨迹, Q+<think>+SQL); 其余 5 视角与原版"
                "eval_5p_t10 完全一致(import build_prompt_variants 复用); "
                "6 视角 × 1 采样(默认温度 0 贪心) → 执行分组投票, 口径与 5p 基线一致。"
                "few-shot 示例严禁 dev 轨迹(收集端 dev.json 重叠过滤 + 评估端重叠中止)。"
            ),
        }, f, ensure_ascii=False, indent=2)
    with open(out_dir / "items.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
