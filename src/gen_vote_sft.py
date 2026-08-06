#!/usr/bin/env python3
"""投票自蒸馏：用训练后模型的 3prompt 投票结果生成 SFT 数据

原理（来自我们的投票一致度证据）:
    模型"内部会答"70%+ 的题，但单次生成只能表达 ~50%。
    投票胜出的 SQL = 模型自己能生成的正确答案 → 蒸馏回模型，
    把投票能力压进单次输出。

用法:
    python src/gen_vote_sft.py \
        --lora-path checkpoints/p2a_500/checkpoint-25 \
        --num-train 200 --n-prompts 3 --min-votes 2 \
        --output data/vote_sft_data.json
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from reasoning_generator_agent import ReasoningGeneratorAgent
from spider_utils import SpiderLoader, DatabaseExecutor
from eval_5prompt_agent import build_prompt_variants

BASE_MODEL = str(PROJECT / "models" / "Qwen2.5-Coder-3B-Instruct")
SPIDER = str(PROJECT / "data" / "spider_data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--num-train", type=int, default=200,
                        help="用 Spider train 前 N 条生成投票 SFT 数据")
    parser.add_argument("--n-prompts", type=int, default=3, choices=[3, 5])
    parser.add_argument("--min-votes", type=int, default=2,
                        help="至少几票一致才保留（质量门槛）")
    parser.add_argument("--output", default=str(PROJECT / "data" / "vote_sft_data.json"))
    parser.add_argument("--base-model", default=BASE_MODEL)
    args = parser.parse_args()

    print(f"投票自蒸馏数据生成 | lora={args.lora_path} | train前{args.num_train}条 | {args.n_prompts}p | min_votes={args.min_votes}")

    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                device_map={"": 0}, local_files_only=True,
                                                trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.lora_path)
    model.eval()
    model.config.pad_token_id = tokenizer.eos_token_id

    loader = SpiderLoader(SPIDER)
    executor = DatabaseExecutor(SPIDER)
    with open(Path(SPIDER) / "train_spider.json") as f:
        train = json.load(f)[:args.num_train]

    records = []
    n_voted = 0
    t0 = time.time()

    for i, item in enumerate(train):
        db_id, question, gold_sql = item["db_id"], item["question"], item["query"]
        try:
            ddl, _ = loader.get_ddl_with_source(db_id)
        except RuntimeError:
            continue

        prompts = build_prompt_variants(question, ddl)[:args.n_prompts]
        exec_results = []  # (full_rows_tuple, sql)
        for p in prompts:
            chat = tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                                 tokenize=False, add_generation_prompt=True)
            enc = tokenizer(chat, return_tensors="pt", truncation=True,
                            max_length=1536).to("cuda:0")
            in_len = enc["input_ids"].shape[1]
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
            parsed = ReasoningGeneratorAgent.extract_sql(text)
            if parsed["parse_success"]:
                r = executor.execute(db_id, parsed["sql"])
                if r["success"]:
                    exec_results.append((tuple(tuple(row) for row in r["full_rows"]),
                                         parsed["sql"]))

        # 多数投票：取执行结果一致的胜出 SQL
        if not exec_results:
            continue
        mc = Counter(e[0] for e in exec_results).most_common(1)[0]
        if mc[1] < args.min_votes:
            continue  # 低一致度 = 模型不确定 → 不要
        voted_rows = mc[0]
        voted_sql = next(e[1] for e in exec_results if e[0] == voted_rows)
        n_voted += 1

        # SFT 样本：与训练一致的 prompt 格式 + 投票胜出 SQL
        prompt = ReasoningGeneratorAgent.build_prompt(question=question, ddl_schema=ddl,
                                                      dialect="sqlite")
        completion = "```sql\n" + voted_sql.strip().rstrip(";") + "\n```"
        records.append({"text": prompt + "\n\n" + completion,
                        "db_id": db_id, "votes": mc[1], "gold": gold_sql})

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{args.num_train}] 已收集 {len(records)} 条 (用时 {time.time()-t0:.0f}s)")

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"\n=== 完成 ===")
    print(f"生成 SFT 数据: {len(records)} 条（{args.num_train} 题中 {n_voted} 题有足够一致度）")
    print(f"保存到: {out}")
    print(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
