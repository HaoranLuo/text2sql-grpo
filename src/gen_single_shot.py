#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单次贪心生成评估（Spider dev，官方 EX 口径）——RetrySQL GO/NO-GO 门统一脚本。

- 模型：plain HF 目录（合并后 sft_v3 或 CPT checkpoint），vLLM 贪心 T=0 单样本
- prompt = 候选池生成同款 canonical prompt（ReasoningGeneratorAgent.build_prompt，
  schema_links=None/evidence=None/dialect=sqlite），截断 max_length=1536
- 解析 = finer_port.sampler.VavSampler.extract_sql（与池生成口径一致）
- 输出 items.json（dataset_index/db_id/question/predicted_sql）→ eval_official.sh 兼容
用法:
    python src/gen_single_shot.py --model-path checkpoints/sft_v3_merged \
        --output-dir outputs/single_sft_v3 --limit 1034
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from reasoning_generator_agent import ReasoningGeneratorAgent  # noqa: E402
from spider_utils import SpiderLoader, DatabaseExecutor  # noqa: E402
from finer_port.sampler import VavSampler  # noqa: E402

SPIDER = str(PROJECT / "data" / "spider_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=1034)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--exec-workers", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True)
    loader = SpiderLoader(SPIDER)
    items = loader.load_dev(limit=args.limit)

    # 构造 prompt
    per_item = []
    for it in items:
        db_id, question = it["db_id"], it["question"]
        ddl, _ = loader.get_ddl_with_source(db_id)
        p = ReasoningGeneratorAgent.build_prompt(
            question=question, ddl_schema=ddl, schema_links=None,
            evidence=None, dialect="sqlite")
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True)
        ids = tokenizer(chat, truncation=True, max_length=1536)["input_ids"]
        per_item.append((it, ids))

    llm = LLM(model=args.model_path, dtype="bfloat16",
              gpu_memory_utilization=0.9, max_model_len=4096,
              trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)

    prompt_ids = [ids for _, ids in per_item]
    outputs = llm.generate(prompt_token_ids=prompt_ids, sampling_params=sp)
    texts = [o.outputs[0].text for o in outputs]

    # 解析 SQL + 执行（线程池）
    executor = DatabaseExecutor(SPIDER)

    def process(job):
        it, text = job
        sql = VavSampler.extract_sql(text)
        if not sql:
            return it, "", False, None
        try:
            rows = executor.execute(it["db_id"], sql)
            ok = True
        except Exception as e:  # 执行异常按空结果处理，不计入 parse 失败
            rows, ok = None, False
        return it, sql, True, rows

    results = []
    with ThreadPoolExecutor(max_workers=args.exec_workers) as pool:
        for r in pool.map(process, zip([it for it, _ in per_item], texts)):
            results.append(r)

    out_items = []
    n_parse_ok = 0
    for it, sql, parsed, _rows in results:
        if parsed:
            n_parse_ok += 1
        out_items.append({
            "dataset_index": it["dataset_index"],
            "db_id": it["db_id"],
            "question": it["question"],
            "predicted_sql": sql,
        })
    (out_dir / "items.json").write_text(
        json.dumps(out_items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[single-shot] {len(out_items)} 题 | parse 成功 {n_parse_ok} "
          f"({n_parse_ok / max(1, len(out_items)):.3f})")
    print(f"[single-shot] items -> {out_dir / 'items.json'}（接 eval_official.sh）")


if __name__ == "__main__":
    main()
