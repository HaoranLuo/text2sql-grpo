#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RetrySQL 全参短 CPT（P1-3 牌一训练）。

配方（RetrySQL arXiv:2507.02529 + 项目适配）：
- 全参持续预训练（LoRA 学不会自纠错，论文实测 r=8~256 全败）
- lr 5e-5 cosine、wd 0.1、batch 128 有效（论文）；本脚本 micro 2 × accum 32 = 64
  （A40 48G 全参 3B 约束），如实记录差异
- [BACK] 注册为特殊 token（单 token）；chat 格式 + LM 全序列训练（CPT 口径）
- 8bit adam（bitsandbytes）以在 48G 上装下全参优化器状态
用法:
    python src/train_retry_cpt.py --data data/retry_cpt_train.json \
        --model-path checkpoints/sft_v3_merged --output checkpoints/retry_cpt \
        --epochs 2 --batch-size 2 --grad-accum 32 --lr 5e-5 --save-steps 50
"""
import argparse
import json
import sys

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="调试：只取前 N 条")
    args = ap.parse_args()

    data = json.load(open(args.data, encoding="utf-8"))
    if args.limit:
        data = data[:args.limit]
    print(f"[retry-cpt] 数据条数: {len(data)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True)
    n_new = tokenizer.add_special_tokens({"additional_special_tokens": ["[BACK]"]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[retry-cpt] 新增特殊 token: {n_new}（[BACK] id={tokenizer.convert_tokens_to_ids('[BACK]')}）")

    def to_text(msgs):
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=False)

    ds = Dataset.from_list([{"text": to_text(e["messages"])} for e in data])

    def tokenize_fn(ex):
        enc = tokenizer(ex["text"], truncation=True, max_length=args.max_length)
        enc["labels"] = enc["input_ids"]  # CPT：全序列 LM（无掩码）
        return enc

    ds = ds.map(tokenize_fn, remove_columns=["text"],
                desc="tokenize", num_proc=1)

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        local_files_only=True, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))
    model.enable_input_require_grads()  # 梯度检查点必需

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_steps=args.save_steps,
        save_total_limit=6,
        bf16=True,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[retry-cpt] done -> {args.output}")


if __name__ == "__main__":
    main()
