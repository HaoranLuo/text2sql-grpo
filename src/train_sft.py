#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) cold-start before GRPO.
Trains the model to imitate DeepSeek API-generated reasoning + SQL.

Usage:
    python src/train_sft.py --data data/api_sft_data.json --output checkpoints/sft_coldstart
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"


def build_sft_dataset(data_path: str) -> Dataset:
    """Convert training data to simple text format.

    Supports three input schemas:
    - {"messages": [{"role": "user"/"assistant", ...}]}  (chat-format, eval-consistent)
    - {"text": "full prompt+completion"}  (pre-built)
    - {"ddl": ..., "question": ..., "api_response": ...}  (legacy API format)
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: List[Dict] = []
    for item in data:
        if "messages" in item:
            records.append({"messages": item["messages"]})
        elif "text" in item:
            text = item["text"]
            records.append({"text": text})
        else:
            prompt = (
                f"You are an expert Text-to-SQL assistant. "
                f"Given a database schema and a question, "
                f"think step-by-step and generate the correct SQLite query.\n\n"
                f"Database Schema:\n{item['ddl']}\n\n"
                f"Question: {item['question']}\n\n"
                f"Answer:"
            )
            completion = item.get("api_response") or item.get("response") or ""
            text = prompt + "\n" + completion
            records.append({"text": text})

    print(f"Built SFT dataset: {len(records)} examples")
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="API-generated data JSON")
    parser.add_argument("--output", required=True, help="Output dir for SFT model")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--eval-split", type=float, default=0.05,
                        help="dev split fraction for early stopping (0 = disable)")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_sft_dataset(args.data)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize dataset (chat messages via apply_chat_template = eval-consistent)
    def tokenize_fn(example):
        if "messages" in example:
            # return_dict=True 必须: 默认返回 list, datasets map 会抛 TypeError
            enc = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=True,
                return_dict=True,
                truncation=True,
                max_length=2048,
            )
            return dict(enc)
        return tokenizer(
            example["text"],
            truncation=True,
            padding=False,
            max_length=2048,
        )

    # P0 修复(监管审查): 只删除实际存在的列, 否则纯 messages 数据会因
    # "Column to remove ['text'] not in the dataset" 直接崩溃
    drop_cols = [c for c in ("messages", "text") if c in dataset.column_names]
    tokenized_dataset = dataset.map(tokenize_fn, remove_columns=drop_cols)

    eval_dataset = None
    if args.eval_split > 0 and len(tokenized_dataset) >= 20:
        split = tokenized_dataset.train_test_split(
            test_size=args.eval_split, seed=42)
        train_dataset, eval_dataset = split["train"], split["test"]
        print(f"Dev split: train={len(train_dataset)} eval={len(eval_dataset)}")
    else:
        train_dataset = tokenized_dataset
        print(f"No dev split ({len(tokenized_dataset)} examples)")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch" if eval_dataset is not None else "no",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        bf16=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print(f"Starting SFT training ({len(dataset)} examples, {args.epochs} epochs)...")
    trainer.train()
    trainer.save_model(args.output)
    print(f"SFT model saved to: {args.output}")


if __name__ == "__main__":
    main()
