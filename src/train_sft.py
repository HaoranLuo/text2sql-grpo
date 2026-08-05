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
MODEL_PATH = PROJECT_ROOT / "models" / "qwen2.5-coder-7b-instruct"


def build_sft_dataset(data_path: str) -> Dataset:
    """Convert training data to simple text format.

    Supports two input schemas:
    - {"text": "full prompt+completion"}  (pre-built)
    - {"ddl": ..., "question": ..., "api_response": ...}  (legacy API format)
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: List[Dict] = []
    for item in data:
        if "text" in item:
            text = item["text"]
        else:
            prompt = (
                f"You are an expert Text-to-SQL assistant. "
                f"Given a database schema and a question, "
                f"think step-by-step and generate the correct SQLite query.\n\n"
                f"Database Schema:\n{item['ddl']}\n\n"
                f"Question: {item['question']}\n\n"
                f"Answer:"
            )
            completion = item["api_response"]
            text = prompt + "\n" + completion
        records.append({"text": text})

    print(f"Built SFT dataset: {len(records)} examples")
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="API-generated data JSON")
    parser.add_argument("--output", required=True, help="Output dir for SFT model")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
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
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize dataset
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=2048,
        )

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

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
        save_strategy="no",
        bf16=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print(f"Starting SFT training ({len(dataset)} examples, {args.epochs} epochs)...")
    trainer.train()
    trainer.save_model(args.output)
    print(f"SFT model saved to: {args.output}")


if __name__ == "__main__":
    main()
