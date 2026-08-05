import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = (
    "/gpfs/work/aac/jiahuiwang24/"
    "reasoning_generator_3b/models/Qwen2.5-Coder-3B-Instruct"
)

RESULT_PATH = (
    "/gpfs/work/aac/jiahuiwang24/"
    "reasoning_generator_3b/results/smoke_test_result.json"
)


DDL_SCHEMA = """
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    major TEXT
);

CREATE TABLE scores (
    score_id INTEGER PRIMARY KEY,
    student_id INTEGER,
    course TEXT,
    score REAL,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);
""".strip()


QUESTION = (
    "List the names of students who are older than 20 "
    "and major in Computer Science. Order the names alphabetically."
)


PROMPT = f"""
Task Overview:
You are a Text-to-SQL reasoning generator. Given a SQLite database
schema and a natural-language question, reason about the required
tables, columns, filters, and ordering, and then generate one valid
SQLite query.

Database Engine:
SQLite

Database Schema:
{DDL_SCHEMA}

Question:
{QUESTION}

Instructions:
1. Understand which tables and columns are required.
2. Do not invent tables or columns.
3. Generate exactly one SQL query.
4. Put the final SQL query inside a ```sql code block.
5. Do not include multiple alternative SQL queries.
""".strip()


def extract_sql(text: str) -> str:
    """Extract SQL from a Markdown SQL code block."""
    match = re.search(r"```sql\s*(.*?)```", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*(.*?)```", text, flags=re.S)
    if match:
        return match.group(1).strip()

    return text.strip()


def main():
    print("Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    print("Step 2: Loading 3B model onto the A40...")
    start_load = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()

    load_seconds = time.time() - start_load
    print(f"Model loaded in {load_seconds:.2f} seconds")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "GPU memory after loading:",
        f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB",
    )

    messages = [{"role": "user", "content": PROMPT}]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
    ).to("cuda:0")

    print("Step 3: Generating SQL...")
    start_generate = time.time()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generation_seconds = time.time() - start_generate

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    extracted_sql = extract_sql(raw_response)

    result = {
        "model": "Qwen2.5-Coder-3B-Instruct",
        "model_path": MODEL_PATH,
        "question": QUESTION,
        "ddl_schema": DDL_SCHEMA,
        "raw_response": raw_response,
        "extracted_sql": extracted_sql,
        "model_load_seconds": round(load_seconds, 2),
        "generation_seconds": round(generation_seconds, 2),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": round(
            torch.cuda.memory_allocated(0) / 1024**3,
            2,
        ),
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\n===== Raw model response =====")
    print(raw_response)

    print("\n===== Extracted SQL =====")
    print(extracted_sql)

    print(f"\nGeneration time: {generation_seconds:.2f} seconds")
    print(f"Result saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
