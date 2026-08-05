import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = (
    "/gpfs/work/aac/jiahuiwang24/"
    "reasoning_generator_3b/models/Qwen2.5-Coder-3B-Instruct"
)


class ReasoningGeneratorAgent:
    """
    A minimal Reasoning Generator Agent for Text-to-SQL.

    Responsibilities:
    1. Receive a question and DDL schema.
    2. Build a Text-to-SQL reasoning prompt.
    3. Call a local 3B language model.
    4. Extract the final SQL query.
    5. Return a structured result.

    This agent does not:
    1. Train or fine-tune the model.
    2. Execute SQL.
    3. Select between DIN and Reasoning Generator results.
    4. Use Gold SQL during inference.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        max_new_tokens: int = 512,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Run this agent on a GPU node."
            )

        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        print("Loading model...")
        load_start = time.time()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model.eval()

        # We use deterministic greedy decoding.
        # Clear sampling parameters inherited from generation_config.json.
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        self.load_seconds = time.time() - load_start

        print(f"Model loaded in {self.load_seconds:.2f} seconds")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            "GPU memory:",
            f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB",
        )

    @staticmethod
    def format_optional_list(
        values: Optional[List[str]],
    ) -> str:
        if not values:
            return "Not provided"

        return "\n".join(
            f"- {value}"
            for value in values
        )

    def build_prompt(
        self,
        question: str,
        ddl_schema: str,
        schema_links: Optional[List[str]] = None,
        evidence: Optional[str] = None,
        dialect: str = "sqlite",
    ) -> str:
        schema_links_text = self.format_optional_list(schema_links)
        evidence_text = evidence if evidence else "Not provided"

        return f"""
Task Overview:
You are a Text-to-SQL Reasoning Generator Agent.
Given a database schema and a natural-language question,
reason about the required tables, columns, joins, filters,
grouping, aggregation, and ordering. Then generate exactly
one valid SQL query.

Database Dialect:
{dialect}

Database Schema (DDL):
{ddl_schema}

Question:
{question}

Optional Schema Links:
{schema_links_text}

Optional External Evidence:
{evidence_text}

Instructions:
1. Use only tables and columns that appear in the schema.
2. Do not invent database objects.
3. Explain the reasoning before producing the final SQL.
4. Generate exactly one final SQL query.
5. Put the final SQL inside a ```sql code block.
6. Do not use Gold SQL or expected answers.
7. The final query must use the {dialect} dialect.
""".strip()

    @staticmethod
    def extract_sql(response: str) -> Dict[str, Any]:
        sql_match = re.search(
            r"```sql\s*(.*?)```",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if sql_match:
            sql = sql_match.group(1).strip()
            return {
                "sql": sql,
                "parse_success": True,
                "parse_method": "sql_code_block",
            }

        generic_match = re.search(
            r"```\s*(.*?)```",
            response,
            flags=re.DOTALL,
        )

        if generic_match:
            sql = generic_match.group(1).strip()
            return {
                "sql": sql,
                "parse_success": True,
                "parse_method": "generic_code_block",
            }

        plain_match = re.search(
            r"\b(SELECT|WITH)\b.*?;",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if plain_match:
            return {
                "sql": plain_match.group(0).strip(),
                "parse_success": True,
                "parse_method": "plain_sql",
            }

        return {
            "sql": "",
            "parse_success": False,
            "parse_method": "not_found",
        }

    def generate(
        self,
        question: str,
        ddl_schema: str,
        schema_links: Optional[List[str]] = None,
        evidence: Optional[str] = None,
        dialect: str = "sqlite",
        candidate_count: int = 1,
    ) -> Dict[str, Any]:
        if candidate_count != 1:
            raise ValueError(
                "The current prototype supports candidate_count=1 only."
            )

        prompt = self.build_prompt(
            question=question,
            ddl_schema=ddl_schema,
            schema_links=schema_links,
            evidence=evidence,
            dialect=dialect,
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            chat_text,
            return_tensors="pt",
        ).to("cuda:0")

        generation_start = time.time()

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generation_seconds = time.time() - generation_start

        generated_ids = output_ids[0][
            inputs["input_ids"].shape[1]:
        ]

        raw_response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        parsed = self.extract_sql(raw_response)

        candidate = {
            "candidate_id": 0,
            "raw_response": raw_response,
            "sql": parsed["sql"],
            "parse_success": parsed["parse_success"],
            "parse_method": parsed["parse_method"],
        }

        return {
            "model": "Qwen2.5-Coder-3B-Instruct",
            "model_path": self.model_path,
            "input": {
                "question": question,
                "ddl_schema": ddl_schema,
                "schema_links": schema_links or [],
                "evidence": evidence,
                "dialect": dialect,
                "candidate_count": candidate_count,
            },
            "candidates": [candidate],
            "metadata": {
                "model_load_seconds": round(
                    self.load_seconds,
                    2,
                ),
                "generation_seconds": round(
                    generation_seconds,
                    2,
                ),
                "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_gib": round(
                    torch.cuda.memory_allocated(0) / 1024**3,
                    2,
                ),
                "decoding": "greedy",
            },
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run the 3B Reasoning Generator Agent."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to the local model.",
    )

    args = parser.parse_args()

    with open(
        args.input_json,
        "r",
        encoding="utf-8",
    ) as file:
        request = json.load(file)

    agent = ReasoningGeneratorAgent(
        model_path=args.model_path,
    )

    result = agent.generate(
        question=request["question"],
        ddl_schema=request["ddl_schema"],
        schema_links=request.get("schema_links"),
        evidence=request.get("evidence"),
        dialect=request.get("dialect", "sqlite"),
        candidate_count=request.get(
            "candidate_count",
            1,
        ),
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n===== Reasoning Generator result =====")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
