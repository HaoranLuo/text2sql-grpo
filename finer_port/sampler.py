"""
P1: n 候选采样器（vav 投票评估的采样端，HF transformers 路径）。

对应 PLAN §3.1.2：替代现有贪心 candidate_count=1——
    model.generate(do_sample=True, temperature=1.0, top_p=1.0,
                   num_return_sequences=n, max_new_tokens=2048)
单次 forward 产 n 条候选，逐条独立 extract_sql。

要点：
  - 默认 prompt 复用 src/reasoning_generator_agent.py 的 ReasoningGeneratorAgent.build_prompt
    （canonical prompt）；`prompt_style="finer"` 时用 FINER 系统提示模板（P1 对照臂：
    FINER 权重用其训练分布的模板更贴近官方 85.0% 口径）。
  - 提取双兼容：响应含 `</think>` 时取 `rfind('</think>')` 后内容（FINER 权重格式）；
    否则走 ReasoningGeneratorAgent.extract_sql（```sql 块 / plain SQL）。
  - 可选 LoRA（--lora-path），加载方式与 ReasoningGeneratorAgent 一致。

注意：本模块需要 GPU（构造时校验 CUDA）；纯逻辑自测请跑 vav_voting.py。
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    _PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PEFT_AVAILABLE = False

# 复用项目 agent 的 prompt / 提取逻辑（sys.path 引导）
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from reasoning_generator_agent import ReasoningGeneratorAgent  # noqa: E402

# ---------------------------------------------------------------------------
# FINER 系统提示模板（PLAN §4.2，与 FINER-SQL 训练分布一致）
# ---------------------------------------------------------------------------
FINER_SYSTEM_PROMPT = (
    "You are a meticulous SQL expert. Generate a single, correct SQL query "
    "for the user question and the provided database schema. Rules:\n"
    "- Output exactly one SQL statement.\n"
    "- The SQL must be executable on SQLite.\n"
    "- Do not include any explanatory text.\n"
    "- Output one SQL statement only. Do not include any extra text, tags, "
    "or code fences."
)
FINER_USER_PROMPT = "Database Schema:\n{ddl}\nQuestion: {question}"


class VavSampler:
    """
    n 候选采样器：加载模型（可选 LoRA），按 prompt 风格构造对话，
    单次 forward 采样 n 条并逐条提取 SQL。
    """

    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        prompt_style: str = "default",
        max_prompt_tokens: int = 1536,
        seed: int = 0,
        local_files_only: bool = True,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Run the vav evaluator on a GPU node."
            )
        if prompt_style not in ("default", "finer"):
            raise ValueError(f"Unknown --prompt-style: {prompt_style!r}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.model_path = model_path
        self.lora_path = lora_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.prompt_style = prompt_style
        self.max_prompt_tokens = max_prompt_tokens

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        # CRITICAL: unify pad/eos（Qwen 默认 pad=<|endoftext|> != eos=<|im_end|>）
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._orig_padding_side = self.tokenizer.padding_side

        print("Loading model...")
        load_start = time.time()
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        # CRITICAL: model.config.pad_token_id 保留旧 pad（151643）也在 eos_token_id 里，
        # 不覆盖会导致 padding 被当作 EOS。
        base_model.config.pad_token_id = self.tokenizer.eos_token_id

        if lora_path is not None:
            if not _PEFT_AVAILABLE:
                raise ImportError(
                    "peft is required to load a LoRA adapter. "
                    "Install it with: pip install peft"
                )
            print(f"Loading LoRA adapter from: {lora_path}")
            self.model = PeftModel.from_pretrained(base_model, lora_path)
            print("LoRA adapter loaded (unmerged).")
        else:
            self.model = base_model

        self.model.eval()
        self.load_seconds = time.time() - load_start
        print(f"Model loaded in {self.load_seconds:.2f} seconds")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            "GPU memory:",
            f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GiB",
        )

    # ------------------------------------------------------------------
    # Prompt 构造
    # ------------------------------------------------------------------
    def build_prompt(self, question: str, ddl_schema: str) -> str:
        """纯文本 prompt（default 风格 = 项目 canonical prompt）。"""
        return ReasoningGeneratorAgent.build_prompt(
            question=question,
            ddl_schema=ddl_schema,
            schema_links=None,
            evidence=None,
            dialect="sqlite",
        )

    def build_chat_text(self, question: str, ddl_schema: str) -> str:
        """按 prompt_style 构造对话并应用 chat template（含 generation prompt）。"""
        if self.prompt_style == "finer":
            messages = [
                {"role": "system", "content": FINER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": FINER_USER_PROMPT.format(
                        ddl=ddl_schema, question=question
                    ),
                },
            ]
        else:
            messages = [
                {"role": "user", "content": self.build_prompt(question, ddl_schema)}
            ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------------
    # SQL 提取（双兼容：<think> 标签 / 现有 ```sql 块逻辑）
    # ------------------------------------------------------------------
    @staticmethod
    def extract_sql(response: str) -> Dict[str, Any]:
        """
        提取 SQL：
          - 响应含 `</think>`（FINER 权重格式）：取 rfind('</think>') 后内容，
            优先走现有 extract_sql；若提取失败则整段视为 SQL。
          - 无标签：走 ReasoningGeneratorAgent.extract_sql（```sql 块 / plain）。
        """
        if "</think>" in response:
            tail = response[response.rfind("</think>") + len("</think>"):].strip()
            if tail:
                parsed = ReasoningGeneratorAgent.extract_sql(tail)
                if parsed["parse_success"]:
                    return parsed
                return {
                    "sql": tail,
                    "parse_success": True,
                    "parse_method": "after_think_tag",
                }
        return ReasoningGeneratorAgent.extract_sql(response)

    # ------------------------------------------------------------------
    # n 候选采样
    # ------------------------------------------------------------------
    def _generate(self, encodings: Dict[str, torch.Tensor], n: int) -> torch.Tensor:
        with torch.inference_mode():
            return self.model.generate(
                **encodings,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.eos_token_id,
            )

    @staticmethod
    def _decode_candidates(
        tokenizer: AutoTokenizer,
        output_ids: torch.Tensor,
        n: int,
        padded_input_len: int,
    ) -> List[Dict[str, Any]]:
        """把一次 forward 的 [B*n, T] 输出解码成逐条候选（含独立 SQL 提取）。"""
        candidates: List[Dict[str, Any]] = []
        for k in range(n):
            generated_ids = output_ids[k][padded_input_len:]
            raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            parsed = VavSampler.extract_sql(raw_response)
            candidates.append(
                {
                    "candidate_id": k,
                    "raw_response": raw_response,
                    "sql": parsed["sql"],
                    "parse_success": parsed["parse_success"],
                    "parse_method": parsed["parse_method"],
                }
            )
        return candidates

    def sample_batch(
        self,
        chat_texts: List[str],
        n: int,
    ) -> List[List[Dict[str, Any]]]:
        """
        对一批 prompt 各采样 n 条候选（一次 forward 产 B*n 条）。

        chat_texts: build_chat_text 的输出列表
        n: 每个 prompt 的候选数（num_return_sequences）
        返回: [[候选 dict, ...] × n, ...]（与输入顺序一致）
        """
        if not chat_texts:
            return []
        if n < 1:
            raise ValueError("n must be >= 1")

        self.tokenizer.padding_side = "left"
        try:
            encodings = self.tokenizer(
                chat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_prompt_tokens,
            ).to("cuda:0")
        finally:
            self.tokenizer.padding_side = self._orig_padding_side

        output_ids = self._generate(encodings, n)
        padded_input_len = encodings["input_ids"].shape[1]
        batch_size = len(chat_texts)

        results: List[List[Dict[str, Any]]] = []
        for i in range(batch_size):
            # num_return_sequences 时输出按输入展开：输入 i 的 n 条在行 [i*n, (i+1)*n)
            rows = output_ids[i * n:(i + 1) * n]
            results.append(
                self._decode_candidates(
                    self.tokenizer, rows, n, padded_input_len
                )
            )
        return results

    def sample(self, chat_text: str, n: int) -> List[Dict[str, Any]]:
        """单 prompt 采样 n 条候选。"""
        return self.sample_batch([chat_text], n)[0]
