#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/train_orm.py — T2 自训 ORM（GradeSQL 式 outcome reward model）训练 + dev 评估。

设计决策：生成式 Yes/No（自回归二分类），而非 num_labels=2 序列分类头。依据：
  1. GradeSQL（arXiv:2509.01308，ACL 2026）原版做法：ORM = 与 SQL 生成对齐的基座
     LLM 经 LoRA 微调的自回归二分类器，输出 Yes/No，测试时以
     P("Yes") / (P("Yes") + P("No")) 打分排序（ORM-BoN：Spider +2.10pp vs ex-BoN、
     +0.93pp vs Majority Vote；BIRD +4.33/+2.91）。本实现沿用该方案。
  2. GenRM（DeepMind，"Generative Verifiers: Reward Modeling as Next-Token
     Prediction"，arXiv:2410.12832）：把验证建模为 next-token 预测比判别式分类头
     更好地复用预训练先验，且在 1.5-3B 小模型上收益最显著——与本项目 3B 规模吻合。
  3. 判别式分类头需随机初始化 2×hidden 线性层 + 池化策略（末 token/mean）取舍，
     对 decoder-only 基座丢弃了生成分布对齐；生成式的训练目标即因果 LM 损失，
     prompt 构造（canonical build_prompt + chat template）与生成端/推理端完全一致。
  4. 推理成本：Qwen2.5 词表 "Yes"/"No" 均为单 token（已核实 id 9454/2753），打分
     只需一次前向，与分类头同价，无需自回归解码。

训练数据：data/orm_train.json（src/label_orm_data.py 产物：chat 格式 + label）。
dev 划分：按【题】切（question-level split）——rank 评估必须题目不相交，否则同题
样本泄漏导致虚高。dev 上报告（compute_metrics + 训练后权威重算并存 JSON）：
  - acc                       分类准确率（P(Yes)>0.5）
  - rank_acc                  ORM 每题 argmax P(Yes) 候选正确率
  - rank_acc_recoverable      有正确候选的题（平票/败局时 ORM 有无可能救回）
  - rank_acc_maj_wrong        arm_maj 败局题子集（裁决器已知答案，ORM 能否选对）
    （需 data/orm_questions.json 含 maj_correct 字段；无该字段则此项不报）

数据卫生：样本不含 gold_sql；label_orm_data.py 已整题剔除 gold 执行失败/无实例题。

用法
  # 训练（3 epochs, LoRA r=32/alpha=64, lr 1e-5, question-level dev 5%）
  envs/reasoning3b/bin/python src/train_orm.py \
      --data data/orm_train.json --questions data/orm_questions.json \
      --output checkpoints/orm_b1 \
      --model-path models/Qwen2.5-Coder-3B-Instruct
  # 只评估已有适配器（不训练）
  envs/reasoning3b/bin/python src/train_orm.py \
      --data data/orm_train.json --questions data/orm_questions.json \
      --output checkpoints/orm_b1 --eval-only
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "Qwen2.5-Coder-3B-Instruct"

YES_STR, NO_STR = "Yes", "No"


# ===================================================================
# 数据集
# ===================================================================


def load_data(data_path: str, questions_path: Optional[str]) -> \
        (List[Dict[str, Any]], Dict[int, Dict[str, Any]]):
    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    questions: Dict[int, Dict[str, Any]] = {}
    if questions_path and Path(questions_path).exists():
        with open(questions_path, "r", encoding="utf-8") as f:
            for q in json.load(f):
                questions[int(q["question_id"])] = q
    return samples, questions


def split_by_question(samples: List[Dict[str, Any]], dev_frac: float,
                      dev_questions: Optional[int], seed: int) -> \
        (List[Dict[str, Any]], List[Dict[str, Any]]):
    """question-level split：同题全部候选必须在同一侧（否则同题样本泄漏）。"""
    qids = sorted({int(s["question_id"]) for s in samples})
    rng = random.Random(seed)
    rng.shuffle(qids)
    k = dev_questions if dev_questions is not None else max(1, int(round(len(qids) * dev_frac)))
    k = min(k, len(qids) - 1) if len(qids) > 1 else k
    dev_qids = set(qids[:k])
    train = [s for s in samples if int(s["question_id"]) not in dev_qids]
    dev = [s for s in samples if int(s["question_id"]) in dev_qids]
    return train, dev, dev_qids


def build_dataset(samples: List[Dict[str, Any]], tokenizer, max_length: int,
                  for_train: bool) -> Dataset:
    """训练：user+gen_prompt 后接答案 token（labels 仅监督答案）；
    打分/评估：只到 gen_prompt 结尾（labels 占位 = input_ids，供 collator）。
    注意：map 后必须移除标量「label」列——DataCollatorForSeq2Seq 见到 label 键
    会优先把它当 label 序列（而非 labels），与 int 类别列冲突。"""
    records = [{
        "prompt": s["messages"][0]["content"],
        "label": int(s["label"]),
        "question_id": int(s["question_id"]),
        "is_correct": int(s["label"]),
    } for s in samples]
    ds = Dataset.from_list(records)

    def tokenize_train(example):
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=True, add_generation_prompt=True, return_dict=True)
        ans = YES_STR if example["label"] == 1 else NO_STR
        ans_ids = tokenizer.encode(ans, add_special_tokens=False)
        input_ids = enc["input_ids"] + ans_ids
        labels = [-100] * len(enc["input_ids"]) + ans_ids
        if len(input_ids) > max_length:
            cut = len(input_ids) - max_length  # 左截断：保住候选 SQL 与答案
            input_ids = input_ids[cut:]
            labels = labels[cut:]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
                "labels": labels}

    def tokenize_score(example):
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=True, add_generation_prompt=True, return_dict=True)
        input_ids = enc["input_ids"]
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]  # 左截断：保住候选 SQL 与指令
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
                "labels": list(input_ids)}

    fn = tokenize_train if for_train else tokenize_score
    return ds.map(fn, remove_columns=["prompt", "label"])


# ===================================================================
# 打分与评估指标
# ===================================================================


def _p_yes_from_logits(logits, yes_id: int, no_id: int) -> np.ndarray:
    """P(Yes) = softmax(Yes, No)。logits: (n, V) 末位置 logits。"""
    sub = np.asarray(logits[:, [yes_id, no_id]], dtype=np.float32)
    # 数值稳定：sigmoid(logit_yes - logit_no)
    return 1.0 / (1.0 + np.exp(sub[:, 1] - sub[:, 0]))


def ranking_metrics(p_yes: np.ndarray, is_correct: np.ndarray,
                    qids: List[int], questions: Dict[int, Dict[str, Any]],
                    prefix: str = "") -> Dict[str, Any]:
    """分类 acc + 按题 argmax 排名指标（含败局题子集）。"""
    n = len(p_yes)
    acc = float(np.mean((p_yes > 0.5).astype(int) == is_correct))
    by_q: Dict[int, List[tuple]] = defaultdict(list)
    for p, c, q in zip(p_yes, is_correct, qids):
        by_q[int(q)].append((float(p), int(c)))
    n_questions = len(by_q)
    rank_correct = rec_correct = maj_wrong_ok = ties = 0
    n_recoverable = n_maj_wrong = 0
    for q, rows in by_q.items():
        rows_sorted = sorted(rows, key=lambda r: -r[0])
        best_ok = bool(rows_sorted[0][1])
        rank_correct += int(best_ok)
        if any(r[1] for r in rows):
            n_recoverable += 1
            rec_correct += int(best_ok)
        if len(rows_sorted) >= 2 and (rows_sorted[0][0] - rows_sorted[1][0]) < 0.1:
            ties += 1
        m = questions.get(q) or {}
        if m.get("maj_correct") is False:
            n_maj_wrong += 1
            maj_wrong_ok += int(best_ok)
    out: Dict[str, Any] = {
        prefix + "acc": round(acc, 4),
        prefix + "rank_acc": round(rank_correct / n_questions, 4) if n_questions else None,
        prefix + "rank_acc_recoverable": round(rec_correct / n_recoverable, 4)
        if n_recoverable else None,
        prefix + "n_questions": n_questions,
        prefix + "n_recoverable": n_recoverable,
        prefix + "tie_frac": round(ties / n_questions, 4) if n_questions else None,
    }
    if n_maj_wrong:
        out[prefix + "rank_acc_maj_wrong"] = round(maj_wrong_ok / n_maj_wrong, 4)
        out[prefix + "n_maj_wrong"] = n_maj_wrong
    return out


def make_compute_metrics(tokenizer, eval_ds: Dataset,
                         questions: Dict[int, Dict[str, Any]]):
    yes_ids = tokenizer.encode(YES_STR, add_special_tokens=False)
    no_ids = tokenizer.encode(NO_STR, add_special_tokens=False)
    assert len(yes_ids) == 1 and len(no_ids) == 1, \
        f"Yes/No 必须单 token，实际 {yes_ids}/{no_ids}"
    yes_id, no_id = yes_ids[0], no_ids[0]
    qids = [int(x) for x in eval_ds["question_id"]]
    is_correct = np.asarray(eval_ds["is_correct"], dtype=int)

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        p_yes = _p_yes_from_logits(logits, yes_id, no_id)
        return ranking_metrics(p_yes, is_correct, qids, questions, prefix="eval_")
    return compute_metrics


# ===================================================================
# Trainer：类别权重 + 只保留末位置 logits（left padding 下 = 打分位置）
# ===================================================================


class OrmTrainer(Trainer):
    """compute_loss：answer token 上带 pos_weight 的 CE（prompt 全部 ignore）。
    prediction_step：截取每行末位置 logits 交给 compute_metrics，避免
    (n × seq × vocab) 整张 logits 内存爆炸（3B × 152k 词表 × 2048 seq）。"""

    def __init__(self, pos_weight: float = 1.0, yes_id: Optional[int] = None,
                 no_id: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = float(pos_weight)
        self.yes_id = yes_id
        self.no_id = no_id

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # 权重必须按词表 id 给 Yes/No 两个 token（CE 的 weight 是 class-indexed，
        # [1.0, pos_weight] 只会加权词表第 0/1 个 token，与 Yes/No 无关）。
        if self.yes_id is not None and self.no_id is not None:
            weight = torch.ones(logits.size(-1), dtype=logits.dtype,
                                device=logits.device)
            weight[self.yes_id] = self.pos_weight
            weight[self.no_id] = 1.0
        else:
            weight = None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, logits, labels = super().prediction_step(
            model, inputs, True, ignore_keys)
        if logits is None:
            # prediction_loss_only 路径（Trainer 终轮 eval）不返回 logits
            return (loss, None, None)
        last = logits[:, -1:, :] if logits.dim() == 3 else logits
        return (loss, last, None)


# ===================================================================
# 训练后权威评估（自有前向循环，与 compute_metrics 相互印证）
# ===================================================================


@torch.no_grad()
def evaluate_dev(model, tokenizer, dev_ds: Dataset,
                 questions: Dict[int, Dict[str, Any]], batch_size: int,
                 yes_id: int, no_id: int, device: torch.device) -> Dict[str, Any]:
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True,
                                      label_pad_token_id=-100)
    p_list: List[float] = []
    c_list: List[int] = []
    q_list: List[int] = []
    for i in range(0, len(dev_ds), batch_size):
        n = min(batch_size, len(dev_ds) - i)
        # 按索引取行（此 datasets 版本迭代切片会产出列名字符串，见 2095363 教训）
        features = [{k: dev_ds[i + j][k]
                     for k in ("input_ids", "attention_mask", "labels")}
                    for j in range(n)]
        batch_in = collator(features)
        out = model(input_ids=batch_in["input_ids"].to(device),
                    attention_mask=batch_in["attention_mask"].to(device),
                    use_cache=False)
        logits = out.logits[:, -1, :].float()  # left padding → 末位 = 打分位
        p_yes = _p_yes_from_logits(logits.cpu().numpy(), yes_id, no_id)
        p_list.extend(float(x) for x in p_yes)
        c_list.extend(int(dev_ds[i + j]["is_correct"]) for j in range(n))
        q_list.extend(int(dev_ds[i + j]["question_id"]) for j in range(n))
    return ranking_metrics(np.asarray(p_list), np.asarray(c_list, dtype=int),
                           q_list, questions)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="T2 自训 ORM（GradeSQL 式 Yes/No）")
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data" / "orm_train.json"))
    ap.add_argument("--questions",
                    default=str(PROJECT_ROOT / "data" / "orm_questions.json"))
    ap.add_argument("--output", default=str(PROJECT_ROOT / "checkpoints" / "orm_b1"))
    ap.add_argument("--model-path", default=str(MODEL_PATH))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--dev-frac", type=float, default=0.05,
                    help="question-level dev 比例（按题切，防同题泄漏）")
    ap.add_argument("--dev-questions", type=int, default=None,
                    help="dev 题目数（覆盖 --dev-frac）")
    ap.add_argument("--pos-weight", type=float, default=1.0,
                    help="正类（correct）CE 权重；<=0 时自动取 neg/pos（训练集统计）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-only", action="store_true",
                    help="只加载 --output 适配器做 dev 评估（不训练）")
    args = ap.parse_args(argv)

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 末位 = 打分位置，eval/训练统一

    samples, questions = load_data(args.data, args.questions)
    train_samples, dev_samples, dev_qids = split_by_question(
        samples, args.dev_frac, args.dev_questions, args.seed)
    print(f"[orm] 样本 {len(samples)} | train {len(train_samples)} "
          f"| dev {len(dev_samples)}（{len(dev_qids)} 题，题目不相交）")
    tp = sum(1 for s in train_samples if s["label"] == 1)
    pos_weight = args.pos_weight
    if pos_weight <= 0:
        pos_weight = round((len(train_samples) - tp) / max(1, tp), 3)
    print(f"[orm] train 正负比 {tp}/{len(train_samples) - tp} = "
          f"{tp / max(1, len(train_samples) - tp):.3f}（pos_weight={pos_weight}）")

    dev_ds = build_dataset(dev_samples, tokenizer, args.max_length, for_train=False)

    yes_ids = tokenizer.encode(YES_STR, add_special_tokens=False)
    no_ids = tokenizer.encode(NO_STR, add_special_tokens=False)
    assert len(yes_ids) == 1 and len(no_ids) == 1, \
        f"Yes/No 必须单 token，实际 {yes_ids}/{no_ids}"
    yes_id, no_id = yes_ids[0], no_ids[0]

    # ---- 仅评估模式 ----
    if args.eval_only:
        print(f"[orm] eval-only: 加载适配器 {args.output}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
            local_files_only=True, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, args.output)
        model = model.merge_and_unload()
        model.eval()
        metrics = evaluate_dev(model, tokenizer, dev_ds, questions,
                               args.batch_size * 4, yes_id, no_id, device)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "eval_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0

    # ---- 训练模式 ----
    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        local_files_only=True, trust_remote_code=True)
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = build_dataset(train_samples, tokenizer, args.max_length,
                             for_train=True)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,  # rank 指标由训练后 evaluate_dev 权威评估; epoch-eval 只产 eval_loss
        # metric_for_best_model="eval_rank_acc",  # 已禁用: epoch-eval 无此指标(KeyError 教训)
        save_total_limit=2,
        bf16=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
        seed=args.seed,
    )

    trainer = OrmTrainer(
        pos_weight=pos_weight,
        yes_id=yes_id,
        no_id=no_id,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True,
                                             label_pad_token_id=-100),
        compute_metrics=make_compute_metrics(tokenizer, dev_ds, questions),
        tokenizer=tokenizer,
    )

    print(f"[orm] 开始训练（{len(train_ds)} 样本 × {args.epochs} epochs, "
          f"有效 batch {args.batch_size * args.grad_accum}）")
    trainer.train()
    trainer.save_model(args.output)

    # 训练后权威重算（自有前向循环）+ 落盘
    metrics = evaluate_dev(model, tokenizer, dev_ds, questions,
                           args.batch_size * 4, yes_id, no_id, device)
    (Path(args.output) / "eval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== dev 权威指标（{len(dev_samples)} 样本 / {len(dev_qids)} 题）===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"ORM saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
