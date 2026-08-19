#!/usr/bin/env python3
"""
P1#4 FINER 完整配方 GRPO 训练 driver (新文件; 不改 src/train_reasoning_grpo.py).

与 train_reasoning_grpo.main() 保持逐行一致, 仅做三件事:
  A) 新增 FINER 配方必需、而原脚本未暴露的 GRPOConfig 参数:
       --max-grad-norm      TRL 默认 1.0  → FINER 配方 0.1
       --warmup-ratio       TRL GRPOConfig 默认 0.0 → FINER 配方 0.1 (已核实 0.17.0)
       --lr-scheduler-type  TRL 默认 linear → GRPO 论文配方 cosine
       --save-steps         T2.x 硬编码 25 → 本配方 200 (checkpoint 早停粒度)
  B) 默认值切换为 FINER 配方: G=16 / lr=4e-6 / beta=0.01 / T=1.0 / 2000 步。
     注意: per_device_train_batch_size 默认值从原脚本的 G*4 改为 4 (4 提示/步)。
     原公式在 G=16 时 = 64 提示 × 16 = 1024 条完成/步, A40 必 OOM;
     本默认 4 提示 × G16 = 64 条完成/步 == T23 实测足迹 (~59.5s/步, 无 OOM)。
  C) 新增中间早评保护 MidTrainEvalCallback (防 loss=0 空转态 / 越练越差):
       --mid-eval-every (默认 400, 0=关闭) / --mid-eval-limit (默认 100) /
       --early-stop-flat (默认开启)。每 400 步用当前 LoRA 权重对 Spider dev
       前 100 条做贪心快速评估 (复用 evaluate_after_grpo 同口径 match 判定),
       mid_eval_match_rate 打印训练日志并追加写 outputs/mid_eval_results.jsonl
       (供外部守望进程读取); 连续 2 次 ≤ 首次成绩 - 0.5pp 且步数 ≥ 800 → 提前停止。

数据集构建 / 奖励函数 / RewardStdGuard / pad-eos 修复全部复用原模块,
preflight_check.sh 针对 train_reasoning_grpo.py 的静态检查因此仍然有效。
四分量奖励说明: format(解析失败=0 已内建) + exec(compare_execution_results)
+ atomic(sqlglot Jaccard, reward_type=finer 分支 0.5+0.5×atomic) 三份量可用;
memory 分量(教师轨迹 embedding 语义对齐)在 src/ 无实现、HPC 未装
chromadb/sentence-transformers, 已降级为三份量, 影响见提交报告。
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, PeftModel
from trl import GRPOConfig, GRPOTrainer

import train_reasoning_grpo as base
from train_reasoning_grpo import (
    build_dataset,
    create_reward_function,
    RewardStdGuardCallback,
)
from spider_utils import (
    SpiderLoader, DatabaseExecutor, compare_execution_results,
)
from reasoning_generator_agent import ReasoningGeneratorAgent


# ===================================================================
# P1#4 中间早评 (mid-train quick eval) + flat 早停保护
# -------------------------------------------------------------------
# 背景: T2.x 出现过 loss 全程为 0 但权重仍在缓慢更新的"空转态" ——
# 单看 loss/reward_std 不可靠, 必须中途用真实评估 (dev match_rate) 验证
# 训练是否有效; flat 早停则防止"越练越差"白烧 GPU。
#
# 耗时依据: evaluate_after_grpo 历史 1034 题 ≈ 61min (≈3.5s/题) →
# 100 题 ≈ 6min, 加 DDL 加载 / SQLite 执行开销预算 6-10min/次;
# 默认配方 5 次 (400/800/1200/1600/2000) ≈ 30-50min, 相对
# 2000 步 × ~60s/步 ≈ 33h 训练, 开销 < 3% (同步阻塞训练, 可接受)。
# ===================================================================
MID_EVAL_BATCH_SIZE = 8        # 与 evaluate_after_grpo 默认一致 (A40 48G 实测可容纳)
MID_EVAL_MAX_NEW_TOKENS = 512  # 必须与训练 max_completion_length / 推理口径一致
MID_EVAL_MAX_PROMPT_LEN = 1536 # 与训练 max_prompt_length 一致
MID_EVAL_LOG_NAME = "mid_eval_results.jsonl"  # outputs/ 下, 供外部守望进程读取


def _run_mid_eval(
    model,
    tokenizer,
    spider_dir: str,
    limit: int,
    batch_size: int = MID_EVAL_BATCH_SIZE,
) -> Dict[str, Any]:
    """用当前(训练中)权重对 Spider dev 前 limit 条做贪心快速评估。

    判定口径与 src/evaluate_after_grpo.py::_evaluate_one 的
    custom_execution_match 完全一致: pred/gold 均执行成功、均未截断,
    且 compare_execution_results (ORDER BY 感知) 判 match 才计数。
    生成协议与 ReasoningGeneratorAgent.generate_batch 一致:
    chat 模板 + 左 padding + 截断 1536 + greedy(max_new_tokens=512)。

    返回 {"match_rate": 百分点(52.0=52%), "match_count", "total",
          "wall_seconds"}。评估期间 model 置 eval/use_cache=True,
    finally 恢复训练状态, 不污染训练循环。
    """
    loader = SpiderLoader(spider_dir)
    executor = DatabaseExecutor(spider_dir)
    items = loader.load_dev(limit=limit)

    eval_rows: List[Dict[str, Any]] = []
    for item in items:
        try:
            ddl, _source = loader.get_ddl_with_source(item["db_id"])
        except RuntimeError:
            continue  # DDL 缺失极罕见; 不进分母, 与评估脚本口径一致
        prompt = ReasoningGeneratorAgent.build_prompt(
            question=item["question"],
            ddl_schema=ddl,
            schema_links=None,
            evidence=None,
            dialect="sqlite",
        )
        eval_rows.append({
            "messages": [{"role": "user", "content": prompt}],
            "db_id": item["db_id"],
            "gold_sql": item["query"],
        })

    total = len(eval_rows)
    if total == 0:
        return {"match_rate": 0.0, "match_count": 0, "total": 0,
                "wall_seconds": 0.0}

    device = next(model.parameters()).device
    original_padding_side = tokenizer.padding_side
    was_training = model.training
    original_use_cache = getattr(model.config, "use_cache", None)

    # 训练期因 gradient checkpointing 被置 use_cache=False; 评估期临时打开加速生成
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"  # 生成必须左 padding (同 generate_batch)

    match_count = 0
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            for bstart in range(0, total, batch_size):
                batch = eval_rows[bstart:bstart + batch_size]
                chat_texts = [
                    tokenizer.apply_chat_template(
                        r["messages"], tokenize=False,
                        add_generation_prompt=True)
                    for r in batch
                ]
                enc = tokenizer(
                    chat_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=MID_EVAL_MAX_PROMPT_LEN,
                ).to(device)
                out = model.generate(
                    **enc,
                    max_new_tokens=MID_EVAL_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                new_ids = out[:, enc["input_ids"].shape[1]:]
                for row, ids in zip(batch, new_ids):
                    raw = tokenizer.decode(ids, skip_special_tokens=True)
                    sql = base.extract_sql(raw)
                    if not sql:
                        continue
                    pred = executor.execute(row["db_id"], sql)
                    gold = executor.execute(row["db_id"], row["gold_sql"])
                    if not (pred["success"] and gold["success"]):
                        continue
                    if pred["full_rows_truncated"] or gold["full_rows_truncated"]:
                        continue
                    if compare_execution_results(
                        pred["full_rows"], gold["full_rows"],
                        gold_sql=row["gold_sql"],
                    )["match"]:
                        match_count += 1
    finally:
        tokenizer.padding_side = original_padding_side
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        model.train(was_training)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wall = time.perf_counter() - t0
    return {
        "match_rate": round(100.0 * match_count / total, 2),  # 百分点: 52.0 = 52%
        "match_count": match_count,
        "total": total,
        "wall_seconds": round(wall, 2),
    }


class MidTrainEvalCallback(TrainerCallback):
    """每 eval_every 步用当前 LoRA 权重对 Spider dev 快速评估 + flat 早停。

    - 仅在 on_step_end 触发 (global_step > 0 且 % eval_every == 0),
      第 0 步不触发; eval_every=0 时完全不触发。
    - 评估同步阻塞训练 (~6-10min/次), 结果打印训练日志 (键名
      mid_eval_match_rate, 单位百分点) 并追加写 outputs/mid_eval_results.jsonl。
    - 评估异常只记日志并写 mid_eval_error 记录, 不中断训练
      (一次评估失败不能毁掉 ~33h 的训练)。
    """

    # ── flat 早停常量 (依据) ──
    # DELTA 0.5pp: Spider-100 子集上 0.5pp = 0.5 题, 低于该粒度的波动
    #              视为抽样/解码噪声, 不作为"变差"证据;
    # WINDOW 2 次: 单次评估有子集噪声, 连续 2 次确认下降才判"越练越差",
    #              平衡误杀 (浪费已完成的训练) 与白烧 GPU;
    # MIN_STEP 800: 400/800 两次评估仍处 SFT→RL 早期探索, 波动大, 不轻易判死;
    #              默认配方下最早实际触发 = 第 3 次评估 = step 1200 ≥ 800。
    FLAT_DELTA_PP = 0.5
    FLAT_WINDOW = 2
    FLAT_MIN_STEP = 800

    def __init__(
        self,
        spider_dir: str,
        mid_eval_path: Path,
        eval_every: int = 400,
        eval_limit: int = 100,
        early_stop_flat: bool = True,
        evaluator=None,   # (model, tokenizer) -> dict; None = 真实评估 (冒烟测试可注入 stub)
        model=None,
        tokenizer=None,
    ):
        self.spider_dir = spider_dir
        self.mid_eval_path = Path(mid_eval_path)
        self.eval_every = int(eval_every)
        self.eval_limit = int(eval_limit)
        self.early_stop_flat = early_stop_flat
        self._evaluator = evaluator
        self._model = model
        self._tokenizer = tokenizer
        self._trainer = None
        self._scores: List[float] = []   # 历次 mid_eval_match_rate (百分点)
        self._running = False            # 防重入

    def attach_trainer(self, trainer) -> None:
        """GRPOTrainer 构造完成后注入 —— trainer.model 才是实际训练权重."""
        self._trainer = trainer

    def _write_record(self, record: Dict[str, Any]) -> None:
        try:
            self.mid_eval_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.mid_eval_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            print(f"[MidTrainEval] jsonl 写入失败 (不影响训练): {exc}")

    def on_step_end(self, args, state, control, **kwargs):
        if self._running:
            return
        step = getattr(state, "global_step", None)
        if not (self.eval_every > 0 and step is not None
                and step > 0 and step % self.eval_every == 0):
            return

        if self._trainer is not None:
            model = self._trainer.model
            tokenizer = (getattr(self._trainer, "processing_class", None)
                         or getattr(self._trainer, "tokenizer", None))
        else:
            model, tokenizer = self._model, self._tokenizer

        evaluator = self._evaluator
        if evaluator is None:
            evaluator = lambda m, t: _run_mid_eval(  # noqa: E731
                m, t, self.spider_dir, self.eval_limit)

        self._running = True
        try:
            print(f"\n[MidTrainEval] step {step}: 开始快速评估 "
                  f"(Spider dev 前 {self.eval_limit} 条, "
                  f"贪心 batch={MID_EVAL_BATCH_SIZE})...")
            result = evaluator(model, tokenizer)
            rate = float(result["match_rate"])
            record = {
                "event": "mid_eval",
                "step": int(step),
                "mid_eval_match_rate": rate,  # 百分点 (52.0 = 52%)
                "match_count": int(result.get("match_count", 0)),
                "total": int(result.get("total", 0)),
                "wall_seconds": float(result.get("wall_seconds", 0.0)),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "output_dir": (str(self._trainer.args.output_dir)
                               if self._trainer is not None else None),
            }
            self._write_record(record)
            print(f"[MidTrainEval] step {step}: mid_eval_match_rate = "
                  f"{rate:.2f}% ({record['match_count']}/{record['total']}, "
                  f"{record['wall_seconds']:.0f}s)")

            self._scores.append(rate)
            if self.early_stop_flat:
                self._maybe_flat_stop(int(step), control)
        except Exception as exc:
            print(f"[MidTrainEval] step {step}: 评估失败, 跳过本次 (训练继续): "
                  f"{type(exc).__name__}: {exc}")
            self._write_record({
                "event": "mid_eval_error",
                "step": int(step),
                "error": f"{type(exc).__name__}: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        finally:
            self._running = False

    def _maybe_flat_stop(self, step: int, control) -> None:
        """连续 FLAT_WINDOW 次成绩 ≤ 首次 - FLAT_DELTA_PP 且
        step ≥ FLAT_MIN_STEP → 置 control.should_training_stop = True."""
        if len(self._scores) < self.FLAT_WINDOW + 1:
            return
        first = self._scores[0]
        recent = self._scores[-self.FLAT_WINDOW:]
        if all(s <= first - self.FLAT_DELTA_PP for s in recent):
            if step >= self.FLAT_MIN_STEP:
                print(f"[MidTrainEval] flat 早停触发: 最近 {self.FLAT_WINDOW} 次 "
                      f"mid-eval ({recent}) 均 ≤ 首次 ({first}) - "
                      f"{self.FLAT_DELTA_PP}pp 且 step={step} ≥ "
                      f"{self.FLAT_MIN_STEP} → 停止训练 (防越练越差)")
                self._write_record({
                    "event": "early_stop_flat",
                    "step": step,
                    "first_mid_eval_match_rate": first,
                    "recent_mid_eval_match_rates": recent,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                control.should_training_stop = True
            else:
                print(f"[MidTrainEval] 成绩连续下滑但 step={step} < "
                      f"{self.FLAT_MIN_STEP}, 暂不触发早停 (早期探索期)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P1#4 FINER 完整配方 GRPO (在 train_reasoning_grpo 之上增加"
                    "warmup/max_grad_norm/lr_scheduler/save_steps 旋钮"
                    "+ 中间早评/flat 早停保护)",
    )
    parser.add_argument("--num-train", type=int, default=5500,
                        help="训练条数 (默认 5500 = GRPO manifest 全量)")
    parser.add_argument("--num-generations", type=int, default=16,
                        help="每组采样数 G (默认 16; A40 48G 上 G32×4 提示/步会 OOM)")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="训练步数上限 (FINER 配方 2000)")
    parser.add_argument("--save-steps", type=int, default=100,
                        help="checkpoint 保存间隔 (官方每 100 步存档; 2000/100 = 20 个, 选最优粒度更细)")
    parser.add_argument("--learning-rate", type=float, default=4e-6,
                        help="lr (FINER 配方区间 1e-6~8e-6 取中值 4e-6; T2.x 1-3e-6 欠训)")
    parser.add_argument("--beta", type=float, default=0.01,
                        help="KL 系数 (FINER 配方 0.01-0.02 取下限, 锁 SFT 先验)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度 (GRPO 标准 1.0)")
    parser.add_argument("--max-grad-norm", type=float, default=0.1,
                        help="梯度裁剪 (TRL 默认 1.0; FINER/GRPO 论文 0.1, 防单步巨跳)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="warmup 占比 (TRL GRPOConfig 默认 0.0; 配方要求 0.1)")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine",
                        choices=["linear", "cosine"],
                        help="lr 调度 (GRPO 论文 cosine; TRL 默认 linear)")
    parser.add_argument("--output-dir", type=str, default=str(base.DEFAULT_OUTPUT_DIR),
                        help="LoRA 输出目录")
    parser.add_argument("--spider-dir", type=str, default=str(base.SPIDER_DIR))
    parser.add_argument("--train-file", type=str, default=None,
                        help="训练数据 JSON (数据卫生 G3: 必须用 GRPO-manifest 物化子集, "
                             "不能直接取 train_spider 前 N 条)")
    parser.add_argument("--model-path", type=str, default=str(base.MODEL_PATH))
    parser.add_argument("--loss-type", type=str, default="dr_grpo",
                        choices=["grpo", "bnpo", "dr_grpo"],
                        help="TRL 0.17 支持: grpo/bnpo/dr_grpo; dapo 需 >=0.18 勿用")
    parser.add_argument("--scale-rewards", type=str, default="none",
                        choices=["default", "none"],
                        help="去 std (P7 梯度修复): none=不缩放(源码 if self.scale_rewards "
                             "才除 std+1e-4; 零 std 组 → 全零优势, 无 NaN)")
    parser.add_argument("--reward-type", type=str, default="finer",
                        choices=["binary", "three_level", "partial", "atomic", "finer"])
    parser.add_argument("--train-batch-size", type=int, default=4,
                        help="每步提示数 (默认 4 → G16×4 = 64 条完成/步 == T23 实测足迹)")
    parser.add_argument("--lora-init", type=str, default=None,
                        help="SFT 冷启动 LoRA (本配方: checkpoints/sft_v3)")
    parser.add_argument("--filter-gold", action="store_true",
                        help="剔除 gold SQL 执行失败样本 (FINER Step-1 执行过滤同款纪律)")
    parser.add_argument("--mid-eval-every", type=int, default=400,
                        help="中间早评间隔(步): 每 N 步对 Spider dev 前 --mid-eval-limit 条 "
                             "做贪心快速评估 (默认 400 → 400/800/1200/1600/2000 共 5 次, "
                             "每次约 6-10min, 同步阻塞训练; 0 = 完全关闭)")
    parser.add_argument("--mid-eval-limit", type=int, default=100,
                        help="中间早评的 dev 条数 (默认 100, 与 evaluate_after_grpo "
                             "基线同切片 dev[:100])")
    parser.add_argument("--early-stop-flat", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="flat 早停保护: 连续 2 次 mid-eval 成绩 ≤ 首次成绩 - 0.5pp "
                             "且步数 ≥ 800 时提前停止训练, 防越练越差白烧 GPU "
                             "(--no-early-stop-flat 关闭)")
    return parser


def main() -> None:
    # ── GPU performance: TF32 + autotune (同原脚本) ──
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    args = build_parser().parse_args()

    spider_dir = args.spider_dir
    model_path = args.model_path
    output_dir = args.output_dir

    # ------------------------------------------------------------------
    # 1. Build dataset (复用原模块, 保证 prompt/奖励一致性)
    # ------------------------------------------------------------------
    print(f"Building dataset from: {args.train_file or spider_dir}")
    dataset = build_dataset(spider_dir, limit=args.num_train,
                            filter_gold=args.filter_gold,
                            train_file=args.train_file)
    print(f"Dataset: {len(dataset)} examples")
    print(f"First db_id: {dataset[0]['db_id']}")

    # ------------------------------------------------------------------
    # 2. Load model & tokenizer (同原脚本: SFT merge + pad/eos 修复)
    # ------------------------------------------------------------------
    print(f"\nLoading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=True,
    )
    if args.lora_init:
        print(f"Loading existing LoRA (SFT cold-start): {args.lora_init}")
        tmp = PeftModel.from_pretrained(model, args.lora_init)
        model = tmp.merge_and_unload()
        print("SFT adapter merged into base weights (cold-start 起点)")
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    # CRITICAL (同原脚本): Qwen pad(151643)!=eos(151645) 且 config 保留旧 pad,
    # 导致 padding 位置被当 EOS、生成 ~29 token 即停。必须 tokenizer+config 都改。
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded. GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 3. LoRA configuration (同原脚本 r=16/alpha=32)
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ------------------------------------------------------------------
    # 4. GRPO configuration (原脚本 + FINER 配方 4 个新旋钮)
    # ------------------------------------------------------------------
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,           # FINER 0.1 (TRL 默认 0.0)
        lr_scheduler_type=args.lr_scheduler_type,  # FINER cosine (TRL 默认 linear)
        max_grad_norm=args.max_grad_norm,          # FINER 0.1 (TRL 默认 1.0)
        logging_steps=5,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_prompt_length=1536,
        max_completion_length=512,  # 必须与推理 max_new_tokens (512) 一致
        temperature=args.temperature,
        beta=args.beta,
        loss_type=args.loss_type,       # dr_grpo: token-mean 去长度归一化
        scale_rewards=(False if args.scale_rewards == "none" else True),
        remove_unused_columns=False,   # 保留 query, db_id 供奖励函数使用
        bf16=True,
        dataloader_num_workers=2,
        report_to="none",
    )

    print(f"\nGRPO: G={args.num_generations}, batch_prompts={args.train_batch_size}, "
          f"completions/step={args.train_batch_size * args.num_generations}, "
          f"max_steps={args.max_steps}, lr={args.learning_rate}, "
          f"beta={args.beta}, T={args.temperature}, "
          f"grad_norm={args.max_grad_norm}, warmup={args.warmup_ratio}, "
          f"sched={args.lr_scheduler_type}, loss={args.loss_type}, "
          f"scale_rewards={args.scale_rewards}, reward={args.reward_type}")

    # ------------------------------------------------------------------
    # 5. Reward function (复用原模块: finer = format+exec+atomic 三份量)
    # ------------------------------------------------------------------
    reward_func = create_reward_function(spider_dir, reward_type=args.reward_type)
    print(f"\nReward function: {args.reward_type} "
          f"(finer: 1.0 匹配 / 0.5+0.5×atomic 可执行但错 / 0.0 失败; memory 分量缺实现)")

    # ------------------------------------------------------------------
    # 6. Trainer (RewardStdGuard: reward_std 连续 3 次为 0 即提前停止;
    #    MidTrainEvalCallback: 中间早评 + flat 早停)
    # ------------------------------------------------------------------
    mid_eval_cb = None
    mid_eval_path = base.PROJECT_ROOT / "outputs" / MID_EVAL_LOG_NAME
    if args.mid_eval_every > 0:
        mid_eval_cb = MidTrainEvalCallback(
            spider_dir=spider_dir,
            mid_eval_path=mid_eval_path,
            eval_every=args.mid_eval_every,
            eval_limit=args.mid_eval_limit,
            early_stop_flat=args.early_stop_flat,
            model=model,
            tokenizer=tokenizer,
        )
        print(f"\nMid-train eval: every {args.mid_eval_every} step(s) on "
              f"dev[:{args.mid_eval_limit}], greedy batch={MID_EVAL_BATCH_SIZE} "
              f"(~6-10min/次), jsonl -> {mid_eval_path}, "
              f"early_stop_flat={args.early_stop_flat}")

    callbacks = [RewardStdGuardCallback()]
    if mid_eval_cb is not None:
        callbacks.append(mid_eval_cb)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=callbacks,
    )
    if mid_eval_cb is not None:
        mid_eval_cb.attach_trainer(trainer)

    print("\n" + "=" * 60)
    print("Starting P1#4 FINER GRPO training...")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
    except Exception as exc:
        print(f"\nTraining error: {type(exc).__name__}: {exc}")
        raise

    peak_gib = torch.cuda.max_memory_allocated(0) / 1024 ** 3
    print(f"\nPeak GPU memory: {peak_gib:.2f} GiB (A40 48G)")

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    print(f"\nSaving LoRA adapter to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "recipe": "P1#4 FINER full (3-component reward: format+exec+atomic; memory missing)",
        "base_model": model_path,
        "lora_init": args.lora_init,
        "train_file": args.train_file,
        "num_train_examples": args.num_train,
        "num_generations": args.num_generations,
        "train_batch_size": args.train_batch_size,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "temperature": args.temperature,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "loss_type": args.loss_type,
        "scale_rewards": args.scale_rewards,
        "reward_type": args.reward_type,
        "filter_gold": args.filter_gold,
        "mid_eval_every": args.mid_eval_every,
        "mid_eval_limit": args.mid_eval_limit,
        "early_stop_flat": args.early_stop_flat,
        "mid_eval_jsonl": str(mid_eval_path),
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": list(lora_config.target_modules),
        "peak_gpu_gib": peak_gib,
    }
    meta_path = Path(output_dir) / "training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"Metadata saved to: {meta_path}")
    print("\nDone.")
    print(f"\nNext step: evaluate checkpoints (slurm 内自动执行)")
    print(f"  python src/evaluate_after_grpo.py --lora-path {output_dir}")
    print(f"\nMid-train eval log (供外部守望进程读取): {mid_eval_path}")


if __name__ == "__main__":
    main()
