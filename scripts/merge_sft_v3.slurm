#!/bin/bash
# 合并 sft_v3 LoRA 为全参数模型（RetrySQL CPT 起点）
#SBATCH --job-name=merge_sftv3
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/merge_sftv3_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/merge_sftv3_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=$BASE/envs/reasoning3b/bin/python

$PY - <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = "models/Qwen2.5-Coder-3B-Instruct"
ckpt = "checkpoints/sft_v3"
out = "checkpoints/sft_v3_merged"
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                         local_files_only=True, trust_remote_code=True)
m = PeftModel.from_pretrained(m, ckpt)
m = m.merge_and_unload()
m.save_pretrained(out)
# tokenizer 用 checkpoint 版（含 SFT 期新增 special tokens）
tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True, trust_remote_code=True)
tok.save_pretrained(out)
print("merged ->", out)
EOF

echo "Done: $(date)"
