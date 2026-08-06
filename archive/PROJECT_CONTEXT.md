# Reasoning Generator GRPO Experiment

## 1. Project Overview

This project is a small-scale validation experiment for improving Text-to-SQL with a Reasoning Generator Agent.

The main research idea:

Use a smaller language model (Qwen2.5-Coder-3B-Instruct) as a replacement for the original large model and verify whether training a reasoning generation agent with GRPO can improve Text-to-SQL performance.

The final experiment compares:

Before training:
- Run the Reasoning Generator on Spider examples.
- Record baseline performance.

After training:
- Train the model using GRPO.
- Run the same evaluation examples.
- Compare performance improvement.

---

# 2. Project Location

HPC project path:

/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b


Project structure:

reasoning_generator_3b/

├── src/
├── data/
├── models/
├── outputs/
├── results/
└── envs/


---

# 3. Computing Environment

Platform:

XJTLU High Performance Computing Platform


Python environment:

/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/envs/reasoning3b


Python version:

Python 3.10.20


Installed packages:

torch==2.5.1+cu124

transformers==4.48.3

accelerate==1.3.0

trl==0.15.2

peft==0.14.0

datasets==3.2.0


Verified:

from trl import GRPOTrainer, GRPOConfig

works successfully.


---

# 4. Model

Model:

Qwen2.5-Coder-3B-Instruct


Location:

models/Qwen2.5-Coder-3B-Instruct


Purpose:

Used as a smaller validation model instead of OmniSQL-32B.


---

# 5. Existing Code

## reasoning_generator_agent.py

Purpose:

Inference-only Reasoning Generator Agent.


Functions:

1. Load local Qwen2.5-Coder-3B-Instruct.

2. Receive:
- natural language question
- database schema
- optional schema links
- evidence


3. Generate reasoning.

4. Extract final SQL.


Important:

This file does NOT perform training.

It only performs inference.


---

## spider_utils.py

Purpose:

Provide Spider dataset and database utilities.


Contains:

SpiderLoader

Functions:

- load Spider data
- obtain database DDL schema
- execute SQL


This file should be reused.

Do not rewrite existing Spider loading logic.


---

# 6. Dataset

Dataset:

Spider


Location:

data/spider_data/


Important files:

train_spider.json

train_others.json

tables.json

dev.json


Each training item contains:

{
    "db_id",
    "question",
    "query",
    "sql"
}


Example:

question:

How many heads of departments are older than 56?


Gold SQL:

SELECT count(*) FROM head WHERE age > 56


Important:

Gold SQL should NOT be provided to the model during generation.

Gold SQL is only used for reward calculation.


---

# 7. Current Progress

Completed:

## Environment

- HPC account configured.
- Project directory created.
- Conda environment created.
- Python dependencies installed.


## Model

- Qwen2.5-Coder-3B-Instruct downloaded.
- Local inference verified.


## Agent

Completed:

reasoning_generator_agent.py


The agent can:

- load model
- generate reasoning
- generate SQL
- output structured JSON


## Data

Completed:

- Spider dataset prepared.
- DDL extraction verified.


## GRPO preparation

Completed:

- Installed TRL.
- Verified GRPOTrainer.
- Verified GRPOConfig.
- Verified LoRA support.


---

# 8. Current Development Task

Main file:

src/train_reasoning_grpo.py


Goal:

Implement GRPO training pipeline.


Development should follow phases.


---

# Phase 1: Dataset Pipeline

Goal:

Create HuggingFace Dataset.


Input:

Spider dataset.


Output format:

Each sample:

{
    "prompt": "...",
    "question": "...",
    "query": "...",
    "db_id": "...",
    "ddl": "..."
}


Only build dataset.

Do NOT train yet.


---

# Phase 2: Add GRPO Training Framework

Add:

- GRPOConfig
- GRPOTrainer
- LoRA configuration


Use:

Qwen2.5-Coder-3B-Instruct


Do small smoke tests first.


---

# Phase 3: Reward Function

Reward design:

Generated SQL:

↓

Execute SQL on SQLite database

↓

Compare execution result with Gold SQL execution result


Reward:

Correct execution:

1.0


Wrong execution:

0.0


The reward function should be simple and interpretable.


---

# Phase 4: Training Experiment

Run small experiment first:

Dataset:

100 Spider examples


Training:

few steps only


Compare:

Before training:

baseline result


After training:

GRPO result


Metrics:

- Execution Accuracy
- Exact Match if available
- Reward change


---

# 9. Development Rules

Important:

Do NOT:

- rewrite existing inference code
- change dataset format
- download another model
- introduce unnecessary frameworks
- use Gold SQL as model input


Always:

- make small changes
- test after every modification
- explain why each change is needed


---

# 10. Instructions for Coding Assistant

Before modifying code:

First analyze:

1. Current code structure.
2. Existing functions.
3. Possible impact.


After every modification provide:

1. Changed files.
2. Reason for modification.
3. Testing command.
4. Expected output.


Keep the implementation minimal and reproducible.
