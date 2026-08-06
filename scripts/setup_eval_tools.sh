#!/bin/bash
# 安装官方 test-suite 评估器依赖 + 验证
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
envs/reasoning3b/bin/pip install sqlparse nltk -q 2>&1 | tail -1
envs/reasoning3b/bin/python -c "import sqlparse, nltk; print('deps OK')"
