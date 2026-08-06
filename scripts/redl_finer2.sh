#!/bin/bash
# 彻底重下 FINER 权重：杀残留进程 + 清目录 + 完整下载 + 校验
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
pkill -f "huggingface-cli download griffith-bigdata/FINER-SQL-3B-Spider" 2>/dev/null
sleep 2
rm -rf models/FINER-SQL-3B-Spider
mkdir -p models/FINER-SQL-3B-Spider
nohup envs/reasoning3b/bin/huggingface-cli download griffith-bigdata/FINER-SQL-3B-Spider \
    --local-dir models/FINER-SQL-3B-Spider > logs/dl_finer3.log 2>&1 &
echo "REDL_V3_STARTED"
