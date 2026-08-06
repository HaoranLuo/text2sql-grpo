#!/bin/bash
# 补下 FINER-SQL-3B-Spider 缺失分片（后台，登录节点）
cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
rm -rf models/FINER-SQL-3B-Spider
nohup envs/reasoning3b/bin/huggingface-cli download griffith-bigdata/FINER-SQL-3B-Spider \
    --local-dir models/FINER-SQL-3B-Spider > logs/dl_finer2.log 2>&1 &
echo "REDL_STARTED"
