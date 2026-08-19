#!/bin/bash
# HPC 连通性守望：每 2 分钟查公告板 Gist 端口 + 试连 ssh；端口漂移自动改 ~/.ssh/config；恢复后报队列并退出
LOG=C:/Users/13389/Desktop/女朋友/reasoning_generator_3b/logs/hpc_watch.log
mkdir -p "$(dirname "$LOG")"
CUR=11545
echo "$(date '+%F %T') WATCH_START cur_port=$CUR" >> "$LOG"
while true; do
  PORT=$(curl -s --max-time 10 https://gist.githubusercontent.com/raw/94fb05cee1ed4fe1b0b9edfbb3d718e2 2>/dev/null | grep -oE '[0-9]+$' | head -1)
  if [ -n "$PORT" ] && [ "$PORT" != "$CUR" ]; then
    echo "$(date '+%F %T') PORT_CHANGE $CUR -> $PORT" >> "$LOG"
    sed -i "s/^    Port [0-9]*/    Port $PORT/" ~/.ssh/config
    CUR=$PORT
  fi
  if ssh -o ConnectTimeout=8 -o BatchMode=yes hpc "echo ALIVE" >> "$LOG" 2>&1; then
    echo "$(date '+%F %T') HPC_RESTORED" >> "$LOG"
    ssh hpc "squeue -u jiahuiwang24 -o '%.10i %.20j %.8T %.10M %R'" >> "$LOG" 2>&1
    exit 0
  fi
  sleep 120
done
