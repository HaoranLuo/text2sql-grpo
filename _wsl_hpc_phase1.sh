#!/bin/bash
set -u
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cp -f /mnt/c/Users/13389/.ssh/id_ed25519 ~/.ssh/id_ed25519
cp -f /mnt/c/Users/13389/.ssh/id_ed25519.pub ~/.ssh/id_ed25519.pub
chmod 600 ~/.ssh/id_ed25519
H="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o ServerAliveInterval=20 -o ServerAliveCountMax=5 -J ASUS@6.tcp.cpolar.cn:10889 -i ~/.ssh/id_ed25519 jiahuiwang24@login.hpc.xjtlu.edu.cn"

echo "=== PHASE1 start $(date '+%H:%M:%S') ==="
$H 'echo "HPC=$(hostname)"
pwd
if [ -n "$DEEPSEEK_API_KEY" ]; then echo "KEY_SET=yes"; else echo "KEY_SET=no"; fi
echo "KEYVAR_NAMES=$(env | grep -i deepseek | grep -o "^[A-Z_]*" | tr "\n" " ")"
ls -d ~/.deepseek* 2>/dev/null
P=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
[ -d "$P" ] && echo "PROJECT=ok" || echo "PROJECT=MISSING"
ls -la "$P/envs/reasoning3b/bin/python" "$P/data/spider_data/train_spider.json" 2>&1
echo "PHASE1_DONE"'
echo "=== PHASE1 done $(date '+%H:%M:%S') ==="
