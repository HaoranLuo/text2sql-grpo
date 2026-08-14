#!/bin/bash
set -u
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cp -f /mnt/c/Users/13389/.ssh/id_ed25519 ~/.ssh/id_ed25519
cp -f /mnt/c/Users/13389/.ssh/id_ed25519.pub ~/.ssh/id_ed25519.pub
chmod 600 ~/.ssh/id_ed25519

SRC="/mnt/c/Users/13389/Desktop/女朋�?reasoning_generator_3b/src/gen_reasoning_data.py"
DST="jiahuiwang24@login.hpc.xjtlu.edu.cn:/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/src/gen_reasoning_data.py"

echo "=== PHASE2 upload $(date '+%H:%M:%S') ==="
scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 \
    -J ASUS@6.tcp.cpolar.cn:12592 -i ~/.ssh/id_ed25519 "$SRC" "$DST"
echo "scp exit=$?"

H="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o ServerAliveInterval=20 -o ServerAliveCountMax=5 -J ASUS@6.tcp.cpolar.cn:12592 -i ~/.ssh/id_ed25519 jiahuiwang24@login.hpc.xjtlu.edu.cn"

echo "=== verify upload + dry-run ==="
$H 'cd /gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
echo "REMOTE_SIZE=$(wc -c < src/gen_reasoning_data.py)"
echo "--- dry run (no API calls) ---"
envs/reasoning3b/bin/python src/gen_reasoning_data.py --num-train 1000 --model deepseek-chat --dry-run 2>&1 | tail -20
echo "PHASE2_DONE"'
echo "=== PHASE2 done $(date '+%H:%M:%S') ==="
