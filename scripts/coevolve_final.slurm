#!/bin/bash
# co-evolve 选择侧封盘实验（零 GPU：复用适配判卷打分缓存 + CPU final + 官方 EX）
#SBATCH --job-name=coevolve_final
#SBATCH --partition=cpu6348
#SBATCH --qos=32cores
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/coevolve_final_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/coevolve_final_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
cd $BASE
PY=$BASE/envs/reasoning3b/bin/python

echo "=== co-evolve final（选择侧，复用适配判卷缓存）==="
$PY src/bird_coevolve.py --phase final \
    --out-dir outputs/bird_coevolve \
    --reuse-work outputs/bird_select_ormbird_bird_bal2/work \
    --num-cpus 12

echo "Done: $(date)"
