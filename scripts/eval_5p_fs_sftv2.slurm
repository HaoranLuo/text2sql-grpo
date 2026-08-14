#!/bin/bash
# 瀹為獙 2: few-shot 杞ㄨ抗绗?6 瑙嗚鎶曠エ 鈥?SFT v2 妫€鏌ョ偣, 鍏ㄩ噺 1034 鏉?# 鐩殑: 楠岃瘉"瑙嗚澶氭牱鎬?(鑰岄潪閲囨牱鍣０)鏄?5p 鎶曠エ澧炵泭鐨勬瑙?鏂囩尞 Reasoning-SQL
#       Table 4 + 缁撹 11), 鏈熸湜 vs 5p 鍩虹嚎(eval_5p_sft_v2) +1~2pp銆?# 姝ラ: 1) collect_fs_examples.py 浠?train 渚?sft_v2_mix.json 鏀堕泦 3 鏉℃墽琛屾纭?#          鐨?<think>+SQL 杞ㄨ抗(鏄?涓?闅惧悇涓€, dev 閲嶅彔瀹堝崼, CPU only);
#       2) eval_5p_fs.py 6 瑙嗚 脳 1 閲囨牱(娓╁害 0 璐績, 涓庡熀绾夸竴鑷? 鈫?鍒嗙粍鎶曠エ;
#       3) eval_official.sh 瀹樻柟 test-suite 澶嶈瘎銆?# 鏃堕棿棰勪及: 1034脳6=6204 娆¤椽蹇冪敓鎴?脳 ~4.4s 鈮?7.6h(5p 鍏ㄩ噺 5170 娆＄害 7.5h 宸茶窇閫?
#SBATCH --job-name=5p_fs_sftv2
#SBATCH --partition=aiaca40
#SBATCH --qos=1a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5p_fs_sftv2_%j.out
#SBATCH --error=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/logs/5p_fs_sftv2_%j.err

set -e
BASE=/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b
PYTHON=$BASE/envs/reasoning3b/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 瀹為獙 2 few-shot 绗?6 瑙嗚: 鏀堕泦绀轰緥 (CPU) ==="
$PYTHON $BASE/src/collect_fs_examples.py \
    --source $BASE/data/sft_v2_mix.json \
    --output $BASE/outputs/fewshot_examples.json \
    --min-examples 3

echo "=== 6 瑙嗚鎶曠エ璇勪及 (SFT v2, 璐績) ==="
$PYTHON $BASE/src/eval_5p_fs.py \
    --lora-path $BASE/checkpoints/sft_v2 \
    --output-dir $BASE/outputs/eval_5p_fs_sft_v2 \
    --fewshot-file $BASE/outputs/fewshot_examples.json \
    --n-prompts 5 --limit 1034 --temperature 0.0

echo "=== 瀹樻柟鍙ｅ緞 ==="
bash $BASE/scripts/eval_official.sh \
    $BASE/outputs/eval_5p_fs_sft_v2/items.json \
    $BASE/outputs/official_5p_fs_sft_v2

echo "Done: $(date)"
