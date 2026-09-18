#!/usr/bin/env bash
# 固定从项目根目录运行，数据和配置路径以根目录为基准。
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_ROOT" || exit 1

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

DATASET="Industrial_and_Scientific"
# DATASET="Office_Products"
# DATASET="Toys_and_Games"

MODEL="Qwen2.5-1.5B"

for category in $DATASET; do
    
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    torchrun --nproc_per_node 8 \
            -m minionerec.experiments.ts.sft \
            --base_model ${MODEL} \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ${MODEL}_ts_rec_sft_${category} \
            --wandb_project MiniOneRec_SFT \
            --wandb_run_name ${MODEL}_ts_rec_sft_${category} \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/${category}.index.json \
            --item_meta_path ./data/Amazon/index/${category}.item.json \
            --freeze_LLM False \
            --description_path ./ts_rec_data/${category}.description_keywords.json \
            --learning_rate 3e-4 "$@"
done
