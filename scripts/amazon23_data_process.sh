#!/usr/bin/env bash
# 固定从项目根目录运行，数据和配置路径以根目录为基准。
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_ROOT" || exit 1


python -m minionerec.preprocessing.amazon23 \
    --dataset {domain} \
    --metadata_file ./meta_{domain}.jsonl \
    --reviews_file ./{domain}.jsonl \
    --user_k 5 \
    --st_year 2018 \
    --st_month 10 \
    --ed_year 2023 \
    --ed_month 9 \
    --output_path ./data/Amazon23 "$@"
