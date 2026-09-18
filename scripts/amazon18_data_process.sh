#!/usr/bin/env bash
# 固定从项目根目录运行，数据和配置路径以根目录为基准。
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_ROOT" || exit 1

python -m minionerec.preprocessing.amazon18 \
    --dataset Industrial_and_Scientific \
    --metadata_file ./meta_Industrial_and_Scientific.json \
    --reviews_file ./Industrial_and_Scientific.json \
    --user_k 5 \
    --item_k 5 \
    --st_year 1996 \
    --st_month 10 \
    --ed_year 2018 \
    --ed_month 10 \
    --output_path ./data/Amazon18 "$@"
