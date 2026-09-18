#!/usr/bin/env bash
# 固定从项目根目录运行，数据和配置路径以根目录为基准。
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_ROOT" || exit 1



PYTHON_MODULE="minionerec.preprocessing.convert_dataset"

INPUT_DIR="data/Amazon18/Industrial_and_Scientific"

OUTPUT_DIR="data/Amazon18"

DATASET_NAME="Industrial_and_Scientific"

# ===========================================

echo "Start converting $DATASET_NAME ..."

python -m "$PYTHON_MODULE" \
    --dataset_name $DATASET_NAME \
    --data_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --category $DATASET_NAME \
    --seed 42 "$@"

echo "Finished!"
