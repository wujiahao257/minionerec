#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m rq.kmeans.generate_indices_plus \
  --data_path data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
  --ckpt_path "${1:?Usage: bash rq/scripts/generate_indices_plus.sh /path/to/best_collision_model.pth}" \
  --num_emb_list 256 256 256 \
  --device cuda:0
