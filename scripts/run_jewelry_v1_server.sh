#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-sam3}"
SAM3_REPO="${SAM3_REPO:-/home/zirui/KABUDA/sam3}"
CHECKPOINT="${CHECKPOINT:-/home/zirui/KABUDA/models/sam3/sam3.pt}"
INPUT_DIR="${INPUT_DIR:-/home/zirui/KABUDA/jewelry}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zirui/KABUDA/debug_outputs/jewelry_v1}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/jewelry_v1.json}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
LIMIT="${LIMIT:-5}"

export PYTHONPATH="${SAM3_REPO}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${PROJECT_ROOT}/scripts/infer_jewelry_v1.py" \
    --input "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --config "${CONFIG}" \
    --checkpoint-path "${CHECKPOINT}" \
    --device cuda \
    --dtype bfloat16 \
    --limit "${LIMIT}" \
    --max-masks-per-prompt "${MAX_MASKS_PER_PROMPT:-5}" \
    --pixel-threshold "${PIXEL_THRESHOLD:-0.46}" \
    --dedup-iou "${DEDUP_IOU:-0.84}" \
    --max-border-contacts "${MAX_BORDER_CONTACTS:-2}"

echo "Done. Results: ${OUTPUT_DIR}"
