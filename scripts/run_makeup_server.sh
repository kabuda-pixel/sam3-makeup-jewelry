#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-sam3}"
SAM3_REPO="${SAM3_REPO:-/home/zirui/KABUDA/sam3}"
CHECKPOINT="${CHECKPOINT:-/home/zirui/KABUDA/models/sam3/sam3.pt}"
INPUT_DIR="${INPUT_DIR:-/home/zirui/KABUDA/makeup}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zirui/KABUDA/debug_outputs/makeup}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/makeup_complex.json}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
LIMIT="${LIMIT:-5}"

export PYTHONPATH="${SAM3_REPO}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${PROJECT_ROOT}/scripts/infer_makeup.py" \
    --input "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --config "${CONFIG}" \
    --checkpoint-path "${CHECKPOINT}" \
    --device cuda \
    --dtype bfloat16 \
    --limit "${LIMIT}" \
    --max-masks-per-prompt "${MAX_MASKS_PER_PROMPT:-2}" \
    --min-visual-evidence "${MIN_VISUAL_EVIDENCE:-0.08}" \
    --min-candidate-quality "${MIN_CANDIDATE_QUALITY:-0.48}" \
    --pixel-threshold "${PIXEL_THRESHOLD:-0.45}" \
    --precise-max-face-coverage "${PRECISE_MAX_FACE_COVERAGE:-0.12}" \
    --regional-max-face-coverage "${REGIONAL_MAX_FACE_COVERAGE:-0.35}" \
    --global-max-face-coverage "${GLOBAL_MAX_FACE_COVERAGE:-0.92}" \
    --max-crop-coverage "${MAX_CROP_COVERAGE:-0.78}" \
    --eye-exclusion "${EYE_EXCLUSION:-on}" \
    --eye-exclusion-min-score "${EYE_EXCLUSION_MIN_SCORE:-0.35}" \
    --eye-exclusion-min-precision "${EYE_EXCLUSION_MIN_PRECISION:-0.35}" \
    --eye-exclusion-min-guard-recall "${EYE_EXCLUSION_MIN_GUARD_RECALL:-0.10}" \
    --eye-exclusion-max-crop-coverage "${EYE_EXCLUSION_MAX_CROP_COVERAGE:-0.45}"

echo "Done. Results: ${OUTPUT_DIR}"
