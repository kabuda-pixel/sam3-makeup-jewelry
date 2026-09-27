#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
args=("$PROJECT_ROOT/scripts/infer_masks.py")
[[ -z "${INPUT_DIR:-}" ]] || args+=(--input "$INPUT_DIR")
[[ -z "${OUTPUT_DIR:-}" ]] || args+=(--output-dir "$OUTPUT_DIR")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint-path "$CHECKPOINT")
[[ -z "${SAM3_REPO:-}" ]] || args+=(--sam3-repo "$SAM3_REPO")
[[ -z "${CONFIG:-}" ]] || args+=(--config "$CONFIG")
[[ -z "${TASK:-}" ]] || args+=(--task "$TASK")
if [[ -n "${CUDA_DEVICE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

# Use the active Python environment, or explicitly select a Conda environment.
runner=("${PYTHON:-python}")
if [[ -n "${CONDA_ENV:-}" ]]; then
  runner=(conda run --no-capture-output -n "$CONDA_ENV" "${PYTHON:-python}")
fi
exec "${runner[@]}" "${args[@]}" "$@"
