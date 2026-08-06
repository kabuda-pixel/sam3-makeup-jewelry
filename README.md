# SAM3 Makeup and Jewelry Segmentation

This repository provides reproducible inference code for pixel-level segmentation of complex facial makeup and jewelry with SAM3 and MediaPipe Face Mesh.

## 1. Requirements

- Linux
- Python 3.10 or later
- A CUDA-capable GPU
- A working CUDA-compatible PyTorch installation
- A local SAM3 source checkout
- A compatible SAM3 checkpoint

The number and IDs of available GPUs are not fixed. Each inference process uses the GPU selected through `CUDA_VISIBLE_DEVICES`.

## 2. Clone the repositories

Clone this repository:

```bash
git clone https://github.com/kabuda-pixel/sam3-makeup-jewelry.git
cd sam3-makeup-jewelry
```

Clone and install SAM3 by following the official SAM3 repository instructions. For exact reproduction, check out the commit recorded in this repository:

```bash
git -C /path/to/sam3 checkout "$(cat SAM3_COMMIT.txt)"
```

Download the SAM3 checkpoint using the official instructions. The checkpoint is not distributed by this repository.

If `SAM3_CHECKPOINT_SHA256.txt` contains the checksum for the checkpoint release being used, verify it with:

```bash
cd /directory/containing/checkpoint
sha256sum -c /path/to/sam3-makeup-jewelry/SAM3_CHECKPOINT_SHA256.txt
```

## 3. Create the environment

The recommended method is to reproduce the exported Conda environment:

```bash
conda env create -f environment.yml
conda activate sam3
```

If SAM3 already has a working environment, install only this repository's direct dependencies:

```bash
python -m pip install -r requirements.txt
```

PyTorch and SAM3 should be installed according to the CUDA version and installation instructions of the SAM3 repository. They are intentionally not installed by `requirements.txt` to avoid replacing a working CUDA-specific PyTorch build.

## 4. Configure paths

Set paths for the current machine:

```bash
export PROJECT_ROOT=/path/to/sam3-makeup-jewelry
export SAM3_REPO=/path/to/sam3
export CHECKPOINT=/path/to/sam3.pt
export PYTHONPATH="${SAM3_REPO}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
```

Verify the environment:

```bash
test -d "$SAM3_REPO"
test -f "$CHECKPOINT"

python - <<'PY'
import mediapipe
import numpy
import PIL
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY
```

## 5. Run tests

```bash
cd "$PROJECT_ROOT"

python -m py_compile \
  scripts/infer_makeup_v2.py \
  scripts/infer_jewelry_v1.py \
  src/sam3_face_attributes/core.py

bash -n \
  scripts/run_makeup_v2_server.sh \
  scripts/run_jewelry_v1_server.sh

pytest -q \
  tests/test_core.py \
  tests/test_makeup_v2.py \
  tests/test_jewelry_v1.py
```

## 6. Makeup inference

The input may be one image, a directory of images, or a text file containing one image path per line.

```bash
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=0 python scripts/infer_makeup_v2.py \
  --input /path/to/makeup_images \
  --output-dir /path/to/makeup_outputs \
  --config configs/makeup_complex.json \
  --checkpoint-path "$CHECKPOINT" \
  --device cuda \
  --dtype bfloat16 \
  --recursive \
  --max-masks-per-prompt 2 \
  --min-visual-evidence 0.08 \
  --min-candidate-quality 0.48 \
  --pixel-threshold 0.45 \
  --precise-max-face-coverage 0.12 \
  --regional-max-face-coverage 0.35 \
  --global-max-face-coverage 0.92 \
  --max-crop-coverage 0.78 \
  --eye-exclusion on \
  --eye-exclusion-min-score 0.35 \
  --eye-exclusion-min-precision 0.35 \
  --eye-exclusion-min-guard-recall 0.10 \
  --eye-exclusion-max-crop-coverage 0.45
```

To use another GPU, replace `CUDA_VISIBLE_DEVICES=0` with the desired GPU ID.

Makeup outputs:

```text
makeup_outputs/
├── combined_masks/makeup/
├── combined_overlays/makeup/
├── probability_maps/makeup/
├── exclusion_masks/eyes/
└── predictions.jsonl
```

## 7. Jewelry inference

```bash
cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=0 python scripts/infer_jewelry_v1.py \
  --input /path/to/jewelry_images \
  --output-dir /path/to/jewelry_outputs \
  --config configs/jewelry_v1.json \
  --checkpoint-path "$CHECKPOINT" \
  --device cuda \
  --dtype bfloat16 \
  --recursive \
  --max-masks-per-prompt 5 \
  --pixel-threshold 0.46 \
  --dedup-iou 0.84 \
  --max-border-contacts 2
```

Jewelry outputs:

```text
jewelry_outputs/
├── combined_masks/jewelry/
├── combined_overlays/jewelry/
├── probability_maps/jewelry/
├── attribute_masks/jewelry/<attribute>/
└── predictions.jsonl
```

## 8. Output interpretation

The files under `combined_masks` are binary PNG masks aligned with the source image:

- pixel value `255`: predicted makeup or jewelry region;
- pixel value `0`: background.

Each line of `predictions.jsonl` corresponds to one source image and records:

- source image path;
- whether face landmarks were detected;
- detected face count;
- accepted candidate count;
- combined mask area ratio;
- mask, overlay and probability-map paths;
- accepted candidates for each semantic attribute;
- rejected candidates and rejection reasons.

Load a binary mask with:

```python
import numpy as np
from PIL import Image

mask = np.asarray(Image.open(mask_path).convert("L")) > 0
```

## 9. Reproducibility record

Record the following items when reporting results:

- commit hash of this repository;
- commit hash recorded in `SAM3_COMMIT.txt`;
- checkpoint SHA-256 recorded in `SAM3_CHECKPOINT_SHA256.txt`;
- `environment.yml` or installed package versions;
- unchanged JSON prompt configuration;
- complete inference command and parameter values;
- GPU model, CUDA version and PyTorch version.

The generated masks are automatic predictions or pseudo-labels. Quantitative use should include validation on manually inspected or manually annotated samples.
