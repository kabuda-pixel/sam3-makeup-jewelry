from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .paths import iter_images


@dataclass(frozen=True)
class AttributeSpec:
    category: str
    name: str
    prompts: tuple[str, ...]
    threshold: float


def load_specs(path: Path) -> list[AttributeSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for category, attributes in raw.items():
        for attribute in attributes:
            specs.append(
                AttributeSpec(
                    category=category,
                    name=attribute["name"],
                    prompts=tuple(attribute["prompts"]),
                    threshold=float(attribute["threshold"]),
                )
            )
    return specs


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            value = value.float()
        value = value.cpu().numpy()
    return np.asarray(value)


def normalize_masks(value: Any) -> np.ndarray:
    masks = to_numpy(value)
    if masks.size == 0:
        return np.zeros((0, 1, 1), dtype=bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    return masks > 0


def normalize_scores(value: Any, count: int) -> np.ndarray:
    if value is None:
        return np.ones(count, dtype=np.float32)
    scores = to_numpy(value).reshape(-1).astype(np.float32)
    if len(scores) >= count:
        return scores[:count]
    return np.pad(scores, (0, count - len(scores)), constant_values=0.0)


def aggregate_evidence(
    prompt_outputs: list[tuple[str, float, np.ndarray]],
    image_shape: tuple[int, int],
) -> tuple[float, float, np.ndarray]:
    height, width = image_shape
    union = np.zeros((height, width), dtype=bool)
    if not prompt_outputs:
        return 0.0, 0.0, union
    scores = []
    for _, score, mask in prompt_outputs:
        union |= mask
        scores.append(np.clip(score, 0.0, 1.0))
    confidence = float(1.0 - np.prod(1.0 - np.asarray(scores)))
    return confidence, float(union.mean()), union


def visual_evidence(image: Image.Image, mask: np.ndarray) -> dict[str, float]:
    mask = mask.astype(bool)
    if not mask.any():
        return {
            "evidence": 0.0,
            "color_delta": 0.0,
            "saturation_delta": 0.0,
            "brightness_delta": 0.0,
            "texture_delta": 0.0,
        }

    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    ring_image = mask_image.filter(ImageFilter.MaxFilter(19))
    ring = (np.asarray(ring_image) > 0) & ~mask
    if ring.sum() < 32:
        ring = ~mask
    if ring.sum() < 32:
        return {
            "evidence": 0.0,
            "color_delta": 0.0,
            "saturation_delta": 0.0,
            "brightness_delta": 0.0,
            "texture_delta": 0.0,
        }

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    gradient = np.zeros_like(gray)
    gradient[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1])
    gradient[1:, :] += np.abs(gray[1:, :] - gray[:-1, :])
    gradient *= 0.5

    color_delta = float(np.linalg.norm(rgb[mask].mean(0) - rgb[ring].mean(0)) / np.sqrt(3.0))
    saturation_delta = float(abs(hsv[..., 1][mask].mean() - hsv[..., 1][ring].mean()))
    brightness_delta = float(abs(gray[mask].mean() - gray[ring].mean()))
    texture_delta = float(abs(gradient[mask].mean() - gradient[ring].mean()))
    evidence = max(
        color_delta,
        saturation_delta,
        brightness_delta,
        min(1.0, texture_delta * 3.0),
    )
    return {
        "evidence": evidence,
        "color_delta": color_delta,
        "saturation_delta": saturation_delta,
        "brightness_delta": brightness_delta,
        "texture_delta": texture_delta,
    }


def build_processor(device: str, checkpoint_path: Path | None) -> Any:
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    target = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    kwargs = {}
    if checkpoint_path is not None:
        kwargs = {"checkpoint_path": str(checkpoint_path), "load_from_HF": False}
    model = build_sam3_image_model(device=target, **kwargs)
    return Sam3Processor(model, device=target)


def autocast_context(device: str, dtype: str):
    import torch

    if device != "cuda" or not torch.cuda.is_available() or dtype == "float32":
        return contextlib.nullcontext()
    torch_dtype = torch.bfloat16 if dtype in {"auto", "bfloat16"} else torch.float16
    return torch.autocast(device_type="cuda", dtype=torch_dtype)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
