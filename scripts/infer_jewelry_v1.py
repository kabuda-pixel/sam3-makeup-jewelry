#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam3_face_attributes.core import (
    autocast_context,
    build_processor,
    iter_images,
    normalize_masks,
    normalize_scores,
    save_mask,
    save_overlay,
)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_BROW = [70, 63, 105, 66, 107]
RIGHT_BROW = [336, 296, 334, 293, 300]
OUTER_LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]
NOSE = [1, 2, 4, 5, 45, 48, 64, 94, 97, 98, 168, 195, 197, 275, 278, 294, 326, 327]
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


ROUTE_MAP = {
    "ear_jewelry": ("left_ear", "right_ear"),
    "nose_jewelry": ("nose",),
    "lip_jewelry": ("lips",),
    "eyebrow_jewelry": ("left_brow", "right_brow"),
    "face_gems": ("face",),
    "forehead_jewelry": ("forehead",),
    "nose_chain": ("face",),
    "neck_jewelry": ("neck",),
}


@dataclass(frozen=True)
class JewelrySpec:
    name: str
    prompts: tuple[str, ...]
    min_sam_score: float
    min_quality: float
    max_crop_coverage: float
    max_face_coverage: float


@dataclass(frozen=True)
class Roi:
    name: str
    box: tuple[int, int, int, int]
    branch_prior: float


@dataclass
class Candidate:
    face_index: int
    attribute: str
    prompt: str
    roi: str
    sam_score: float
    quality: float
    mask: np.ndarray
    crop_coverage: float
    face_coverage: float
    border_contacts: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent SAM3 jewelry segmentation.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "jewelry_v1.json",
    )
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-masks-per-prompt", type=int, default=5)
    parser.add_argument("--pixel-threshold", type=float, default=0.46)
    parser.add_argument("--dedup-iou", type=float, default=0.84)
    parser.add_argument("--max-border-contacts", type=int, default=2)
    return parser.parse_args()


def load_jewelry_specs(path: Path) -> list[JewelrySpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        JewelrySpec(
            name=item["name"],
            prompts=tuple(item["prompts"]),
            min_sam_score=float(item["min_sam_score"]),
            min_quality=float(item["min_quality"]),
            max_crop_coverage=float(item["max_crop_coverage"]),
            max_face_coverage=float(item["max_face_coverage"]),
        )
        for item in raw["jewelry"]
    ]


def load_face_mesh() -> Any:
    import mediapipe as mp

    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


def landmark_faces(
    image: Image.Image,
    face_mesh: Any,
) -> list[list[tuple[float, float]]]:
    results = face_mesh.process(np.asarray(image.convert("RGB")))
    if not results.multi_face_landmarks:
        return []
    return [
        [
            (landmark.x * image.width, landmark.y * image.height)
            for landmark in face_landmarks.landmark
        ]
        for face_landmarks in results.multi_face_landmarks
    ]


def clamp_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(width - 1, int(round(x0))))
    y0 = max(0, min(height - 1, int(round(y0))))
    x1 = max(x0 + 1, min(width, int(round(x1))))
    y1 = max(y0 + 1, min(height, int(round(y1))))
    return x0, y0, x1, y1


def landmark_box(
    points: list[tuple[float, float]],
    indices: list[int],
    pad_x: float,
    pad_y: float,
    image: Image.Image,
) -> tuple[int, int, int, int]:
    selected = np.asarray([points[index] for index in indices], dtype=np.float32)
    x0, y0 = selected.min(axis=0)
    x1, y1 = selected.max(axis=0)
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    return clamp_box(
        (
            x0 - width * pad_x,
            y0 - height * pad_y,
            x1 + width * pad_x,
            y1 + height * pad_y,
        ),
        image.width,
        image.height,
    )


def build_rois(points: list[tuple[float, float]], image: Image.Image) -> dict[str, Roi]:
    face = landmark_box(points, FACE_OVAL, 0.10, 0.10, image)
    nose = landmark_box(points, NOSE, 0.65, 0.70, image)
    lips = landmark_box(points, OUTER_LIPS, 0.55, 0.70, image)
    left_brow = landmark_box(points, LEFT_BROW, 0.65, 1.20, image)
    right_brow = landmark_box(points, RIGHT_BROW, 0.65, 1.20, image)

    fx0, fy0, fx1, fy1 = face
    face_width = fx1 - fx0
    face_height = fy1 - fy0
    left_ear = clamp_box(
        (
            fx0 - 0.28 * face_width,
            fy0 + 0.20 * face_height,
            fx0 + 0.14 * face_width,
            fy0 + 0.88 * face_height,
        ),
        image.width,
        image.height,
    )
    right_ear = clamp_box(
        (
            fx1 - 0.14 * face_width,
            fy0 + 0.20 * face_height,
            fx1 + 0.28 * face_width,
            fy0 + 0.88 * face_height,
        ),
        image.width,
        image.height,
    )
    upper_eye_y = min(
        min(points[index][1] for index in LEFT_EYE),
        min(points[index][1] for index in RIGHT_EYE),
    )
    forehead = clamp_box(
        (
            fx0 + 0.08 * face_width,
            fy0 - 0.05 * face_height,
            fx1 - 0.08 * face_width,
            upper_eye_y + 0.12 * face_height,
        ),
        image.width,
        image.height,
    )
    neck = clamp_box(
        (
            fx0 - 0.18 * face_width,
            fy1 - 0.04 * face_height,
            fx1 + 0.18 * face_width,
            fy1 + 0.72 * face_height,
        ),
        image.width,
        image.height,
    )
    return {
        "left_ear": Roi("left_ear", left_ear, 0.94),
        "right_ear": Roi("right_ear", right_ear, 0.94),
        "nose": Roi("nose", nose, 0.98),
        "lips": Roi("lips", lips, 0.98),
        "left_brow": Roi("left_brow", left_brow, 0.96),
        "right_brow": Roi("right_brow", right_brow, 0.96),
        "forehead": Roi("forehead", forehead, 0.92),
        "face": Roi("face", face, 0.88),
        "neck": Roi("neck", neck, 0.90),
        "full_image": Roi(
            "full_image",
            (0, 0, image.width, image.height),
            0.62,
        ),
    }


def route_names(attribute: str, has_landmarks: bool) -> tuple[str, ...]:
    if not has_landmarks:
        return ("full_image",)
    return ROUTE_MAP.get(attribute, ("face",))


def paste_crop_mask(
    crop_mask: np.ndarray,
    box: tuple[int, int, int, int],
    image: Image.Image,
) -> np.ndarray:
    x0, y0, x1, y1 = box
    resized = Image.fromarray(crop_mask.astype(np.uint8) * 255).resize(
        (x1 - x0, y1 - y0),
        Image.Resampling.NEAREST,
    )
    full = np.zeros((image.height, image.width), dtype=bool)
    full[y0:y1, x0:x1] = np.asarray(resized) > 0
    return full


def count_border_contacts(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    return sum(
        (
            bool(mask[0].any()),
            bool(mask[-1].any()),
            bool(mask[:, 0].any()),
            bool(mask[:, -1].any()),
        )
    )


def candidate_quality(
    sam_score: float,
    branch_prior: float,
    crop_coverage: float,
    max_crop_coverage: float,
) -> float:
    compactness = max(0.0, 1.0 - crop_coverage / max(1e-6, max_crop_coverage))
    return float(
        np.clip(
            0.75 * sam_score + 0.15 * branch_prior + 0.10 * compactness,
            0.0,
            1.0,
        )
    )


def run_branch(
    processor: Any,
    image: Image.Image,
    roi: Roi,
    state: dict[str, Any],
    face_index: int,
    face_area: int,
    spec: JewelrySpec,
    args: argparse.Namespace,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    accepted = []
    rejected = []
    for prompt in spec.prompts:
        with autocast_context(args.device, args.dtype):
            result = processor.set_text_prompt(state=state, prompt=prompt)
        masks = normalize_masks(result.get("masks", []))
        scores = normalize_scores(result.get("scores"), len(masks))
        for mask_index in np.argsort(-scores)[: args.max_masks_per_prompt]:
            crop_mask = masks[mask_index]
            full_mask = paste_crop_mask(crop_mask, roi.box, image)
            sam_score = float(scores[mask_index])
            crop_coverage = float(crop_mask.mean())
            face_coverage = float(full_mask.sum() / max(1, face_area))
            border_contacts = count_border_contacts(crop_mask)
            quality = candidate_quality(
                sam_score,
                roi.branch_prior,
                crop_coverage,
                spec.max_crop_coverage,
            )
            reason = None
            if sam_score < spec.min_sam_score:
                reason = "low_sam_score"
            elif crop_coverage > spec.max_crop_coverage:
                reason = "excessive_crop_coverage"
            elif face_coverage > spec.max_face_coverage:
                reason = "excessive_face_coverage"
            elif border_contacts > args.max_border_contacts and crop_coverage > 0.20:
                reason = "mask_touches_too_many_crop_borders"
            elif quality < spec.min_quality:
                reason = "low_candidate_quality"
            record = {
                "face_index": face_index,
                "attribute": spec.name,
                "prompt": prompt,
                "roi": roi.name,
                "sam_score": sam_score,
                "quality": quality,
                "crop_coverage": crop_coverage,
                "face_coverage": face_coverage,
                "border_contacts": border_contacts,
            }
            if reason is not None:
                record["reason"] = reason
                rejected.append(record)
                continue
            accepted.append(
                Candidate(
                    face_index=face_index,
                    attribute=spec.name,
                    prompt=prompt,
                    roi=roi.name,
                    sam_score=sam_score,
                    quality=quality,
                    mask=full_mask,
                    crop_coverage=crop_coverage,
                    face_coverage=face_coverage,
                    border_contacts=border_contacts,
                )
            )
    return accepted, rejected


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def deduplicate(
    candidates: list[Candidate],
    iou_threshold: float,
) -> list[Candidate]:
    kept = []
    for candidate in sorted(candidates, key=lambda item: item.quality, reverse=True):
        if any(mask_iou(candidate.mask, selected.mask) >= iou_threshold for selected in kept):
            continue
        kept.append(candidate)
    return kept


def fuse_candidates(
    candidates: list[Candidate],
    shape: tuple[int, int],
    pixel_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.zeros(shape, dtype=np.float32)
    for candidate in candidates:
        probability = np.maximum(
            probability,
            candidate.quality * candidate.mask.astype(np.float32),
        )
    return probability, probability >= pixel_threshold


def save_probability(probability: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(probability * 255.0, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    specs = load_jewelry_specs(args.config)
    images = iter_images(args.input, args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = build_processor(args.device, args.checkpoint_path)
    face_mesh = load_face_mesh()

    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as output_file:
        for image_index, image_path in enumerate(images, 1):
            image = Image.open(image_path).convert("RGB")
            faces = landmark_faces(image, face_mesh)
            face_inputs: list[tuple[int, dict[str, Roi], int, bool]] = []
            if faces:
                for face_index, points in enumerate(faces):
                    rois = build_rois(points, image)
                    face_box = rois["face"].box
                    face_area = (face_box[2] - face_box[0]) * (
                        face_box[3] - face_box[1]
                    )
                    face_inputs.append((face_index, rois, face_area, True))
            else:
                face_inputs.append(
                    (
                        0,
                        {
                            "full_image": Roi(
                                "full_image",
                                (0, 0, image.width, image.height),
                                0.62,
                            )
                        },
                        image.width * image.height,
                        False,
                    )
                )

            candidates_by_attribute = {spec.name: [] for spec in specs}
            rejected = []
            for face_index, rois, face_area, has_landmarks in face_inputs:
                required_rois = {
                    roi_name
                    for spec in specs
                    for roi_name in route_names(spec.name, has_landmarks)
                }
                roi_states = {}
                for roi_name in required_rois:
                    crop = image.crop(rois[roi_name].box)
                    with autocast_context(args.device, args.dtype):
                        roi_states[roi_name] = processor.set_image(crop)

                for spec in specs:
                    for roi_name in route_names(spec.name, has_landmarks):
                        branch_candidates, branch_rejected = run_branch(
                            processor,
                            image,
                            rois[roi_name],
                            roi_states[roi_name],
                            face_index,
                            face_area,
                            spec,
                            args,
                        )
                        candidates_by_attribute[spec.name].extend(branch_candidates)
                        rejected.extend(branch_rejected)

            all_candidates = []
            attribute_records = []
            attribute_masks = {}
            for spec in specs:
                candidates = deduplicate(
                    candidates_by_attribute[spec.name],
                    args.dedup_iou,
                )
                all_candidates.extend(candidates)
                _, attribute_mask = fuse_candidates(
                    candidates,
                    (image.height, image.width),
                    args.pixel_threshold,
                )
                attribute_masks[spec.name] = attribute_mask
                attribute_records.append(
                    {
                        "attribute": spec.name,
                        "accepted_count": len(candidates),
                        "accepted": [
                            {
                                "face_index": candidate.face_index,
                                "prompt": candidate.prompt,
                                "roi": candidate.roi,
                                "sam_score": candidate.sam_score,
                                "quality": candidate.quality,
                                "crop_coverage": candidate.crop_coverage,
                                "face_coverage": candidate.face_coverage,
                                "border_contacts": candidate.border_contacts,
                            }
                            for candidate in candidates
                        ],
                    }
                )

            all_candidates = deduplicate(all_candidates, 0.92)
            probability, combined_mask = fuse_candidates(
                all_candidates,
                (image.height, image.width),
                args.pixel_threshold,
            )
            stem = image_path.stem
            combined_mask_path = (
                args.output_dir / "combined_masks" / "jewelry" / f"{stem}.png"
            )
            combined_overlay_path = (
                args.output_dir / "combined_overlays" / "jewelry" / f"{stem}.jpg"
            )
            probability_path = (
                args.output_dir / "probability_maps" / "jewelry" / f"{stem}.png"
            )
            save_mask(combined_mask, combined_mask_path)
            save_overlay(image, combined_mask, combined_overlay_path)
            save_probability(probability, probability_path)
            for attribute, attribute_mask in attribute_masks.items():
                save_mask(
                    attribute_mask,
                    args.output_dir
                    / "attribute_masks"
                    / "jewelry"
                    / attribute
                    / f"{stem}.png",
                )

            output_file.write(
                json.dumps(
                    {
                        "image": str(image_path),
                        "landmarks_detected": bool(faces),
                        "face_count": len(faces),
                        "accepted_candidate_count": len(all_candidates),
                        "combined_area_ratio": float(combined_mask.mean()),
                        "combined_mask_path": str(combined_mask_path),
                        "combined_overlay_path": str(combined_overlay_path),
                        "probability_map_path": str(probability_path),
                        "attributes": attribute_records,
                        "rejected_candidates": rejected,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"[{image_index}/{len(images)}] {image_path}")

    face_mesh.close()


if __name__ == "__main__":
    main()
