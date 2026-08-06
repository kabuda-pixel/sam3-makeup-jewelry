#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam3_face_attributes.core import (
    autocast_context,
    build_processor,
    iter_images,
    load_specs,
    normalize_masks,
    normalize_scores,
    save_mask,
    save_overlay,
    visual_evidence,
)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
NOSE = [1, 2, 4, 5, 45, 48, 64, 94, 97, 98, 168, 195, 197, 275, 278, 294, 326, 327]
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


ROUTE_MAP = {
    "upper_eyelid_eyeshadow": ("left_eye", "right_eye"),
    "lower_eyelid_eyeshadow": ("left_eye", "right_eye"),
    "eyeliner": ("left_eye", "right_eye"),
    "false_eyelashes": ("left_eye", "right_eye"),
    "lip_color": ("lips",),
    "lip_gloss": ("lips",),
    "lip_liner": ("lips",),
    "cheek_blush": ("left_cheek", "right_cheek"),
    "cheek_contour": ("left_cheek", "right_cheek"),
    "nose_contour": ("nose",),
    "facial_highlighter": ("left_cheek", "right_cheek", "nose", "forehead"),
}

PRECISE_ATTRIBUTES = {
    "eyeliner",
    "false_eyelashes",
    "lip_color",
    "lip_gloss",
    "lip_liner",
}

REGIONAL_ATTRIBUTES = set(ROUTE_MAP) - PRECISE_ATTRIBUTES

ATTRIBUTE_MAX_FACE_COVERAGE = {
    "upper_eyelid_eyeshadow": 0.12,
    "lower_eyelid_eyeshadow": 0.10,
    "eyeliner": 0.06,
    "false_eyelashes": 0.08,
    "lip_color": 0.12,
    "lip_gloss": 0.12,
    "lip_liner": 0.08,
    "cheek_blush": 0.25,
    "cheek_contour": 0.22,
    "nose_contour": 0.12,
    "facial_highlighter": 0.30,
}

EYE_EXCLUSION_PROMPT_GROUPS = {
    "eye_interior": (
        "visible interior of the eye opening",
        "ocular surface inside the eyelids",
    ),
    "sclera": (
        "white sclera of the eye",
        "white of the eye excluding eyelid skin",
    ),
    "iris_pupil": (
        "iris and pupil of the eye",
    ),
}


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
    score: float
    quality: float
    mask: np.ndarray
    metrics: dict[str, float]
    face_coverage: float
    crop_coverage: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe-guided SAM3 makeup segmentation.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "makeup_complex.json",
    )
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-masks-per-prompt", type=int, default=2)
    parser.add_argument("--min-visual-evidence", type=float, default=0.08)
    parser.add_argument("--min-candidate-quality", type=float, default=0.48)
    parser.add_argument("--pixel-threshold", type=float, default=0.45)
    parser.add_argument("--precise-max-face-coverage", type=float, default=0.12)
    parser.add_argument("--regional-max-face-coverage", type=float, default=0.35)
    parser.add_argument("--global-max-face-coverage", type=float, default=0.92)
    parser.add_argument("--max-crop-coverage", type=float, default=0.78)
    parser.add_argument("--eye-exclusion", choices=("on", "off"), default="on")
    parser.add_argument("--eye-exclusion-min-score", type=float, default=0.35)
    parser.add_argument("--eye-exclusion-min-precision", type=float, default=0.35)
    parser.add_argument("--eye-exclusion-min-guard-recall", type=float, default=0.1)
    parser.add_argument("--eye-exclusion-max-crop-coverage", type=float, default=0.45)
    return parser.parse_args()


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
    face = landmark_box(points, FACE_OVAL, 0.12, 0.12, image)
    left_eye = landmark_box(points, LEFT_EYE, 1.35, 1.8, image)
    right_eye = landmark_box(points, RIGHT_EYE, 1.35, 1.8, image)
    lips = landmark_box(points, OUTER_LIPS, 0.5, 0.6, image)
    nose = landmark_box(points, NOSE, 0.75, 0.8, image)

    fx0, fy0, fx1, fy1 = face
    face_width = fx1 - fx0
    face_height = fy1 - fy0
    nose_x = points[1][0]
    eye_y = float(np.mean([points[index][1] for index in LEFT_EYE + RIGHT_EYE]))
    mouth_y = float(np.mean([points[index][1] for index in OUTER_LIPS]))
    cheek_y0 = eye_y - 0.02 * face_height
    cheek_y1 = mouth_y + 0.38 * face_height
    left_cheek = clamp_box(
        (fx0, cheek_y0, nose_x + 0.12 * face_width, cheek_y1),
        image.width,
        image.height,
    )
    right_cheek = clamp_box(
        (nose_x - 0.12 * face_width, cheek_y0, fx1, cheek_y1),
        image.width,
        image.height,
    )
    upper_eye_y = min(left_eye[1], right_eye[1])
    forehead = clamp_box(
        (fx0 + 0.05 * face_width, fy0, fx1 - 0.05 * face_width, upper_eye_y + 0.18 * face_height),
        image.width,
        image.height,
    )
    return {
        "left_eye": Roi("left_eye", left_eye, 0.98),
        "right_eye": Roi("right_eye", right_eye, 0.98),
        "lips": Roi("lips", lips, 0.98),
        "nose": Roi("nose", nose, 0.92),
        "left_cheek": Roi("left_cheek", left_cheek, 0.92),
        "right_cheek": Roi("right_cheek", right_cheek, 0.92),
        "forehead": Roi("forehead", forehead, 0.9),
        "face": Roi("face", face, 0.86),
        "full_image": Roi("full_image", (0, 0, image.width, image.height), 0.68),
    }


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


def eye_aperture_guard(
    points: list[tuple[float, float]],
    indices: list[int],
    image: Image.Image,
) -> np.ndarray:
    polygon = [points[index] for index in indices]
    mask_image = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask_image).polygon(polygon, fill=255)
    eye_width = max(point[0] for point in polygon) - min(point[0] for point in polygon)
    radius = max(1, int(round(eye_width * 0.06)))
    filter_size = 2 * radius + 1
    return np.asarray(mask_image.filter(ImageFilter.MaxFilter(filter_size))) > 0


def segment_eye_exclusion(
    processor: Any,
    image: Image.Image,
    roi: Roi,
    state: dict[str, Any],
    guard: np.ndarray,
    face_index: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    component_masks: dict[str, np.ndarray] = {}
    component_qualities: dict[str, float] = {}
    candidates = []
    guard_area = int(guard.sum())
    for component, prompts in EYE_EXCLUSION_PROMPT_GROUPS.items():
        best_component_quality = -1.0
        best_component_mask = np.zeros((image.height, image.width), dtype=bool)
        required_guard_recall = (
            args.eye_exclusion_min_guard_recall
            if component == "eye_interior"
            else args.eye_exclusion_min_guard_recall * 0.25
        )
        for prompt in prompts:
            with autocast_context(args.device, args.dtype):
                result = processor.set_text_prompt(state=state, prompt=prompt)
            masks = normalize_masks(result.get("masks", []))
            scores = normalize_scores(result.get("scores"), len(masks))
            for mask_index in np.argsort(-scores)[: args.max_masks_per_prompt]:
                crop_mask = masks[mask_index]
                full_mask = paste_crop_mask(crop_mask, roi.box, image)
                clipped_mask = full_mask & guard
                score = float(scores[mask_index])
                crop_coverage = float(crop_mask.mean())
                candidate_area = int(full_mask.sum())
                clipped_area = int(clipped_mask.sum())
                guard_precision = float(clipped_area / max(1, candidate_area))
                guard_recall = float(clipped_area / max(1, guard_area))
                quality = float(score * (0.5 + 0.5 * guard_precision))
                accepted = (
                    score >= args.eye_exclusion_min_score
                    and crop_coverage <= args.eye_exclusion_max_crop_coverage
                    and guard_precision >= args.eye_exclusion_min_precision
                    and guard_recall >= required_guard_recall
                )
                candidates.append(
                    {
                        "component": component,
                        "prompt": prompt,
                        "sam_score": score,
                        "crop_coverage": crop_coverage,
                        "guard_precision": guard_precision,
                        "guard_recall": guard_recall,
                        "required_guard_recall": required_guard_recall,
                        "quality": quality,
                        "accepted": accepted,
                    }
                )
                if accepted and quality > best_component_quality:
                    best_component_quality = quality
                    best_component_mask = clipped_mask
        if best_component_mask.any():
            component_masks[component] = best_component_mask
            component_qualities[component] = best_component_quality

    exclusion_mask = np.zeros((image.height, image.width), dtype=bool)
    for component_mask in component_masks.values():
        exclusion_mask |= component_mask
    return exclusion_mask, {
        "face_index": face_index,
        "roi": roi.name,
        "selected": bool(exclusion_mask.any()),
        "selected_components": {
            component: {
                "quality": component_qualities[component],
                "area": int(component_mask.sum()),
            }
            for component, component_mask in component_masks.items()
        },
        "excluded_area": int(exclusion_mask.sum()),
        "candidates": candidates,
    }


def apply_eye_exclusion(
    probability: np.ndarray,
    combined_mask: np.ndarray,
    eye_exclusion_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    filtered_probability = probability.copy()
    filtered_probability[eye_exclusion_mask] = 0.0
    return filtered_probability, combined_mask & ~eye_exclusion_mask


def route_names(attribute: str, has_landmarks: bool) -> tuple[str, ...]:
    if not has_landmarks:
        return ("full_image",)
    if attribute in ROUTE_MAP:
        return ROUTE_MAP[attribute]
    return ("face",)


def max_face_coverage(attribute: str, args: argparse.Namespace) -> float:
    if attribute in ATTRIBUTE_MAX_FACE_COVERAGE:
        return ATTRIBUTE_MAX_FACE_COVERAGE[attribute]
    if attribute in PRECISE_ATTRIBUTES:
        return args.precise_max_face_coverage
    if attribute in REGIONAL_ATTRIBUTES:
        return args.regional_max_face_coverage
    return args.global_max_face_coverage


def candidate_quality(
    sam_score: float,
    metrics: dict[str, float],
    branch_prior: float,
    crop_coverage: float,
) -> float:
    coverage_penalty = max(0.0, 1.0 - max(0.0, crop_coverage - 0.45) / 0.55)
    return float(
        np.clip(
            (0.58 * sam_score + 0.27 * metrics["evidence"] + 0.15 * branch_prior)
            * coverage_penalty,
            0.0,
            1.0,
        )
    )


def run_candidate_branch(
    processor: Any,
    image: Image.Image,
    roi: Roi,
    state: dict[str, Any],
    face_index: int,
    attribute: str,
    prompts: tuple[str, ...],
    face_area: int,
    args: argparse.Namespace,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    accepted = []
    rejected = []
    for prompt in prompts:
        with autocast_context(args.device, args.dtype):
            result = processor.set_text_prompt(state=state, prompt=prompt)
        masks = normalize_masks(result.get("masks", []))
        scores = normalize_scores(result.get("scores"), len(masks))
        for mask_index in np.argsort(-scores)[: args.max_masks_per_prompt]:
            crop_mask = masks[mask_index]
            full_mask = paste_crop_mask(crop_mask, roi.box, image)
            score = float(scores[mask_index])
            crop_coverage = float(crop_mask.mean())
            face_coverage = float(full_mask.sum() / max(1, face_area))
            metrics = visual_evidence(image, full_mask)
            quality = candidate_quality(score, metrics, roi.branch_prior, crop_coverage)
            record = {
                "face_index": face_index,
                "attribute": attribute,
                "prompt": prompt,
                "roi": roi.name,
                "sam_score": score,
                "quality": quality,
                "crop_coverage": crop_coverage,
                "face_coverage": face_coverage,
                "visual_evidence": metrics,
            }
            reason = None
            if crop_coverage > args.max_crop_coverage:
                reason = "excessive_crop_coverage"
            elif face_coverage > max_face_coverage(attribute, args):
                reason = "excessive_face_coverage"
            elif metrics["evidence"] < args.min_visual_evidence:
                reason = "insufficient_visual_evidence"
            elif quality < args.min_candidate_quality:
                reason = "low_candidate_quality"
            elif (
                attribute not in PRECISE_ATTRIBUTES
                and attribute not in REGIONAL_ATTRIBUTES
                and face_coverage > 0.35
                and quality
                < args.min_candidate_quality
                + 0.35 * (face_coverage - 0.35) / 0.65
            ):
                reason = "low_quality_for_large_coverage"
            if reason:
                record["reason"] = reason
                rejected.append(record)
                continue
            accepted.append(
                Candidate(
                    face_index=face_index,
                    attribute=attribute,
                    prompt=prompt,
                    roi=roi.name,
                    score=score,
                    quality=quality,
                    mask=full_mask,
                    metrics=metrics,
                    face_coverage=face_coverage,
                    crop_coverage=crop_coverage,
                )
            )
    return accepted, rejected


def deduplicate(candidates: list[Candidate], iou_threshold: float = 0.82) -> list[Candidate]:
    kept = []
    for candidate in sorted(candidates, key=lambda item: item.quality, reverse=True):
        duplicate = False
        for selected in kept:
            intersection = np.logical_and(candidate.mask, selected.mask).sum()
            union = np.logical_or(candidate.mask, selected.mask).sum()
            if union and intersection / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def fuse_candidates(
    candidates: list[Candidate],
    shape: tuple[int, int],
    pixel_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.zeros(shape, dtype=np.float32)
    attributes = sorted({candidate.attribute for candidate in candidates})
    for attribute in attributes:
        attribute_probability = np.zeros(shape, dtype=np.float32)
        for candidate in candidates:
            if candidate.attribute != attribute:
                continue
            attribute_probability = np.maximum(
                attribute_probability,
                candidate.quality * candidate.mask.astype(np.float32),
            )
        probability = 1.0 - (1.0 - probability) * (
            1.0 - attribute_probability
        )
    return probability, probability >= pixel_threshold


def save_probability(probability: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(probability * 255.0, 0, 255).astype(np.uint8)).save(path)


def main() -> None:
    args = parse_args()
    specs = [spec for spec in load_specs(args.config) if spec.category == "makeup"]
    images = iter_images(args.input, args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = build_processor(args.device, args.checkpoint_path)
    face_mesh = load_face_mesh()

    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as output_file:
        for image_index, image_path in enumerate(images, 1):
            image = Image.open(image_path).convert("RGB")
            faces = landmark_faces(image, face_mesh)
            face_inputs: list[
                tuple[int, dict[str, Roi], int, bool, list[tuple[float, float]] | None]
            ] = []
            if faces:
                for face_index, points in enumerate(faces):
                    rois = build_rois(points, image)
                    face_box = rois["face"].box
                    face_area = (face_box[2] - face_box[0]) * (
                        face_box[3] - face_box[1]
                    )
                    face_inputs.append((face_index, rois, face_area, True, points))
            else:
                face_inputs.append(
                    (
                        0,
                        {
                            "full_image": Roi(
                                "full_image",
                                (0, 0, image.width, image.height),
                                0.68,
                            )
                        },
                        image.width * image.height,
                        False,
                        None,
                    )
                )

            candidates_by_attribute = {spec.name: [] for spec in specs}
            rejected = []
            eye_exclusion_mask = np.zeros((image.height, image.width), dtype=bool)
            eye_exclusion_records = []
            for face_index, rois, face_area, has_landmarks, points in face_inputs:
                required_roi_names = {
                    roi_name
                    for spec in specs
                    for roi_name in route_names(spec.name, has_landmarks)
                }
                if args.eye_exclusion == "on" and has_landmarks:
                    required_roi_names.update(("left_eye", "right_eye"))
                roi_states = {}
                for roi_name in required_roi_names:
                    crop = image.crop(rois[roi_name].box)
                    with autocast_context(args.device, args.dtype):
                        roi_states[roi_name] = processor.set_image(crop)

                if args.eye_exclusion == "on" and points is not None:
                    for roi_name, landmark_indices in (
                        ("left_eye", LEFT_EYE),
                        ("right_eye", RIGHT_EYE),
                    ):
                        exclusion, exclusion_record = segment_eye_exclusion(
                            processor,
                            image,
                            rois[roi_name],
                            roi_states[roi_name],
                            eye_aperture_guard(points, landmark_indices, image),
                            face_index,
                            args,
                        )
                        eye_exclusion_mask |= exclusion
                        eye_exclusion_records.append(exclusion_record)

                for spec in specs:
                    for roi_name in route_names(spec.name, has_landmarks):
                        branch_candidates, branch_rejected = run_candidate_branch(
                            processor,
                            image,
                            rois[roi_name],
                            roi_states[roi_name],
                            face_index,
                            spec.name,
                            spec.prompts,
                            face_area,
                            args,
                        )
                        candidates_by_attribute[spec.name].extend(branch_candidates)
                        rejected.extend(branch_rejected)

            all_candidates = []
            attribute_records = []
            for spec in specs:
                attribute_candidates = deduplicate(candidates_by_attribute[spec.name])
                all_candidates.extend(attribute_candidates)
                attribute_records.append(
                    {
                        "attribute": spec.name,
                        "accepted_count": len(attribute_candidates),
                        "accepted": [
                            {
                                "face_index": candidate.face_index,
                                "prompt": candidate.prompt,
                                "roi": candidate.roi,
                                "sam_score": candidate.score,
                                "quality": candidate.quality,
                                "face_coverage": candidate.face_coverage,
                                "crop_coverage": candidate.crop_coverage,
                                "visual_evidence": candidate.metrics,
                            }
                            for candidate in attribute_candidates
                        ],
                    }
                )

            all_candidates = deduplicate(all_candidates, iou_threshold=0.9)
            probability, combined_mask = fuse_candidates(
                all_candidates,
                (image.height, image.width),
                args.pixel_threshold,
            )
            probability, combined_mask = apply_eye_exclusion(
                probability,
                combined_mask,
                eye_exclusion_mask,
            )
            stem = image_path.stem
            mask_path = args.output_dir / "combined_masks" / "makeup" / f"{stem}.png"
            overlay_path = args.output_dir / "combined_overlays" / "makeup" / f"{stem}.jpg"
            probability_path = args.output_dir / "probability_maps" / "makeup" / f"{stem}.png"
            eye_exclusion_path = (
                args.output_dir / "exclusion_masks" / "eyes" / f"{stem}.png"
            )
            save_mask(combined_mask, mask_path)
            save_overlay(image, combined_mask, overlay_path)
            save_probability(probability, probability_path)
            save_mask(eye_exclusion_mask, eye_exclusion_path)

            output_file.write(
                json.dumps(
                    {
                        "image": str(image_path),
                        "landmarks_detected": bool(faces),
                        "face_count": len(faces),
                        "accepted_candidate_count": len(all_candidates),
                        "combined_area_ratio": float(combined_mask.mean()),
                        "combined_mask_path": str(mask_path),
                        "combined_overlay_path": str(overlay_path),
                        "probability_map_path": str(probability_path),
                        "eye_exclusion_enabled": args.eye_exclusion == "on",
                        "eye_exclusion_area_ratio": float(eye_exclusion_mask.mean()),
                        "eye_exclusion_mask_path": str(eye_exclusion_path),
                        "eye_exclusion": eye_exclusion_records,
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
