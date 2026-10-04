import sys
import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from infer_makeup import (
    Candidate,
    apply_eye_exclusion,
    deduplicate,
    eye_aperture_guard,
    fuse_candidates,
    is_confident_full_face_paint,
    limit_total_face_coverage,
    max_face_coverage,
    parse_args,
    required_visual_evidence,
    route_names,
    run_candidate_branch,
    Roi,
)


def candidate(mask, quality, attribute="eyeliner"):
    return Candidate(
        face_index=0,
        attribute=attribute,
        prompt="eyeliner",
        roi="left_eye",
        score=quality,
        quality=quality,
        mask=mask,
        metrics={},
        face_coverage=0.01,
        crop_coverage=0.1,
    )


def test_local_attribute_routes_to_both_eyes():
    assert route_names("eyeliner", True) == ("left_eye", "right_eye")


def test_global_attribute_keeps_face_and_full_image_branches():
    assert route_names("painted_face_pattern", True) == ("face",)


def test_deduplicate_keeps_best_overlapping_candidate():
    mask = np.ones((4, 4), dtype=bool)
    result = deduplicate([candidate(mask, 0.4), candidate(mask, 0.8)])
    assert len(result) == 1
    assert result[0].quality == 0.8


def test_low_quality_full_mask_does_not_pass_pixel_threshold():
    mask = np.ones((4, 4), dtype=bool)
    probability, combined = fuse_candidates([candidate(mask, 0.2)], (4, 4), 0.45)
    assert np.allclose(probability, 0.2)
    assert not combined.any()


def test_same_attribute_synonyms_do_not_accumulate():
    mask = np.ones((4, 4), dtype=bool)
    _, combined = fuse_candidates(
        [candidate(mask, 0.3), candidate(mask, 0.3)],
        (4, 4),
        0.45,
    )
    assert not combined.any()


def test_different_attributes_can_accumulate():
    mask = np.ones((4, 4), dtype=bool)
    _, combined = fuse_candidates(
        [
            candidate(mask, 0.3, "eyeliner"),
            candidate(mask, 0.3, "eyeshadow"),
        ],
        (4, 4),
        0.45,
    )
    assert combined.all()


def test_lip_gloss_has_local_face_coverage_limit():
    class Args:
        precise_max_face_coverage = 0.12
        regional_max_face_coverage = 0.35
        global_max_face_coverage = 0.92

    assert max_face_coverage("lip_gloss", Args()) == 0.12
    assert max_face_coverage("fake_blood", Args()) == 0.15
    assert max_face_coverage("painted_freckles", Args()) == 0.08


def test_eye_exclusion_removes_only_sam3_eye_pixels():
    probability = np.full((3, 3), 0.8, dtype=np.float32)
    combined = np.ones((3, 3), dtype=bool)
    eye_mask = np.zeros((3, 3), dtype=bool)
    eye_mask[1, 1] = True

    filtered_probability, filtered_mask = apply_eye_exclusion(
        probability,
        combined,
        eye_mask,
    )

    assert filtered_probability[1, 1] == 0.0
    assert not filtered_mask[1, 1]
    assert np.all(filtered_probability[~eye_mask] == 0.8)
    assert filtered_mask[~eye_mask].all()


def test_default_reviews_more_candidates_and_limits_global_coverage():
    args = parse_args(["--input", "/tmp/input.png", "--output-dir", "/tmp/masks",
                       "--checkpoint-path", "/tmp/sam3.pt"])
    assert args.max_masks_per_prompt == 5
    assert args.global_max_face_coverage < 0.5


def test_low_contrast_gate_uses_score_and_region_size():
    assert np.isclose(required_visual_evidence("cheek_blush", 0.64, 0.037, 0.08), 0.02)
    assert np.isclose(required_visual_evidence("upper_eyelid_eyeshadow", 0.75, 0.006, 0.08), 0.02)
    assert np.isclose(required_visual_evidence("cheek_blush", 0.55, 0.037, 0.08), 0.056)
    assert np.isclose(required_visual_evidence("cheek_blush", 0.64, 0.001, 0.08), 0.056)


def test_full_face_candidate_is_rejected_but_small_pigment_survives():
    image = Image.new("RGB", (100, 100), "white")
    masks = np.zeros((3, 100, 100), dtype=bool)
    masks[0, 10:90, 10:90] = True
    masks[1, 15:85, 15:85] = True
    masks[2, 20:30, 20:30] = True
    processor = SimpleNamespace(set_text_prompt=lambda **_: {
        "masks": masks, "scores": np.array([0.95, 0.8, 0.65]),
    })
    args = SimpleNamespace(device="cpu", dtype="float32", max_masks_per_prompt=5,
                           no_landmark_max_image_coverage=0.18,
                           max_crop_coverage=0.78,
                           precise_max_face_coverage=0.12,
                           regional_max_face_coverage=0.35,
                           global_max_face_coverage=0.38,
                           min_visual_evidence=0.08,
                           min_candidate_quality=0.48)
    with patch("infer_makeup.autocast_context", return_value=contextlib.nullcontext()), \
         patch("infer_makeup.visual_evidence", return_value={"evidence": 0.1}):
        accepted, rejected = run_candidate_branch(
            processor, image, Roi("face", (0, 0, 100, 100), 0.86), {}, 0,
            "painted_face_pattern", ("painted pattern",), 10000, args,
            min_sam_score=0.48,
        )
    assert len(accepted) == 1
    assert accepted[0].mask.sum() == 100
    assert [item["reason"] for item in rejected] == [
        "excessive_face_coverage", "excessive_face_coverage",
    ]


def test_without_landmarks_rejects_broad_image_mask():
    image = Image.new("RGB", (100, 100), "white")
    processor = SimpleNamespace(set_text_prompt=lambda **_: {
        "masks": np.ones((1, 100, 100), dtype=bool), "scores": [0.9],
    })
    args = SimpleNamespace(device="cpu", dtype="float32", max_masks_per_prompt=5,
                           no_landmark_max_image_coverage=0.18,
                           max_crop_coverage=0.78,
                           precise_max_face_coverage=0.12,
                           regional_max_face_coverage=0.35,
                           global_max_face_coverage=0.38,
                           min_visual_evidence=0.08,
                           min_candidate_quality=0.48)
    with patch("infer_makeup.autocast_context", return_value=contextlib.nullcontext()), \
         patch("infer_makeup.visual_evidence", return_value={"evidence": 0.1}):
        accepted, rejected = run_candidate_branch(
            processor, image, Roi("full_image", (0, 0, 100, 100), 0.68), {}, 0,
            "painted_face_pattern", ("painted pattern",), 10000, args,
            min_sam_score=0.48, has_landmarks=False,
        )
    assert not accepted
    assert rejected[0]["reason"] == "excessive_image_coverage_without_landmarks"


def test_local_low_contrast_mask_can_pass_when_sam_score_is_strong():
    image = Image.new("RGB", (100, 100), "white")
    mask = np.zeros((1, 100, 100), dtype=bool)
    mask[20:25, 20:30] = True
    processor = SimpleNamespace(set_text_prompt=lambda **_: {
        "masks": mask, "scores": [0.55],
    })
    args = SimpleNamespace(device="cpu", dtype="float32", max_masks_per_prompt=5,
                           no_landmark_max_image_coverage=0.18,
                           max_crop_coverage=0.78,
                           precise_max_face_coverage=0.12,
                           regional_max_face_coverage=0.35,
                           global_max_face_coverage=0.38,
                           min_visual_evidence=0.08,
                           min_candidate_quality=0.48)
    with patch("infer_makeup.autocast_context", return_value=contextlib.nullcontext()), \
         patch("infer_makeup.visual_evidence", return_value={"evidence": 0.04}):
        accepted, rejected = run_candidate_branch(
            processor, image, Roi("left_eye", (0, 0, 100, 100), 0.98), {}, 0,
            "eyeliner", ("eyeliner",), 10000, args, min_sam_score=0.42,
        )
    assert len(accepted) == 1
    assert not rejected


def test_attribute_sam_threshold_is_applied():
    image = Image.new("RGB", (10, 10), "white")
    mask = np.zeros((1, 10, 10), dtype=bool)
    mask[2:4, 2:4] = True
    processor = SimpleNamespace(set_text_prompt=lambda **_: {
        "masks": mask, "scores": [0.45],
    })
    args = SimpleNamespace(device="cpu", dtype="float32", max_masks_per_prompt=5,
                           no_landmark_max_image_coverage=0.18,
                           max_crop_coverage=0.78,
                           precise_max_face_coverage=0.12,
                           regional_max_face_coverage=0.35,
                           global_max_face_coverage=0.38,
                           min_visual_evidence=0.08,
                           min_candidate_quality=0.48)
    with patch("infer_makeup.autocast_context", return_value=contextlib.nullcontext()), \
         patch("infer_makeup.visual_evidence", return_value={"evidence": 0.2}):
        accepted, rejected = run_candidate_branch(
            processor, image, Roi("face", (0, 0, 10, 10), 0.86), {}, 0,
            "painted_face_pattern", ("painted pattern",), 100, args,
            min_sam_score=0.5,
        )
    assert not accepted
    assert rejected[0]["reason"] == "below_attribute_sam_score"


def test_total_coverage_limit_keeps_strongest_regions():
    first = np.zeros((10, 10), dtype=bool)
    second = np.zeros((10, 10), dtype=bool)
    first[:, :4] = True
    second[:, 4:8] = True
    accepted, rejected = limit_total_face_coverage(
        [candidate(first, 0.9), candidate(second, 0.8)], {0: 100}, 0.55,
    )
    assert len(accepted) == 1
    assert rejected[0]["reason"] == "excessive_total_face_coverage"


def test_total_coverage_reserves_space_for_local_makeup():
    broad = np.zeros((10, 10), dtype=bool)
    eyeliner = np.zeros((10, 10), dtype=bool)
    broad[:, :5] = True
    eyeliner[:, 5:7] = True
    accepted, rejected = limit_total_face_coverage(
        [candidate(broad, 0.95, "painted_face_pattern"),
         candidate(eyeliner, 0.6, "eyeliner")],
        {0: 100}, 0.55,
    )
    assert [item.attribute for item in accepted] == ["eyeliner"]
    assert rejected[0]["attribute"] == "painted_face_pattern"


def test_confident_theatrical_face_paint_can_cover_a_large_area():
    local = np.zeros((10, 10), dtype=bool)
    paint = np.zeros((10, 10), dtype=bool)
    local[:, :1] = True
    paint[:, 1:7] = True
    paint_candidate = candidate(paint, 0.9, "white_face_pigment")
    paint_candidate.metrics = {"evidence": 0.25}
    assert is_confident_full_face_paint(
        paint_candidate.attribute, paint_candidate.score, paint_candidate.metrics,
    )
    accepted, rejected = limit_total_face_coverage(
        [candidate(local, 0.6), paint_candidate], {0: 100}, 0.55,
    )
    assert len(accepted) == 2
    assert not rejected
    assert not is_confident_full_face_paint("white_face_pigment", 0.7, {"evidence": 0.3})
    assert not is_confident_full_face_paint("fake_blood", 0.9, {"evidence": 0.3})


def test_large_candidate_requires_a_confident_paint_attribute():
    image = Image.new("RGB", (10, 10), "white")
    mask = np.zeros((1, 10, 10), dtype=bool)
    mask[0, 2:8, :] = True
    score = [0.9]
    processor = SimpleNamespace(set_text_prompt=lambda **_: {
        "masks": mask, "scores": score,
    })
    args = SimpleNamespace(device="cpu", dtype="float32", max_masks_per_prompt=5,
                           no_landmark_max_image_coverage=0.18,
                           max_crop_coverage=0.78,
                           precise_max_face_coverage=0.12,
                           regional_max_face_coverage=0.35,
                           global_max_face_coverage=0.38,
                           min_visual_evidence=0.08,
                           min_candidate_quality=0.48)
    with patch("infer_makeup.autocast_context", return_value=contextlib.nullcontext()), \
         patch("infer_makeup.visual_evidence", return_value={"evidence": 0.25}):
        paint, _ = run_candidate_branch(
            processor, image, Roi("face", (0, 0, 10, 10), 0.86), {}, 0,
            "white_face_pigment", ("white paint",), 100, args,
        )
        score[0] = 0.7
        ordinary, ordinary_rejected = run_candidate_branch(
            processor, image, Roi("face", (0, 0, 10, 10), 0.86), {}, 0,
            "white_face_pigment", ("white paint",), 100, args,
        )
        score[0] = 0.9
        fake_blood, blood_rejected = run_candidate_branch(
            processor, image, Roi("face", (0, 0, 10, 10), 0.86), {}, 0,
            "fake_blood", ("fake blood",), 100, args,
        )
    assert len(paint) == 1
    assert not ordinary and ordinary_rejected[0]["reason"] == "excessive_face_coverage"
    assert not fake_blood and blood_rejected[0]["reason"] == "excessive_face_coverage"


def test_eye_aperture_guard_stays_inside_eye_boundary():
    image = Image.new("RGB", (40, 40))
    points = [(10, 10), (30, 10), (30, 30), (10, 30)]
    guard = eye_aperture_guard(points, [0, 1, 2, 3], image)
    assert guard[20, 20]
    assert not guard[10, 20]
