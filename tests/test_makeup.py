import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from infer_makeup import (
    Candidate,
    apply_eye_exclusion,
    deduplicate,
    fuse_candidates,
    max_face_coverage,
    route_names,
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
