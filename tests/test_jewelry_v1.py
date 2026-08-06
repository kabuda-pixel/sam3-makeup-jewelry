import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from infer_jewelry_v1 import (
    Candidate,
    candidate_quality,
    count_border_contacts,
    deduplicate,
    fuse_candidates,
    route_names,
)


def candidate(mask, quality=0.7, attribute="nose_jewelry"):
    return Candidate(
        face_index=0,
        attribute=attribute,
        prompt="nose ring",
        roi="nose",
        sam_score=quality,
        quality=quality,
        mask=mask,
        crop_coverage=0.05,
        face_coverage=0.01,
        border_contacts=0,
    )


def test_jewelry_routes_are_independent():
    assert route_names("ear_jewelry", True) == ("left_ear", "right_ear")
    assert route_names("neck_jewelry", True) == ("neck",)
    assert route_names("nose_jewelry", False) == ("full_image",)


def test_full_crop_touches_four_borders():
    assert count_border_contacts(np.ones((4, 4), dtype=bool)) == 4


def test_small_center_mask_touches_no_borders():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    assert count_border_contacts(mask) == 0


def test_duplicate_synonyms_keep_best_candidate():
    mask = np.ones((4, 4), dtype=bool)
    result = deduplicate(
        [candidate(mask, 0.5), candidate(mask, 0.8)],
        iou_threshold=0.84,
    )
    assert len(result) == 1
    assert result[0].quality == 0.8


def test_weak_synonyms_do_not_accumulate():
    mask = np.ones((4, 4), dtype=bool)
    probability, combined = fuse_candidates(
        [candidate(mask, 0.3), candidate(mask, 0.3)],
        (4, 4),
        pixel_threshold=0.46,
    )
    assert np.allclose(probability, 0.3)
    assert not combined.any()


def test_candidate_quality_rewards_small_precise_masks():
    small = candidate_quality(0.5, 0.95, 0.05, 0.3)
    large = candidate_quality(0.5, 0.95, 0.25, 0.3)
    assert small > large
