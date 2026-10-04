import sys
import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from infer_jewelry import (
    Candidate,
    JewelrySpec,
    Roi,
    candidate_quality,
    count_border_contacts,
    deduplicate,
    fuse_candidates,
    lip_anatomy_masks,
    lip_boundary_fractions,
    load_jewelry_specs,
    piercing_highlight_contrast,
    route_names,
    run_branch,
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


def test_lip_jewelry_must_cross_lip_boundary():
    surface = np.zeros((10, 10), dtype=bool)
    vicinity = np.zeros((10, 10), dtype=bool)
    surface[4:6, 3:7] = True
    vicinity[2:8, 1:9] = True
    inside = np.zeros((10, 10), dtype=bool)
    inside[4, 4] = True
    crossing = np.zeros((10, 10), dtype=bool)
    crossing[5:7, 4] = True
    assert lip_boundary_fractions(inside, surface, vicinity) == (0.0, 1.0)
    assert lip_boundary_fractions(crossing, surface, vicinity) == (0.5, 1.0)


def test_lip_vicinity_extends_beyond_lip_surface():
    from infer_jewelry import OUTER_LIPS

    points = [(0.0, 0.0)] * 478
    for i, index in enumerate(OUTER_LIPS):
        angle = 2 * np.pi * i / len(OUTER_LIPS)
        points[index] = (25 + 10 * np.cos(angle), 25 + 4 * np.sin(angle))
    surface, vicinity = lip_anatomy_masks(points, Image.new("RGB", (50, 50)))
    assert surface.any()
    assert np.all(vicinity[surface])
    assert vicinity.sum() > surface.sum()


def test_piercing_highlight_rejects_dark_brow_hair():
    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 10] = True
    gray = np.full((20, 20), 0.6, dtype=np.float32)
    gray[10, 10] = 0.2
    assert piercing_highlight_contrast(gray, mask) < 0
    gray[10, 10] = 0.95
    assert piercing_highlight_contrast(gray, mask) > 0.08


def test_branch_filters_lip_surface_even_when_sam_score_is_high():
    image = Image.new("RGB", (20, 20), "white")
    mask = np.zeros((1, 20, 20), dtype=bool)
    mask[0, 10, 10] = True
    processor = SimpleNamespace(set_text_prompt=lambda **_: {
        "masks": mask, "scores": [0.9],
    })
    spec = JewelrySpec("lip_jewelry", ("lip ring",), 0.3, 0.45, 0.03, 0.004)
    args = SimpleNamespace(device="cpu", dtype="float32", max_masks_per_prompt=5,
                           max_border_contacts=2, min_jewelry_highlight=0.08)
    surface = np.zeros((20, 20), dtype=bool)
    surface[9:12, 8:13] = True
    vicinity = np.zeros((20, 20), dtype=bool)
    vicinity[7:14, 6:15] = True
    with patch("infer_jewelry.autocast_context", return_value=contextlib.nullcontext()):
        accepted, rejected = run_branch(
            processor, image, Roi("lips", (0, 0, 20, 20), 0.98), {}, 0,
            400, spec, args, lip_surface=surface, lip_vicinity=vicinity,
        )
    assert not accepted
    assert rejected[0]["reason"] == "mask_is_lip_surface"


def test_branch_keeps_small_bright_piercing_at_lip_edge():
    rgb = np.full((20, 20, 3), 80, dtype=np.uint8)
    rgb[10, 10] = 245
    rgb[12, 10] = 245
    image = Image.fromarray(rgb)
    mask = np.zeros((1, 20, 20), dtype=bool)
    mask[0, 10, 10] = True
    mask[0, 12, 10] = True
    processor = SimpleNamespace(set_text_prompt=lambda **_: {
        "masks": mask, "scores": [0.9],
    })
    spec = JewelrySpec("lip_jewelry", ("lip ring",), 0.3, 0.45, 0.03, 0.004)
    args = SimpleNamespace(device="cpu", dtype="float32", max_masks_per_prompt=5,
                           max_border_contacts=2, min_jewelry_highlight=0.08)
    surface = np.zeros((20, 20), dtype=bool)
    surface[9:12, 8:13] = True
    vicinity = np.zeros((20, 20), dtype=bool)
    vicinity[7:14, 6:15] = True
    with patch("infer_jewelry.autocast_context", return_value=contextlib.nullcontext()):
        accepted, rejected = run_branch(
            processor, image, Roi("lips", (0, 0, 20, 20), 0.98), {}, 0,
            1000, spec, args, lip_surface=surface, lip_vicinity=vicinity,
        )
    assert len(accepted) == 1
    assert not rejected


def test_piercing_coverage_limits_are_localized():
    specs = {item.name: item for item in load_jewelry_specs(
        Path(__file__).resolve().parents[1] / "configs" / "jewelry.json"
    )}
    assert specs["lip_jewelry"].max_face_coverage == 0.004
    assert specs["eyebrow_jewelry"].max_face_coverage == 0.003
    assert specs["nose_jewelry"].max_face_coverage == 0.08
    assert specs["face_gems"].min_sam_score == 0.60


def test_face_gem_outside_face_is_rejected():
    rgb = np.full((20, 20, 3), 80, dtype=np.uint8)
    rgb[2, 2] = 245
    image = Image.fromarray(rgb)
    mask = np.zeros((1, 20, 20), dtype=bool)
    mask[0, 2, 2] = True
    processor = SimpleNamespace(set_text_prompt=lambda **_: {
        "masks": mask, "scores": [0.9],
    })
    spec = JewelrySpec("face_gems", ("face gem",), 0.6, 0.47, 0.03, 0.03)
    args = SimpleNamespace(device="cpu", dtype="float32", max_masks_per_prompt=5,
                           max_border_contacts=2, min_jewelry_highlight=0.08)
    face = np.zeros((20, 20), dtype=bool)
    face[5:18, 5:18] = True
    lips = np.zeros((20, 20), dtype=bool)
    with patch("infer_jewelry.autocast_context", return_value=contextlib.nullcontext()):
        accepted, rejected = run_branch(
            processor, image, Roi("face", (0, 0, 20, 20), 0.88), {}, 0,
            400, spec, args, lip_surface=lips, face_surface=face,
        )
    assert not accepted
    assert rejected[0]["reason"] == "gem_outside_face"
