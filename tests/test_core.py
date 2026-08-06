import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from sam3_face_attributes.core import (
    aggregate_evidence,
    normalize_masks,
    normalize_scores,
    visual_evidence,
)


def test_normalize_masks_adds_batch_dimension():
    mask = np.array([[0, 1], [1, 0]])
    result = normalize_masks(mask)
    assert result.shape == (1, 2, 2)
    assert result.dtype == bool


def test_normalize_scores_pads_missing_values():
    result = normalize_scores([0.7], 3)
    np.testing.assert_allclose(result, [0.7, 0.0, 0.0])


def test_aggregate_evidence_unions_masks_and_scores():
    left = np.array([[1, 0], [0, 0]], dtype=bool)
    right = np.array([[0, 1], [0, 0]], dtype=bool)
    confidence, area_ratio, union = aggregate_evidence(
        [("a", 0.5, left), ("b", 0.5, right)], (2, 2)
    )
    assert confidence == 0.75
    assert area_ratio == 0.5
    assert union.sum() == 2


def test_visual_evidence_is_low_on_uniform_region():
    image = Image.new("RGB", (32, 32), (128, 128, 128))
    mask = np.zeros((32, 32), dtype=bool)
    mask[10:20, 10:20] = True
    result = visual_evidence(image, mask)
    assert result["evidence"] < 1e-6
