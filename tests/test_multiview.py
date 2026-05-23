# tests/test_multiview.py
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import pytest
import torch
from PIL import Image


def test_identity_only_weighted_equals_single_view():
    from experiments.multiview import fuse_score_matrices

    scores = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    fused = fuse_score_matrices({"identity": scores}, fusion="weighted")
    assert torch.equal(fused, scores)


def test_fusion_strategies_keep_shape():
    from experiments.multiview import fuse_score_matrices

    score_matrices = {
        "identity": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "nlmeans": torch.tensor([[2.0, 1.0], [5.0, 1.0]]),
    }

    assert fuse_score_matrices(score_matrices, fusion="max").shape == (2, 2)
    assert fuse_score_matrices(score_matrices, fusion="mean").shape == (2, 2)
    assert fuse_score_matrices(score_matrices, fusion="weighted").shape == (2, 2)


def test_weighted_fusion_normalizes_selected_weights():
    from experiments.multiview import fuse_score_matrices, normalize_view_weights

    weights = normalize_view_weights(["identity", "nlmeans"])
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert abs(weights["identity"] - 0.875) < 1e-6
    assert abs(weights["nlmeans"] - 0.125) < 1e-6

    score_matrices = {
        "identity": torch.ones((1, 2)),
        "nlmeans": torch.full((1, 2), 9.0),
    }
    fused = fuse_score_matrices(score_matrices, fusion="weighted")
    assert torch.allclose(fused, torch.full((1, 2), 2.0))


def test_invalid_view_name_raises():
    from experiments.multiview import validate_views

    with pytest.raises(ValueError, match="Unknown multiview view"):
        validate_views(["identity", "bogus"])


def test_duplicate_view_name_raises():
    from experiments.multiview import validate_views

    with pytest.raises(ValueError, match="Duplicate multiview view"):
        validate_views(["identity", "identity"])


def test_identity_view_keeps_image_unchanged():
    from experiments.multiview import build_view_batch

    image = Image.new("RGB", (8, 8), "white")
    assert build_view_batch([image], "identity") == [image]
