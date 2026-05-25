import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import pytest
import torch
import torch.nn.functional as F


def test_calibrator_initializes_as_identity_for_normalized_vectors():
    from experiments.invariant_calibration import TokenwiseLinearCalibrator

    calibrator = TokenwiseLinearCalibrator(dim=4)
    embeddings = F.normalize(torch.randn(3, 4), p=2, dim=-1)
    assert torch.allclose(calibrator(embeddings), embeddings, atol=1e-6)


def test_calibrator_keeps_shape_and_normalizes_output():
    from experiments.invariant_calibration import TokenwiseLinearCalibrator

    calibrator = TokenwiseLinearCalibrator(dim=4)
    output = calibrator(torch.randn(5, 4))
    assert output.shape == (5, 4)
    assert torch.allclose(output.norm(dim=-1), torch.ones(5), atol=1e-6)


def test_info_nce_loss_is_finite():
    from experiments.invariant_calibration import info_nce_loss

    clean = F.normalize(torch.randn(4, 8), p=2, dim=-1)
    degraded = F.normalize(torch.randn(4, 8), p=2, dim=-1)
    loss = info_nce_loss(clean, degraded)
    assert torch.isfinite(loss)


def test_calibration_loss_is_finite():
    from experiments.invariant_calibration import TokenwiseLinearCalibrator, calibration_loss

    calibrator = TokenwiseLinearCalibrator(dim=8)
    clean = [F.normalize(torch.randn(6, 8), p=2, dim=-1) for _ in range(3)]
    degraded = [F.normalize(torch.randn(6, 8), p=2, dim=-1) for _ in range(3)]
    losses = calibration_loss(calibrator, clean, degraded)
    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.info_nce)
    assert torch.isfinite(losses.cosine_alignment)
    assert torch.isfinite(losses.identity_regularization)


def test_checkpoint_round_trip_preserves_output(tmp_path):
    from experiments.invariant_calibration import TokenwiseLinearCalibrator

    calibrator = TokenwiseLinearCalibrator(dim=4)
    embeddings = torch.randn(2, 4)
    expected = calibrator(embeddings)

    path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": calibrator.state_dict(), "dim": 4}, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    loaded = TokenwiseLinearCalibrator(dim=payload["dim"])
    loaded.load_state_dict(payload["state_dict"])
    assert torch.allclose(loaded(embeddings), expected)


def test_embedding_cache_rejects_metadata_mismatch(tmp_path):
    from experiments.invariant_embeddings import load_embedding_cache, save_embedding_cache

    cache_path = tmp_path / "cache.pt"
    page_paths = [tmp_path / "page_001.png"]
    save_embedding_cache(
        cache_path,
        doc_id="doc-a",
        mode="clean",
        variant="clean",
        page_paths=page_paths,
        embeddings=[torch.randn(2, 4)],
    )

    with pytest.raises(ValueError, match="doc_id"):
        load_embedding_cache(cache_path, doc_id="doc-b", mode="clean", variant="clean", page_paths=page_paths)
