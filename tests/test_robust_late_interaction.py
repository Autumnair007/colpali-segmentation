import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import pytest
import torch


def test_robust_max_matches_standard_fixed_length():
    from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
    from experiments.robust_late_interaction import score_multi_vector_robust

    qs = [torch.randn(3, 4), torch.randn(3, 4)]
    ps = [torch.randn(5, 4), torch.randn(5, 4)]

    expected = BaseVisualRetrieverProcessor.score_multi_vector(qs, ps, batch_size=2, device="cpu")
    actual = score_multi_vector_robust(qs, ps, reduction="max", batch_size=2, device="cpu")

    assert torch.allclose(actual, expected)


def test_topk_mean_averages_strongest_doc_tokens():
    from experiments.robust_late_interaction import score_multi_vector_robust

    qs = [torch.tensor([[1.0, 0.0]])]
    ps = [torch.tensor([[1.0, 0.0], [0.5, 0.0], [-1.0, 0.0]])]

    scores = score_multi_vector_robust(qs, ps, reduction="topk_mean", top_k=2, device="cpu")

    assert torch.allclose(scores, torch.tensor([[0.75]]))


def test_topk_mean_ignores_padding_tokens():
    from experiments.robust_late_interaction import score_multi_vector_robust

    qs = [torch.tensor([[1.0, 0.0]])]
    ps = [
        torch.tensor([[-1.0, 0.0]]),
        torch.tensor([[-1.0, 0.0], [-0.5, 0.0], [-0.25, 0.0]]),
    ]

    scores = score_multi_vector_robust(qs, ps, reduction="topk_mean", top_k=2, batch_size=2, device="cpu")

    assert torch.allclose(scores[0, 0], torch.tensor(-1.0))
    assert torch.allclose(scores[0, 1], torch.tensor(-0.375))


def test_smoothmax_stays_finite():
    from experiments.robust_late_interaction import score_multi_vector_robust

    qs = [torch.randn(3, 8)]
    ps = [torch.randn(4, 8), torch.randn(5, 8)]

    scores = score_multi_vector_robust(qs, ps, reduction="smoothmax", temperature=0.1, device="cpu")

    assert scores.shape == (1, 2)
    assert torch.isfinite(scores).all()


def test_invalid_reduction_raises():
    from experiments.robust_late_interaction import score_multi_vector_robust

    with pytest.raises(ValueError, match="Unknown reduction"):
        score_multi_vector_robust([torch.randn(1, 2)], [torch.randn(1, 2)], reduction="bad", device="cpu")
