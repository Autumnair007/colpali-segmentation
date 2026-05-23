"""Utilities for training-free multi-view score fusion."""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import torch
from PIL import Image

from robust.restoration.deblur import deblur_wiener
from robust.restoration.denoise import denoise_gaussian, denoise_nlmeans


ViewPreprocessor = Callable[[Image.Image], Image.Image]

AVAILABLE_VIEWS = ("identity", "nlmeans", "gaussian", "wiener")
DEFAULT_VIEW_WEIGHTS = {
    "identity": 0.70,
    "nlmeans": 0.10,
    "gaussian": 0.10,
    "wiener": 0.10,
}


def validate_views(views: Sequence[str]) -> List[str]:
    """Validate and normalize a view list while preserving order."""
    if not views:
        raise ValueError("At least one view must be provided.")

    normalized = list(views)
    unknown = sorted(set(normalized) - set(AVAILABLE_VIEWS))
    if unknown:
        raise ValueError(f"Unknown multiview view(s): {unknown}. Available: {list(AVAILABLE_VIEWS)}")

    duplicates = sorted({view for view in normalized if normalized.count(view) > 1})
    if duplicates:
        raise ValueError(f"Duplicate multiview view(s): {duplicates}")

    return normalized


def get_view_preprocessor(view: str) -> ViewPreprocessor:
    """Return a single-image preprocessor for one multiview branch."""
    validate_views([view])
    if view == "identity":
        return lambda img: img
    if view == "nlmeans":
        return lambda img: denoise_nlmeans(img, h=5)
    if view == "gaussian":
        return lambda img: denoise_gaussian(img, sigma=0.8)
    if view == "wiener":
        return lambda img: deblur_wiener(img, noise_power=0.01)
    raise AssertionError(f"Unhandled view: {view}")


def build_view_batch(images: Sequence[Image.Image], view: str) -> List[Image.Image]:
    """Apply one view preprocessor to a batch of images."""
    preprocessor = get_view_preprocessor(view)
    return [preprocessor(image) for image in images]


def normalize_view_weights(
    views: Sequence[str],
    base_weights: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Return selected view weights normalized to sum to 1."""
    selected = validate_views(views)
    weights = base_weights or DEFAULT_VIEW_WEIGHTS

    missing = sorted(set(selected) - set(weights))
    if missing:
        raise ValueError(f"Missing weight(s) for view(s): {missing}")

    selected_weights = {view: float(weights[view]) for view in selected}
    if any(weight < 0 for weight in selected_weights.values()):
        raise ValueError("View weights must be non-negative.")

    total = sum(selected_weights.values())
    if total <= 0:
        raise ValueError("At least one selected view must have a positive weight.")

    return {view: weight / total for view, weight in selected_weights.items()}


def fuse_score_matrices(
    score_matrices: Dict[str, torch.Tensor],
    fusion: str,
    weights: Dict[str, float] | None = None,
) -> torch.Tensor:
    """Fuse per-view score matrices into one retrieval score matrix."""
    if not score_matrices:
        raise ValueError("No score matrices provided.")

    views = list(score_matrices)
    shapes = {tuple(scores.shape) for scores in score_matrices.values()}
    if len(shapes) != 1:
        raise ValueError(f"All score matrices must have the same shape, got: {sorted(shapes)}")

    if fusion == "max":
        return torch.stack([score_matrices[view] for view in views], dim=0).max(dim=0).values

    if fusion == "mean":
        return torch.stack([score_matrices[view] for view in views], dim=0).mean(dim=0)

    if fusion == "weighted":
        normalized_weights = normalize_view_weights(views, weights)
        fused = None
        for view in views:
            term = score_matrices[view] * normalized_weights[view]
            fused = term if fused is None else fused + term
        assert fused is not None
        return fused

    raise ValueError(f"Unknown fusion strategy: {fusion!r}. Available: ['max', 'mean', 'weighted']")
