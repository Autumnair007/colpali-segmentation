"""Degradation-invariant representation learning utilities."""

from robust.invariant_learning.adapter import ResidualEmbeddingAdapter, apply_adapter_to_embeddings
from robust.invariant_learning.losses import (
    InvariantAdapterLoss,
    identity_loss,
    invariant_adapter_loss,
    late_interaction_scores,
    multi_positive_ranking_loss,
    score_distillation_loss,
)

__all__ = [
    "InvariantAdapterLoss",
    "ResidualEmbeddingAdapter",
    "apply_adapter_to_embeddings",
    "identity_loss",
    "invariant_adapter_loss",
    "late_interaction_scores",
    "multi_positive_ranking_loss",
    "score_distillation_loss",
]
