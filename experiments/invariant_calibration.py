"""Lightweight degradation-invariant embedding calibration utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


DEFAULT_TEMPERATURE = 0.07
DEFAULT_ALPHA = 0.1
DEFAULT_BETA = 0.01
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16


class TokenwiseLinearCalibrator(torch.nn.Module):
    """A small token-wise linear map for frozen ColQwen2 document embeddings."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.projection = torch.nn.Linear(dim, dim, bias=False)
        torch.nn.init.eye_(self.projection.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.shape[-1] != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {embeddings.shape[-1]}.")
        projected = self.projection(embeddings.float())
        return F.normalize(projected, p=2, dim=-1)


@dataclass
class CalibrationLoss:
    total: torch.Tensor
    info_nce: torch.Tensor
    cosine_alignment: torch.Tensor
    identity_regularization: torch.Tensor


def mean_pool_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Mean-pool a multi-vector page embedding into one normalized page vector."""
    if embedding.ndim != 2:
        raise ValueError(f"Expected a 2D embedding tensor, got shape {tuple(embedding.shape)}.")
    return F.normalize(embedding.float().mean(dim=0), p=2, dim=0)


def stack_page_vectors(embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([mean_pool_embedding(embedding) for embedding in embeddings], dim=0)


def info_nce_loss(
    clean_page_vectors: torch.Tensor,
    calibrated_page_vectors: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    if clean_page_vectors.shape != calibrated_page_vectors.shape:
        raise ValueError(
            "clean_page_vectors and calibrated_page_vectors must have the same shape, "
            f"got {tuple(clean_page_vectors.shape)} and {tuple(calibrated_page_vectors.shape)}."
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    clean_page_vectors = F.normalize(clean_page_vectors.float(), p=2, dim=-1)
    calibrated_page_vectors = F.normalize(calibrated_page_vectors.float(), p=2, dim=-1)
    logits = calibrated_page_vectors @ clean_page_vectors.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def cosine_alignment_loss(clean_page_vectors: torch.Tensor, calibrated_page_vectors: torch.Tensor) -> torch.Tensor:
    clean_page_vectors = F.normalize(clean_page_vectors.float(), p=2, dim=-1)
    calibrated_page_vectors = F.normalize(calibrated_page_vectors.float(), p=2, dim=-1)
    return 1.0 - F.cosine_similarity(clean_page_vectors, calibrated_page_vectors, dim=-1).mean()


def identity_regularization(calibrator: TokenwiseLinearCalibrator) -> torch.Tensor:
    weight = calibrator.projection.weight
    eye = torch.eye(weight.shape[0], dtype=weight.dtype, device=weight.device)
    return torch.linalg.matrix_norm(weight - eye, ord="fro")


def calibration_loss(
    calibrator: TokenwiseLinearCalibrator,
    clean_embeddings: Sequence[torch.Tensor],
    degraded_embeddings: Sequence[torch.Tensor],
    temperature: float = DEFAULT_TEMPERATURE,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> CalibrationLoss:
    if len(clean_embeddings) != len(degraded_embeddings):
        raise ValueError("clean_embeddings and degraded_embeddings must have the same length.")
    if len(clean_embeddings) == 0:
        raise ValueError("At least one clean/degraded pair is required.")

    device = calibrator.projection.weight.device
    clean_page_vectors = stack_page_vectors([embedding.to(device) for embedding in clean_embeddings])
    calibrated_embeddings = [calibrator(embedding.to(device)) for embedding in degraded_embeddings]
    calibrated_page_vectors = stack_page_vectors(calibrated_embeddings)

    nce = info_nce_loss(clean_page_vectors, calibrated_page_vectors, temperature)
    align = cosine_alignment_loss(clean_page_vectors, calibrated_page_vectors)
    identity = identity_regularization(calibrator)
    total = nce + alpha * align + beta * identity
    return CalibrationLoss(total=total, info_nce=nce, cosine_alignment=align, identity_regularization=identity)


def apply_calibrator(
    calibrator: TokenwiseLinearCalibrator,
    embeddings: Sequence[torch.Tensor],
    device: str | torch.device,
) -> list[torch.Tensor]:
    calibrator.eval()
    calibrated: list[torch.Tensor] = []
    with torch.no_grad():
        for embedding in embeddings:
            calibrated.append(calibrator(embedding.to(device)).cpu().float())
    return calibrated
