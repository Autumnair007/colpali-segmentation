"""Losses and differentiable scoring for invariant adapter training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


DEFAULT_DISTILL_TEMPERATURE = 0.07


@dataclass
class InvariantAdapterLoss:
    total: torch.Tensor
    score_distill: torch.Tensor
    qrels_rank: torch.Tensor
    identity: torch.Tensor


def _length_mask(lengths: Sequence[int], device: str | torch.device) -> torch.Tensor:
    max_len = max(lengths)
    positions = torch.arange(max_len, device=device)
    return positions.unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)


def late_interaction_scores(
    query_embeddings: Sequence[torch.Tensor],
    doc_embeddings: Sequence[torch.Tensor],
    device: str | torch.device,
    doc_batch_size: int = 16,
) -> torch.Tensor:
    if not query_embeddings:
        raise ValueError("No query embeddings provided.")
    if not doc_embeddings:
        raise ValueError("No document embeddings provided.")
    if doc_batch_size <= 0:
        raise ValueError("doc_batch_size must be positive.")

    query_lengths = [embedding.shape[0] for embedding in query_embeddings]
    query_padded = torch.nn.utils.rnn.pad_sequence(
        [embedding.to(device).float() for embedding in query_embeddings],
        batch_first=True,
        padding_value=0,
    )
    query_mask = _length_mask(query_lengths, device)

    score_parts = []
    for start in range(0, len(doc_embeddings), doc_batch_size):
        doc_batch = doc_embeddings[start : start + doc_batch_size]
        doc_lengths = [embedding.shape[0] for embedding in doc_batch]
        doc_padded = torch.nn.utils.rnn.pad_sequence(
            [embedding.to(device).float() for embedding in doc_batch],
            batch_first=True,
            padding_value=0,
        )
        doc_mask = _length_mask(doc_lengths, device)
        similarities = torch.einsum("bnd,csd->bcns", query_padded, doc_padded)
        similarities = similarities.masked_fill(~doc_mask.unsqueeze(0).unsqueeze(2), float("-inf"))
        token_scores = similarities.max(dim=3).values
        token_scores = token_scores.masked_fill(~query_mask.unsqueeze(1), 0)
        score_parts.append(token_scores.sum(dim=2))
    return torch.cat(score_parts, dim=1)


def score_distillation_loss(
    adapted_scores: torch.Tensor,
    clean_teacher_scores: torch.Tensor,
    temperature: float = DEFAULT_DISTILL_TEMPERATURE,
) -> torch.Tensor:
    if adapted_scores.shape != clean_teacher_scores.shape:
        raise ValueError(f"Score shapes must match, got {tuple(adapted_scores.shape)} and {tuple(clean_teacher_scores.shape)}.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    student_log_probs = F.log_softmax(adapted_scores / temperature, dim=1)
    teacher_probs = F.softmax(clean_teacher_scores.detach() / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def multi_positive_ranking_loss(scores: torch.Tensor, relevant_pages: Sequence[set[int]]) -> torch.Tensor:
    if scores.shape[0] != len(relevant_pages):
        raise ValueError("scores rows must match relevant_pages length.")
    losses = []
    for row_idx, relevant in enumerate(relevant_pages):
        if not relevant:
            continue
        valid = sorted(page for page in relevant if 0 <= page < scores.shape[1])
        if not valid:
            continue
        numerator = torch.logsumexp(scores[row_idx, valid], dim=0)
        denominator = torch.logsumexp(scores[row_idx], dim=0)
        losses.append(-(numerator - denominator))
    if not losses:
        return scores.sum() * 0
    return torch.stack(losses).mean()


def identity_loss(original_embeddings: Sequence[torch.Tensor], adapted_embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(original_embeddings) != len(adapted_embeddings):
        raise ValueError("original_embeddings and adapted_embeddings must have the same length.")
    losses = []
    for original, adapted in zip(original_embeddings, adapted_embeddings):
        losses.append(1.0 - F.cosine_similarity(original.to(adapted.device).float(), adapted.float(), dim=-1).mean())
    return torch.stack(losses).mean()


def invariant_adapter_loss(
    adapted_scores: torch.Tensor,
    clean_teacher_scores: torch.Tensor,
    relevant_pages: Sequence[set[int]],
    original_embeddings: Sequence[torch.Tensor],
    adapted_embeddings: Sequence[torch.Tensor],
    temperature: float = DEFAULT_DISTILL_TEMPERATURE,
    score_weight: float = 1.0,
    rank_weight: float = 0.5,
    identity_weight: float = 0.05,
) -> InvariantAdapterLoss:
    distill = score_distillation_loss(adapted_scores, clean_teacher_scores, temperature)
    rank = multi_positive_ranking_loss(adapted_scores, relevant_pages)
    identity = identity_loss(original_embeddings, adapted_embeddings)
    total = score_weight * distill + rank_weight * rank + identity_weight * identity
    return InvariantAdapterLoss(total=total, score_distill=distill, qrels_rank=rank, identity=identity)
