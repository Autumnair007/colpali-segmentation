"""Robust late-interaction scoring utilities for multi-vector retrieval."""
from __future__ import annotations

from typing import List, Sequence

import torch

from colpali_engine.utils.torch_utils import get_torch_device


VALID_REDUCTIONS = ("max", "topk_mean", "smoothmax")


def _as_tensor_list(values: torch.Tensor | Sequence[torch.Tensor], name: str) -> List[torch.Tensor]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 3:
            raise ValueError(f"{name} tensor must have shape (batch, seq_len, dim), got {tuple(values.shape)}.")
        return [item for item in values]

    values = list(values)
    if not values:
        raise ValueError(f"No {name} embeddings provided.")
    if any(value.ndim != 2 for value in values):
        shapes = [tuple(value.shape) for value in values]
        raise ValueError(f"{name} embeddings must be 2D tensors, got shapes: {shapes}.")
    return values


def _length_mask(lengths: Sequence[int], device: torch.device | str) -> torch.Tensor:
    max_len = max(lengths)
    positions = torch.arange(max_len, device=device)
    return positions.unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)


def _aggregate_doc_tokens(
    similarities: torch.Tensor,
    doc_mask: torch.Tensor,
    reduction: str,
    top_k: int,
    temperature: float,
) -> torch.Tensor:
    if reduction not in VALID_REDUCTIONS:
        raise ValueError(f"Unknown reduction {reduction!r}. Available: {list(VALID_REDUCTIONS)}.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    doc_mask_4d = doc_mask.unsqueeze(0).unsqueeze(2)

    if reduction == "max":
        masked = similarities.masked_fill(~doc_mask_4d, float("-inf"))
        return masked.max(dim=3).values

    if reduction == "topk_mean":
        masked = similarities.masked_fill(~doc_mask_4d, float("-inf"))
        k = min(top_k, similarities.shape[3])
        top_values = masked.topk(k=k, dim=3).values
        valid = torch.isfinite(top_values)
        counts = valid.sum(dim=3).clamp_min(1)
        return top_values.masked_fill(~valid, 0).sum(dim=3) / counts

    logits = similarities.masked_fill(~doc_mask_4d, float("-inf")) / temperature
    weights = torch.softmax(logits, dim=3).masked_fill(~doc_mask_4d, 0)
    masked_similarities = similarities.masked_fill(~doc_mask_4d, 0)
    return (weights * masked_similarities).sum(dim=3)


def score_multi_vector_robust(
    qs: torch.Tensor | Sequence[torch.Tensor],
    ps: torch.Tensor | Sequence[torch.Tensor],
    reduction: str = "topk_mean",
    top_k: int = 3,
    temperature: float = 0.05,
    batch_size: int = 128,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Compute late-interaction scores with a configurable document-token reducer.

    The standard ColBERT/ColPali reducer is ``max``. ``topk_mean`` averages the
    strongest k document-token responses for each query token, and ``smoothmax``
    uses a temperature-controlled softmax-weighted average. Both variants aim to
    reduce sensitivity to single noisy document patches without changing the
    frozen encoder outputs.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    query_embeddings = _as_tensor_list(qs, "query")
    passage_embeddings = _as_tensor_list(ps, "passage")
    device = device or get_torch_device("auto")

    scores_list: List[torch.Tensor] = []
    with torch.no_grad():
        for query_start in range(0, len(query_embeddings), batch_size):
            query_batch = query_embeddings[query_start : query_start + batch_size]
            query_lengths = [embedding.shape[0] for embedding in query_batch]
            query_padded = torch.nn.utils.rnn.pad_sequence(
                query_batch,
                batch_first=True,
                padding_value=0,
            ).to(device)
            query_mask = _length_mask(query_lengths, device=device)

            score_parts: List[torch.Tensor] = []
            for passage_start in range(0, len(passage_embeddings), batch_size):
                passage_batch = passage_embeddings[passage_start : passage_start + batch_size]
                passage_lengths = [embedding.shape[0] for embedding in passage_batch]
                passage_padded = torch.nn.utils.rnn.pad_sequence(
                    passage_batch,
                    batch_first=True,
                    padding_value=0,
                ).to(device)
                passage_mask = _length_mask(passage_lengths, device=device)

                similarities = torch.einsum("bnd,csd->bcns", query_padded, passage_padded)
                token_scores = _aggregate_doc_tokens(
                    similarities,
                    doc_mask=passage_mask,
                    reduction=reduction,
                    top_k=top_k,
                    temperature=temperature,
                )
                token_scores = token_scores.masked_fill(~query_mask.unsqueeze(1), 0)
                score_parts.append(token_scores.sum(dim=2))

            scores_list.append(torch.cat(score_parts, dim=1).cpu())

    scores = torch.cat(scores_list, dim=0).to(torch.float32)
    if scores.shape != (len(query_embeddings), len(passage_embeddings)):
        raise AssertionError(f"Expected {(len(query_embeddings), len(passage_embeddings))}, got {tuple(scores.shape)}.")
    return scores
