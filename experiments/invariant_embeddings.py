"""Embedding cache helpers for clean/degraded page-pair experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from experiments.run_local_hr_benchmark import (
    DEFAULT_DOC_ID,
    DEFAULT_OUTPUT_DIR,
    build_page_paths,
    encode_documents,
)


DEFAULT_CACHE_ROOT = DEFAULT_OUTPUT_DIR.parent / "invariant_cache"


def mean_pool_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Mean-pool a multi-vector page embedding into one normalized page vector."""
    if embedding.ndim != 2:
        raise ValueError(f"Expected a 2D embedding tensor, got shape {tuple(embedding.shape)}.")
    return F.normalize(embedding.float().mean(dim=0), p=2, dim=0)


def embedding_cache_path(cache_root: Path, doc_id: str, mode: str, variant: str) -> Path:
    variant_tag = "clean" if mode == "clean" else variant
    return cache_root / doc_id / f"{mode}_{variant_tag}.pt"


def encode_page_embeddings(
    model,
    processor,
    page_paths: Sequence[Path],
    batch_size: int,
    device: str,
) -> list[torch.Tensor]:
    return encode_documents(
        model=model,
        processor=processor,
        page_paths=page_paths,
        batch_size=batch_size,
        device=device,
        preprocess_view="identity",
    )


def save_embedding_cache(
    path: Path,
    doc_id: str,
    mode: str,
    variant: str,
    page_paths: Sequence[Path],
    embeddings: Sequence[torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_id": doc_id,
        "mode": mode,
        "variant": variant,
        "page_paths": [str(path) for path in page_paths],
        "embeddings": [embedding.cpu().float() for embedding in embeddings],
    }
    torch.save(payload, path)


def load_embedding_cache(
    path: Path,
    doc_id: str,
    mode: str,
    variant: str,
    page_paths: Sequence[Path] | None = None,
    expected_count: int | None = None,
) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    for key, expected in (("doc_id", doc_id), ("mode", mode), ("variant", variant)):
        if payload.get(key) != expected:
            raise ValueError(f"Cache metadata mismatch for {key}: expected {expected!r}, got {payload.get(key)!r}.")

    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list):
        raise ValueError("Cache payload is missing an embeddings list.")

    if expected_count is not None and len(embeddings) != expected_count:
        raise ValueError(f"Cache page count mismatch: expected {expected_count}, got {len(embeddings)}.")

    if page_paths is not None:
        expected_paths = [str(path) for path in page_paths]
        if payload.get("page_paths") != expected_paths:
            raise ValueError("Cache page paths do not match the requested page paths.")

    payload["embeddings"] = [embedding.cpu().float() for embedding in embeddings]
    return payload


def load_or_encode_page_embeddings(
    model,
    processor,
    dataset_root: Path,
    doc_id: str = DEFAULT_DOC_ID,
    mode: str = "clean",
    variant: str = "clean",
    batch_size: int = 4,
    device: str = "cuda:1",
    cache_root: Path = DEFAULT_CACHE_ROOT,
    max_docs: int | None = None,
    overwrite_cache: bool = False,
) -> tuple[list[torch.Tensor], list[Path], Path]:
    page_paths = build_page_paths(dataset_root, doc_id, mode, variant, max_docs)
    cache_path = embedding_cache_path(cache_root, doc_id, mode, variant)

    if cache_path.exists() and not overwrite_cache:
        payload = load_embedding_cache(
            cache_path,
            doc_id=doc_id,
            mode=mode,
            variant=variant,
            page_paths=page_paths,
            expected_count=len(page_paths),
        )
        return payload["embeddings"], page_paths, cache_path

    if model is None or processor is None:
        raise ValueError(f"Cache not found at {cache_path}; model and processor are required to encode embeddings.")

    embeddings = encode_page_embeddings(model, processor, page_paths, batch_size, device)
    save_embedding_cache(cache_path, doc_id, mode, variant, page_paths, embeddings)
    return embeddings, page_paths, cache_path
