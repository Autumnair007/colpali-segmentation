"""Embedding cache helpers for clean/degraded page-pair experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from experiments.config import DEVICE
from experiments.run_local_hr_benchmark import (
    DEFAULT_DOC_ID,
    DEFAULT_OUTPUT_DIR,
    build_page_paths,
    encode_documents,
    open_rgb_image,
)
from robust.restoration.pipeline import RestorationPipeline


DEFAULT_CACHE_ROOT = DEFAULT_OUTPUT_DIR.parent / "invariant_cache"
RESTORED_PROXY_CONFIGS = {
    "nlmeans": [("nlmeans", {"h": 10})],
    "gaussian": [("gaussian", {"sigma": 1.5})],
    "wiener": [("wiener", {})],
}


def mean_pool_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Mean-pool a multi-vector page embedding into one normalized page vector."""
    if embedding.ndim != 2:
        raise ValueError(f"Expected a 2D embedding tensor, got shape {tuple(embedding.shape)}.")
    return F.normalize(embedding.float().mean(dim=0), p=2, dim=0)


def embedding_cache_path(cache_root: Path, doc_id: str, mode: str, variant: str, max_docs: int | None = None) -> Path:
    variant_tag = "clean" if mode == "clean" else variant
    max_docs_tag = "" if max_docs is None else f"_first{max_docs}"
    return cache_root / doc_id / f"{mode}_{variant_tag}{max_docs_tag}.pt"


def restored_proxy_variant_tag(base_variant: str, restoration: str) -> str:
    return f"{base_variant}__{restoration}"


def build_restoration_preprocessor(restoration: str) -> Callable[[Image.Image], Image.Image]:
    if restoration not in RESTORED_PROXY_CONFIGS:
        raise ValueError(f"Unknown restored proxy {restoration!r}. Available: {sorted(RESTORED_PROXY_CONFIGS)}")
    pipeline = RestorationPipeline(RESTORED_PROXY_CONFIGS[restoration])
    return pipeline


def encode_page_embeddings(
    model,
    processor,
    page_paths: Sequence[Path],
    batch_size: int,
    device: str,
    preprocess_fn: Callable[[Image.Image], Image.Image] | None = None,
) -> list[torch.Tensor]:
    if preprocess_fn is not None:
        embeddings: list[torch.Tensor] = []
        for start in tqdm(range(0, len(page_paths), batch_size), desc="Encoding pages"):
            images = [preprocess_fn(open_rgb_image(path)) for path in page_paths[start : start + batch_size]]
            batch = processor.process_images(images).to(device)
            with torch.no_grad():
                batch_vecs = model(**batch)
            embeddings.extend([vec.cpu().float() for vec in batch_vecs])
        return embeddings

    return encode_documents(
        model=model,
        processor=processor,
        page_paths=page_paths,
        batch_size=batch_size,
        device=device,
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
    device: str = DEVICE,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    max_docs: int | None = None,
    overwrite_cache: bool = False,
    preprocess_fn: Callable[[Image.Image], Image.Image] | None = None,
) -> tuple[list[torch.Tensor], list[Path], Path]:
    path_mode = "degraded" if mode == "restored_proxy" else mode
    path_variant = variant.split("__", 1)[0] if mode == "restored_proxy" else variant
    page_paths = build_page_paths(dataset_root, doc_id, path_mode, path_variant, max_docs)
    cache_path = embedding_cache_path(cache_root, doc_id, mode, variant, max_docs=max_docs)

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

    embeddings = encode_page_embeddings(model, processor, page_paths, batch_size, device, preprocess_fn=preprocess_fn)
    save_embedding_cache(cache_path, doc_id, mode, variant, page_paths, embeddings)
    return embeddings, page_paths, cache_path


def load_or_encode_restored_proxy_embeddings(
    model,
    processor,
    dataset_root: Path,
    doc_id: str,
    degraded_variant: str,
    restoration: str,
    batch_size: int = 4,
    device: str = DEVICE,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    max_docs: int | None = None,
    overwrite_cache: bool = False,
) -> tuple[list[torch.Tensor], list[Path], Path]:
    return load_or_encode_page_embeddings(
        model=model,
        processor=processor,
        dataset_root=dataset_root,
        doc_id=doc_id,
        mode="restored_proxy",
        variant=restored_proxy_variant_tag(degraded_variant, restoration),
        batch_size=batch_size,
        device=device,
        cache_root=cache_root,
        max_docs=max_docs,
        overwrite_cache=overwrite_cache,
        preprocess_fn=build_restoration_preprocessor(restoration),
    )
