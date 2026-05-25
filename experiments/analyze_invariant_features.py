#!/usr/bin/env python
"""Analyze clean/degraded embedding drift for the local HR page subset."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from experiments.invariant_calibration import mean_pool_embedding
from experiments.invariant_embeddings import DEFAULT_CACHE_ROOT, load_or_encode_page_embeddings
from experiments.run_local_hr_benchmark import (
    DATASET_ROOT,
    DEFAULT_DOC_ID,
    DEFAULT_MODEL_PATH,
    DEFAULT_VARIANT,
    compute_metrics,
    encode_queries,
    load_model,
    load_tables,
    select_queries,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "invariant_features"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze clean/degraded invariant-feature drift.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=1)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--include-cross-doc-queries", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--skip-score-analysis", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def pearson_corr(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().flatten()
    right = right.float().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denom = left.norm() * right.norm()
    if denom.item() == 0:
        return 0.0
    return float((left @ right / denom).item())


def simple_rank(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten()
    order = torch.argsort(flat)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(len(flat), dtype=torch.float32)
    return ranks.reshape(values.shape)


def spearman_corr(left: torch.Tensor, right: torch.Tensor) -> float:
    return pearson_corr(simple_rank(left), simple_rank(right))


def first_relevant_rank(scores: Sequence[float], relevant_pages: set[int]) -> int | None:
    ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    for pos, doc_idx in enumerate(ranked, start=1):
        if doc_idx in relevant_pages:
            return pos
    return None


def write_page_similarity(
    output_path: Path,
    clean_embeddings: Sequence[torch.Tensor],
    degraded_embeddings: Sequence[torch.Tensor],
) -> dict[str, float]:
    rows = []
    cosines = []
    distances = []
    for page_idx, (clean_embedding, degraded_embedding) in enumerate(zip(clean_embeddings, degraded_embeddings)):
        clean_vector = mean_pool_embedding(clean_embedding)
        degraded_vector = mean_pool_embedding(degraded_embedding)
        cosine = float(F.cosine_similarity(clean_vector, degraded_vector, dim=0).item())
        distance = float(torch.linalg.vector_norm(clean_vector - degraded_vector).item())
        rows.append({"page_index": page_idx, "page_file_index": page_idx + 1, "cosine": cosine, "l2": distance})
        cosines.append(cosine)
        distances.append(distance)

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["page_index", "page_file_index", "cosine", "l2"])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "page_count": len(rows),
        "mean_page_cosine": sum(cosines) / len(cosines),
        "min_page_cosine": min(cosines),
        "mean_page_l2": sum(distances) / len(distances),
        "max_page_l2": max(distances),
    }


def write_query_drop_cases(
    output_path: Path,
    clean_scores: torch.Tensor,
    degraded_scores: torch.Tensor,
    query_ids: Sequence[int],
    relevant_pages: dict[int, set[int]],
) -> dict[str, int]:
    rows = []
    worsened = 0
    improved = 0
    unchanged = 0
    for row_idx, query_id in enumerate(query_ids):
        relevant = relevant_pages[int(query_id)]
        clean_rank = first_relevant_rank(clean_scores[row_idx].tolist(), relevant)
        degraded_rank = first_relevant_rank(degraded_scores[row_idx].tolist(), relevant)
        if clean_rank is not None and degraded_rank is not None:
            if degraded_rank > clean_rank:
                worsened += 1
            elif degraded_rank < clean_rank:
                improved += 1
            else:
                unchanged += 1
        rows.append(
            {
                "query_id": int(query_id),
                "clean_first_relevant_rank": clean_rank,
                "degraded_first_relevant_rank": degraded_rank,
                "rank_delta": None if clean_rank is None or degraded_rank is None else degraded_rank - clean_rank,
                "relevant_pages": " ".join(str(page) for page in sorted(relevant)),
            }
        )

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query_id",
                "clean_first_relevant_rank",
                "degraded_first_relevant_rank",
                "rank_delta",
                "relevant_pages",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {"worsened_queries": worsened, "improved_queries": improved, "unchanged_queries": unchanged}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor_name = args.processor or args.model
    model, processor = load_model(args.model, processor_name, args.device, args.local_files_only)

    clean_embeddings, clean_paths, clean_cache = load_or_encode_page_embeddings(
        model=model,
        processor=processor,
        dataset_root=args.dataset_root,
        doc_id=args.doc_id,
        mode="clean",
        variant="clean",
        batch_size=args.batch_size,
        device=args.device,
        cache_root=args.cache_root,
        max_docs=args.max_docs,
        overwrite_cache=args.overwrite_cache,
    )
    degraded_embeddings, degraded_paths, degraded_cache = load_or_encode_page_embeddings(
        model=model,
        processor=processor,
        dataset_root=args.dataset_root,
        doc_id=args.doc_id,
        mode="degraded",
        variant=args.variant,
        batch_size=args.batch_size,
        device=args.device,
        cache_root=args.cache_root,
        max_docs=args.max_docs,
        overwrite_cache=args.overwrite_cache,
    )

    page_summary = write_page_similarity(
        args.output_dir / "page_level_similarity.csv",
        clean_embeddings,
        degraded_embeddings,
    )
    summary = {
        "doc_id": args.doc_id,
        "variant": args.variant,
        "clean_cache": str(clean_cache),
        "degraded_cache": str(degraded_cache),
        "clean_page_count": len(clean_paths),
        "degraded_page_count": len(degraded_paths),
        "page_similarity": page_summary,
    }

    if not args.skip_score_analysis:
        tables = load_tables(args.dataset_root)
        selected_queries, relevant_pages, query_summary = select_queries(
            tables["corpus"],
            tables["queries"],
            tables["qrels"],
            doc_id=args.doc_id,
            include_cross_doc_queries=args.include_cross_doc_queries,
            max_queries=args.max_queries,
        )
        query_texts = selected_queries["query"].tolist()
        query_ids = selected_queries["query_id"].tolist()
        query_embeddings = encode_queries(model, processor, query_texts, args.query_batch_size, args.device)

        clean_scores = processor.score_multi_vector(
            query_embeddings,
            clean_embeddings,
            batch_size=args.score_batch_size,
            device=args.device,
        )
        degraded_scores = processor.score_multi_vector(
            query_embeddings,
            degraded_embeddings,
            batch_size=args.score_batch_size,
            device=args.device,
        )

        summary["query_summary"] = query_summary
        summary["clean_metrics"] = compute_metrics(clean_scores, query_ids, relevant_pages)
        summary["degraded_metrics"] = compute_metrics(degraded_scores, query_ids, relevant_pages)
        summary["score_correlation"] = {
            "pearson": pearson_corr(clean_scores, degraded_scores),
            "spearman": spearman_corr(clean_scores, degraded_scores),
        }
        summary["retrieval_drop_cases"] = write_query_drop_cases(
            args.output_dir / "query_drop_cases.csv",
            clean_scores,
            degraded_scores,
            query_ids,
            relevant_pages,
        )

    summary_path = args.output_dir / "feature_drift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
