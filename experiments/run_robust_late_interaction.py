#!/usr/bin/env python
"""Evaluate robust late-interaction scoring on the local HR degraded subset."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.invariant_embeddings import DEFAULT_CACHE_ROOT, load_or_encode_page_embeddings
from experiments.robust_late_interaction import VALID_REDUCTIONS, score_multi_vector_robust
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "robust_late_interaction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robust late-interaction scoring on cached local HR embeddings.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-batch-size", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=1)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--reduction", choices=VALID_REDUCTIONS, default="topk_mean")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--include-cross-doc-queries", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def recovery(method_value: float, clean_value: float, degraded_value: float) -> float | None:
    loss = clean_value - degraded_value
    if abs(loss) < 1e-12:
        return None
    return (method_value - degraded_value) / loss


def build_comparison(clean_metrics: dict[str, float], degraded_metrics: dict[str, float], method_metrics: dict[str, float]):
    comparison: dict[str, dict[str, float | None]] = {}
    for metric in ("ndcg@5", "recall@5", "mrr"):
        clean_value = clean_metrics[metric]
        degraded_value = degraded_metrics[metric]
        method_value = method_metrics[metric]
        comparison[metric] = {
            "degradation_loss": clean_value - degraded_value,
            "robust_gain": method_value - degraded_value,
            "recovery": recovery(method_value, clean_value, degraded_value),
        }
    return comparison


def save_payload(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scoring = payload["scoring"]
    if scoring["reduction"] == "topk_mean":
        suffix = f"topk{scoring['top_k']}"
    elif scoring["reduction"] == "smoothmax":
        suffix = f"smoothmax_tau{str(scoring['temperature']).replace('.', 'p')}"
    else:
        suffix = "max"
    path = output_dir / f"{timestamp}_robust_late_interaction_{suffix}_{payload['variant']}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def main() -> None:
    args = parse_args()
    processor_name = args.processor or args.model
    model, processor = load_model(args.model, processor_name, args.device, args.local_files_only)

    clean_embeddings, _, clean_cache = load_or_encode_page_embeddings(
        model=model,
        processor=processor,
        dataset_root=args.dataset_root,
        doc_id=args.doc_id,
        mode="clean",
        variant="clean",
        batch_size=args.model_batch_size,
        device=args.device,
        cache_root=args.cache_root,
        max_docs=args.max_docs,
        overwrite_cache=args.overwrite_cache,
    )
    degraded_embeddings, _, degraded_cache = load_or_encode_page_embeddings(
        model=model,
        processor=processor,
        dataset_root=args.dataset_root,
        doc_id=args.doc_id,
        mode="degraded",
        variant=args.variant,
        batch_size=args.model_batch_size,
        device=args.device,
        cache_root=args.cache_root,
        max_docs=args.max_docs,
        overwrite_cache=args.overwrite_cache,
    )

    tables = load_tables(args.dataset_root)
    selected_queries, relevant_pages, query_summary = select_queries(
        tables["corpus"],
        tables["queries"],
        tables["qrels"],
        doc_id=args.doc_id,
        include_cross_doc_queries=args.include_cross_doc_queries,
        max_queries=args.max_queries,
    )
    query_embeddings = encode_queries(
        model,
        processor,
        selected_queries["query"].tolist(),
        args.query_batch_size,
        args.device,
    )
    query_ids = selected_queries["query_id"].tolist()

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
    robust_scores = score_multi_vector_robust(
        query_embeddings,
        degraded_embeddings,
        reduction=args.reduction,
        top_k=args.top_k,
        temperature=args.temperature,
        batch_size=args.score_batch_size,
        device=args.device,
    )

    clean_metrics = compute_metrics(clean_scores, query_ids, relevant_pages)
    degraded_metrics = compute_metrics(degraded_scores, query_ids, relevant_pages)
    robust_metrics = compute_metrics(robust_scores, query_ids, relevant_pages)
    payload = {
        "method": "robust_late_interaction",
        "doc_id": args.doc_id,
        "variant": args.variant,
        "device": args.device,
        "model": args.model,
        "local_files_only": args.local_files_only,
        "clean_cache": str(clean_cache),
        "degraded_cache": str(degraded_cache),
        "query_summary": query_summary,
        "scoring": {
            "reduction": args.reduction,
            "top_k": args.top_k,
            "temperature": args.temperature,
        },
        "clean_metrics": clean_metrics,
        "degraded_baseline_metrics": degraded_metrics,
        "metrics": robust_metrics,
        "comparison": build_comparison(clean_metrics, degraded_metrics, robust_metrics),
    }

    output_path = save_payload(args.output_dir, payload)
    print(json.dumps(payload["metrics"], indent=2))
    print(json.dumps(payload["comparison"], indent=2))
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
