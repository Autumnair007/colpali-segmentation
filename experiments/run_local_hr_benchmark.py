#!/usr/bin/env python
"""
Local benchmark runner for the 111-page degraded HR document subset.

This script evaluates ColQwen2 on the clean pages that align with the degraded
subset, or on one degraded variant at a time. It reuses the existing model and
metrics code, but loads the local parquet / PNG files instead of the remote
ViDoRe subsets.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Set

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from colpali_engine.models import ColQwen2, ColQwen2Processor
from robust.evaluation.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "degraded_dataset"
DEFAULT_DOC_ID = "employment_and_social_developments_in_europe_2024-KEBD24002ENN"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "colqwen2-v1.0"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "local_hr"
DEFAULT_VARIANT = "PD_MB_GN_JC_LR_CS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local ColQwen2 retrieval on the degraded HR subset.")
    parser.add_argument("--mode", choices=["clean", "degraded"])
    parser.add_argument("--variant", default=DEFAULT_VARIANT, help="Degraded variant suffix to evaluate.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--processor", default=None, help="Defaults to the same path as --model.")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument(
        "--include-cross-doc-queries",
        action="store_true",
        help="Keep queries that also have relevant pages in other documents.",
    )
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--local-files-only", action="store_true", help="Use only local Hugging Face files.")
    parser.add_argument("--list-variants", action="store_true", help="Print available degraded variants and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset wiring without loading the model.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_tables(dataset_root: Path) -> Dict[str, pd.DataFrame]:
    return {
        "corpus": pd.read_parquet(dataset_root / "corpus" / "test-00000-of-00001.parquet"),
        "queries": pd.read_parquet(dataset_root / "queries" / "test-00000-of-00001.parquet"),
        "qrels": pd.read_parquet(dataset_root / "qrels" / "test-00000-of-00001.parquet"),
    }


def list_available_variants(dataset_root: Path) -> List[str]:
    sample_dir = dataset_root / "degraded_image" / "page_001"
    variants = []
    for path in sorted(sample_dir.glob("page_001_*.png")):
        variants.append(path.stem.removeprefix("page_001_"))
    return variants


def select_queries(
    corpus_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    doc_id: str,
    include_cross_doc_queries: bool,
    max_queries: int | None,
) -> tuple[pd.DataFrame, Dict[int, Set[int]], Dict[str, int]]:
    corpus_meta = corpus_df[["corpus_id", "doc_id", "page_number_in_doc"]]
    joined = qrels_df.merge(corpus_meta, on="corpus_id", how="left")

    focus_qrels = joined[joined["doc_id"] == doc_id].copy()
    focus_query_ids = sorted(focus_qrels["query_id"].unique().tolist())

    doc_counts = joined[joined["query_id"].isin(focus_query_ids)].groupby("query_id")["doc_id"].nunique()
    single_doc_query_ids = sorted(doc_counts[doc_counts == 1].index.tolist())

    if include_cross_doc_queries:
        selected_query_ids = focus_query_ids
    else:
        selected_query_ids = single_doc_query_ids

    if max_queries is not None:
        selected_query_ids = selected_query_ids[:max_queries]

    selected_qrels = focus_qrels[focus_qrels["query_id"].isin(selected_query_ids)]
    selected_queries = (
        queries_df[queries_df["query_id"].isin(selected_query_ids)][["query_id", "query"]]
        .sort_values("query_id")
        .reset_index(drop=True)
    )

    relevant_pages: Dict[int, Set[int]] = {}
    for query_id, group in selected_qrels.groupby("query_id"):
        relevant_pages[int(query_id)] = {int(page) for page in group["page_number_in_doc"].tolist()}

    summary = {
        "focus_query_count": len(focus_query_ids),
        "focus_single_doc_query_count": len(single_doc_query_ids),
        "focus_cross_doc_query_count": len(focus_query_ids) - len(single_doc_query_ids),
        "selected_query_count": len(selected_query_ids),
        "selected_qrel_count": len(selected_qrels),
    }
    return selected_queries, relevant_pages, summary


def build_page_paths(dataset_root: Path, doc_id: str, mode: str, variant: str, max_docs: int | None) -> List[Path]:
    if mode == "clean":
        base_dir = dataset_root / "images" / doc_id
        paths = sorted(base_dir.glob("page_*.png"))
    else:
        base_dir = dataset_root / "degraded_image"
        paths = [base_dir / f"page_{page_idx:03d}" / f"page_{page_idx:03d}_{variant}.png" for page_idx in range(1, 112)]

    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} page images, first missing path: {missing[0]}")

    if len(paths) != 111:
        raise ValueError(f"Expected 111 pages, found {len(paths)}")

    if max_docs is not None:
        paths = paths[:max_docs]

    return paths


def load_model(model_name: str, processor_name: str, device: str, local_files_only: bool):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device '{device}' requested, but torch.cuda.is_available() is False.")

    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=local_files_only,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(processor_name, local_files_only=local_files_only)
    return model, processor


def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def encode_queries(
    model: ColQwen2,
    processor: ColQwen2Processor,
    queries: Sequence[str],
    batch_size: int,
    device: str,
) -> List[torch.Tensor]:
    embeddings: List[torch.Tensor] = []
    for start in range(0, len(queries), batch_size):
        batch = processor.process_queries(list(queries[start : start + batch_size])).to(device)
        with torch.no_grad():
            batch_vecs = model(**batch)
        embeddings.extend([vec.cpu().float() for vec in batch_vecs])
    return embeddings


def encode_documents(
    model: ColQwen2,
    processor: ColQwen2Processor,
    page_paths: Sequence[Path],
    batch_size: int,
    device: str,
) -> List[torch.Tensor]:
    embeddings: List[torch.Tensor] = []
    for start in tqdm(range(0, len(page_paths), batch_size), desc="Encoding pages"):
        images = [open_rgb_image(path) for path in page_paths[start : start + batch_size]]
        batch = processor.process_images(images).to(device)
        with torch.no_grad():
            batch_vecs = model(**batch)
        embeddings.extend([vec.cpu().float() for vec in batch_vecs])
    return embeddings


def compute_metrics(scores_matrix: torch.Tensor, query_ids: Sequence[int], relevant_pages: Dict[int, Set[int]]) -> Dict[str, float]:
    ndcg_scores: List[float] = []
    recall_scores: List[float] = []
    mrr_scores: List[float] = []

    for row_idx, query_id in enumerate(query_ids):
        scores = scores_matrix[row_idx].tolist()
        ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        relevant = relevant_pages[int(query_id)]
        ndcg_scores.append(ndcg_at_k(scores, relevant, k=5))
        recall_scores.append(recall_at_k(ranked, relevant, k=5))
        mrr_scores.append(mean_reciprocal_rank(ranked, relevant))

    count = len(query_ids)
    return {
        "ndcg@5": sum(ndcg_scores) / count,
        "recall@5": sum(recall_scores) / count,
        "mrr": sum(mrr_scores) / count,
        "n_queries": count,
        "n_docs": scores_matrix.shape[1],
    }


def save_results(output_dir: Path, payload: Dict[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_tag = "singleview"
    variant = payload["variant"] if payload["mode"] == "degraded" else "clean"
    filename = f"{timestamp}_{payload['mode']}_{method_tag}_{variant}.json"
    path = output_dir / filename
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def main() -> None:
    args = parse_args()

    if args.list_variants:
        print("\n".join(list_available_variants(args.dataset_root)))
        return

    if args.mode is None:
        raise ValueError("--mode is required unless --list-variants is used.")

    tables = load_tables(args.dataset_root)
    selected_queries, relevant_pages, summary = select_queries(
        corpus_df=tables["corpus"],
        queries_df=tables["queries"],
        qrels_df=tables["qrels"],
        doc_id=args.doc_id,
        include_cross_doc_queries=args.include_cross_doc_queries,
        max_queries=args.max_queries,
    )
    page_paths = build_page_paths(args.dataset_root, args.doc_id, args.mode, args.variant, args.max_docs)

    payload: Dict[str, object] = {
        "mode": args.mode,
        "variant": args.variant,
        "doc_id": args.doc_id,
        "device": args.device,
        "model": args.model,
        "local_files_only": args.local_files_only,
        "include_cross_doc_queries": args.include_cross_doc_queries,
        "page_count": len(page_paths),
        "method": "singleview",
        "query_summary": summary,
    }

    print("=" * 60)
    print("Local HR benchmark")
    print("=" * 60)
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.dry_run:
        return

    processor_name = args.processor or args.model
    model, processor = load_model(
        model_name=args.model,
        processor_name=processor_name,
        device=args.device,
        local_files_only=args.local_files_only,
    )

    query_texts = selected_queries["query"].tolist()
    query_ids = selected_queries["query_id"].tolist()

    query_embeddings = encode_queries(model, processor, query_texts, args.batch_size, args.device)
    doc_embeddings = encode_documents(
        model,
        processor,
        page_paths,
        args.batch_size,
        args.device,
    )
    scores_matrix = processor.score_multi_vector(
        query_embeddings,
        doc_embeddings,
        batch_size=args.score_batch_size,
        device=args.device,
    )

    metrics = compute_metrics(scores_matrix, query_ids, relevant_pages)
    payload["metrics"] = metrics

    output_path = save_results(args.output_dir, payload)
    print(json.dumps(metrics, indent=2))
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
