#!/usr/bin/env python
"""Evaluate a trained invariant adapter on raw degraded and restored proxy embeddings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.invariant_embeddings import (
    DEFAULT_CACHE_ROOT,
    RESTORED_PROXY_CONFIGS,
    load_or_encode_page_embeddings,
    load_or_encode_restored_proxy_embeddings,
)
from experiments.invariant_splits import DEFAULT_OUTPUT_DIR, DEFAULT_SEED, load_or_build_manifest
from experiments.config import DEVICE
from experiments.run_invariant_adapter_training import (
    evaluate_adapter_on_embeddings,
    filter_queries_by_ids,
    query_ids_from_groups,
)
from experiments.run_local_hr_benchmark import (
    DATASET_ROOT,
    DEFAULT_DOC_ID,
    DEFAULT_MODEL_PATH,
    DEFAULT_VARIANT,
    PROJECT_ROOT,
    encode_queries,
    load_model,
    load_tables,
    select_queries,
)
from robust.invariant_learning import ResidualEmbeddingAdapter


DEFAULT_ADAPTER_CHECKPOINT = PROJECT_ROOT / "artifacts" / "invariant_adapter" / "tune_distill6_seed13" / "adapter_checkpoint.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained invariant adapter.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ADAPTER_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-batch-size", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=1)
    parser.add_argument("--doc-batch-size", type=int, default=16)
    parser.add_argument("--eval-raw-variant", default=DEFAULT_VARIANT)
    parser.add_argument("--eval-raw-variants", nargs="+", default=None)
    parser.add_argument("--restored-proxies", nargs="+", default=list(RESTORED_PROXY_CONFIGS))
    parser.add_argument("--skip-restored-proxies", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-query-groups", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=None)
    return parser.parse_args()


def load_adapter(checkpoint_path: Path, device: str) -> ResidualEmbeddingAdapter:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    adapter = ResidualEmbeddingAdapter(dim=int(payload["dim"]), hidden_dim=int(payload["hidden_dim"]))
    adapter.load_state_dict(payload["state_dict"])
    return adapter.to(device).eval()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, _ = load_or_build_manifest(
        path=args.split_manifest,
        dataset_root=args.dataset_root,
        doc_id=args.doc_id,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    test_groups = manifest["query_splits"]["test"][: args.max_query_groups]
    test_query_ids = query_ids_from_groups(test_groups)

    tables = load_tables(args.dataset_root)
    selected_queries, relevant_pages, _ = select_queries(
        tables["corpus"],
        tables["queries"],
        tables["qrels"],
        doc_id=args.doc_id,
        include_cross_doc_queries=False,
        max_queries=None,
    )
    processor_name = args.processor or args.model
    model, processor = load_model(args.model, processor_name, args.device, args.local_files_only)
    query_frame = filter_queries_by_ids(selected_queries, test_query_ids)
    query_embeddings = encode_queries(model, processor, query_frame["query"].tolist(), args.query_batch_size, args.device)
    adapter = load_adapter(args.checkpoint, args.device)

    clean_embeddings = load_or_encode_page_embeddings(
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
    )[0]
    raw_variants = args.eval_raw_variants or [args.eval_raw_variant]
    raw_paths = []
    restored_paths = []
    raw_metrics = {}

    for raw_variant in raw_variants:
        raw_embeddings = load_or_encode_page_embeddings(
            model=model,
            processor=processor,
            dataset_root=args.dataset_root,
            doc_id=args.doc_id,
            mode="degraded",
            variant=raw_variant,
            batch_size=args.model_batch_size,
            device=args.device,
            cache_root=args.cache_root,
            max_docs=args.max_docs,
            overwrite_cache=args.overwrite_cache,
        )[0]
        raw_payload = evaluate_adapter_on_embeddings(
            adapter,
            query_embeddings,
            test_query_ids,
            relevant_pages,
            clean_embeddings,
            raw_embeddings,
            device=args.device,
            doc_batch_size=args.doc_batch_size,
            max_docs=args.max_docs,
        )
        raw_payload.update({"method": "invariant_adapter", "variant": raw_variant, "checkpoint": str(args.checkpoint)})
        raw_path = args.output_dir / f"eval_raw_{raw_variant}.json"
        raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False))
        raw_paths.append(str(raw_path))
        raw_metrics[raw_variant] = raw_payload["metrics"]

        if args.skip_restored_proxies:
            continue

        for restoration in args.restored_proxies:
            restored_embeddings = load_or_encode_restored_proxy_embeddings(
                model=model,
                processor=processor,
                dataset_root=args.dataset_root,
                doc_id=args.doc_id,
                degraded_variant=raw_variant,
                restoration=restoration,
                batch_size=args.model_batch_size,
                device=args.device,
                cache_root=args.cache_root,
                max_docs=args.max_docs,
                overwrite_cache=args.overwrite_cache,
            )[0]
            payload = evaluate_adapter_on_embeddings(
                adapter,
                query_embeddings,
                test_query_ids,
                relevant_pages,
                clean_embeddings,
                restored_embeddings,
                device=args.device,
                doc_batch_size=args.doc_batch_size,
                max_docs=args.max_docs,
            )
            payload.update(
                {
                    "method": "invariant_adapter",
                    "variant": raw_variant,
                    "restored_proxy": restoration,
                    "checkpoint": str(args.checkpoint),
                }
            )
            suffix = restoration if len(raw_variants) == 1 else f"{restoration}_{raw_variant}"
            path = args.output_dir / f"eval_restored_proxy_{suffix}.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            restored_paths.append(str(path))

    print(json.dumps({"raw": raw_paths, "restored": restored_paths, "raw_metrics": raw_metrics}, indent=2))


if __name__ == "__main__":
    main()
