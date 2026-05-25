#!/usr/bin/env python
"""Train and evaluate a lightweight degradation-invariant embedding calibrator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
from tqdm import tqdm

from experiments.invariant_calibration import (
    DEFAULT_ALPHA,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BETA,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_TEMPERATURE,
    TokenwiseLinearCalibrator,
    apply_calibrator,
    calibration_loss,
)
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "invariant_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen-backbone invariant embedding calibration.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    parser.add_argument("--train-variants", nargs="+", default=[DEFAULT_VARIANT])
    parser.add_argument("--eval-variant", default=DEFAULT_VARIANT)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-batch-size", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=1)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--include-cross-doc-queries", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def extend_training_pairs(
    clean_embeddings: Sequence[torch.Tensor],
    degraded_embeddings_by_variant: Sequence[Sequence[torch.Tensor]],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    clean_train: list[torch.Tensor] = []
    degraded_train: list[torch.Tensor] = []
    for degraded_embeddings in degraded_embeddings_by_variant:
        if len(clean_embeddings) != len(degraded_embeddings):
            raise ValueError("Clean and degraded embedding counts must match.")
        clean_train.extend(clean_embeddings)
        degraded_train.extend(degraded_embeddings)
    return clean_train, degraded_train


def train_calibrator(
    clean_train: Sequence[torch.Tensor],
    degraded_train: Sequence[torch.Tensor],
    args: argparse.Namespace,
) -> tuple[TokenwiseLinearCalibrator, list[dict[str, float]]]:
    device = torch.device(args.device)
    calibrator = TokenwiseLinearCalibrator(dim=clean_train[0].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    train_log: list[dict[str, float]] = []

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training calibrator"):
        order = torch.randperm(len(clean_train), generator=generator).tolist()
        totals = {"loss": 0.0, "info_nce": 0.0, "cosine_alignment": 0.0, "identity_regularization": 0.0}
        batch_count = 0

        calibrator.train()
        for start in range(0, len(order), args.batch_size):
            batch_indices = order[start : start + args.batch_size]
            clean_batch = [clean_train[idx] for idx in batch_indices]
            degraded_batch = [degraded_train[idx] for idx in batch_indices]

            optimizer.zero_grad(set_to_none=True)
            losses = calibration_loss(
                calibrator,
                clean_batch,
                degraded_batch,
                temperature=args.temperature,
                alpha=args.alpha,
                beta=args.beta,
            )
            losses.total.backward()
            optimizer.step()

            totals["loss"] += float(losses.total.detach().cpu())
            totals["info_nce"] += float(losses.info_nce.detach().cpu())
            totals["cosine_alignment"] += float(losses.cosine_alignment.detach().cpu())
            totals["identity_regularization"] += float(losses.identity_regularization.detach().cpu())
            batch_count += 1

        train_log.append({"epoch": epoch, **{key: value / batch_count for key, value in totals.items()}})

    return calibrator, train_log


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    train_degraded_embeddings = []
    train_caches = []
    for variant in args.train_variants:
        embeddings, _, cache_path = load_or_encode_page_embeddings(
            model=model,
            processor=processor,
            dataset_root=args.dataset_root,
            doc_id=args.doc_id,
            mode="degraded",
            variant=variant,
            batch_size=args.model_batch_size,
            device=args.device,
            cache_root=args.cache_root,
            max_docs=args.max_docs,
            overwrite_cache=args.overwrite_cache,
        )
        train_degraded_embeddings.append(embeddings)
        train_caches.append(str(cache_path))

    clean_train, degraded_train = extend_training_pairs(clean_embeddings, train_degraded_embeddings)
    calibrator, train_log = train_calibrator(clean_train, degraded_train, args)

    eval_degraded_embeddings, _, eval_cache = load_or_encode_page_embeddings(
        model=model,
        processor=processor,
        dataset_root=args.dataset_root,
        doc_id=args.doc_id,
        mode="degraded",
        variant=args.eval_variant,
        batch_size=args.model_batch_size,
        device=args.device,
        cache_root=args.cache_root,
        max_docs=args.max_docs,
        overwrite_cache=args.overwrite_cache,
    )
    calibrated_eval_embeddings = apply_calibrator(calibrator, eval_degraded_embeddings, args.device)

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
    scores_matrix = processor.score_multi_vector(
        query_embeddings,
        calibrated_eval_embeddings,
        batch_size=args.score_batch_size,
        device=args.device,
    )
    metrics = compute_metrics(scores_matrix, query_ids, relevant_pages)

    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "beta": args.beta,
        "seed": args.seed,
    }
    payload = {
        "method": "invariant_calibration",
        "doc_id": args.doc_id,
        "train_variants": args.train_variants,
        "eval_variant": args.eval_variant,
        "device": args.device,
        "model": args.model,
        "local_files_only": args.local_files_only,
        "clean_cache": str(clean_cache),
        "train_caches": train_caches,
        "eval_cache": str(eval_cache),
        "query_summary": query_summary,
        "training_config": training_config,
        "metrics": metrics,
    }

    checkpoint_path = args.output_dir / "calibration_checkpoint.pt"
    torch.save(
        {
            "state_dict": calibrator.state_dict(),
            "dim": calibrator.dim,
            "training_config": training_config,
            "payload": payload,
        },
        checkpoint_path,
    )
    (args.output_dir / "train_log.json").write_text(json.dumps(train_log, indent=2, ensure_ascii=False))
    eval_path = args.output_dir / f"eval_{args.eval_variant}.json"
    eval_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(json.dumps(metrics, indent=2))
    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved eval to: {eval_path}")


if __name__ == "__main__":
    main()
