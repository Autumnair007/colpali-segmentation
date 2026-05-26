#!/usr/bin/env python
"""Train a frozen-backbone retrieval-distillation adapter for degraded embeddings."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import torch

from experiments.invariant_embeddings import (
    DEFAULT_CACHE_ROOT,
    RESTORED_PROXY_CONFIGS,
    load_or_encode_page_embeddings,
    load_or_encode_restored_proxy_embeddings,
)
from experiments.config import DEVICE
from experiments.invariant_splits import DEFAULT_OUTPUT_DIR, DEFAULT_SEED, flatten_query_ids, load_or_build_manifest
from experiments.robust_late_interaction import score_multi_vector_robust
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
from robust.invariant_learning import (
    ResidualEmbeddingAdapter,
    apply_adapter_to_embeddings,
    invariant_adapter_loss,
    late_interaction_scores,
)


DEFAULT_TEMPERATURE = 0.07
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 10


def checkpoint_metadata(value: Any) -> Any:
    """Convert metadata to PyTorch weights-only-safe Python containers."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): checkpoint_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [checkpoint_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train invariant adapter on cached clean/degraded embeddings.")
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
    parser.add_argument("--batch-query-groups", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--identity-weight", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--init-scale", type=float, default=0.1)
    parser.add_argument("--eval-raw-variant", default=DEFAULT_VARIANT)
    parser.add_argument("--restored-proxies", nargs="+", default=list(RESTORED_PROXY_CONFIGS))
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-query-groups", type=int, default=None)
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument(
        "--variant-order",
        choices=("manifest", "hard"),
        default="manifest",
        help="Order train variants before applying --max-variants. 'hard' prioritizes higher-order degradations.",
    )
    parser.add_argument("--train-variants", nargs="+", default=None, help="Explicit train variant list from the train split.")
    parser.add_argument("--val-variants", nargs="+", default=None, help="Explicit val variant list from the val split.")
    return parser.parse_args()


def filter_queries_by_ids(selected_queries, query_ids: Sequence[int]):
    order = {int(query_id): idx for idx, query_id in enumerate(query_ids)}
    filtered = selected_queries[selected_queries["query_id"].isin(order)].copy()
    filtered["__order"] = filtered["query_id"].map(order)
    return filtered.sort_values("__order").drop(columns="__order").reset_index(drop=True)


def relevant_pages_for_query_ids(relevant_pages: dict[int, set[int]], query_ids: Sequence[int], max_docs: int | None = None) -> list[set[int]]:
    rows = []
    for query_id in query_ids:
        pages = set(relevant_pages[int(query_id)])
        if max_docs is not None:
            pages = {page for page in pages if page < max_docs}
        rows.append(pages)
    return rows


def encode_query_set(model, processor, selected_queries, query_ids: Sequence[int], batch_size: int, device: str) -> dict[int, torch.Tensor]:
    filtered = filter_queries_by_ids(selected_queries, query_ids)
    embeddings = encode_queries(model, processor, filtered["query"].tolist(), batch_size, device)
    return {int(query_id): embedding for query_id, embedding in zip(filtered["query_id"].tolist(), embeddings)}


def batch_groups(groups: list[dict], batch_size: int) -> list[list[dict]]:
    return [groups[start : start + batch_size] for start in range(0, len(groups), batch_size)]


def query_ids_from_groups(groups: Sequence[dict]) -> list[int]:
    return [int(query_id) for group in groups for query_id in group["query_ids"]]


def degradation_order(variant: str) -> int:
    return len([part for part in variant.split("_") if part])


def select_variants(
    manifest_variants: Sequence[str],
    explicit_variants: Sequence[str] | None = None,
    max_variants: int | None = None,
    variant_order: str = "manifest",
) -> list[str]:
    available = list(manifest_variants)
    if explicit_variants is not None:
        missing = sorted(set(explicit_variants) - set(available))
        if missing:
            raise ValueError(f"Requested variants are not in this split: {missing}")
        selected = list(explicit_variants)
    else:
        selected = available

    if variant_order == "hard" and explicit_variants is None:
        selected = sorted(selected, key=lambda variant: (-degradation_order(variant), variant))
    if max_variants is not None:
        selected = selected[:max_variants]
    return selected


def evaluate_adapter_on_embeddings(
    adapter: ResidualEmbeddingAdapter,
    query_embeddings: Sequence[torch.Tensor],
    query_ids: Sequence[int],
    relevant_pages_map: dict[int, set[int]],
    clean_embeddings: Sequence[torch.Tensor],
    target_embeddings: Sequence[torch.Tensor],
    device: str,
    doc_batch_size: int,
    max_docs: int | None = None,
) -> dict[str, Any]:
    adapter.eval()
    relevant = relevant_pages_for_query_ids(relevant_pages_map, query_ids, max_docs=max_docs)
    with torch.no_grad():
        clean_scores = late_interaction_scores(query_embeddings, clean_embeddings, device=device, doc_batch_size=doc_batch_size).cpu()
        target_scores = late_interaction_scores(query_embeddings, target_embeddings, device=device, doc_batch_size=doc_batch_size).cpu()
        adapted_embeddings = apply_adapter_to_embeddings(adapter, list(target_embeddings), device)
        adapted_scores = late_interaction_scores(query_embeddings, adapted_embeddings, device=device, doc_batch_size=doc_batch_size).cpu()
    topk_scores = score_multi_vector_robust(
        query_embeddings,
        target_embeddings,
        reduction="topk_mean",
        top_k=3,
        batch_size=doc_batch_size,
        device=device,
    )
    return {
        "clean_metrics": compute_metrics(clean_scores, query_ids, dict(zip(query_ids, relevant))),
        "target_original_metrics": compute_metrics(target_scores, query_ids, dict(zip(query_ids, relevant))),
        "target_topk3_metrics": compute_metrics(topk_scores, query_ids, dict(zip(query_ids, relevant))),
        "metrics": compute_metrics(adapted_scores, query_ids, dict(zip(query_ids, relevant))),
        "rank_delta": summarize_rank_delta(target_scores, adapted_scores, query_ids, relevant),
    }


def first_relevant_rank(scores: torch.Tensor, relevant: set[int]) -> int | None:
    ranked = torch.argsort(scores, descending=True).tolist()
    for rank, doc_idx in enumerate(ranked, start=1):
        if doc_idx in relevant:
            return rank
    return None


def summarize_rank_delta(
    original_scores: torch.Tensor,
    adapted_scores: torch.Tensor,
    query_ids: Sequence[int],
    relevant_pages: Sequence[set[int]],
) -> dict[str, Any]:
    rows = []
    improved = worsened = unchanged = 0
    for row_idx, query_id in enumerate(query_ids):
        original_rank = first_relevant_rank(original_scores[row_idx], relevant_pages[row_idx])
        adapted_rank = first_relevant_rank(adapted_scores[row_idx], relevant_pages[row_idx])
        delta = None if original_rank is None or adapted_rank is None else adapted_rank - original_rank
        if delta is not None:
            if delta < 0:
                improved += 1
            elif delta > 0:
                worsened += 1
            else:
                unchanged += 1
        rows.append(
            {
                "query_id": int(query_id),
                "original_first_relevant_rank": original_rank,
                "adapted_first_relevant_rank": adapted_rank,
                "rank_delta": delta,
                "relevant_pages": sorted(relevant_pages[row_idx]),
            }
        )
    return {"improved": improved, "worsened": worsened, "unchanged": unchanged, "rows": rows}


def average_metric(payloads: list[dict[str, Any]], metric: str = "ndcg@5") -> float:
    return sum(payload["metrics"][metric] for payload in payloads) / len(payloads)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest, manifest_file = load_or_build_manifest(
        path=args.split_manifest,
        dataset_root=args.dataset_root,
        doc_id=args.doc_id,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    tables = load_tables(args.dataset_root)
    selected_queries, relevant_pages, query_summary = select_queries(
        tables["corpus"],
        tables["queries"],
        tables["qrels"],
        doc_id=args.doc_id,
        include_cross_doc_queries=False,
        max_queries=None,
    )

    train_groups = manifest["query_splits"]["train"][: args.max_query_groups]
    val_groups = manifest["query_splits"]["val"][: args.max_query_groups]
    test_groups = manifest["query_splits"]["test"][: args.max_query_groups]
    train_variants = select_variants(
        manifest["variant_splits"]["train"],
        explicit_variants=args.train_variants,
        max_variants=args.max_variants,
        variant_order=args.variant_order,
    )
    val_variants = select_variants(
        manifest["variant_splits"]["val"],
        explicit_variants=args.val_variants,
        max_variants=args.max_variants,
        variant_order="manifest",
    )

    processor_name = args.processor or args.model
    model, processor = load_model(args.model, processor_name, args.device, args.local_files_only)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    all_query_ids = sorted(set(query_ids_from_groups(train_groups + val_groups + test_groups)))
    query_by_id = encode_query_set(model, processor, selected_queries, all_query_ids, args.query_batch_size, args.device)

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
    print(f"[cache] clean -> {clean_cache}", flush=True)
    train_embeddings = {
        variant: load_or_encode_page_embeddings(
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
        )[0]
        for variant in train_variants
    }
    print(f"[cache] train variants: {', '.join(train_variants)}", flush=True)
    val_embeddings = {
        variant: load_or_encode_page_embeddings(
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
        )[0]
        for variant in val_variants
    }
    print(f"[cache] val variants: {', '.join(val_variants)}", flush=True)

    adapter = ResidualEmbeddingAdapter(dim=clean_embeddings[0].shape[-1], hidden_dim=args.hidden_dim, init_scale=args.init_scale).to(args.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    train_log: list[dict[str, float]] = []
    val_log: list[dict[str, Any]] = []
    best_ndcg = float("-inf")
    best_epoch = 0
    best_checkpoint = None

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        rng = random.Random(args.seed + epoch)
        groups = list(train_groups)
        rng.shuffle(groups)
        variants = list(train_variants)
        rng.shuffle(variants)
        totals = {"loss": 0.0, "score_distill": 0.0, "qrels_rank": 0.0, "identity": 0.0}
        steps = 0

        for variant in variants:
            target_embeddings = train_embeddings[variant]
            for group_batch in batch_groups(groups, args.batch_query_groups):
                query_ids = query_ids_from_groups(group_batch)
                query_embeddings = [query_by_id[query_id] for query_id in query_ids]
                relevant = relevant_pages_for_query_ids(relevant_pages, query_ids, max_docs=args.max_docs)

                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    clean_scores = late_interaction_scores(
                        query_embeddings,
                        clean_embeddings,
                        device=args.device,
                        doc_batch_size=args.doc_batch_size,
                    )
                adapted_embeddings = apply_adapter_to_embeddings(adapter, list(target_embeddings), args.device)
                adapted_scores = late_interaction_scores(
                    query_embeddings,
                    adapted_embeddings,
                    device=args.device,
                    doc_batch_size=args.doc_batch_size,
                )
                losses = invariant_adapter_loss(
                    adapted_scores=adapted_scores,
                    clean_teacher_scores=clean_scores,
                    relevant_pages=relevant,
                    original_embeddings=target_embeddings,
                    adapted_embeddings=adapted_embeddings,
                    temperature=args.temperature,
                    score_weight=args.score_weight,
                    rank_weight=args.rank_weight,
                    identity_weight=args.identity_weight,
                )
                losses.total.backward()
                optimizer.step()
                totals["loss"] += float(losses.total.detach().cpu())
                totals["score_distill"] += float(losses.score_distill.detach().cpu())
                totals["qrels_rank"] += float(losses.qrels_rank.detach().cpu())
                totals["identity"] += float(losses.identity.detach().cpu())
                steps += 1

        train_log.append({"epoch": epoch, **{key: value / max(1, steps) for key, value in totals.items()}})

        val_payloads = []
        val_query_ids = query_ids_from_groups(val_groups)
        val_query_embeddings = [query_by_id[query_id] for query_id in val_query_ids]
        for variant, embeddings in val_embeddings.items():
            payload = evaluate_adapter_on_embeddings(
                adapter,
                val_query_embeddings,
                val_query_ids,
                relevant_pages,
                clean_embeddings,
                embeddings,
                device=args.device,
                doc_batch_size=args.doc_batch_size,
                max_docs=args.max_docs,
            )
            payload["variant"] = variant
            val_payloads.append(payload)
        val_ndcg = average_metric(val_payloads, "ndcg@5") if val_payloads else float("-inf")
        val_log.append({"epoch": epoch, "mean_ndcg@5": val_ndcg, "variants": val_payloads})
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_log[-1]["loss"],
                    "val_ndcg@5": val_ndcg,
                    "best_ndcg@5": max(best_ndcg, val_ndcg),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_epoch = epoch
            best_checkpoint = {key: value.detach().cpu() for key, value in adapter.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break

    if best_checkpoint is not None:
        adapter.load_state_dict(best_checkpoint)

    checkpoint_path = args.output_dir / "adapter_checkpoint.pt"
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "dim": adapter.dim,
            "hidden_dim": adapter.hidden_dim,
            "best_epoch": best_epoch,
            "best_val_ndcg@5": best_ndcg,
            "manifest_path": str(manifest_file),
            "training_args": checkpoint_metadata(vars(args)),
            "query_summary": checkpoint_metadata(query_summary),
            "clean_cache": str(clean_cache),
        },
        checkpoint_path,
    )
    (args.output_dir / "train_log.json").write_text(json.dumps(train_log, indent=2, ensure_ascii=False))
    (args.output_dir / "val_metrics_by_epoch.json").write_text(json.dumps(val_log, indent=2, ensure_ascii=False))

    test_query_ids = query_ids_from_groups(test_groups)
    test_query_embeddings = [query_by_id[query_id] for query_id in test_query_ids]
    raw_embeddings = load_or_encode_page_embeddings(
        model=model,
        processor=processor,
        dataset_root=args.dataset_root,
        doc_id=args.doc_id,
        mode="degraded",
        variant=args.eval_raw_variant,
        batch_size=args.model_batch_size,
        device=args.device,
        cache_root=args.cache_root,
        max_docs=args.max_docs,
        overwrite_cache=args.overwrite_cache,
    )[0]
    raw_payload = evaluate_adapter_on_embeddings(
        adapter,
        test_query_embeddings,
        test_query_ids,
        relevant_pages,
        clean_embeddings,
        raw_embeddings,
        device=args.device,
        doc_batch_size=args.doc_batch_size,
        max_docs=args.max_docs,
    )
    raw_payload.update({"method": "invariant_adapter", "variant": args.eval_raw_variant, "checkpoint": str(checkpoint_path)})
    raw_path = args.output_dir / f"test_raw_{args.eval_raw_variant}.json"
    raw_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False))

    for restoration in args.restored_proxies:
        restored_embeddings = load_or_encode_restored_proxy_embeddings(
            model=model,
            processor=processor,
            dataset_root=args.dataset_root,
            doc_id=args.doc_id,
            degraded_variant=args.eval_raw_variant,
            restoration=restoration,
            batch_size=args.model_batch_size,
            device=args.device,
            cache_root=args.cache_root,
            max_docs=args.max_docs,
            overwrite_cache=args.overwrite_cache,
        )[0]
        restored_payload = evaluate_adapter_on_embeddings(
            adapter,
            test_query_embeddings,
            test_query_ids,
            relevant_pages,
            clean_embeddings,
            restored_embeddings,
            device=args.device,
            doc_batch_size=args.doc_batch_size,
            max_docs=args.max_docs,
        )
        restored_payload.update(
            {
                "method": "invariant_adapter",
                "variant": args.eval_raw_variant,
                "restored_proxy": restoration,
                "checkpoint": str(checkpoint_path),
            }
        )
        (args.output_dir / f"test_restored_proxy_{restoration}.json").write_text(
            json.dumps(restored_payload, indent=2, ensure_ascii=False)
        )

    print(json.dumps({"checkpoint": str(checkpoint_path), "best_epoch": best_epoch, "best_val_ndcg@5": best_ndcg, "raw_test": raw_payload["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
