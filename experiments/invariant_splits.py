"""Deterministic query and degradation splits for invariant adapter experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from experiments.run_local_hr_benchmark import DEFAULT_DOC_ID, DATASET_ROOT, list_available_variants, select_queries


DEFAULT_SEED = 13
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "invariant_learning"
HELD_OUT_MAIN_VARIANT = "PD_MB_GN_JC_LR_CS"


def stable_hash(value: str, seed: int = DEFAULT_SEED) -> str:
    return hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def variant_order(values: Iterable[str], seed: int = DEFAULT_SEED) -> list[str]:
    return sorted(values, key=lambda value: (stable_hash(value, seed), value))


def variant_components(variant: str) -> tuple[str, ...]:
    return tuple(part for part in variant.split("_") if part)


def build_query_groups(selected_queries: pd.DataFrame, all_queries: pd.DataFrame, seed: int = DEFAULT_SEED) -> list[dict]:
    merged = selected_queries[["query_id", "query"]].merge(
        all_queries[["query_id", "language", "raw_answers"]],
        on="query_id",
        how="left",
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in merged.sort_values("query_id").to_dict("records"):
        group_key = stable_json(row["raw_answers"])
        groups.setdefault(group_key, []).append(
            {
                "query_id": int(row["query_id"]),
                "language": row.get("language"),
                "query": row.get("query"),
            }
        )

    ordered = []
    for group_key, rows in groups.items():
        rows = sorted(rows, key=lambda item: item["query_id"])
        ordered.append(
            {
                "group_key": group_key,
                "group_hash": stable_hash(group_key, seed),
                "query_ids": [row["query_id"] for row in rows],
                "languages": [row["language"] for row in rows],
                "queries": rows,
            }
        )
    return sorted(ordered, key=lambda item: (item["group_hash"], item["group_key"]))


def split_query_groups(groups: list[dict]) -> dict[str, list[dict]]:
    if len(groups) < 3:
        raise ValueError("At least 3 query groups are required for train/val/test splitting.")
    if len(groups) >= 29:
        return {"train": groups[:17], "val": groups[17:23], "test": groups[23:29]}

    train_count = max(1, int(len(groups) * 0.6))
    val_count = max(1, int(len(groups) * 0.2))
    if train_count + val_count >= len(groups):
        val_count = 1
        train_count = len(groups) - 2
    return {
        "train": groups[:train_count],
        "val": groups[train_count : train_count + val_count],
        "test": groups[train_count + val_count :],
    }


def split_variants(variants: list[str], seed: int = DEFAULT_SEED) -> dict[str, list[str]]:
    variants = sorted(set(variants))
    by_size: dict[int, list[str]] = {}
    for variant in variants:
        if variant == HELD_OUT_MAIN_VARIANT:
            continue
        by_size.setdefault(len(variant_components(variant)), []).append(variant)

    singles = sorted(by_size.get(1, []))
    doubles = sorted(by_size.get(2, []))
    triples = variant_order(by_size.get(3, []), seed)
    fours = variant_order(by_size.get(4, []), seed)
    fives = variant_order(by_size.get(5, []), seed)

    train = singles + doubles + triples[:9]
    val = triples[9:] + fours[:5]
    test_extra = fives[:5]
    if len(test_extra) < 5:
        test_extra.extend([variant for variant in fours[5:] if variant not in val][: 5 - len(test_extra)])
    test = [HELD_OUT_MAIN_VARIANT] + test_extra

    overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
    if overlap:
        raise ValueError(f"Variant splits overlap: {sorted(overlap)}")
    if HELD_OUT_MAIN_VARIANT in set(train) | set(val):
        raise ValueError(f"{HELD_OUT_MAIN_VARIANT} must not appear in train/val variants.")
    return {"train": train, "val": val, "test": test}


def flatten_query_ids(groups: list[dict]) -> list[int]:
    query_ids: list[int] = []
    for group in groups:
        query_ids.extend(int(query_id) for query_id in group["query_ids"])
    return query_ids


def build_invariant_split_manifest(
    dataset_root: Path = DATASET_ROOT,
    doc_id: str = DEFAULT_DOC_ID,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    corpus = pd.read_parquet(dataset_root / "corpus" / "test-00000-of-00001.parquet")
    queries = pd.read_parquet(dataset_root / "queries" / "test-00000-of-00001.parquet")
    qrels = pd.read_parquet(dataset_root / "qrels" / "test-00000-of-00001.parquet")
    selected_queries, _, query_summary = select_queries(
        corpus_df=corpus,
        queries_df=queries,
        qrels_df=qrels,
        doc_id=doc_id,
        include_cross_doc_queries=False,
        max_queries=None,
    )
    query_groups = build_query_groups(selected_queries, queries, seed)
    query_splits = split_query_groups(query_groups)
    variant_splits = split_variants(list_available_variants(dataset_root), seed)

    return {
        "seed": seed,
        "doc_id": doc_id,
        "query_group_key": "raw_answers",
        "query_summary": query_summary,
        "query_splits": query_splits,
        "query_split_counts": {
            split: {
                "groups": len(groups),
                "queries": len(flatten_query_ids(groups)),
            }
            for split, groups in query_splits.items()
        },
        "variant_splits": variant_splits,
        "variant_split_counts": {split: len(values) for split, values in variant_splits.items()},
        "held_out_main_variant": HELD_OUT_MAIN_VARIANT,
    }


def manifest_path(output_dir: Path = DEFAULT_OUTPUT_DIR, seed: int = DEFAULT_SEED) -> Path:
    return output_dir / f"splits_seed{seed}.json"


def save_manifest(manifest: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    path = manifest_path(output_dir, int(manifest["seed"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return path


def load_or_build_manifest(
    path: Path | None = None,
    dataset_root: Path = DATASET_ROOT,
    doc_id: str = DEFAULT_DOC_ID,
    seed: int = DEFAULT_SEED,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[dict[str, Any], Path]:
    resolved = path or manifest_path(output_dir, seed)
    if resolved.exists():
        return json.loads(resolved.read_text()), resolved
    manifest = build_invariant_split_manifest(dataset_root=dataset_root, doc_id=doc_id, seed=seed)
    saved = save_manifest(manifest, output_dir=output_dir)
    return manifest, saved
