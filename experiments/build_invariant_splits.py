#!/usr/bin/env python
"""Build deterministic train/val/test splits for invariant adapter experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.invariant_splits import DEFAULT_OUTPUT_DIR, DEFAULT_SEED, build_invariant_split_manifest, save_manifest
from experiments.run_local_hr_benchmark import DATASET_ROOT, DEFAULT_DOC_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build invariant-learning split manifest.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_invariant_split_manifest(dataset_root=args.dataset_root, doc_id=args.doc_id, seed=args.seed)
    path = save_manifest(manifest, output_dir=args.output_dir)
    print(json.dumps({"path": str(path), "query_split_counts": manifest["query_split_counts"], "variant_split_counts": manifest["variant_split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
