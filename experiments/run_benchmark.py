# experiments/run_benchmark.py
"""
Main robustness benchmark runner.

Uses colpali_engine for model + scoring.
Intercepts process_images() to inject degradation/restoration/segmentation.

Usage:
    python experiments/run_benchmark.py --condition clean
    python experiments/run_benchmark.py --condition degraded --deg heavy_noise
    python experiments/run_benchmark.py --condition restored --deg heavy_noise --rest nlmeans
    python experiments/run_benchmark.py --condition segmented
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
import argparse
import json
import torch
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from typing import List, Callable

from experiments.config import (
    MODEL_NAME, PROCESSOR_NAME, DEVICE, BATCH_SIZE, RESULTS_DIR, VIDORE_SUBSETS
)
from colpali_engine.models import ColQwen2, ColQwen2Processor
from robust.evaluation.metrics import ndcg_at_k, recall_at_k, mean_reciprocal_rank


def load_model():
    print(f"Loading {MODEL_NAME}...")
    model = ColQwen2.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE,
    ).eval()
    processor = ColQwen2Processor.from_pretrained(PROCESSOR_NAME)
    return model, processor


def get_preprocessor(condition: str, deg_type: str = "", rest_type: str = "", **kwargs) -> Callable:
    """Return a function: List[Image] -> List[Image] based on condition."""
    if condition == "clean":
        return lambda imgs: imgs

    DEG_CONFIGS = {
        "light_noise":  [("gaussian_noise",   {"std": 15})],
        "heavy_noise":  [("gaussian_noise",   {"std": 50})],
        "motion_blur":  [("motion_blur",      {"kernel_size": 15, "angle": 45})],
        "tilt":         [("tilt",             {"angle": 15})],
        "jpeg_low":     [("jpeg_compression", {"quality": 5})],
        "combined":     [("gaussian_noise",   {"std": 25}),
                         ("motion_blur",      {"kernel_size": 9, "angle": 30})],
    }

    if condition in ("degraded", "restored"):
        from robust.degradation.pipeline import DegradationPipeline
        deg_pipeline = DegradationPipeline(DEG_CONFIGS[deg_type])

        if condition == "degraded":
            return lambda imgs: [deg_pipeline(img) for img in imgs]

        from robust.restoration.pipeline import RestorationPipeline
        REST_CONFIGS = {
            "nlmeans":  [("nlmeans",  {"h": 10})],
            "gaussian": [("gaussian", {"sigma": 1.5})],
            "wiener":   [("wiener",   {})],
        }
        rest_pipeline = RestorationPipeline(REST_CONFIGS[rest_type])
        return lambda imgs: [rest_pipeline(deg_pipeline(img)) for img in imgs]

    if condition == "segmented":
        seg_method = kwargs.get("seg_method", "adaptive")
        if seg_method == "adaptive":
            from robust.segmentation.adaptive_seg import adaptive_segment
            seg_fn = adaptive_segment
        elif seg_method == "grabcut":
            from robust.segmentation.grabcut_seg import grabcut_segment
            seg_fn = grabcut_segment
        elif seg_method == "edge":
            from robust.segmentation.edge_seg import edge_segment
            seg_fn = edge_segment
        else:
            raise ValueError(f"Unknown seg_method: {seg_method!r}")
        seg_params = {k: v for k, v in kwargs.items()
                      if k not in ("seg_method",)}
        return lambda imgs: [seg_fn(img, **seg_params) for img in imgs]

    raise ValueError(f"Unknown condition: {condition!r}")


def evaluate_subset(model, processor, subset_name: str, preprocess_fn: Callable) -> dict:
    print(f"  Loading {subset_name}...")
    ds = load_dataset(subset_name, split="test")
    queries = [row["query"] for row in ds]
    images  = [row["image"].convert("RGB") for row in ds]
    n = len(queries)

    # Encode queries
    all_q_vecs = []
    for i in range(0, n, BATCH_SIZE):
        inputs = processor.process_queries(queries[i:i+BATCH_SIZE]).to(DEVICE)
        with torch.no_grad():
            vecs = model(**inputs)
        all_q_vecs.extend([v.cpu().float() for v in vecs])

    # Encode documents (with preprocessing intercept)
    all_d_vecs = []
    for i in tqdm(range(0, n, BATCH_SIZE), desc="  Encoding docs"):
        preprocessed = preprocess_fn(images[i:i+BATCH_SIZE])
        inputs = processor.process_images(preprocessed).to(DEVICE)
        with torch.no_grad():
            vecs = model(**inputs)
        all_d_vecs.extend([v.cpu().float() for v in vecs])

    # Score (n_q, n_d) — pass as list to handle variable-length sequences
    scores_matrix = processor.score_multi_vector(all_q_vecs, all_d_vecs)

    # Compute metrics (query i -> document i is the correct match)
    ndcg_list, rec_list, mrr_list = [], [], []
    for i in range(n):
        s = scores_matrix[i].tolist()
        ranked = sorted(range(n), key=lambda j: s[j], reverse=True)
        ndcg_list.append(ndcg_at_k(s, {i}, k=5))
        rec_list.append(recall_at_k(ranked, {i}, k=5))
        mrr_list.append(mean_reciprocal_rank(ranked, {i}))

    return {
        "ndcg@5":    sum(ndcg_list) / n,
        "recall@5":  sum(rec_list)  / n,
        "mrr":       sum(mrr_list)  / n,
        "n_samples": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="clean",
                        choices=["clean", "degraded", "restored", "segmented"])
    parser.add_argument("--deg",  default="heavy_noise",
                        help="Degradation type (for degraded/restored)")
    parser.add_argument("--rest", default="nlmeans",
                        help="Restoration type (for restored)")
    parser.add_argument("--seg_method", default="adaptive",
                        choices=["adaptive", "grabcut", "edge"],
                        help="Segmentation method (for segmented)")
    parser.add_argument("--subsets", nargs="+", default=VIDORE_SUBSETS)
    args = parser.parse_args()

    model, processor = load_model()
    preprocess_fn = get_preprocessor(args.condition, args.deg, args.rest,
                                     seg_method=args.seg_method)

    results = {}
    for subset in args.subsets:
        print(f"\nEvaluating: {subset.split('/')[-1]}")
        metrics = evaluate_subset(model, processor, subset, preprocess_fn)
        results[subset.split("/")[-1]] = metrics
        print(f"  nDCG@5={metrics['ndcg@5']:.4f}  Recall@5={metrics['recall@5']:.4f}  MRR={metrics['mrr']:.4f}")

    # Save results with descriptive filename
    tag = args.condition
    if args.condition == "degraded":
        tag += f"_{args.deg}"
    elif args.condition == "restored":
        tag += f"_{args.deg}_{args.rest}"
    elif args.condition == "segmented":
        tag += f"_{args.seg_method}"

    # Save to timestamped directory (archive)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_dir = RESULTS_DIR / timestamp
    ts_dir.mkdir(parents=True, exist_ok=True)
    (ts_dir / f"results_{tag}.json").write_text(json.dumps(results, indent=2))

    # Also save flat for backward compatibility (visualize_results.py)
    out = RESULTS_DIR / f"results_{tag}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")
    print(f"Archived to {ts_dir / f'results_{tag}.json'}")


if __name__ == "__main__":
    main()
