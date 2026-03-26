# experiments/config.py
"""Shared configuration for all experiment scripts."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR      = PROJECT_ROOT / "data"
RESULTS_DIR   = PROJECT_ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Model — using ColQwen2 (better performance than original ColPali)
MODEL_NAME = "vidore/colqwen2-v1.0"
PROCESSOR_NAME = "vidore/colqwen2-v1.0"
DEVICE = "cuda:1"        # Change to "mps" for Apple Silicon

# ViDoRe evaluation subsets
VIDORE_SUBSETS = [
    "vidore/docvqa_test_subsampled",
    "vidore/infovqa_test_subsampled",
]

# Batch size for model inference
BATCH_SIZE = 4
