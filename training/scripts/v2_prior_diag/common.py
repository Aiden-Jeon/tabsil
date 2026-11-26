"""Shared helpers of the v2 prior diagnostics (`README.md` in this folder)."""

from __future__ import annotations

import os

import numpy as np
import torch

DEVICE = os.environ.get("DIAG_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
# Output folder of the CSV files (run_all.py logs it to the MLflow run)
OUT = os.environ.get("DIAG_OUT", "/tmp/v2_prior_diag")
os.makedirs(OUT, exist_ok=True)


def log_density(log_probs: torch.Tensor, borders: torch.Tensor, y: np.ndarray) -> np.ndarray:
    """Per-row log density of y under a bucket distribution (`predict_distribution`: log
    probabilities on borders in original units), uniform within a bucket. A value outside the
    borders gets the outermost bucket's density."""
    lp = log_probs.double().cpu().numpy()
    b = borders.double().cpu().numpy()
    idx = np.clip(np.searchsorted(b, y, side="right") - 1, 0, len(b) - 2)
    width = np.maximum(b[idx + 1] - b[idx], 1e-300)
    return lp[np.arange(len(y)), idx] - np.log(width)
