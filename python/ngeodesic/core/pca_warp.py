from __future__ import annotations
import numpy as np
from typing import Tuple

def radius_from_sample_energy(energies: np.ndarray, eps: float = 1e-9) -> float:
    """
    Crude scalar 'radius' estimate from per-channel residual energies.

    Useful for linking a sample's residual signature to a radial prior (Stage-11).
    """
    e = np.asarray(energies, dtype=float).ravel()
    return float(np.linalg.norm(e) + eps)

def _lateral_inhibition(X: np.ndarray, k: int = 5, lam: float = 0.1) -> np.ndarray:
    """
    Private helper: a tiny KNN smoothing/inhibition pass used during warp.
    For each row vector, subtract a fraction of its k-neighbor mean.
    """
    X = np.asarray(X, dtype=float)
    if len(X) <= k:
        return X.copy()
    # naive L2 KNN per-row (CPU-friendly for small N)
    out = X.copy()
    norms = (X @ X.T)
    diag = np.diag(norms).copy()
    d2 = (diag[:, None] + diag[None, :] - 2 * norms).clip(min=0.0)
    np.fill_diagonal(d2, np.inf)
    idx = np.argpartition(d2, kth=k, axis=1)[:, :k]
    for i in range(len(X)):
        m = X[idx[i]].mean(axis=0)
        out[i] = X[i] - lam * m
    return out

# keep/export your existing pca3_and_warp(...) here

