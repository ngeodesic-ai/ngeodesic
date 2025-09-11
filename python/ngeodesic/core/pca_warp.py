# At top of file
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

__all__ = ["pca3_and_warp"]  # ensure it exports

def pca3_and_warp(H: np.ndarray, energy: np.ndarray | None = None) -> Tuple[np.ndarray, Dict, Dict]:
    """
    Minimal PCA(3) + isotropic warp (whiten each PC). Returns:
      X3_warp : (N, 3) whitened coordinates
      well_metrics : dict with explained variance etc.
      info : dict with 'center' (2D center for radial profile), 'pcs' (3xD), 'scales' (3,)
    """
    H = np.asarray(H, dtype=float)
    if H.ndim != 2:
        raise ValueError("H must be 2D (N, D)")
    N, D = H.shape
    mu = H.mean(axis=0, keepdims=True)
    X = H - mu

    # PCA via SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)  # X ≈ U S Vt
    pcs = Vt[:3, :]                                   # (3, D)
    X3 = X @ pcs.T                                    # (N, 3)

    # Whiten (isotropize) each PC axis
    scales = X3.std(axis=0) + 1e-8
    X3_warp = X3 / scales

    # 2D center for radial profile (use mean in PC1–PC2 plane)
    c2 = X3_warp[:, :2].mean(axis=0)

    # metrics
    var_total = (S**2).sum()
    var_3 = (S[:3]**2).sum()
    well_metrics = {
        "explained_var_3": float(var_3 / (var_total + 1e-12)),
        "std_scales": scales.tolist(),
        "n": int(N),
        "d": int(D),
    }
    info = {
        "center": c2,          # 2D center for r = ||(x[:2] - center)||
        "pcs": pcs,            # (3, D)
        "scales": scales,      # (3,)
        "mean": mu.squeeze(),  # (D,)
    }
    return X3_warp, well_metrics, info
