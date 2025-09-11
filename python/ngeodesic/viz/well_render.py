# -*- coding: utf-8 -*-
"""
# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Copyright 2025 Ian C. Moore (Provisional Patents #63/864,726, #63/865,437, #63/871,647 and #63/872,334)
# Email: ngeodesic@gmail.com
# Part of Noetic Geodesic Framework (NGF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import annotations
import warnings
from typing import Tuple
import numpy as np

# Try matplotlib; render headlessly if present
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

def _pca3_whiten(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lightweight PCA→3D with whitening (no sklearn dependency).
    Returns (Y, comps, mean) where Y is (N,3).
    """
    X = np.asarray(X, float)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V3 = Vt[:3, :]                # (3, D)
    Y  = Xc @ V3.T                # (N, 3)
    # whiten by component std
    std = Y.std(axis=0, ddof=1) + 1e-8
    Yw  = Y / std
    return Yw, V3, mu.squeeze(0)

def render_pca_well(out_raw: str, out_fit: str, H: np.ndarray, E: np.ndarray) -> None:
    """
    Render a PCA(3) view of feature cloud H with energy coloring E, and a
    simple radial quantile 'funnel' surface fit. Saves two PNGs:
      - out_raw:  3D scatter in PCA space colored by E
      - out_fit:  surface of z(r) at the 0.65 quantile with scatter overlay
    This matches the Stage-11 demo behavior without external deps.
    """
    if plt is None:
        warnings.warn("matplotlib not available; skipping render")
        return

    H = np.asarray(H, float)
    E = np.asarray(E, float)
    if H.ndim != 2 or H.shape[0] != E.shape[0]:
        raise ValueError("H must be (N,D) and E must be (N,)")

    # PCA→3D (whitened)
    Y, comps, mu = _pca3_whiten(H)  # (N,3)

    # --- Figure 1: raw PCA scatter ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=E, cmap="viridis", s=10, alpha=0.85)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("Stage-11 — Warped manifold (PCA view)")
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.05, label="energy")
    fig.tight_layout()
    fig.savefig(out_raw, dpi=220)
    plt.close(fig)

    # --- Figure 2: radial 0.65-quantile 'funnel' surface + scatter ---
    r = np.linalg.norm(Y[:, :2], axis=1)
    z = Y[:, 2]

    # bin radii and compute quantile per annulus
    nb = max(24, int(np.sqrt(len(Y))))
    bins = np.linspace(r.min(), r.max(), nb + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    zq = np.full_like(centers, np.nan, dtype=float)
    for i, (b0, b1) in enumerate(zip(bins[:-1], bins[1:])):
        m = (r >= b0) & (r < b1)
        if np.any(m):
            zq[i] = np.quantile(z[m], 0.65)

    # fill gaps by interpolation if needed
    if np.any(np.isnan(zq)):
        valid = ~np.isnan(zq)
        zq[~valid] = np.interp(centers[~valid], centers[valid], zq[valid])

    th = np.linspace(0, 2*np.pi, 200)
    R, TH = np.meshgrid(centers, th)
    Z = np.tile(zq, (len(th), 1))
    Xs = R * np.cos(TH); Ys = R * np.sin(TH)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xs, Ys, Z, cmap="viridis", alpha=0.90, linewidth=0, antialiased=True)
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=E, cmap="viridis", s=10, alpha=0.60)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("Stage-11 — Data-fit funnel (proxy)")
    fig.tight_layout()
    fig.savefig(out_fit, dpi=220)
    plt.close(fig)
