# python/ngeodesic/core/funnel_profile.py
from __future__ import annotations
from typing import Iterable, Dict
import numpy as np

__all__ = [
    "fit_radial_profile",
    "analytic_core_template",
    "priors_from_profile",
    "blend_profiles",          # NEW
    "build_polar_surface",     # NEW
]

def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Compute weighted quantile of x with non-negative weights w."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    w = np.clip(w, 0.0, None)
    if x.size == 0 or w.sum() <= 0.0:
        return 0.0
    order = np.argsort(x)
    x_ord = x[order]
    w_ord = w[order]
    cdf = np.cumsum(w_ord) / (w_ord.sum() + 1e-12)
    i = np.searchsorted(cdf, float(q), side="left")
    i = int(np.clip(i, 0, x_ord.size - 1))
    return float(x_ord[i])

def fit_radial_profile(
    X3: np.ndarray,
    center_2d: Iterable[float],
    r_grid: np.ndarray,
    *,
    h: float,             # bandwidth as absolute distance (e.g., 0.3 * r_max)
    q: float = 0.65,      # upper envelope quantile along z
    r0_frac: float = 0.14,
    core_k: float = 0.18,
    core_p: float = 1.7,
) -> np.ndarray:
    """
    Robust radial envelope: for each radius r0 in r_grid, compute a
    kernel-weighted quantile (q) of z = max(0, X3[:,2]).
    """
    X3 = np.asarray(X3, float)
    r_grid = np.asarray(r_grid, float)
    c = np.asarray(center_2d, float).reshape(2)
    if X3.ndim != 2 or X3.shape[1] < 3:
        raise ValueError("X3 must be (N,3) or (N,>=3) PCA-warped coordinates")

    z = np.maximum(0.0, X3[:, 2])
    r = np.linalg.norm(X3[:, :2] - c[None, :], axis=1)

    bw = float(max(h, 1e-6))
    out = np.empty_like(r_grid, dtype=float)
    for k, r0 in enumerate(r_grid):
        # Gaussian kernel in radius
        w = np.exp(-0.5 * ((r - r0) / bw) ** 2)
        out[k] = _weighted_quantile(z, w, q)

    # Optional: gentle core sharpening (kept tiny by defaults)
    if core_k > 0:
        r_max = float(r_grid.max() + 1e-12)
        r0 = max(r0_frac * r_max, 1e-6)
        core_boost = 1.0 + core_k * np.exp(- (r_grid / r0) ** core_p)
        out = out * core_boost

    # Clamp non-negative
    return np.clip(out, 0.0, None)

def analytic_core_template(
    r_grid: np.ndarray,
    *,
    D: float = 1.2,
    p: float = 1.6,
    r0_frac: float = 0.14,
) -> np.ndarray:
    """
    Simple analytic decay centered at r=0:
      z(r) = (1 + (r / r0)^p)^(-D), scaled to [0,1] at r=0.
    """
    r_grid = np.asarray(r_grid, float)
    r_max = float(r_grid.max() + 1e-12)
    r0 = max(r0_frac * r_max, 1e-6)
    z = (1.0 + (r_grid / r0) ** p) ** (-D)
    # Normalize to 1 at r=0
    return z / (z[:1] + 1e-12)


def blend_profiles(z_data: np.ndarray, z_template: np.ndarray, w: float = 0.25) -> np.ndarray:
    """
    Linear blend of two radial profiles of identical shape:
        z_blend = (1 - w) * z_data + w * z_template
    w in [0,1]. Output is clamped non-negative.
    """
    z_data = np.asarray(z_data, float)
    z_template = np.asarray(z_template, float)
    if z_data.shape != z_template.shape:
        raise ValueError(f"shape mismatch: data {z_data.shape} vs template {z_template.shape}")
    w = float(np.clip(w, 0.0, 1.0))
    z = (1.0 - w) * z_data + w * z_template
    return np.clip(z, 0.0, None)

def build_polar_surface(r_grid: np.ndarray, z_profile: np.ndarray, n_theta: int = 181):
    """
    Revolve a radial profile z(r) around the z-axis to produce a surface.
    Returns X, Y, Z with shapes (len(r_grid), n_theta).
    Useful for quick plotting/meshes.
    """
    r = np.asarray(r_grid, float)
    z = np.asarray(z_profile, float)
    if r.shape != z.shape:
        raise ValueError(f"shape mismatch: r {r.shape} vs z {z.shape}")
    n_theta = int(max(8, n_theta))
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    R, Th = np.meshgrid(r, theta, indexing="ij")
    X = R * np.cos(Th)
    Y = R * np.sin(Th)
    Z = np.tile(z[:, None], (1, n_theta))
    return X, Y, Z

def priors_from_profile(r_grid: np.ndarray, z_profile: np.ndarray) -> Dict[str, list]:
    """
    Package-independent, serializable prior object. The current parser stub
    ignores priors, but we return a consistent structure for future use.
    """
    r_grid = np.asarray(r_grid, float)
    z_profile = np.asarray(z_profile, float)
    # Normalize profile to [0,1] for convenience
    z_norm = z_profile / (z_profile.max() + 1e-12)
    return {
        "r": r_grid.astype(float).tolist(),
        "z": z_norm.astype(float).tolist(),
    }
