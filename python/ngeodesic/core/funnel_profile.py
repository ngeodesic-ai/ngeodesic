# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

__all__ = [
    "fit_radial_profile",
    "analytic_core_template",
    "blend_profiles",
    "priors_from_profile",
    "attach_projection_info",
]

def fit_radial_profile(
    r: np.ndarray,
    z: np.ndarray,
    *,
    n_r: int = 220,
    fit_quantile: float = 0.65,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin radii and compute a quantile z(r). Lightweight, no external deps.
    Returns (r_grid, z_profile).
    """
    r = np.asarray(r, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1)
    assert r.size == z.size, "r and z must have same length"

    rmin, rmax = float(r.min()), float(r.max())
    bins = np.linspace(rmin, rmax, int(n_r) + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    zq = np.full_like(centers, np.nan, dtype=float)

    for i, (b0, b1) in enumerate(zip(bins[:-1], bins[1:])):
        m = (r >= b0) & (r < b1)
        if np.any(m):
            zq[i] = np.quantile(z[m], float(fit_quantile))

    # fill any gaps by linear interpolation
    if np.any(np.isnan(zq)):
        valid = ~np.isnan(zq)
        zq[~valid] = np.interp(centers[~valid], centers[valid], zq[valid])

    return centers, zq

def analytic_core_template(
    r_grid: np.ndarray,
    *,
    k: float = 0.18,
    p: float = 1.7,
    r0_frac: float = 0.14,
) -> np.ndarray:
    """
    Simple monotone-decreasing core template in z vs radius:
      z_core(r) = k * exp(- (r / r0)^p ), with r0 = r0_frac * r_max
    Keeps units consistent and positive.
    """
    r = np.asarray(r_grid, float)
    r0 = max(1e-8, float(r0_frac) * max(1e-8, r.max()))
    return float(k) * np.exp(- (r / r0) ** float(p))

def blend_profiles(z_data: np.ndarray, z_core: np.ndarray, *, blend_core: float = 0.25) -> np.ndarray:
    """
    Blend data quantile profile with analytic core.
    z_blend = (1 - w) * z_data + w * z_core
    """
    z_data = np.asarray(z_data, float)
    z_core = np.asarray(z_core, float)
    w = float(blend_core)
    return (1.0 - w) * z_data + w * z_core

def priors_from_profile(r_grid: np.ndarray, z_profile: np.ndarray) -> Dict[str, list]:
    """
    Normalize z_profile to [0,1] for prior weighting and pack with r grid.
    """
    r = np.asarray(r_grid, float)
    z = np.asarray(z_profile, float)
    z_min, z_ptp = float(z.min()), float(z.ptp())
    z_norm = (z - z_min) / (z_ptp + 1e-9)
    return {"r": r.tolist(), "z": z_norm.tolist()}

def attach_projection_info(priors: Dict[str, list], info: Dict[str, np.ndarray]) -> Dict[str, list]:
    """
    Merge PCA warp info (mean/pcs/scales/center) into priors dict.
    """
    out = dict(priors)
    out["_proj"] = {
        "center": np.asarray(info["center"], float).tolist(),
        "pcs":    np.asarray(info["pcs"], float).tolist(),
        "scales": np.asarray(info["scales"], float).tolist(),
        "mean":   np.asarray(info["mean"], float).tolist(),
    }
    return out
