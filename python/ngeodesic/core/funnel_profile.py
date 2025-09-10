from __future__ import annotations
import numpy as np
from typing import Tuple

def blend_profiles(z_core: np.ndarray, z_shell: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    Blend two 1D radial profiles (same shape) into a single funnel profile.

    Parameters
    ----------
    z_core : (R,) array
        Steep 'core' profile near r≈0 (e.g., analytic_core_template output).
    z_shell : (R,) array
        Flatter 'shell' profile (e.g., empirical average across samples).
    alpha : float
        Blend weight: 0 => shell only, 1 => core only.

    Returns
    -------
    z_blend : (R,) array
    """
    assert z_core.shape == z_shell.shape
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * z_core + (1.0 - a) * z_shell


def build_polar_surface(r_grid: np.ndarray, z_profile: np.ndarray, n_theta: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a polar 'well' surface from a 1D radial profile z(r).

    Returns (R, THETA, Z) grids suitable for visualization or sampling.
    """
    r = np.asarray(r_grid, dtype=float)
    z = np.asarray(z_profile, dtype=float)
    assert r.ndim == 1 and z.ndim == 1 and r.shape == z.shape

    theta = np.linspace(0, 2 * np.pi, int(n_theta), endpoint=False)
    R, TH = np.meshgrid(r, theta, indexing="ij")
    # z(r) replicated across theta
    Z = np.tile(z[:, None], (1, theta.size))
    return R, TH, Z
