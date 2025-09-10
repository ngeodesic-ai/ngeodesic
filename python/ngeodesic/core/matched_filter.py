from __future__ import annotations
import numpy as np
from typing import Literal

def half_sine_proto(n: int) -> np.ndarray:
    t = np.linspace(0, np.pi, int(n), endpoint=False)
    return np.sin(t)

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = v.std()
    return (v - v.mean()) / (s + 1e-12)

def nxcorr(x: np.ndarray, q: np.ndarray, mode: Literal["same","valid"]="same") -> np.ndarray:
    """
    Normalized cross-correlation (z-scored inputs), scaled by len(q).
    """
    xz, qz = _normalize(x), _normalize(q)
    out = np.correlate(xz, qz, mode=mode)
    return out / (len(qz) + 1e-12)

def corr_at(x: np.ndarray, q: np.ndarray, center_idx: int, half_width: int) -> float:
    """
    Corr 'at' a center by correlating a window around it with q (normalized).
    """
    L = int(half_width)
    lo, hi = max(0, center_idx - L), min(len(x), center_idx + L + 1)
    w = x[lo:hi]
    q_resized = q
    if len(q_resized) != len(w):
        # simple linear resample to match length
        idx = np.linspace(0, len(q) - 1, num=len(w))
        q_resized = np.interp(idx, np.arange(len(q)), q)
    return float(nxcorr(w, q_resized, mode="valid").max())

def null_threshold(
    x: np.ndarray,
    q: np.ndarray,
    shifts: int = 64,
    z: float = 3.0,
    mode: Literal["circular","perm"]="circular",
) -> float:
    """
    Estimate an absolute decision threshold from a null distribution
    produced via circular shifts or random permutations.
    """
    N = len(x)
    scores = []
    if mode == "circular":
        step = max(1, N // max(1, shifts))
        for k in range(shifts):
            xr = np.roll(x, k * step)
            scores.append(float(nxcorr(xr, q, mode="same").max()))
    else:  # perm
        rng = np.random.default_rng(0)
        for _ in range(shifts):
            xr = x.copy()
            rng.shuffle(xr)
            scores.append(float(nxcorr(xr, q, mode="same").max()))
    mu, sd = np.mean(scores), np.std(scores) + 1e-12
    return float(mu + z * sd)
