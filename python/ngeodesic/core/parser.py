# python/ngeodesic/core/parser.py
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import numpy as np
from .smoothing import moving_average  # used for light smoothing

# ---------- small helpers ----------
def _ensure_dict(traces: Iterable[np.ndarray] | Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    if isinstance(traces, dict):
        keys = list(traces.keys())
        return {k: np.asarray(traces[k], float) for k in keys}, keys
    arrs = [np.asarray(t, float) for t in traces]
    keys = [str(i) for i in range(len(arrs))]
    return dict(zip(keys, arrs)), keys

def _common_mode(X: Dict[str, np.ndarray]) -> np.ndarray:
    M = np.vstack([X[k] for k in X.keys()])
    return M.mean(axis=0)

def _residual_energy(X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    cm = _common_mode(X)
    return {k: (X[k] - cm) ** 2 for k in X.keys()}

def half_sine_proto(width: int) -> np.ndarray:
    P = np.sin(np.linspace(0, np.pi, int(width)))
    return P / (np.linalg.norm(P) + 1e-8)

def _corr_at(sig: np.ndarray, proto: np.ndarray, idx: int, width: int, T: int) -> float:
    a, b = max(0, idx - width // 2), min(T, idx + width // 2)
    w = sig[a:b]
    if len(w) < 3:
        return 0.0
    w = w - w.mean()
    pr = proto[: len(w)] - proto[: len(w)].mean()
    denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
    return float(np.dot(w, pr) / denom)

def _circ_shift(x: np.ndarray, k: int) -> np.ndarray:
    k = int(k) % len(x)
    if k == 0:
        return x
    return np.concatenate([x[-k:], x[:-k]])

def _circ_null_z(sig: np.ndarray, proto: np.ndarray, idx: int, width: int, n_draws: int = 120) -> float:
    T = len(sig)
    obs = _corr_at(sig, proto, idx, width, T)
    rng = np.random.default_rng(0)
    null = np.empty(n_draws, float)
    for i in range(n_draws):
        shift = rng.integers(1, max(2, T - 1))
        null[i] = _corr_at(_circ_shift(sig, int(shift)), proto, idx, width, T)
    mu, sd = float(null.mean()), float(null.std() + 1e-8)
    return (obs - mu) / sd

# ---------- public API ----------
def stock_parse(traces: Iterable[np.ndarray] | Dict[str, np.ndarray], *, sigma: int = 9, proto_width: int = 64):
    """Baseline: correlate each channel with half-sine; keep all; order by peak time."""
    X, keys = _ensure_dict(traces)
    proto = half_sine_proto(proto_width)
    peaks = []
    for k in keys:
        x = moving_average(X[k], k=sigma)
        c = np.correlate(x, proto, mode="same")
        peaks.append((k, int(np.argmax(c))))
    order = [k for (k, j) in sorted(peaks, key=lambda t: t[1])]
    keep = keys[:]  # keep everything
    return keep, order

def geodesic_parse_report(
    traces: Iterable[np.ndarray] | Dict[str, np.ndarray],
    *,
    sigma: int = 9,
    proto_width: int = 64,
    allow_empty: bool = False,
):
    """
    Stage-11 baseline (rebased):
      - residual energy (channel - common mode)^2 and raw channel, both smoothed
      - z-score of local correlation vs circular-shift null
      - combined score: z_res + 0.4*z_raw - 0.3*max(0, z_cm)
      - keep: score >= 0.5 * max(score); fallback to top-1 unless allow_empty=True
      - order: by peak index
    Returns (keep_keys, ordered_keys) as strings.
    """
    X, keys = _ensure_dict(traces)
    T = len(next(iter(X.values())))
    proto = half_sine_proto(proto_width)

    # smoothed residual energy & raw + common mode
    Eres = _residual_energy(X)
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    Sraw = {p: moving_average(X[p],     k=sigma) for p in keys}
    Scm  = moving_average(_common_mode(X), k=sigma)

    # peak indices from residual energy path
    peak_idx = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in keys}

    # circ-shift null z-scores
    z_res = {p: _circ_null_z(Sres[p], proto, peak_idx[p], proto_width) for p in keys}
    z_raw = {p: _circ_null_z(Sraw[p], proto, peak_idx[p], proto_width) for p in keys}
    # probe CM at any channel's index (use first key’s index)
    any_idx = peak_idx[keys[0]]
    z_cm  = {p: _circ_null_z(Scm,      proto, any_idx,     proto_width) for p in keys}

    score = {p: 1.0 * z_res[p] + 0.4 * z_raw[p] - 0.3 * max(0.0, z_cm[p]) for p in keys}

    smax = max(score.values()) if score else 0.0

    if allow_empty:
        # Absolute presence check: at least one channel must show strong residual evidence.
        Z_MIN = 2.8  # tuned to reject pure-noise peaks; real lobes easily exceed this
        present = [p for p in keys if z_res[p] >= Z_MIN]
        if not present:
            return [], []
        # Apply the original relative gate, but only among present channels.
        base = max(score[p] for p in present)
        keep = [p for p in present if score[p] >= 0.5 * (base + 1e-12)]
    else:
        # Original Stage-11 behavior (never empty): relative gate across all channels,
        # with fallback to top-1 if none clear the relative threshold.
        keep = [p for p in keys if score[p] >= 0.5 * (smax + 1e-12)]
        if not keep and keys:
            keep = [max(keys, key=lambda p: score[p])]

    order = sorted(keep, key=lambda p: peak_idx[p])
    return keep, order


def geodesic_parse_with_prior(traces, priors=None, **kw):
    """Stub: kept for API stability; currently just calls geodesic_parse_report."""
    return geodesic_parse_report(traces, **kw)
