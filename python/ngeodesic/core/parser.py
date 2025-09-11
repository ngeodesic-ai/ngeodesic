# python/ngeodesic/core/parser.py
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np

# -----------------------
# Small utilities (local)
# -----------------------

def _as_dict(
    traces: Union[Dict[str, np.ndarray], Sequence[np.ndarray]]
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Normalize input to a dict[str, np.ndarray] and return (dict, keys_in_order).
    Accepts either a dict of named channels or a list/tuple of arrays.
    """
    if isinstance(traces, dict):
        keys = list(traces.keys())
        td = {k: np.asarray(traces[k], float) for k in keys}
        return td, keys
    # list/tuple of arrays -> "0","1","2",...
    arrs = list(traces)
    keys = [str(i) for i in range(len(arrs))]
    td = {k: np.asarray(arrs[i], float) for i, k in enumerate(keys)}
    return td, keys

def moving_average(x: np.ndarray, k: int = 9) -> np.ndarray:
    if k <= 1:
        return np.asarray(x, float).copy()
    x = np.asarray(x, float)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k) / k, mode="valid")

def common_mode(traces: Dict[str, np.ndarray]) -> np.ndarray:
    return np.stack(list(traces.values()), axis=0).mean(axis=0)

def perpendicular_energy(traces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mu = common_mode(traces)
    return {k: np.clip(v - mu, 0.0, None) for k, v in traces.items()}

def half_sine_proto(width: int) -> np.ndarray:
    w = int(max(3, width))
    P = np.sin(np.linspace(0.0, np.pi, w))
    return P / (np.linalg.norm(P) + 1e-8)

def _corr_at(sig: np.ndarray, proto: np.ndarray, idx: int, width: int, T: int) -> float:
    a, b = max(0, idx - width // 2), min(T, idx + width // 2)
    w = sig[a:b]
    if w.size < 3:
        return 0.0
    w = w - w.mean()
    pr = proto[: w.size] - proto[: w.size].mean()
    denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
    return float(np.dot(w, pr) / denom)

def _circ_shift(x: np.ndarray, k: int) -> np.ndarray:
    k = int(k) % len(x)
    if k == 0:
        return x
    return np.concatenate([x[-k:], x[:-k]])


# ------------------------------------
# Known-working Stage-11 report parser
# ------------------------------------
def geodesic_parse_report(
    traces: Union[Dict[str, np.ndarray], Sequence[np.ndarray]],
    *,
    sigma: int = 9,
    proto_width: int = 160,
) -> Tuple[List[str], List[str]]:
    """
    Stage-11 consolidated behavior:
      - residual (perpendicular) energy vs common mode
      - smoothed residual + smoothed raw + smoothed CM
      - local correlation at peak with half-sine prototype
      - permutation/circular-shift null → Z-scores
      - blended score: 1.0*z_res + 0.4*z_raw - 0.3*max(0, z_cm)
      - keep: >= 0.5 * best score; order: by peak time
    Returns: (keep_keys, order_keys) as strings matching input keys.
    """
    Tdict, keys = _as_dict(traces)
    T = len(next(iter(Tdict.values())))
    proto = half_sine_proto(int(proto_width))

    # Residual energy (positive part), and smoothed views
    Eres = perpendicular_energy(Tdict)
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    Sraw = {p: moving_average(Tdict[p], k=sigma) for p in keys}
    Scm  = moving_average(common_mode(Tdict), k=sigma)

    # Peak indices (in residual view)
    peak_idx = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in keys}

    # Permutation/circ-shift null Z for (residual/raw/cm)
    def perm_null_z(sig: np.ndarray, idx: int, n: int = 120) -> float:
        obs = _corr_at(sig, proto, idx, int(proto_width), T)
        null = np.empty(n, float)
        rng_local = np.random.default_rng(0)
        for i in range(n):
            shift = rng_local.integers(1, max(2, T - 1))
            null[i] = _corr_at(_circ_shift(sig, int(shift)), proto, idx, int(proto_width), T)
        mu, sd = float(null.mean()), float(null.std() + 1e-8)
        return (obs - mu) / sd

    # Use the same CM index for all (matches the consolidated script)
    cm_idx = peak_idx[keys[0]] if keys else 0

    z_res = {p: perm_null_z(Sres[p], peak_idx[p]) for p in keys}
    z_raw = {p: perm_null_z(Sraw[p], peak_idx[p]) for p in keys}
    z_cm  = {p: perm_null_z(Scm,      cm_idx)     for p in keys}

    # Blended score
    score = {
        p: 1.0 * z_res[p] + 0.4 * z_raw[p] - 0.3 * max(0.0, z_cm[p])
        for p in keys
    }

    # Keep: relative to best; Order: by peak time
    smax = max(score.values()) + 1e-12 if score else 0.0
    keep = [p for p in keys if score[p] >= 0.5 * smax]
    if not keep and keys:
        keep = [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak_idx[p])

    return keep, order





# --------------------------------------------------------
# Minimal "stock" baseline kept for backward-compat import
# --------------------------------------------------------
def stock_parse(
    traces: Union[Dict[str, np.ndarray], Sequence[np.ndarray]],
    *,
    sigma: int = 9,
    proto_width: int = 160,
) -> Tuple[List[str], List[str]]:
    Tdict, keys = _as_dict(traces)
    proto = half_sine_proto(int(proto_width))
    S = {p: moving_average(Tdict[p], k=sigma) for p in keys}
    peak = {p: int(np.argmax(np.correlate(S[p], proto, mode="same"))) for p in keys}
    score = {p: float(np.max(np.correlate(S[p], proto, mode="same"))) for p in keys}
    smax = max(score.values()) + 1e-12 if score else 0.0
    keep = [p for p in keys if score[p] >= 0.6 * smax]
    if not keep and keys:
        keep = [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak[p])
    return keep, order

def geodesic_parse_with_prior(
    traces,
    priors=None,
    **kwargs,
):
    import numpy as np

    # knobs (with safe defaults)
    sigma        = int(kwargs.get("sigma", 9))
    proto_width  = int(kwargs.get("proto_width", 160))
    allow_empty  = bool(kwargs.get("allow_empty", True))
    rel_floor    = float(kwargs.get("rel_floor", 0.70))
    margin_floor = float(kwargs.get("margin_floor", 0.03))
    z_null       = float(kwargs.get("z", 2.4))
    alpha        = float(kwargs.get("alpha", 0.05))
    beta_s       = float(kwargs.get("beta_s", 0.25))
    q_s          = float(kwargs.get("q_s", 2.0))

    # normalize traces to dict + keys (reuse helper from this file)
    Tdict, keys = _as_dict(traces)
    if not keys:
        return [], []
    T = len(next(iter(Tdict.values())))
    proto = half_sine_proto(int(proto_width))
    half = max(1, int(proto_width // 2))

    # prior grids
    r_grid = np.asarray((priors or {}).get("r", []), float)
    z_grid = np.asarray((priors or {}).get("z", []), float)

    # projection info (optional but recommended)
    P = (priors or {}).get("_proj", {})
    center = np.asarray(P.get("center", [0.0, 0.0]), float)
    pcs    = np.asarray(P.get("pcs", np.eye(3)), float)         # (3, D=3)
    scales = np.asarray(P.get("scales", np.ones(3)), float)     # (3,)
    mean   = np.asarray(P.get("mean", np.zeros(3)), float)      # (3,)

    def _interp_prior(r: float) -> float:
        if r_grid.size == 0:
            return 1.0
        if r <= r_grid[0]:   return float(z_grid[0])
        if r >= r_grid[-1]:  return float(z_grid[-1])
        k = int(np.searchsorted(r_grid, r) - 1)
        t = (r - r_grid[k]) / (r_grid[k+1] - r_grid[k] + 1e-12)
        return float((1 - t) * z_grid[k] + t * z_grid[k+1])

    def _features(x: np.ndarray) -> tuple[float, float, float]:
        pos = np.maximum(0.0, x)
        w = max(1, int(proto_width))
        ma = moving_average(pos, k=w)
        j  = int(np.argmax(ma))
        halfw = max(1, w // 2)
        area  = float(pos[max(0, j - halfw): j + halfw + 1].sum())
        meanp = float(pos.mean())
        return (j / max(1, len(x) - 1), area, meanp)

    def perm_null_z(sig: np.ndarray, idx: int, n: int = 120) -> float:
        obs = _corr_at(sig, proto, idx, int(proto_width), T)
        null = np.empty(n, float)
        rng_local = np.random.default_rng(0)
        for i in range(n):
            shift = rng_local.integers(1, max(2, T - 1))
            null[i] = _corr_at(_circ_shift(sig, int(shift)), proto, idx, int(proto_width), T)
        mu, sd = float(null.mean()), float(null.std() + 1e-8)
        return (obs - mu) / sd

    # residual/raw/cm prep
    Eres = perpendicular_energy(Tdict)
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    Sraw = {p: moving_average(Tdict[p], k=sigma) for p in keys}
    Scm  = moving_average(common_mode(Tdict), k=sigma)

    # peak indices
    peak_idx = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in keys}
    cm_idx   = peak_idx[keys[0]] if keys else 0

    # baseline z-scores
    z_res = {p: perm_null_z(Sres[p], peak_idx[p]) for p in keys}
    z_raw = {p: perm_null_z(Sraw[p], peak_idx[p]) for p in keys}
    z_cm  = {p: perm_null_z(Scm,      cm_idx)     for p in keys}

    # baseline blended score
    s_base = np.array([1.0 * z_res[p] + 0.4 * z_raw[p] - 0.3 * max(0.0, z_cm[p]) for p in keys], float)

    # prior term per channel
    radii = []
    for p in keys:
        f = np.asarray(_features(Sraw[p]), float)  # (3,)
        X3 = ((f - mean) @ pcs.T) / (scales + 1e-8)
        r  = float(np.linalg.norm(X3[:2] - center))
        radii.append(r)
    s_prior = np.array([(_interp_prior(r) ** q_s) * beta_s for r in radii], float)

    # mixed score
    scores = ((1.0 - alpha) * s_base + alpha * s_prior).tolist()

    # area around peak + null threshold (for absolute gate)
    areas, thrs = [], []
    for p in keys:
        c = np.correlate(Sres[p], proto, mode="same")  # local corr curve for area approx
        j = peak_idx[p]
        w = c[max(0, j - half): j + half + 1]
        a = float(np.clip(w, 0.0, None).sum())
        t = float(z_null)  # use z as a margin proxy here; report uses explicit null, but we keep it simple
        areas.append(a)
        thrs.append(t)

    # gating
    best = max(scores) if scores else 0.0
    area_min = 6.0
    keep_mask = []
    for s, a, t in zip(scores, areas, thrs):
        rel_ok = (s >= rel_floor * best) if best > 0 else False
        abs_ok = (s >= t)                 # s already z-like; treat z as floor
        area_ok = (a >= area_min)
        keep_mask.append(rel_ok and abs_ok and area_ok)

    order_idx = [i for i, _ in sorted(enumerate(keys), key=lambda kv: peak_idx[kv[1]]) if keep_mask[i]]

    if not order_idx and not allow_empty:
        jbest = int(np.argmax(scores)) if scores else 0
        keep_mask = [False] * len(keys)
        if scores:
            keep_mask[jbest] = True
        order_idx = [jbest] if scores else []

    keep_keys  = [keys[i] for i, k in enumerate(keep_mask) if k]
    order_keys = [keys[i] for i in order_idx]
    return keep_keys, order_keys

def geodesic_parse_report_conf(
    traces,
    *,
    sigma: int = 9,
    proto_width: int = 160,
):
    """
    Same decision logic as geodesic_parse_report, but also returns per-channel
    diagnostics: z_res, z_raw, z_cm, blended score, and peak index.
    Returns: (keep_keys, order_keys, debug_dict)
    """
    import numpy as np

    Tdict, keys = _as_dict(traces)
    if not keys:
        return [], [], {"channels": {}}

    T = len(next(iter(Tdict.values())))
    proto = half_sine_proto(int(proto_width))

    # Residual energy & smoothed views
    Eres = perpendicular_energy(Tdict)
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    Sraw = {p: moving_average(Tdict[p], k=sigma) for p in keys}
    Scm  = moving_average(common_mode(Tdict), k=sigma)

    # Peak indices (correlate residual with prototype)
    peak_idx = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in keys}
    cm_idx   = peak_idx[keys[0]] if keys else 0

    # Permutation/circ-shift null Z
    def perm_null_z(sig: np.ndarray, idx: int, n: int = 120) -> float:
        obs = _corr_at(sig, proto, idx, int(proto_width), T)
        null = np.empty(n, float)
        rng_local = np.random.default_rng(0)
        for i in range(n):
            shift = rng_local.integers(1, max(2, T - 1))
            null[i] = _corr_at(_circ_shift(sig, int(shift)), proto, idx, int(proto_width), T)
        mu, sd = float(null.mean()), float(null.std() + 1e-8)
        return (obs - mu) / sd

    z_res = {p: perm_null_z(Sres[p], peak_idx[p]) for p in keys}
    z_raw = {p: perm_null_z(Sraw[p], peak_idx[p]) for p in keys}
    z_cm  = {p: perm_null_z(Scm,      cm_idx)     for p in keys}

    score = {p: 1.0 * z_res[p] + 0.4 * z_raw[p] - 0.3 * max(0.0, z_cm[p]) for p in keys}

    # Keep + order (identical to geodesic_parse_report)
    smax = max(score.values()) + 1e-12 if score else 0.0
    keep = [p for p in keys if score[p] >= 0.5 * smax]
    if not keep and keys:
        keep = [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak_idx[p])

    debug = {
        "channels": {
            p: {
                "peak_idx": int(peak_idx[p]),
                "z_res": float(z_res[p]),
                "z_raw": float(z_raw[p]),
                "z_cm":  float(z_cm[p]),
                "score": float(score[p]),
            } for p in keys
        },
        "smax": float(smax),
    }
    return keep, order, debug



