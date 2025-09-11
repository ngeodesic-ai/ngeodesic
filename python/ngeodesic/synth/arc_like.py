# python/ngeodesic/synth/arc_like.py
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional, Union
import numpy as np

# Keep existing simple generator(s) you already have above…
# ------------------------------------------------------------------

PRIMS: Tuple[str, str, str] = ("flip_h", "flip_v", "rotate")

def _gaussian_bump(T: int, center: int, width: int, amp: float = 1.0) -> np.ndarray:
    """FWHM-based Gaussian bump."""
    t = np.arange(T)
    sig2 = (width / 2.355) ** 2  # FWHM→σ
    return amp * np.exp(-((t - center) ** 2) / (2 * sig2))

# public alias (useful in tests/demos)
def gaussian_bump(T: int, center: int, width: int, amp: float = 1.0) -> np.ndarray:
    return _gaussian_bump(T, center, width, amp)

def _as_rng(rng_or_seed: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    if isinstance(rng_or_seed, np.random.Generator):
        return rng_or_seed
    if rng_or_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(rng_or_seed))

def make_synthetic_traces(
    seed: int = 0,
    which: tuple[int, ...] = (1, 2),
    T: int = 160,
    lobe_width: int = 64,
    noise: float = 0.05,
):
    """
    Returns:
      traces: list[np.ndarray] with 3 channels (len T)
      truth:  list[int] e.g. [1,2] indicating active channels in temporal order
    """
    rng = np.random.default_rng(int(seed))
    x = [np.zeros(T, float) for _ in range(3)]

    # half-sine prototype of requested width
    w = int(max(3, lobe_width))
    q = np.sin(np.linspace(0, np.pi, w))

    # centers chosen so order(which) = temporal order
    centers = {1: int(0.40 * T), 2: int(0.70 * T), 0: int(0.55 * T)}

    for ch in which:
        c = centers.get(int(ch), int(0.50 * T))
        s = max(0, c - w // 2); e = min(T, s + w)
        seg = q[: (e - s)]
        x[int(ch)][s:e] += seg

    # light Gaussian noise
    for i in range(3):
        x[i] = x[i] + rng.normal(0.0, noise, size=T)

    truth = list(which)
    return x, truth

def make_synthetic_traces_stage11(
    rng: Optional[Union[int, np.random.Generator]] = None,
    *,
    T: int = 720,
    noise: float = 0.02,
    cm_amp: float = 0.02,
    overlap: float = 0.5,
    amp_jitter: float = 0.4,
    distractor_prob: float = 0.4,
    tasks_k: Tuple[int, int] = (1, 3),
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Stage-11 “hard mode” ARC-like generator (mirrors the consolidated script).

    Returns:
      traces: dict {"flip_h","flip_v","rotate"} -> np.ndarray[T] (nonnegative)
      tasks:  list[str] true primitives in temporal order (subset of PRIMS)
    """
    rng = _as_rng(rng)

    # number of true tasks and their order
    k = int(rng.integers(tasks_k[0], tasks_k[1] + 1))
    tasks = list(rng.choice(PRIMS, size=k, replace=False))
    rng.shuffle(tasks)

    # three canonical centers pulled toward the middle by `overlap`
    base = np.array([0.20, 0.50, 0.80]) * T
    centers = ((1.0 - overlap) * base + overlap * (T * 0.50)).astype(int)
    width = int(max(12, T * 0.10))

    # low-amp common-mode drift
    t = np.arange(T)
    cm = cm_amp * (1.0 + 0.2 * np.sin(2 * np.pi * t / max(30, T // 6)))

    traces = {p: np.zeros(T, float) for p in PRIMS}

    # lay down bumps for true tasks (with center & amplitude jitter)
    for i, prim in enumerate(tasks):
        c0 = centers[i % len(centers)]
        amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
        c = int(np.clip(c0 + rng.integers(-width // 5, width // 5 + 1), 0, T - 1))
        traces[prim] += _gaussian_bump(T, c, width, amp=amp)

    # optional distractor bumps on non-task channels
    for p in PRIMS:
        if p not in tasks and rng.random() < distractor_prob:
            c = int(rng.uniform(T * 0.15, T * 0.85))
            amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
            traces[p] += _gaussian_bump(T, c, width, amp=0.9 * amp)

    # add CM + noise; clamp to nonnegative
    for p in PRIMS:
        traces[p] = np.clip(traces[p] + cm, 0, None)
        traces[p] = traces[p] + rng.normal(0, noise, size=T)
        traces[p] = np.clip(traces[p], 0, None)

    return traces, tasks
