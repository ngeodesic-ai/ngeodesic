from __future__ import annotations
import numpy as np
from typing import List, Tuple
from ..core.matched_filter import half_sine_proto

def make_synthetic_traces(
    n_channels: int = 3,
    T: int = 160,
    which: Tuple[int, ...] = (1, 2),
    noise_sd: float = 0.25,
    lobe_width: int = 64,
    seed: int = 0,
) -> Tuple[List[np.ndarray], List[int]]:
    rng = np.random.default_rng(seed)
    traces: List[np.ndarray] = []
    starts = {}
    for c in range(n_channels):
        x = rng.normal(0, noise_sd, T)
        if c in which:
            q = half_sine_proto(lobe_width)        # e.g., 64
            start = int(rng.integers(8, T - lobe_width - 8))
            x[start:start + lobe_width] += q
            starts[c] = start   
        traces.append(x)
    truth_order = sorted(which, key=lambda ch: starts[ch])
    return traces, truth_order