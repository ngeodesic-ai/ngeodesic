# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np

__all__ = ["phantom_guard"]

def phantom_guard(step_vec: np.ndarray, pos: np.ndarray, descend_fn, k: int = 3, eps: float = 0.02) -> bool:
    """
    Majority-vote directional consistency guard (“phantom well” protection).

    step_vec: proposed update at position `pos`
    pos:      current latent position
    descend_fn(p): vector field toward target from probe position p
    k:        number of stochastic probes
    eps:      probe noise scale (relative to ||pos||)

    Returns True if dot(step_vec, probe_step) > 0 for a majority of probes.
    """
    if k <= 1:
        return True
    denom = float(np.linalg.norm(step_vec) + 1e-9)
    if denom == 0.0:
        return False
    step_dir = step_vec / denom
    base = float(np.linalg.norm(pos) + 1e-9)

    agree = 0
    for _ in range(k):
        delta = np.random.randn(*pos.shape) * eps * base
        probe_step = descend_fn(pos + delta)
        if np.dot(step_dir, probe_step) > 0:
            agree += 1
    return agree >= (k // 2 + 1)
