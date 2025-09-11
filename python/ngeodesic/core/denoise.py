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
from typing import Iterable, Literal, Optional
import numpy as np

__all__ = ["phantom_guard", "snr_db", "TemporalDenoiser"]

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _as_float1d(x: Iterable[float]) -> np.ndarray:
    x = np.asarray(x, float).reshape(-1)
    return x

def phantom_guard(x: Iterable[float], z: float = 3.0) -> np.ndarray:
    """
    Z-score clamp to suppress 'phantom' spikes before smoothing.
    x' = mean + clip((x - mean)/std, [-z, +z]) * std
    """
    x = _as_float1d(x)
    mu = x.mean()
    sd = x.std() + 1e-8
    zed = (x - mu) / sd
    zed = np.clip(zed, -float(z), float(z))
    return mu + zed * sd

def snr_db(signal: Iterable[float], noise: Optional[Iterable[float]] = None) -> float:
    """
    10*log10(P_signal / P_noise). If noise is None, returns +inf.
    Typical usage: snr_db(clean, noisy - clean)
    """
    s = _as_float1d(signal)
    if noise is None:
        return float("inf")
    n = _as_float1d(noise)
    ps = float(np.mean(s * s))
    pn = float(np.mean(n * n)) + 1e-20
    return 10.0 * np.log10(ps / pn)

# ------------------------------------------------------------
# Core filters
# ------------------------------------------------------------

def _ema(x: np.ndarray, alpha: float = 0.20) -> np.ndarray:
    """
    Exponential moving average with bias correction on the first few samples.
    """
    x = _as_float1d(x)
    y = np.empty_like(x)
    acc = 0.0
    w = 0.0
    a = float(alpha)
    for i, xi in enumerate(x):
        w = a + (1.0 - a) * w
        acc = a * xi + (1.0 - a) * acc
        y[i] = acc / (w + 1e-12)   # mild warm-up correction
    return y

def _median_filter(x: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Sliding-window median (odd k), reflect-padded.
    """
    x = _as_float1d(x)
    k = int(max(1, k))
    if k == 1:
        return x.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    # rolling median (vectorized extraction)
    strides = xp.strides[0]
    n = x.size
    windows = np.lib.stride_tricks.as_strided(xp, shape=(n, k), strides=(strides, strides))
    return np.median(windows, axis=1)

# ------------------------------------------------------------
# High-level denoiser
# ------------------------------------------------------------

class TemporalDenoiser:
    """
    Unified temporal denoiser:
      - method="ema":    EMA(alpha)
      - method="median": median(k)
      - method="hybrid": phantom_guard -> median(k) -> EMA(alpha)
    """
    def __init__(
        self,
        *,
        method: Literal["ema", "median", "hybrid"] = "hybrid",
        ema_alpha: float = 0.20,
        median_k: int = 5,
        hybrid_k: int = 5,
        guard_z: Optional[float] = 3.0,
    ) -> None:
        self.method = method
        self.ema_alpha = float(ema_alpha)
        self.median_k = int(median_k)
        self.hybrid_k = int(hybrid_k)
        self.guard_z = guard_z if (guard_z is None) else float(guard_z)

    def smooth(self, x: Iterable[float]) -> np.ndarray:
        x = _as_float1d(x)
        if self.method == "ema":
            xg = phantom_guard(x, self.guard_z) if (self.guard_z is not None) else x
            return _ema(xg, alpha=self.ema_alpha)

        if self.method == "median":
            xg = phantom_guard(x, self.guard_z) if (self.guard_z is not None) else x
            return _median_filter(xg, k=self.median_k)

        # hybrid: guard -> median -> ema
        xg = phantom_guard(x, self.guard_z) if (self.guard_z is not None) else x
        xm = _median_filter(xg, k=self.hybrid_k)
        return _ema(xm, alpha=self.ema_alpha)

    # convenience alias
    __call__ = smooth
