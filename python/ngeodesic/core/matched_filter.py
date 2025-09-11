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
import numpy as np
from typing import Literal

__all__ = ["half_sine_proto", "nxcorr", "null_threshold"]

def half_sine_proto(width: int) -> np.ndarray:
    """
    Unit-norm half-sine prototype of length `width`.
    """
    w = int(max(3, width))
    p = np.sin(np.linspace(0.0, np.pi, w))
    return p / (np.linalg.norm(p) + 1e-8)

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-8)

def nxcorr(x: np.ndarray, q: np.ndarray, mode: Literal["same","valid","full"]="same") -> np.ndarray:
    """
    Quick normalized cross-correlation (whole-signal z-norm on both sides).
    Good default for scoring & ranking peaks.
    """
    xs = _zscore(np.asarray(x, float))
    qs = _zscore(np.asarray(q, float))
    return np.correlate(xs, qs[::-1], mode=mode)

def _circ_shift(x: np.ndarray, k: int) -> np.ndarray:
    k = int(k) % len(x)
    if k == 0:
        return x
    return np.concatenate([x[-k:], x[:-k]])

def null_threshold(
    x: np.ndarray,
    q: np.ndarray,
    *,
    shifts: int = 600,
    z: float = 2.2,
    mode: Literal["perm","circ"] = "perm",
) -> float:
    """
    Permutation/circular-shift null model for correlation maxima.
    Returns a scalar threshold = mu + z * sd over the null distribution of max nxcorr.
    """
    x = np.asarray(x, float)
    q = np.asarray(q, float)
    T = len(x)
    rng = np.random.default_rng(0)

    null_max = np.empty(shifts, float)
    for i in range(shifts):
        if mode == "circ":
            xi = _circ_shift(x, rng.integers(1, max(2, T - 1)))
        else:
            xi = rng.permutation(x)
        c = nxcorr(xi, q, mode="same")
        null_max[i] = float(np.max(c))

    mu = float(null_max.mean())
    sd = float(null_max.std() + 1e-8)
    return mu + z * sd
