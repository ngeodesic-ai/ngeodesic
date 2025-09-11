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

import numpy as np

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")

def ema(x: np.ndarray, decay: float) -> np.ndarray:
    y = np.empty_like(x, dtype=float)
    a = float(decay)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * y[i-1] + (1 - a) * x[i]
    return y

def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    r = k // 2
    y = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lo, hi = max(0, i - r), min(len(x), i + r + 1)
        y[i] = np.median(x[lo:hi])
    return y

