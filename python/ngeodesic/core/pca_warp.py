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
from typing import Dict, Tuple, Optional
import numpy as np

__all__ = ["pca3_and_warp"]

def _pca3_whiten(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA -> 3D (no sklearn). Returns (Yw, pcs, scales, mean)
      mean   : (D,)
      pcs    : (3, D) principal axes
      scales : (3,) per-component std for whitening
      Yw     : (N, 3) whitened 3D scores
    """
    X = np.asarray(X, float)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # SVD economy
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Vt[:3, :]                   # (3, D)
    Y  = Xc @ pcs.T                   # (N, 3)
    scales = Y.std(axis=0, ddof=1) + 1e-8
    Yw = Y / scales
    return Yw, pcs, scales, mu.reshape(-1)

def pca3_and_warp(H: np.ndarray, E: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Minimal Stage-11 PCA warp packer.
    Args:
      H : (N, D) feature rows, e.g. concatenated z-scored residuals
      E : optional (N,) weights (unused for now; reserved for center finding)
    Returns:
      dict with projection info for prior mixing:
        - "mean":   (D,)
        - "pcs":    (3, D)
        - "scales": (3,)
        - "center": (2,) center in (PC1,PC2) after whitening; we use [0,0].
    """
    Yw, pcs, scales, mu = _pca3_whiten(H)
    # For Stage-11 use we take the radial center at the origin in whitened (PC1,PC2).
    center = np.array([0.0, 0.0], dtype=float)
    return dict(mean=mu, pcs=pcs, scales=scales, center=center)
