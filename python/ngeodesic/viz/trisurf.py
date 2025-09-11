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

__all__ = ["plot_trisurf"]

def plot_trisurf(X3: np.ndarray, energy: np.ndarray | None = None,
                 elev: float = 22.0, azim: float = -60.0, s: int = 6):
    """
    Minimal 3D scatter as a stand-in for a real trisurf. Avoids SciPy dependency.
    Returns (fig, ax). If 'energy' provided, uses it for color; else uses PC3.
    """
    import matplotlib.pyplot as plt  # optional dependency
    X3 = np.asarray(X3, float)
    if X3.ndim != 2 or X3.shape[1] < 3:
        raise ValueError("X3 must be (N,3)+")
    c = energy if energy is not None else X3[:, 2]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=s, c=c)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    return fig, ax

