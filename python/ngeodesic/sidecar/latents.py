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
from typing import List, Tuple
import numpy as np

__all__ = ["build_latent_arc_set"]

def build_latent_arc_set(dim: int, seed: int = 0) -> Tuple[List[str], List[np.ndarray]]:
    """
    Five canonical Stage-11 latent targets with simple geometry.
    Returns (names, targets)
    """
    rng = np.random.default_rng(seed)
    # A: axis-aligned pull
    xA = np.zeros(dim); xA[0] = 1.0; xA[1] = 0.5
    # B: quadrant target
    xB = np.zeros(dim); xB[0] = -0.8; xB[1] = 0.9
    # C: ring-radius target (SW)
    xC = np.zeros(dim); r = 1.2; ang = np.deg2rad(225); xC[0] = r*np.cos(ang); xC[1] = r*np.sin(ang)
    # D: shallow well near origin
    xD = np.zeros(dim); xD[0] = 0.25; xD[1] = -0.15
    # E: deep well at edge
    xE = np.zeros(dim); xE[0] = 1.8; xE[1] = -1.4
    targets = [xA, xB, xC, xD, xE]
    names = ["axis_pull","quad_NE","ring_SW","shallow_origin","deep_edge"]
    return names, targets
