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
from dataclasses import dataclass
from typing import Dict, Protocol, Tuple
import math
import numpy as np

__all__ = ["ModelHooks", "DemoHooks"]

class ModelHooks(Protocol):
    """
    Plug your model/agent here. Implement:
      - propose_step(x_t, x_star, cfg) -> (dx_raw, conf_rel, logits_opt)
      - descend_vector(p, x_star, cfg) -> descent direction (for phantom_guard probes)
      - score_sample(x_final, x_star)  -> dict of training metrics
    """
    def propose_step(self, x_t: np.ndarray, x_star: np.ndarray, cfg) -> Tuple[np.ndarray, float, np.ndarray | None]: ...
    def descend_vector(self, p: np.ndarray, x_star: np.ndarray, cfg) -> np.ndarray: ...
    def score_sample(self, x_final: np.ndarray, x_star: np.ndarray) -> Dict[str, float]: ...

# --- Default reference hooks (ported from Stage-11 script) ---
@dataclass
class DemoHooks:
    """Reference hooks so runner works out-of-the-box."""
    def propose_step(self, x_t: np.ndarray, x_star: np.ndarray, cfg) -> Tuple[np.ndarray, float, None]:
        direction = x_star - x_t
        dist = float(np.linalg.norm(direction) + 1e-9)
        unit = direction / (dist + 1e-9)
        step_mag = min(1.0, 0.1 + 0.9 * math.tanh(dist / (cfg.proto_width + 1e-9)))
        noise = np.random.normal(scale=cfg.sigma * 1e-3, size=x_t.shape)
        dx_raw = step_mag * unit + noise
        conf_rel = float(max(0.0, min(1.0, 1.0 - math.exp(-dist / (cfg.proto_width + 1e-9)))))
        return dx_raw, conf_rel, None

    def descend_vector(self, p: np.ndarray, x_star: np.ndarray, cfg) -> np.ndarray:
        return (x_star - p)

    def score_sample(self, x_final: np.ndarray, x_star: np.ndarray) -> Dict[str, float]:
        err = float(np.linalg.norm(x_final - x_star))
        accuracy_exact = 1.0 if err < 0.05 else 0.0
        hallucination_rate = max(0.0, min(1.0, err)) * 0.2
        omission_rate = max(0.0, min(1.0, err)) * 0.1
        precision = max(0.0, 1.0 - 0.5 * hallucination_rate)
        recall = max(0.0, 1.0 - 0.5 * omission_rate)
        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        jaccard = f1 / (2 - f1 + 1e-9)
        return {
            "accuracy_exact": accuracy_exact,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": jaccard,
            "hallucination_rate": hallucination_rate,
            "omission_rate": omission_rate,
        }
