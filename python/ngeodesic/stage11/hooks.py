# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
import math

__all__ = ["ModelHooks"]

@dataclass
class ModelHooks:
    """
    Stage-11 minimal hooks used by the denoise runner.

    Methods:
      - propose_step(x_t, x_star, args) -> (dx_raw, conf_rel, logits_or_None)
      - descend_vector(p, x_star, args) -> step vector toward target
      - score_sample(x_final, x_star)   -> metrics dict (toy latent harness)
    """

    def propose_step(self, x_t: np.ndarray, x_star: np.ndarray, args):
        d = x_star - x_t
        dist = float(np.linalg.norm(d) + 1e-9)
        unit = d / (dist + 1e-9)
        step_mag = min(1.0, 0.1 + 0.9 * math.tanh(dist / (getattr(args, "proto_width", 160) + 1e-9)))
        noise = np.random.normal(scale=1e-3, size=x_t.shape)
        dx_raw = step_mag * unit + noise
        conf_rel = float(max(0.0, min(1.0, 1.0 - math.exp(-dist / (getattr(args, "proto_width", 160) + 1e-9)))))
        logits = None  # placeholder for future logits smoothing
        return dx_raw, conf_rel, logits

    def descend_vector(self, p: np.ndarray, x_star: np.ndarray, args) -> np.ndarray:
        return (x_star - p)

    def score_sample(self, x_final: np.ndarray, x_star: np.ndarray) -> Dict[str, float]:
        err = float(np.linalg.norm(x_final - x_star))
        acc = 1.0 if err < 0.05 else 0.0
        hall = max(0.0, min(1.0, err)) * 0.2
        omi  = max(0.0, min(1.0, err)) * 0.1
        P = max(0.0, 1.0 - 0.5 * hall)
        R = max(0.0, 1.0 - 0.5 * omi)
        F1 = (2 * P * R) / (P + R + 1e-9)
        J  = F1 / (2 - F1 + 1e-9)
        return dict(
            accuracy_exact=acc,
            precision=P, recall=R, f1=F1, jaccard=J,
            hallucination_rate=hall, omission_rate=omi
        )
