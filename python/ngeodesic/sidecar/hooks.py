from __future__ import annotations
from typing import Protocol, Mapping, Any, Tuple
import numpy as np

class ModelHooks(Protocol):
    """
    Sidecar interface for small models / adapters.
    - propose_step: suggest a latent delta (or intent), with a confidence.
    - descend_vector: provide a local descent direction (used by phantom-guard probes).
    - score_sample: return task-level scores after a run (for metrics).
    """
    def propose_step(self, x_t: np.ndarray, x_star: np.ndarray, *, cfg: Mapping[str, Any]
                     ) -> Tuple[np.ndarray, float, dict]: ...
    def descend_vector(self, x_t: np.ndarray, x_star: np.ndarray, *, cfg: Mapping[str, Any]
                       ) -> np.ndarray: ...
    def score_sample(self, x_final: np.ndarray, x_star: np.ndarray
                     ) -> Mapping[str, float]: ...
