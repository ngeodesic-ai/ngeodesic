from __future__ import annotations
from typing import Mapping, Any, Sequence
import numpy as np
from .hooks import ModelHooks

class Runner:
    def __init__(self, cfg: Mapping[str, Any], hooks: ModelHooks):
        self.cfg = cfg
        self.hooks = hooks

    def run_sample(self, x0: np.ndarray, x_star: np.ndarray) -> dict:
        # Minimal placeholder loop; you’ll wire NGF parser/denoiser calls in here.
        dx, conf, aux = self.hooks.propose_step(x0, x_star, cfg=self.cfg)
        x1 = x0 + dx
        scores = self.hooks.score_sample(x1, x_star)
        return {"x_final": x1, "confidence": conf, "aux": aux, "scores": dict(scores)}

    def run(self, dataset: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict:
        out = [self.run_sample(x0, x_star) for (x0, x_star) in dataset]
        return {"n": len(out), "results": out}

