# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence
import numpy as np
import math
import logging as pylog

from .hooks import ModelHooks
from .latents import build_latent_arc_set
from ngeodesic.core.denoise import TemporalDenoiser, phantom_guard, snr_db

__all__ = ["RunConfig", "DenoiseRunner"]

@dataclass
class RunConfig:
    # sampling
    samples: int = 100
    T: int = 64
    seed: int = 42
    latent_dim: int = 64
    latent_arc: bool = True
    latent_arc_noise: float = 0.05
    # denoiser/guards
    denoise_mode: str = "off"           # off|ema|median|hybrid
    ema_decay: float = 0.85             # script-style; we map to ema_alpha = 1-ema_decay
    median_k: int = 3
    probe_k: int = 3
    probe_eps: float = 0.02
    conf_gate: float = 0.60
    noise_floor: float = 0.05
    seed_jitter: int = 0
    log_snr: bool = False
    # step shaping (kept for DemoHooks)
    sigma: int = 9
    proto_width: int = 160
    # outputs
    dump_latents: str = ""              # NPZ path (optional)

class DenoiseRunner:
    def __init__(self, cfg: RunConfig, hooks: ModelHooks, logger: Optional[pylog.Logger] = None):
        self.cfg = cfg
        self.hooks = hooks
        self.rng = np.random.default_rng(cfg.seed)
        self.logger = logger or pylog.getLogger("ngeodesic.sidecar.runner")

        self._dump: Optional[List[Tuple[np.ndarray, np.ndarray, str]]] = [] if cfg.dump_latents else None
        self._latent_names: Optional[List[str]] = None
        self._latent_targets: Optional[List[np.ndarray]] = None
        self._latent_idx = 0

        if cfg.latent_arc:
            self._latent_names, self._latent_targets = build_latent_arc_set(cfg.latent_dim, cfg.seed)

    # -- internals ------------------------------------------------------------
    def _init_latents(self) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        """Return (x_t, x_star, name_opt)."""
        if self._latent_targets is None:
            # generic random pair
            x_star = self.rng.uniform(-1.0, 1.0, size=(self.cfg.latent_dim,))
            x0 = x_star + self.rng.normal(scale=self.cfg.latent_arc_noise, size=self.cfg.latent_dim)
            return x0, x_star, None

        j = self._latent_idx % len(self._latent_targets)
        self._latent_idx += 1
        x_star = self._latent_targets[j].copy()
        name = self._latent_names[j]
        x0 = x_star + self.rng.normal(scale=self.cfg.latent_arc_noise, size=self.cfg.latent_dim)
        return x0, x_star, name

    def _make_denoiser(self) -> TemporalDenoiser:
        mode = self.cfg.denoise_mode
        ema_alpha = 1.0 - float(self.cfg.ema_decay)   # map decay (script) -> alpha (our impl)
        k = int(self.cfg.median_k) | 1
        if mode == "off":
            return TemporalDenoiser(method="ema", ema_alpha=1.0)  # no-op behavior
        if mode == "ema":
            return TemporalDenoiser(method="ema", ema_alpha=ema_alpha)
        if mode == "median":
            return TemporalDenoiser(method="median", median_k=k)
        # hybrid
        return TemporalDenoiser(method="hybrid", ema_alpha=ema_alpha, hybrid_k=k)

    # -- public API -----------------------------------------------------------
    def run_sample(self, idx: int) -> Dict[str, float]:
        np.random.seed(self.cfg.seed + idx)  # keep prior behavior for any NumPy RNG in hooks
        x_t, x_star, name = self._init_latents()
        den = self._make_denoiser()

        for t in range(self.cfg.T):
            dx_raw, conf_rel, logits = self.hooks.propose_step(x_t, x_star, self.cfg)
            residual = x_star - x_t
            dx = dx_raw

            if self.cfg.log_snr:
                snr = snr_db(signal=residual, noise=dx - residual)
                self.logger.info(f"[i={idx} t={t}] SNR(dB)={snr:.2f} |res|={np.linalg.norm(residual):.4f} |dx|={np.linalg.norm(dx):.4f} conf={conf_rel:.3f}")

            # gates
            if (conf_rel < self.cfg.conf_gate) or (np.linalg.norm(dx) < self.cfg.noise_floor):
                dx = 0.5 * residual

            # phantom guard
            def _desc(p: np.ndarray) -> np.ndarray:
                return self.hooks.descend_vector(p, x_star, self.cfg)
            if not phantom_guard(dx, x_t, _desc, k=self.cfg.probe_k, eps=self.cfg.probe_eps):
                dx = 0.3 * residual

            # step + denoise (+ optional seed jitter)
            x_next = x_t + dx
            x_next = den.smooth(x_next)

            if self.cfg.seed_jitter > 0:
                xs = [x_next]
                for _ in range(self.cfg.seed_jitter):
                    jitter = np.random.normal(scale=0.01, size=x_next.shape)
                    xs.append(den.smooth(x_t + dx + jitter))
                x_next = np.mean(xs, axis=0)

            x_t = x_next

        if self._dump is not None:
            self._dump.append((x_t.copy(), x_star.copy(), name or ""))

        # metrics
        M = self.hooks.score_sample(x_t, x_star)
        if name is not None:
            M = {**M, "latent_arc": name}
        return M

    def run(self) -> Dict[str, float]:
        metrics_list: List[Dict[str, float]] = [self.run_sample(i) for i in range(self.cfg.samples)]
        keys = [k for k in metrics_list[0].keys() if k != "latent_arc"] if metrics_list else []
        agg = {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

        # breakdown by latent_arc name (if present)
        names_present = any("latent_arc" in m for m in metrics_list)
        if names_present:
            by = {}
            for m in metrics_list:
                nm = m.get("latent_arc", "?")
                by.setdefault(nm, []).append(m)
            agg["latent_arc_breakdown"] = {nm: {k: float(np.mean([x[k] for x in arr])) for k in keys} for nm, arr in by.items()}

        # dump NPZ if requested
        if self._dump is not None and self.cfg.dump_latents:
            labmap = {"axis_pull":0,"quad_NE":1,"ring_SW":2,"shallow_origin":3,"deep_edge":4}
            x0, xs, nm = zip(*self._dump)
            np.savez(self.cfg.dump_latents,
                     x0=np.stack(x0), x_star=np.stack(xs),
                     label=np.array([labmap.get(k, -1) for k in nm], int),
                     name=np.array(nm, dtype="U16"))

        return agg
