# -*- coding: utf-8 -*-
from __future__ import annotations
import logging as pylog
from typing import Callable, Dict, List, Tuple
import numpy as np

from ngeodesic.core.denoise import make_denoiser, snr_db

class Runner:
    """
    Minimal Stage-11 denoise runner moved into the package.

    Dependencies are intentionally injected:
      - hooks: object with propose_step(), descend_vector(), score_sample()
      - guard_fn: callable(step_vec, pos, descend_fn, k, eps) -> bool

    Everything else (phantom_guard, ModelHooks, CLI args, latent ARC specifics)
    stays in your script for now to keep this change isolated.
    """
    def __init__(self, args, hooks, guard_fn: Callable):
        self.args = args
        self.hooks = hooks
        self.guard_fn = guard_fn
        self.rng = np.random.default_rng(args.seed)
        self.logger = pylog.getLogger("stage11.denoise")

        # Keep latent set code local to Runner for now (no extra moves yet)
        self._names, self._targets = ([], np.zeros((0, args.latent_dim)))
        if getattr(args, "latent_arc", False):
            self._names, self._targets = self._build_latent_set(
                args.latent_dim, args.seed, args.latent_arc_noise
            )

    # --- internal (kept here to avoid moving more pieces right now) ---
    def _build_latent_set(self, dim: int, seed: int, noise: float):
        rng = np.random.default_rng(seed)
        xA = np.zeros(dim); xA[0]= 1.0; xA[1]= 0.5
        xB = np.zeros(dim); xB[0]=-0.8; xB[1]= 0.9
        r  = 1.2
        xC = np.zeros(dim); xC[0]= r/np.sqrt(2); xC[1]= r/np.sqrt(2)
        xD = np.zeros(dim); xD[:4]= np.array([0.7,-0.6,0.5,-0.4])
        xE = np.zeros(dim); xE[1]= -1.3
        names = ["A_axis","B_quad","C_ring","D_mix4","E_down"]
        X = np.stack([xA,xB,xC,xD,xE], axis=0)
        if noise and float(noise) != 0.0:
            X = X + rng.normal(scale=noise, size=X.shape)
        return names, X

    # --- public API ---
    def run_sample(self, i: int) -> Dict[str, float]:
        # choose target
        if len(self._targets):
            j = i % len(self._targets)
            x_star = self._targets[j].copy()
            latent_name = self._names[j]
        else:
            x_star = self.rng.uniform(-1.0, 1.0, size=(self.args.latent_dim,))
            latent_name = None

        # init position + denoiser
        x_t = self.rng.uniform(-1.0, 1.0, size=x_star.shape)
        den = make_denoiser(self.args.denoise_mode, self.args.ema_decay, self.args.median_k)
        if hasattr(den, "reset"):
            den.reset()

        # iterate
        for _ in range(50):
            dx_raw, conf_rel, logits = self.hooks.propose_step(x_t, x_star, self.args)
            residual = x_star - x_t
            dx = dx_raw

            if getattr(self.args, "log_snr", 1):
                _ = snr_db(residual, dx_raw)

            if conf_rel < self.args.conf_gate or np.linalg.norm(dx) < self.args.noise_floor:
                dx = 0.5 * residual

            # phantom guard via injected function
            if not self.guard_fn(dx, x_t,
                                 lambda p: self.hooks.descend_vector(p, x_star, self.args),
                                 k=self.args.probe_k, eps=self.args.probe_eps):
                dx = 0.3 * residual

            x_next = x_t + dx
            if hasattr(den, "latent"):
                x_next = den.latent(x_next)

            if self.args.seed_jitter > 0:
                xs = [x_next]
                for _ in range(self.args.seed_jitter):
                    jitter = np.random.normal(scale=0.01, size=x_next.shape)
                    xj = x_t + dx + jitter
                    if hasattr(den, "latent"):
                        xj = den.latent(xj)
                    xs.append(xj)
                x_next = np.mean(xs, axis=0)

            x_t = x_next

        m = self.hooks.score_sample(x_t, x_star)
        if latent_name:
            m["latent_arc"] = latent_name
        return m

    def run(self) -> Dict[str, float]:
        Ms: List[Dict[str, float]] = [self.run_sample(i) for i in range(self.args.samples)]
        keys = [k for k in Ms[0].keys() if k != "latent_arc"] if Ms else []
        agg  = {k: float(np.mean([m[k] for m in Ms])) for k in keys}
        if any("latent_arc" in m for m in Ms):
            by = {}
            for m in Ms:
                nm = m.get("latent_arc", "?")
                by.setdefault(nm, []).append(m)
            agg["latent_arc_breakdown"] = {nm: {k: float(np.mean([x[k] for x in arr])) for k in keys}
                                           for nm, arr in by.items()}
        self.logger.info("[SUMMARY] Geodesic (denoise path): %s", agg)
        print("[SUMMARY] Denoise :", {k: (round(v,3) if isinstance(v,float) else v) for k,v in agg.items()})
        return agg
