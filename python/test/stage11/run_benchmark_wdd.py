# Patch run_benchmark_wdd.py to be constructor-compatible with different
# versions of ngeodesic.core.denoise.TemporalDenoiser.
# - Adds a LocalDenoiser fallback that implements "ema", "median", "hybrid"
# - Adds a make_denoiser() factory that tries the 3-arg ctor, then 0-arg + configure,
#   and finally falls back to LocalDenoiser.
# Save as /mnt/data/run_benchmark_wdd_compat.py
# -*- coding: utf-8 -*-
"""
Stage 11 — Report + Denoise (WDD) — package-based runner (compat)
This version is constructor-compatible with different TemporalDenoiser APIs.

python3 python/test/stage11/run_benchmark_wdd.py \
  --samples 5 --seed 42 \
  --denoise_mode hybrid --ema_decay 0.85 --median_k 3 \
  --probe_k 5 --probe_eps 0.02 --conf_gate 0.65 --noise_floor 0.03 \
  --seed_jitter 2 \
  --latent_arc --latent_arc_noise 0.05 \
  --out_csv latent_arc_denoise.csv \
  --out_json latent_arc_denoise.json

"""

from __future__ import annotations

import argparse, json, os, math, random, logging as pylog
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np

# ---------- ngeodesic imports (no local re-implementations) ----------
from ngeodesic.core.parser import geodesic_parse_report, stock_parse  # report path
from ngeodesic.bench.metrics import set_metrics                      # metrics
from ngeodesic.bench.io import write_rows_csv, write_json            # io
from ngeodesic.core.denoise import TemporalDenoiser, snr_db          # denoiser
from ngeodesic.synth.arc_like import make_synthetic_traces_stage11 as make_synthetic_traces  # generator

try:
    from ngeodesic.viz import collect_HE, render_pca_well            # manifold viz (optional)
except Exception:
    collect_HE = None
    render_pca_well = None

PRIMS = ["flip_h", "flip_v", "rotate"]

# ----------------------------
# Denoiser compatibility layer
# ----------------------------
class LocalDenoiser:
    """Minimal drop-in denoiser that supports latent() and logits().
    Modes: off, ema, median, hybrid.
    """
    def __init__(self, mode: str = "off", ema_decay: float = 0.85, median_k: int = 3):
        self.mode = mode
        self.ema_decay = float(ema_decay)
        self.median_k = int(max(1, median_k))
        self._ema = None
        self._buf = deque(maxlen=self.median_k)

    def reset(self):
        self._ema = None
        self._buf.clear()

    def latent(self, x: np.ndarray) -> np.ndarray:
        if self.mode in ("ema", "hybrid"):
            if self._ema is None:
                self._ema = x.astype(float).copy()
            else:
                self._ema = self.ema_decay * self._ema + (1.0 - self.ema_decay) * x
            x_ema = self._ema
        else:
            x_ema = x

        if self.mode in ("median", "hybrid"):
            self._buf.append(x)
            if len(self._buf) == 1:
                x_med = x
            else:
                x_med = np.median(np.stack(list(self._buf)), axis=0)
        else:
            x_med = x

        if self.mode == "ema":      return x_ema
        if self.mode == "median":   return x_med
        if self.mode == "hybrid":   return 0.5 * (x_ema + x_med)
        return x

    def logits(self, z):
        # No-op path for logits smoothing (not used in this demo).
        return z

def make_denoiser(mode: str, ema_decay: float, median_k: int):
    """Instantiate the package denoiser if possible; otherwise fall back to LocalDenoiser."""
    try:
        # Preferred (consolidated) signature
        return TemporalDenoiser(mode, ema_decay, median_k)
    except TypeError:
        # Try zero-arg construction with a configure/set_mode pattern
        try:
            den = TemporalDenoiser()
            if hasattr(den, "set_mode"):
                den.set_mode(mode=mode, ema_decay=ema_decay, median_k=median_k)
                return den
            if hasattr(den, "configure"):
                den.configure(mode=mode, ema_decay=ema_decay, median_k=median_k)
                return den
        except Exception:
            pass
        # Final fallback that guarantees interface
        return LocalDenoiser(mode, ema_decay, median_k)

# ----------------------------
# Denoiser demo hooks (minimal)
# ----------------------------
@dataclass
class ModelHooks:
    def propose_step(self, x_t: np.ndarray, x_star: np.ndarray, args):
        direction = x_star - x_t
        dist = float(np.linalg.norm(direction) + 1e-9)
        unit = direction / (dist + 1e-9)
        step_mag = min(1.0, 0.1 + 0.9 * math.tanh(dist / (getattr(args,"proto_width",160) + 1e-9)))
        noise = np.random.normal(scale=1e-3, size=x_t.shape)
        dx_raw = step_mag * unit + noise
        conf_rel = float(max(0.0, min(1.0, 1.0 - math.exp(-dist / (getattr(args,"proto_width",160) + 1e-9)))))
        logits = None
        return dx_raw, conf_rel, logits

    def descend_vector(self, p: np.ndarray, x_star: np.ndarray, args) -> np.ndarray:
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
        return dict(accuracy_exact=accuracy_exact, precision=precision, recall=recall,
                    f1=f1, jaccard=jaccard, hallucination_rate=hallucination_rate,
                    omission_rate=omission_rate)

def phantom_guard(step_vec: np.ndarray, pos: np.ndarray, descend_fn, k: int = 3, eps: float = 0.02) -> bool:
    if k <= 1: return True
    denom = float(np.linalg.norm(step_vec) + 1e-9)
    step_dir = step_vec / denom
    agree = 0
    base_scale = float(np.linalg.norm(pos) + 1e-9)
    for _ in range(k):
        delta = np.random.randn(*pos.shape) * eps * base_scale
        probe_step = descend_fn(pos + delta)
        if np.dot(step_dir, probe_step) > 0:
            agree += 1
    return agree >= (k // 2 + 1)

class Runner:
    def __init__(self, args, hooks: ModelHooks):
        self.args = args
        self.hooks = hooks
        self.rng = np.random.default_rng(args.seed)
        self.logger = pylog.getLogger("stage11.denoise")
        self._latent_names, self._latent_targets = ([], [])
        if getattr(args, "latent_arc", False):
            self._latent_names, self._latent_targets = self._build_latent_arc_set(args.latent_dim, args.seed, args.latent_arc_noise)

    def _build_latent_arc_set(self, dim: int, seed: int, noise_scale: float):
        rng = np.random.default_rng(seed)
        names, X = [], []
        # Five simple canonical cases (same spirit as Stage-11)
        xA = np.zeros(dim); xA[0] =  1.0; xA[1] =  0.5
        xB = np.zeros(dim); xB[0] = -0.8; xB[1] =  0.9
        xC = np.zeros(dim); r = 1.2; xC[0] = r/np.sqrt(2); xC[1] = r/np.sqrt(2)
        xD = np.zeros(dim); xD[:4] = np.array([0.7,-0.6,0.5,-0.4])
        xE = np.zeros(dim); xE[0] = 0.0; xE[1] = -1.3
        for nm, x in zip(["A_axis","B_quad","C_ring","D_mix4","E_down"], [xA,xB,xC,xD,xE]):
            X.append(x + rng.normal(scale=noise_scale, size=x.shape)); names.append(nm)
        return names, X

    def run_sample(self, i: int) -> Dict[str, float]:
        dim = getattr(self.args, "latent_dim", 64)
        if self._latent_targets:
            j = i % len(self._latent_targets)
            x_star = np.array(self._latent_targets[j], dtype=float)
            self._last_latent_arc_name = self._latent_names[j]
        else:
            x_star = self.rng.uniform(-1.0, 1.0, size=(dim,))
            self._last_latent_arc_name = None

        x_t = self.rng.uniform(-1.0, 1.0, size=(dim,))
        den = make_denoiser(self.args.denoise_mode, self.args.ema_decay, self.args.median_k)
        if hasattr(den, "reset"):
            den.reset()

        for _ in range(50):
            dx_raw, conf_rel, logits = self.hooks.propose_step(x_t, x_star, self.args)
            residual = (x_star - x_t)
            dx = dx_raw

            if getattr(self.args, "log_snr", 1):
                _ = snr_db(residual, dx_raw)

            if conf_rel < self.args.conf_gate or np.linalg.norm(dx) < self.args.noise_floor:
                dx = 0.5 * residual

            def _desc(p: np.ndarray) -> np.ndarray:
                return self.hooks.descend_vector(p, x_star, self.args)
            if not phantom_guard(dx, x_t, _desc, k=self.args.probe_k, eps=self.args.probe_eps):
                dx = 0.3 * residual

            x_next = x_t + dx
            if hasattr(den, "latent"):
                x_next = den.latent(x_next)
            if logits is not None and hasattr(den, "logits"):
                _ = den.logits(logits)

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
        if self._last_latent_arc_name:
            m["latent_arc"] = self._last_latent_arc_name
        return m

    def run(self) -> Dict[str, float]:
        Ms: List[Dict[str, float]] = []
        names: List[str] = []
        for i in range(self.args.samples):
            m = self.run_sample(i)
            if m.get("latent_arc"):
                names.append(m["latent_arc"])
            Ms.append(m)
        keys = [k for k in Ms[0].keys() if k != "latent_arc"] if Ms else []
        agg = {k: float(np.mean([m[k] for m in Ms])) for k in keys}
        if names:
            by = {}
            for m in Ms:
                nm = m.get("latent_arc", "?")
                by.setdefault(nm, []).append(m)
            agg["latent_arc_breakdown"] = {nm: {k: float(np.mean([x[k] for x in arr])) for k in keys}
                                           for nm, arr in by.items()}
        self.logger.info("[SUMMARY] Geodesic (denoise path): %s", agg)
        print("[SUMMARY] Denoise :", {k: (round(v,3) if isinstance(v,float) else v) for k,v in agg.items()})
        return agg

# ----------------------------
# Report path (stock vs geodesic)
# ----------------------------
def prefix_exact(true_list: List[str], pred_list: List[str]) -> bool:
    return list(true_list) == list(pred_list)

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-11 — package WDD runner (report + denoise)")
    # data
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=720)
    p.add_argument("--sigma", type=int, default=9)
    p.add_argument("--proto_width", type=int, default=160)
    # generator knobs
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--cm_amp", type=float, default=0.02)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--amp_jitter", type=float, default=0.4)
    p.add_argument("--distractor_prob", type=float, default=0.4)
    p.add_argument("--min_tasks", type=int, default=1)
    p.add_argument("--max_tasks", type=int, default=3)

    # outputs
    p.add_argument("--out_csv", type=str, default="stage11_metrics.csv")
    p.add_argument("--out_json", type=str, default="stage11_summary.json")

    # DENOISE & GUARDS
    p.add_argument("--denoise_mode", type=str, default="off",
                   choices=["off", "ema", "median", "hybrid"])
    p.add_argument("--ema_decay", type=float, default=0.85)
    p.add_argument("--median_k", type=int, default=3)
    p.add_argument("--probe_k", type=int, default=5)
    p.add_argument("--probe_eps", type=float, default=0.02)
    p.add_argument("--conf_gate", type=float, default=0.65)
    p.add_argument("--noise_floor", type=float, default=0.03)
    p.add_argument("--seed_jitter", type=int, default=2)
    p.add_argument("--log_snr", type=int, default=1)

    # Latent ARC tests
    p.add_argument("--latent_arc", action="store_true")
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--latent_arc_noise", type=float, default=0.05)

    # Logging
    p.add_argument("--log", type=str, default="INFO")
    return p

def main():
    args = build_argparser().parse_args()
    lvl = getattr(pylog, getattr(args, "log", "INFO").upper(), pylog.INFO)
    pylog.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = pylog.getLogger("stage11.wdd")

    rng = np.random.default_rng(args.seed)

    # -------------------
    # Metrics (report path)
    # -------------------
    rows: List[Dict[str, object]] = []
    agg_geo = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)
    agg_stock = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)

    for i in range(1, args.samples + 1):
        traces, true_order = make_synthetic_traces(
            rng,
            T=args.T,
            noise=args.noise,
            cm_amp=args.cm_amp,
            overlap=args.overlap,
            amp_jitter=args.amp_jitter,
            distractor_prob=args.distractor_prob,
            tasks_k=(args.min_tasks, args.max_tasks),
        )

        keep_g, order_g = geodesic_parse_report(traces, sigma=args.sigma, proto_width=args.proto_width)
        keep_s, order_s = stock_parse(traces, sigma=args.sigma, proto_width=args.proto_width)

        acc_g = int(prefix_exact(true_order, order_g))
        acc_s = int(prefix_exact(true_order, order_s))

        sm_g = set_metrics(true_order, keep_g)
        sm_s = set_metrics(true_order, keep_s)

        for k, v in sm_g.items():
            key = {"precision":"P","recall":"R","f1":"F1","jaccard":"J","hallucination_rate":"H","omission_rate":"O"}[k]
            agg_geo[key] = agg_geo.get(key, 0) + v
        for k, v in sm_s.items():
            key = {"precision":"P","recall":"R","f1":"F1","jaccard":"J","hallucination_rate":"H","omission_rate":"O"}[k]
            agg_stock[key] = agg_stock.get(key, 0) + v

        agg_geo["acc"] += acc_g
        agg_stock["acc"] += acc_s

        rows.append(dict(
            sample=i,
            true="|".join(true_order),
            geodesic_tasks="|".join(keep_g), geodesic_order="|".join(order_g), geodesic_ok=acc_g,
            stock_tasks="|".join(keep_s), stock_order="|".join(order_s), stock_ok=acc_s,
            geodesic_precision=sm_g["precision"], geodesic_recall=sm_g["recall"], geodesic_f1=sm_g["f1"],
            geodesic_jaccard=sm_g["jaccard"], geodesic_hallucination=sm_g["hallucination_rate"], geodesic_omission=sm_g["omission_rate"],
            stock_precision=sm_s["precision"], stock_recall=sm_s["recall"], stock_f1=sm_s["f1"],
            stock_jaccard=sm_s["jaccard"], stock_hallucination=sm_s["hallucination_rate"], stock_omission=sm_s["omission_rate"]))

    n = float(args.samples)
    Sg = dict(
        accuracy_exact=agg_geo["acc"]/n, precision=agg_geo["P"]/n, recall=agg_geo["R"]/n, f1=agg_geo["F1"]/n,
        jaccard=agg_geo["J"]/n, hallucination_rate=agg_geo["H"]/n, omission_rate=agg_geo["O"]/n
    )
    Ss = dict(
        accuracy_exact=agg_stock["acc"]/n, precision=agg_stock["P"]/n, recall=agg_stock["R"]/n, f1=agg_stock["F1"]/n,
        jaccard=agg_stock["J"]/n, hallucination_rate=agg_stock["H"]/n, omission_rate=agg_stock["O"]/n
    )

    if args.out_csv:
        write_rows_csv(args.out_csv, rows)

    summary = dict(samples=int(n), geodesic=Sg, stock=Ss, csv=args.out_csv)
    if args.out_json:
        write_json(args.out_json, summary)

    print("[SUMMARY] Geodesic:", {k: round(v, 3) for k, v in Sg.items()})
    print("[SUMMARY] Stock   :", {k: round(v, 3) for k, v in Ss.items()})
    print(f"[CSV ] {args.out_csv}")
    print(f"[JSON] {args.out_json}")

    # -------------------
    # Denoiser path (optional)
    # -------------------
    if args.denoise_mode != "off" and args.latent_arc:
        hooks = ModelHooks()
        runner = Runner(args, hooks)
        denoise_metrics = runner.run()
        if args.out_json:
            try:
                with open(args.out_json, "r") as f:
                    S = json.load(f)
            except Exception:
                S = {}
            S["denoise"] = denoise_metrics
            write_json(args.out_json, S)
        print("[DENOISE] latent-ARC metrics:", {k: (round(v,3) if isinstance(v,float) else v) for k,v in denoise_metrics.items()})

if __name__ == "__main__":
    main()

