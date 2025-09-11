# Create a compatibility version of run_benchmark_v1.py that avoids importing the
# visualization package at module import time (which currently breaks due to
# missing `perpendicular_energy`). The compat runner keeps the same CLI and
# functionality, but disables the `--render_well` feature gracefully if viz
# isn't available.
#
# Output: /mnt/data/run_benchmark_v1_compat.py

"""
Stage-11 benchmark (report + optional denoise) — compat version
- Avoids importing `ngeodesic.viz` at import time.
- Keeps identical CLI to your v1 runner.
- If --render_well is requested but viz deps are missing, prints a warning and continues.

Apache 2.0 (ngeodesic.ai)

# warp-detect
python3 python/test/stage11/run_benchmark_v2.py \
    --samples 200 --seed 42 --T 720 --sigma 9 --proto_width 160 \
    --out_plot manifold_pca3_mesh_warped.png \
    --out_plot_fit manifold_pca3_mesh_warped_fit.png \
    --out_csv stage11_metrics.csv \
    --out_json stage11_summary.json

# warp-detect-denoise
python3 python/test/stage11/run_benchmark_v2.py \
  --samples 200 --seed 42 \
  --denoise_mode hybrid --ema_decay 0.85 --median_k 3 \
  --probe_k 5 --probe_eps 0.02 --conf_gate 0.65 --noise_floor 0.03 \
  --seed_jitter 2 \
  --latent_arc --latent_arc_noise 0.05 \
  --out_csv latent_arc_denoise.csv \
  --out_json latent_arc_denoise.json

"""
from __future__ import annotations

import argparse, csv, json, os, warnings, math, logging as pylog
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

# --- use package parsers (no local re-definitions) ---
from ngeodesic.core.parser import geodesic_parse_report, stock_parse
from ngeodesic.synth.arc_like import make_synthetic_traces_stage11
from ngeodesic.bench.metrics import set_metrics
from ngeodesic.core.matched_filter import half_sine_proto, nxcorr, null_threshold
from ngeodesic.core.denoise import TemporalDenoiser, snr_db
from ngeodesic.synth import gaussian_bump
from ngeodesic.core.denoise import make_denoiser
from ngeodesic.stage11.runner import Runner 
from ngeodesic.bench.io import write_rows_csv, write_json

# ===== Phantom guard & hooks (denoise path) =====
def phantom_guard(step_vec, pos, descend_fn, k=3, eps=0.02):
    if k <= 1: return True
    step_dir = step_vec / (np.linalg.norm(step_vec)+1e-9)
    base = float(np.linalg.norm(pos)+1e-9)
    agree = 0
    for _ in range(k):
        delta = np.random.randn(*pos.shape)*eps*base
        if np.dot(step_dir, descend_fn(pos+delta)) > 0: agree += 1
    return agree >= (k//2 + 1)

@dataclass
class ModelHooks:
    def propose_step(self, x_t, x_star, args):
        d = x_star - x_t; dist = float(np.linalg.norm(d)+1e-9); unit = d/(dist+1e-9)
        step_mag = min(1.0, 0.1 + 0.9*np.tanh(dist/(getattr(args,"proto_width",160)+1e-9)))
        dx_raw = step_mag*unit + np.random.normal(scale=1e-3, size=x_t.shape)
        conf_rel = float(max(0.0, min(1.0, 1.0 - np.exp(-dist/(getattr(args,"proto_width",160)+1e-9)))))
        return dx_raw, conf_rel, None
    def descend_vector(self, p, x_star, args): return (x_star - p)
    def score_sample(self, x_final, x_star):
        err = float(np.linalg.norm(x_final - x_star))
        acc = 1.0 if err < 0.05 else 0.0
        hall = max(0.0, min(1.0, err))*0.2; omi = max(0.0, min(1.0, err))*0.1
        P = max(0.0, 1.0-0.5*hall); R = max(0.0, 1.0-0.5*omi); F1 = (2*P*R)/(P+R+1e-9); J = F1/(2-F1+1e-9)
        return dict(accuracy_exact=acc, precision=P, recall=R, f1=F1, jaccard=J, hallucination_rate=hall, omission_rate=omi)

# class Runner:
#     def __init__(self, args, hooks):
#         self.args, self.hooks = args, hooks
#         self.rng = np.random.default_rng(args.seed)
#         self._names, self._targets = self._latent_set(args.latent_dim, args.seed, args.latent_arc_noise) if args.latent_arc else ([],[])
#         self.logger = pylog.getLogger("stage11.denoise")
#     def _latent_set(self, dim, seed, noise):
#         rng = np.random.default_rng(seed)
#         xA = np.zeros(dim); xA[0]=1.0; xA[1]=0.5
#         xB = np.zeros(dim); xB[0]=-0.8; xB[1]=0.9
#         xC = np.zeros(dim); r=1.2; xC[0]=r/np.sqrt(2); xC[1]=r/np.sqrt(2)
#         xD = np.zeros(dim); xD[:4]=np.array([0.7,-0.6,0.5,-0.4])
#         xE = np.zeros(dim); xE[1]=-1.3
#         names = ["A_axis","B_quad","C_ring","D_mix4","E_down"]
#         X = [xA,xB,xC,xD,xE]
#         return names, [x + rng.normal(scale=noise, size=dim) for x in X]
#     def run_sample(self, i):
#         dim = self.args.latent_dim
#         if self._targets:
#             j = i % len(self._targets); x_star = np.array(self._targets[j]); last = self._names[j]
#         else:
#             x_star = self.rng.uniform(-1,1,size=dim); last=None
#         x_t = self.rng.uniform(-1,1,size=dim)
#         den = make_denoiser(self.args.denoise_mode, self.args.ema_decay, self.args.median_k); getattr(den,"reset",lambda:None)()
#         for _ in range(50):
#             dx_raw, conf_rel, logits = self.hooks.propose_step(x_t, x_star, self.args)
#             res = x_star - x_t; dx = dx_raw
#             if self.args.log_snr: _ = snr_db(res, dx_raw)
#             if conf_rel < self.args.conf_gate or np.linalg.norm(dx) < self.args.noise_floor: dx = 0.5*res
#             if not phantom_guard(dx, x_t, lambda p: self.hooks.descend_vector(p, x_star, self.args), k=self.args.probe_k, eps=self.args.probe_eps):
#                 dx = 0.3*res
#             x_next = x_t + dx
#             if hasattr(den,"latent"): x_next = den.latent(x_next)
#             if self.args.seed_jitter>0:
#                 xs=[x_next]; 
#                 for _ in range(self.args.seed_jitter):
#                     jitt = np.random.normal(scale=0.01,size=x_next.shape); xj = x_t+dx+jitt; 
#                     if hasattr(den,"latent"): xj = den.latent(xj)
#                     xs.append(xj)
#                 x_next = np.mean(xs,0)
#             x_t = x_next
#         m = self.hooks.score_sample(x_t, x_star); 
#         if last: m["latent_arc"]=last
#         return m
#     def run(self):
#         Ms=[self.run_sample(i) for i in range(self.args.samples)]
#         keys=[k for k in Ms[0].keys() if k!="latent_arc"] if Ms else []
#         agg={k: float(np.mean([m[k] for m in Ms])) for k in keys}
#         if any("latent_arc" in m for m in Ms):
#             by={}
#             for m in Ms: by.setdefault(m.get("latent_arc","?"),[]).append(m)
#             agg["latent_arc_breakdown"]={nm:{k:float(np.mean([x[k] for x in arr])) for k in keys} for nm,arr in by.items()}
#         self.logger.info("[SUMMARY] Geodesic (denoise path): %s", agg)
#         print("[SUMMARY] Denoise :", {k:(round(v,3) if isinstance(v,float) else v) for k,v in agg.items()})
#         return agg

# ============================================================
# Stage-11 synthetic ARC-like generator (RNG-first, hard mode)
# ============================================================

def prefix_exact(true_list: List[str], pred_list: List[str]) -> bool:
    return list(true_list) == list(pred_list)

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-11 report benchmark (package-based, compat)")
    # data
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=720)
    p.add_argument("--sigma", type=int, default=9, help="smoother window for residual energy")
    p.add_argument("--proto_width", type=int, default=160)
    # generator knobs (Stage-11 hard mode)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--cm_amp", type=float, default=0.02)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--amp_jitter", type=float, default=0.4)
    p.add_argument("--distractor_prob", type=float, default=0.4)
    p.add_argument("--min_tasks", type=int, default=1)
    p.add_argument("--max_tasks", type=int, default=3)
    # outputs
    p.add_argument("--out_plot", type=str, default="manifold_pca3_mesh_warped.png")
    p.add_argument("--out_plot_fit", type=str, default="manifold_pca3_mesh_warped_fit.png")
    p.add_argument("--out_csv", type=str, default="stage11_metrics.csv")
    p.add_argument("--out_json", type=str, default="stage11_summary.json")
    # viz toggle
    p.add_argument("--render_well", action="store_true", help="Render PCA well/funnel proxies (if viz extras available)")
    # DENOISE & GUARDS
    p.add_argument("--denoise_mode", type=str, default="off", choices=["off","ema","median","hybrid"])
    p.add_argument("--ema_decay", type=float, default=0.85)
    p.add_argument("--median_k", type=int, default=3)
    p.add_argument("--probe_k", type=int, default=5)
    p.add_argument("--probe_eps", type=float, default=0.02)
    p.add_argument("--conf_gate", type=float, default=0.65)
    p.add_argument("--noise_floor", type=float, default=0.03)
    p.add_argument("--seed_jitter", type=int, default=2)
    p.add_argument("--log_snr", type=int, default=1)
    # Latent ARC
    p.add_argument("--latent_arc", action="store_true")
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--latent_arc_noise", type=float, default=0.05)
    # Logging
    p.add_argument("--log", type=str, default="INFO")
    return p

def main():
    args = build_argparser().parse_args()
    pylog.basicConfig(level=getattr(pylog, args.log.upper(), pylog.INFO),
                      format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, object]] = []
    agg_geo = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)
    agg_stock = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)

    for i in range(1, args.samples + 1):
        traces, true_order = make_synthetic_traces_stage11(
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
            stock_jaccard=sm_s["jaccard"], stock_hallucination=sm_s["hallucination_rate"], stock_omission=sm_s["omission_rate"],
        ))

    n = float(args.samples)
    Sg = dict(
        accuracy_exact=agg_geo["acc"]/n, precision=agg_geo["P"]/n, recall=agg_geo["R"]/n, f1=agg_geo["F1"]/n,
        jaccard=agg_geo["J"]/n, hallucination_rate=agg_geo["H"]/n, omission_rate=agg_geo["O"]/n
    )
    Ss = dict(
        accuracy_exact=agg_stock["acc"]/n, precision=agg_stock["P"]/n, recall=agg_stock["R"]/n, f1=agg_stock["F1"]/n,
        jaccard=agg_stock["J"]/n, hallucination_rate=agg_stock["H"]/n, omission_rate=agg_stock["O"]/n
    )

    # Optional rendering (lazy import; safe failure)
    if args.render_well:
        try:
            from ngeodesic.viz import collect_HE, render_pca_well  # NOTE: may fail if viz extras not installed
            H, E = collect_HE(
                samples=min(max(int(n), 100), 2000),
                rng=np.random.default_rng(args.seed + 777),
                T=args.T, sigma=args.sigma,
                noise=args.noise, cm_amp=args.cm_amp, overlap=args.overlap,
                amp_jitter=args.amp_jitter, distractor_prob=args.distractor_prob,
                tasks_k=(args.min_tasks, args.max_tasks),
            )
            render_pca_well(args.out_plot, args.out_plot_fit, H, E)
        except Exception as e:
            warnings.warn(f"Render disabled (viz extras not available): {e}")

    if args.out_csv:
        write_rows_csv(args.out_csv, rows)

    summary = dict(
        samples=int(n), geodesic=Sg, stock=Ss,
        plot_raw=args.out_plot, plot_fitted=args.out_plot_fit, csv=args.out_csv
    )
    if args.out_json:
        write_json(args.out_json, summary)

    print("[SUMMARY] Geodesic:", {k: round(v, 3) for k, v in Sg.items()})
    print("[SUMMARY] Stock   :", {k: round(v, 3) for k, v in Ss.items()})
    print(f"[PLOT] RAW:     {args.out_plot}")
    print(f"[PLOT] FITTED:  {args.out_plot_fit}")
    print(f"[CSV ] {args.out_csv}")
    print(f"[JSON] {args.out_json}")

    # -------------------
    # Denoiser path (optional)
    # -------------------
    if args.denoise_mode != "off" and args.latent_arc:
        hooks = ModelHooks()
        # runner = Runner(args, hooks)
        # denoise_metrics = runner.run()
        runner = Runner(args, hooks, phantom_guard)  # pass the guard function in
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

