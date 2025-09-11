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

import argparse, csv, json, os, warnings
from typing import Dict, List, Tuple
import numpy as np

# --- use package parsers (no local re-definitions) ---
from ngeodesic.core.parser import geodesic_parse_report, stock_parse
from ngeodesic.synth.arc_like import make_synthetic_traces_stage11
from ngeodesic.viz.well_render import render_pca_well
from ngeodesic.bench.metrics import set_metrics  
from ngeodesic.core.matched_filter import half_sine_proto, nxcorr, null_threshold
from ngeodesic.core.denoise import TemporalDenoiser, snr_db
from ngeodesic.synth import gaussian_bump
from ngeodesic.viz import collect_HE, render_pca_well
from ngeodesic.bench.io import write_rows_csv, write_json

PRIMS = ["flip_h", "flip_v", "rotate"]

# ============================================================
# Stage-11 synthetic ARC-like generator (RNG-first, hard mode)
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-11 report benchmark (package-based)")
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
    p.add_argument("--render_well", action="store_true", help="Render PCA well/funnel proxies")
    return p

def main():
    args = build_argparser().parse_args()
    rng = np.random.default_rng(args.seed)

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

    if args.render_well:
        try:
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
            warnings.warn(f"Render failed: {e}")

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

if __name__ == "__main__":
    main()
