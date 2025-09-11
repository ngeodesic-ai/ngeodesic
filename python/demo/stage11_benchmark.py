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
import argparse, csv, json, sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# --- package imports (all from ngeodesic) ---
from ngeodesic.synth.arc_like import make_synthetic_traces
from ngeodesic.core.parser import geodesic_parse_report
# prior-aware parser is optional; we guard it below
try:
    from ngeodesic.core.parser import geodesic_parse_with_prior
except Exception:
    geodesic_parse_with_prior = None  # type: ignore

from ngeodesic.core.smoothing import moving_average
from ngeodesic.core.pca_warp import pca3_and_warp
from ngeodesic.core.funnel_profile import (
    fit_radial_profile,
    analytic_core_template,
    blend_profiles,
    priors_from_profile,
)

# plotting is optional; avoid import errors
try:
    from ngeodesic.viz.trisurf import plot_trisurf
except Exception:
    plot_trisurf = None


# -------------------------
# Results & metrics helpers
# -------------------------
def keep_to_mask(keep_keys: List[str], n: int) -> List[bool]:
    s = set(keep_keys)
    return [str(i) in s for i in range(n)]

def keys_to_ints(keys: List[str]) -> List[int]:
    return [int(k) for k in keys]

@dataclass
class SampleResult:
    idx: int
    seed: int
    truth_order: List[int]
    pred_order: List[int]
    keep_mask: List[bool]
    exact_prefix_match: bool
    precision: float
    recall: float
    f1: float

@dataclass
class Summary:
    samples: int
    exact_prefix_match_rate: float
    avg_precision: float
    avg_recall: float
    avg_f1: float
    active_prefix_rate: float
    none_empty_rate: float


def prf_from_truth_and_keep(truth: List[int], keep_mask: List[bool]) -> Tuple[float, float, float]:
    n = len(keep_mask)
    truth_present = [i in set(truth) for i in range(n)]
    pred_present  = keep_mask

    tp = sum(t and p for t, p in zip(truth_present, pred_present))
    fp = sum((not t) and p for t, p in zip(truth_present, pred_present))
    fn = sum(t and (not p) for t, p in zip(truth_present, pred_present))

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# ----------------------------------------------------
# Simple well-features extractor (package-only, no ARC)
# ----------------------------------------------------
def _channel_features(x: np.ndarray, proto_width: int) -> Tuple[float, float, float]:
    """
    Lightweight per-channel features:
      - peak position (normalized to [0,1]) using a moving-average over rectified signal
      - local positive area around the peak (proxy for energy)
      - global positive mean
    """
    T = len(x)
    pos = np.maximum(0.0, x)
    w = max(1, int(proto_width))
    ma = moving_average(pos, k=w)  # uses ngeodesic.core.smoothing
    j = int(np.argmax(ma))
    half = max(1, w // 2)
    area = float(pos[max(0, j - half): j + half + 1].sum())
    meanp = float(pos.mean())
    return (j / max(1, T - 1), area, meanp)

def build_well_and_priors(
    samples: int,
    seed: int,
    T: int,
    proto_width: int,
    fit_quantile: float,
    rbf_bw: float,
    core_k: float,
    core_p: float,
    core_r0_frac: float,
    blend_core: float,
    template_D: float,
    template_p: float,
    n_r: int = 220,
    warmup_scale: float = 2.0,
    active_mode: str = "mix",
) -> Tuple[Dict[str, list], Dict]:
    """
    Assemble a simple feature cloud H (N x 3) with energy E, run PCA(3)+warp,
    fit a robust radial envelope, blend with an analytic core, and return priors.
    """
    rng = np.random.default_rng(seed)
    warmup_n = int(max(samples * warmup_scale, 50))
    feats = []
    energies = []
    for i in range(warmup_n):
        # diversify: half active, half none if 'mix'
        if active_mode == "mix":
            which = (1, 2) if (i % 2 == 0) else ()
        elif active_mode == "two":
            which = (1, 2)
        else:
            which = ()
        s = int(rng.integers(0, 2**31 - 1))
        traces, _ = make_synthetic_traces(seed=s, which=which, T=T, lobe_width=proto_width)
        for x in traces:
            f = _channel_features(np.asarray(x, float), proto_width)
            feats.append(f)
            energies.append(f[1])  # use local area as energy proxy

    H = np.asarray(feats, dtype=float)           # (N, 3)
    E = np.asarray(energies, dtype=float)        # (N,)

    X3, well_metrics, info = pca3_and_warp(H, energy=E)

    # radial profile
    r = np.linalg.norm(X3[:, :2] - info["center"][None, :], axis=1)
    r_max = float(np.quantile(r, 0.98))
    r_grid = np.linspace(0.0, r_max, int(max(32, n_r)))
    z_data = fit_radial_profile(
        X3, info["center"], r_grid,
        h=max(1e-6, rbf_bw) * max(1e-6, r_max),
        q=fit_quantile, r0_frac=core_r0_frac, core_k=core_k, core_p=core_p
    )
    z_tmpl = analytic_core_template(r_grid, D=template_D, p=template_p, r0_frac=core_r0_frac)
    z_prof = blend_profiles(z_data, z_tmpl, w=float(np.clip(blend_core, 0.0, 1.0)))

    priors = priors_from_profile(r_grid, z_prof)
    return priors, {"well_metrics": well_metrics, "r_max": r_max}


# -------------------------
# Benchmark core
# -------------------------
def run_one(idx: int, seed: int, T: int, proto_width: int, sigma: int,
            active: str, use_prior: bool, priors: Dict | None) -> SampleResult:
    rng = np.random.default_rng(seed)

    if active == "two":
        which = (1, 2)
    elif active == "none":
        which = ()
    elif active == "mix":
        which = () if rng.random() < 0.5 else (1, 2)
    else:
        raise ValueError(f"Unknown --active mode: {active}")

    traces, truth = make_synthetic_traces(
        seed=seed, which=which, T=T, lobe_width=proto_width
    )

    if use_prior and geodesic_parse_with_prior is not None and priors is not None:
        # NOTE: current package implementation may ignore prior hyperparams; we pass only what's known-safe
        keep_keys, order_keys = geodesic_parse_with_prior(  # type: ignore[call-arg]
            traces, priors, sigma=sigma, proto_width=proto_width
        )
    else:
        keep_keys, order_keys = geodesic_parse_report(
            traces, sigma=sigma, proto_width=proto_width
        )

    order = keys_to_ints(order_keys)
    keep_mask = keep_to_mask(keep_keys, len(traces))
    exact_prefix = order[: len(truth)] == truth
    prec, rec, f1 = prf_from_truth_and_keep(truth, keep_mask)

    return SampleResult(
        idx=idx,
        seed=seed,
        truth_order=list(truth),
        pred_order=order,
        keep_mask=keep_mask,
        exact_prefix_match=bool(exact_prefix),
        precision=prec,
        recall=rec,
        f1=f1,
    )


# -------------------------
# CLI
# -------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="stage11_benchmark.py",
        description="Package-only Stage-11 benchmark using ngeodesic (with optional Well→Priors).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # core
    ap.add_argument("--samples", type=int, default=20, help="Number of synthetic samples.")
    ap.add_argument("--seed", type=int, default=42, help="Global seed.")
    ap.add_argument("--T", type=int, default=160, help="Trace length.")
    ap.add_argument("--sigma", type=int, default=7, help="Smoothing used by parser.")
    ap.add_argument("--proto_width", type=int, default=64, help="Half-sine template width (match lobe width).")
    ap.add_argument("--active", choices=["two", "none", "mix"], default="two",
                    help="Which primitives are active in synthetic data.")
    ap.add_argument("--out_json", type=Path, default=None, help="Write per-sample results + summary to JSON.")
    ap.add_argument("--out_csv", type=Path, default=None, help="Write per-sample results to CSV.")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-sample prints.")

    # WDD / priors flags (accepted for compatibility; parser may ignore some until wired internally)
    ap.add_argument("--use_funnel_prior", type=int, choices=[0,1], default=0, help="Enable prior-aware parsing if available.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Mixing strength for prior (if supported).")
    ap.add_argument("--beta_s", type=float, default=0.25, help="Score shaping (if supported).")
    ap.add_argument("--q_s", type=float, default=2.0, help="Score shaping exponent (if supported).")
    ap.add_argument("--tau_rel", type=float, default=0.60, help="Relative threshold (if supported).")
    ap.add_argument("--tau_abs_q", type=float, default=0.93, help="Absolute quantile threshold (if supported).")
    ap.add_argument("--null_K", type=int, default=40, help="Null resampling count (if supported).")

    # well/profile params
    ap.add_argument("--n_r", type=int, default=220, help="Radial samples for envelope.")
    ap.add_argument("--fit_quantile", type=float, default=0.65, help="Quantile for robust envelope.")
    ap.add_argument("--rbf_bw", type=float, default=0.30, help="RBF bandwidth as fraction of r_max.")
    ap.add_argument("--core_k", type=float, default=0.18, help="Core boost strength.")
    ap.add_argument("--core_p", type=float, default=1.7, help="Core boost exponent.")
    ap.add_argument("--core_r0_frac", type=float, default=0.14, help="Core radius fraction of r_max.")
    ap.add_argument("--blend_core", type=float, default=0.25, help="Blend weight with analytic template [0..1].")
    ap.add_argument("--template_D", type=float, default=1.2, help="Analytic template decay.")
    ap.add_argument("--template_p", type=float, default=1.6, help="Analytic template exponent.")
    ap.add_argument("--out_plot", type=Path, default=None, help="Optional PNG of well scatter (requires matplotlib).")

    return ap


def main(argv: List[str] | None = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(args.samples)]

    # Build priors once if requested
    priors = None
    prior_info = {}
    if args.use_funnel_prior == 1:
        priors, prior_info = build_well_and_priors(
            samples=args.samples,
            seed=args.seed,
            T=args.T,
            proto_width=args.proto_width,
            fit_quantile=args.fit_quantile,
            rbf_bw=args.rbf_bw,
            core_k=args.core_k,
            core_p=args.core_p,
            core_r0_frac=args.core_r0_frac,
            blend_core=args.blend_core,
            template_D=args.template_D,
            template_p=args.template_p,
            n_r=args.n_r,
            active_mode=args.active,
        )

        # optional plot
        if args.out_plot and plot_trisurf is not None:
            # Rebuild the well coordinates to plot
            # (quick reuse of the build function to avoid duplicating logic)
            # This is a lightweight re-run to get X3; skip the plot if you don't need it.
            pass  # kept minimal; use your package's plot helper if desired

    # Run benchmark
    results: List[SampleResult] = []
    for i, s in enumerate(seeds):
        r = run_one(
            idx=i,
            seed=s,
            T=args.T,
            proto_width=args.proto_width,
            sigma=args.sigma,
            active=args.active,
            use_prior=(args.use_funnel_prior == 1),
            priors=priors,
        )
        results.append(r)
        if not args.quiet:
            print(f"[{i:03d}] seed={s} truth={r.truth_order} pred={r.pred_order} keep={r.keep_mask} "
                  f"exact_prefix={r.exact_prefix_match} P/R/F1={r.precision:.2f}/{r.recall:.2f}/{r.f1:.2f}")

    # Summary (including stratified view)
    n = len(results)
    epm = sum(r.exact_prefix_match for r in results) / n if n else 0.0
    avg_p = sum(r.precision for r in results) / n if n else 0.0
    avg_r = sum(r.recall for r in results) / n if n else 0.0
    avg_f = sum(r.f1 for r in results) / n if n else 0.0

    active = [r for r in results if len(r.truth_order) > 0]
    none   = [r for r in results if len(r.truth_order) == 0]
    _avg = lambda xs: (sum(xs)/len(xs) if xs else 0.0)
    active_prefix = _avg([r.exact_prefix_match for r in active])
    none_empty    = _avg([all(b is False for b in r.keep_mask) for r in none])

    summ = Summary(
        samples=n,
        exact_prefix_match_rate=epm,
        avg_precision=avg_p,
        avg_recall=avg_r,
        avg_f1=avg_f,
        active_prefix_rate=active_prefix,
        none_empty_rate=none_empty,
    )

    print("\nSummary:",
          f"samples={summ.samples} exact_prefix_match_rate={summ.exact_prefix_match_rate:.2f} "
          f"avgP={summ.avg_precision:.2f} avgR={summ.avg_recall:.2f} avgF1={summ.avg_f1:.2f}")
    print("Stratified:",
          f"active_prefix={summ.active_prefix_rate:.2f}  none_empty={summ.none_empty_rate:.2f}")

    # Output
    if args.out_json:
        payload = {
            "summary": asdict(summ),
            "results": [asdict(r) for r in results],
            "config": {
                "samples": args.samples, "seed": args.seed, "T": args.T,
                "sigma": args.sigma, "proto_width": args.proto_width, "active": args.active,
                "use_funnel_prior": args.use_funnel_prior,
                "prior_info": prior_info,
                "wdd": {
                    "alpha": args.alpha, "beta_s": args.beta_s, "q_s": args.q_s,
                    "tau_rel": args.tau_rel, "tau_abs_q": args.tau_abs_q, "null_K": args.null_K,
                    "n_r": args.n_r, "fit_quantile": args.fit_quantile, "rbf_bw": args.rbf_bw,
                    "core_k": args.core_k, "core_p": args.core_p, "core_r0_frac": args.core_r0_frac,
                    "blend_core": args.blend_core, "template_D": args.template_D, "template_p": args.template_p,
                },
            },
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON: {args.out_json}")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "seed", "truth_order", "pred_order", "keep_mask",
                        "exact_prefix_match", "precision", "recall", "f1"])
            for r in results:
                w.writerow([
                    r.idx, r.seed,
                    " ".join(map(str, r.truth_order)),
                    " ".join(map(str, r.pred_order)),
                    " ".join("1" if b else "0" for b in r.keep_mask),
                    int(r.exact_prefix_match),
                    f"{r.precision:.4f}", f"{r.recall:.4f}", f"{r.f1:.4f}",
                ])
        print(f"Wrote CSV: {args.out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
