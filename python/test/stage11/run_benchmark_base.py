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
from ngeodesic.core.parser import geodesic_parse_report

# ============================================================
# Stage-11 synthetic ARC-like generator (RNG-first, hard mode)
# (mirrors consolidated script behavior)
# ============================================================

PRIMS = ["flip_h", "flip_v", "rotate"]

def gaussian_bump(T, center, width, amp=1.0):
    t = np.arange(T)
    sig2 = (width/2.355)**2  # FWHM→σ
    return amp * np.exp(-(t-center)**2 / (2*sig2))

def make_synthetic_traces(
    rng,
    T: int = 720,
    noise: float = 0.02,
    cm_amp: float = 0.02,
    overlap: float = 0.5,
    amp_jitter: float = 0.4,
    distractor_prob: float = 0.4,
    tasks_k: Tuple[int, int] = (1, 3),
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Returns:
      traces: dict {flip_h, flip_v, rotate} -> np.ndarray[T]
      tasks:  list[str] true ordered primitives
    """
    # choose k unique tasks and shuffle order
    k = int(rng.integers(tasks_k[0], tasks_k[1] + 1))
    tasks = list(rng.choice(PRIMS, size=k, replace=False))
    rng.shuffle(tasks)

    # schedule centers with an overlap control around middle
    base = np.array([0.20, 0.50, 0.80]) * T
    centers = ((1.0 - overlap) * base + overlap * (T * 0.50)).astype(int)
    width = int(max(12, T * 0.10))

    # low-amplitude common-mode; gentle sinusoid
    t = np.arange(T)
    cm = cm_amp * (1.0 + 0.2 * np.sin(2 * np.pi * t / max(30, T // 6)))

    traces = {p: np.zeros(T, float) for p in PRIMS}

    # lay down bumps for true tasks (with center & amplitude jitter)
    for i, prim in enumerate(tasks):
        c = centers[i % len(centers)]
        amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
        c_jit = int(np.clip(c + rng.integers(-width // 5, width // 5 + 1), 0, T - 1))
        traces[prim] += gaussian_bump(T, c_jit, width, amp=amp)

    # occasional distractor bumps on unused channels
    for p in PRIMS:
        if p not in tasks and rng.random() < distractor_prob:
            c = int(rng.uniform(T * 0.15, T * 0.85))
            amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
            traces[p] += gaussian_bump(T, c, width, amp=0.9 * amp)

    # add common-mode and measurement noise; clamp to nonnegative
    for p in PRIMS:
        traces[p] = np.clip(traces[p] + cm, 0, None)
        traces[p] = traces[p] + rng.normal(0, noise, size=T)
        traces[p] = np.clip(traces[p], 0, None)

    return traces, tasks

# ============================================================
# Geodesic report parser (Stage-11 consolidated behavior)
# ============================================================

def moving_average(x: np.ndarray, k: int = 9) -> np.ndarray:
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k) / k, mode="valid")

def common_mode(traces: Dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([traces[p] for p in PRIMS], 0).mean(0)

def perpendicular_energy(traces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mu = common_mode(traces)
    return {p: np.clip(traces[p] - mu, 0, None) for p in PRIMS}

def half_sine_proto(width: int):
    P = np.sin(np.linspace(0, np.pi, width))
    return P / (np.linalg.norm(P) + 1e-8)

def _corr_at(sig, proto, idx, width, T):
    a, b = max(0, idx - width//2), min(T, idx + width//2)
    w = sig[a:b]
    if len(w) < 3: return 0.0
    w = w - w.mean()
    pr = proto[:len(w)] - proto[:len(w)].mean()
    denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
    return float(np.dot(w, pr) / denom)

def geodesic_parse_report(traces: Dict[str, np.ndarray], *, sigma=9, proto_width=160):
    keys = list(traces.keys())
    T = len(next(iter(traces.values())))
    Eres = perpendicular_energy(traces)
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    Sraw = {p: moving_average(traces[p], k=sigma) for p in keys}
    Scm  = moving_average(common_mode(traces), k=sigma)
    proto = half_sine_proto(proto_width)

    peak_idx = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in keys}

    def circ_shift(x, k):
        k = int(k) % len(x)
        if k == 0: return x
        return np.concatenate([x[-k:], x[:-k]])

    def perm_null_z(sig, idx, n=120):
        T = len(sig); obs = _corr_at(sig, proto, idx, proto_width, T)
        null = np.empty(n, float); rng_local = np.random.default_rng(0)
        for i in range(n):
            shift = rng_local.integers(1, T-1)
            null[i] = _corr_at(circ_shift(sig, shift), proto, idx, proto_width, T)
        mu, sd = float(null.mean()), float(null.std() + 1e-8)
        return (obs - mu) / sd

    z_res = {p: perm_null_z(Sres[p], peak_idx[p]) for p in keys}
    z_raw = {p: perm_null_z(Sraw[p], peak_idx[p]) for p in keys}
    z_cm  = {p: perm_null_z(Scm,      peak_idx[keys[0]]) for p in keys}
    score = {p: 1.0*z_res[p] + 0.4*z_raw[p] - 0.3*max(0.0, z_cm[p]) for p in keys}

    smax = max(score.values()) + 1e-12
    keep = [p for p in keys if score[p] >= 0.5*smax]
    if not keep:
        keep = [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak_idx[p])
    return keep, order

# ============================================================
# Stock baseline + metrics
# ============================================================

def stock_parse(traces: Dict[str, np.ndarray], *, sigma: int = 9, proto_width: int = 160) -> Tuple[List[str], List[str]]:
    keys = list(traces.keys())
    S = {p: moving_average(traces[p], k=sigma) for p in keys}
    proto = half_sine_proto(proto_width)
    peak = {p: int(np.argmax(np.correlate(S[p], proto, mode="same"))) for p in keys}
    score = {p: float(np.max(np.correlate(S[p], proto, mode="same"))) for p in keys}
    smax = max(score.values()) + 1e-12
    keep = [p for p in keys if score[p] >= 0.6 * smax]
    if not keep:
        keep = [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak[p])
    return keep, order

def set_metrics(true_list: List[str], pred_list: List[str]) -> Dict[str, float]:
    Tset, Pset = set(true_list), set(pred_list)
    tp, fp, fn = len(Tset & Pset), len(Pset - Tset), len(Tset - Pset)
    precision = tp / max(1, len(Pset))
    recall    = tp / max(1, len(Tset))
    f1        = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    jaccard   = tp / max(1, len(Tset | Pset))
    return dict(
        precision=precision, recall=recall, f1=f1, jaccard=jaccard,
        hallucination_rate=fp / max(1, len(Pset)),
        omission_rate=fn / max(1, len(Tset)),
    )

def write_rows_csv(path: str, rows: List[Dict[str, object]]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_json(path: str, obj: Dict[str, object]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ============================================================
# Optional PCA-well proxies (kept lightweight)
# ============================================================

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
except Exception:
    plt = None
    PCA = None

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-8)

def render_pca_well(out_raw: str, out_fit: str, H: np.ndarray, E: np.ndarray):
    if plt is None or PCA is None:
        warnings.warn("matplotlib or scikit-learn not available; skipping render")
        return
    pca = PCA(n_components=3, whiten=True, random_state=0)
    Y = pca.fit_transform(H)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=E, cmap="viridis", s=10, alpha=0.85)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("Stage-11 — Warped manifold proxy (PCA view)")
    fig.tight_layout(); fig.savefig(out_raw, dpi=220); plt.close(fig)

    # radial “fit” proxy
    r = np.linalg.norm(Y[:, :2], axis=1)
    z = Y[:, 2]
    nb = max(24, int(np.sqrt(len(Y))))
    bins = np.linspace(r.min(), r.max(), nb + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    zq = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (r >= b0) & (r < b1)
        zq.append(np.quantile(z[m], 0.65) if np.any(m) else np.nan)
    zq = np.array(zq)
    if np.any(np.isnan(zq)):
        valid = ~np.isnan(zq)
        zq[~valid] = np.interp(centers[~valid], centers[valid], zq[valid])

    th = np.linspace(0, 2*np.pi, 180)
    R, TH = np.meshgrid(centers, th)
    Z = np.tile(zq, (len(th), 1))
    Xs = R * np.cos(TH); Ys = R * np.sin(TH)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xs, Ys, Z, cmap="viridis", alpha=0.90, linewidth=0, antialiased=True)
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=E, cmap="viridis", s=10, alpha=0.60)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("Stage-11 — Data-fit funnel (proxy)")
    fig.tight_layout(); fig.savefig(out_fit, dpi=220); plt.close(fig)

def collect_HE(samples: int, rng: np.random.Generator, T: int, sigma: int, **gen_kwargs):
    H_rows, E_vals = [], []
    for _ in range(samples):
        traces, _ = make_synthetic_traces(rng, T=T, **gen_kwargs)
        E_perp = perpendicular_energy(traces)
        S = {p: moving_average(E_perp[p], k=sigma) for p in PRIMS}
        feats = np.concatenate([_z(S[p]) for p in PRIMS], axis=0)
        H_rows.append(feats)
        E_vals.append(float(sum(np.trapz(S[p]) for p in PRIMS)))
    H = np.vstack(H_rows)
    E = np.asarray(E_vals, float)
    E = (E - E.min()) / (E.ptp() + 1e-9)
    return H, E

# ============================================================
# CLI & Main
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-11 report benchmark (consolidated behavior)")
    # data
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=720)
    p.add_argument("--sigma", type=int, default=9, help="smoother window for residual energy")
    p.add_argument("--proto_width", type=int, default=160)
    # generator knobs (exact Stage-11)
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
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, object]] = []
    agg_geo = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)
    agg_stock = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)

    # Use a single RNG and advance it naturally per call (matches Stage-11)
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

        # Geodesic vs Stock
        keep_g, order_g = geodesic_parse_report(traces, sigma=args.sigma, proto_width=args.proto_width)
        keep_s, order_s = stock_parse(traces, sigma=args.sigma, proto_width=args.proto_width)

        acc_g = int(order_g == true_order)
        acc_s = int(order_s == true_order)

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

    # Optional PCA well renders (proxy) using the same generator distribution
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
