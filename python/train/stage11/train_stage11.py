#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-11 training driver (package-based).
Usage example:
    python3 -m pip install -e .
    python3 python/train/stage11/train_stage11.py \
      --samples 200 --seed 42 --T 64 --latent-dim 64 \
      --denoise-mode hybrid --ema-decay 0.85 --median-k 3 \
      --probe-k 5 --probe-eps 0.02 --conf-gate 0.65 --noise-floor 0.03 \
      --seed-jitter 2 \
      --dump-npz out/latent_arc_dump.npz \
      --out-json out/train_metrics.json
"""
from __future__ import annotations
import argparse, logging, sys
from ngeodesic.sidecar import RunConfig, DenoiseRunner, DemoHooks
from ngeodesic.utils import write_json

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-11 training driver (ngeodesic)")
    # core
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=64)
    # latents
    p.add_argument("--no-latent-arc", action="store_true",
                   help="Disable ARC latent set; use random pairs instead.")
    p.add_argument("--latent-arc-noise", type=float, default=0.05)
    # denoise/gates
    p.add_argument("--denoise-mode", choices=["off","ema","median","hybrid"], default="hybrid")
    p.add_argument("--ema-decay", type=float, default=0.85, help="EMA decay; alpha = 1 - decay")
    p.add_argument("--median-k", type=int, default=3)
    p.add_argument("--probe-k", type=int, default=3)
    p.add_argument("--probe-eps", type=float, default=0.02)
    p.add_argument("--conf-gate", type=float, default=0.60)
    p.add_argument("--noise-floor", type=float, default=0.05)
    p.add_argument("--seed-jitter", type=int, default=0)
    p.add_argument("--log-snr", action="store_true")
    # step shaping (kept for DemoHooks)
    p.add_argument("--sigma", type=int, default=9)
    p.add_argument("--proto-width", type=int, default=160)
    # outputs
    p.add_argument("--dump-npz", type=str, default="")
    p.add_argument("--out-json", type=str, default="out/train_metrics.json")
    p.add_argument("--log-level", default="INFO")
    return p

def main():
    ap = build_argparser()
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("train-stage11")

    cfg = RunConfig(
        samples=args.samples,
        seed=args.seed,
        T=args.T,
        latent_dim=args.latent_dim,
        latent_arc=(not args.no_latent_arc),
        latent_arc_noise=args.latent_arc_noise,
        denoise_mode=args.denoise_mode,
        ema_decay=args.ema_decay,
        median_k=args.median_k,
        probe_k=args.probe_k,
        probe_eps=args.probe_eps,
        conf_gate=args.conf_gate,
        noise_floor=args.noise_floor,
        seed_jitter=args.seed_jitter,
        log_snr=args.log_snr,
        sigma=args.sigma,
        proto_width=args.proto_width,
        dump_latents=args.dump_npz,
    )

    runner = DenoiseRunner(cfg, hooks=DemoHooks(), logger=log)
    metrics = runner.run()

    write_json(args.out_json, metrics)
    log.info(f"[OK] wrote metrics JSON -> {args.out_json}")
    if cfg.dump_latents:
        log.info(f"[OK] wrote NPZ dump    -> {cfg.dump_latents}")

if __name__ == "__main__":
    main()
