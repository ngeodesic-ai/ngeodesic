# ngeodesic

**Geometry-First AI Reasoning Engine**  
_Shapes latent geometry to suppress hallucinations and stabilize multi-step plans._

---

## Why ngeodesic?

Modern LLMs are great at language but drift under pressure. **ngeodesic** adds a geometry-first reasoning layer that treats model latents like a physical system:

- **Warp** the manifold to a single dominant “well” (funnel)  
- **Detect** true signals with matched filters + statistically calibrated nulls  
- **Denoise** phantom wells with hybrid smoothing, inhibition guards, and jitter averaging

Use it as a standalone parser/validator or pair it with a small **micro-LLM sidecar** (e.g., for DeFi, ARC-like tasks, coding, or tool plans, etc.).

---

## Install

```bash
# Core (lightweight)
pip install ngeodesic==0.0.2a1

# With optional extras
pip install "ngeodesic[ml]"      # scikit-learn: PCA/whitening
pip install "ngeodesic[viz]"     # matplotlib/scipy: plots
pip install "ngeodesic[dev]"     # pytest, ruff, mypy, etc.
```

Python ≥ 3.10 recommended.

---

## Quickstart

```python
import numpy as np
from ngeodesic.core.denoise import TemporalDenoiser
from ngeodesic.core.parser import geodesic_parse_report  # Stage-10/11 parser
from ngeodesic.core.matched_filter import half_sine_proto

# Example: parse a few synthetic residual traces (one per primitive channel)
rng = np.random.default_rng(42)
n, T = 3, 160
traces = [np.clip(rng.normal(0, 0.3, T) + (i==1)*np.sin(np.linspace(0, np.pi, T)), -2, 2)
          for i in range(n)]

# Detect & order candidate primitives
keep_mask, order = geodesic_parse_report(traces, sigma=9, proto_width=64)

# Optional: smooth a confidence series
den = TemporalDenoiser(mode="hybrid", ema_decay=0.85, median_k=3)
smoothed = den.smooth(np.array([0.1, 0.4, 0.8, 0.6, 0.7], dtype=float))

print("Detected:", keep_mask, "Order:", order)
```

> Tip: In LLM workflows, feed `traces` from a stable hidden-state “tap” (e.g., layer −9, last-k tokens). The parser/denoiser is model-agnostic.

---

## Core ideas (1-minute tour)

- **Warp**: Project latents to PCA(3), fit a radial **funnel** profile, and compute curvature/depth metrics.  
- **Detect**: Build per-primitive residual channels, run a **unimodal matched filter**, gate with dual thresholds (relative vs best channel, absolute vs **null** from circular shifts or permutations).  
- **Denoise**: Hybrid EMA+median smoothing, **phantom-guard** local probes, and **jitter averaging** to enforce stable, repeatable decisions.

---

## API overview

```python
# Parser / detection
from ngeodesic.core.parser import (
    stock_parse,                  # simple baseline
    geodesic_parse_report,        # Stage-10 parser
    geodesic_parse_with_prior,    # Stage-11 (optional priors/funnel)
)

# Funnel & warp
from ngeodesic.core.pca_warp import pca3_and_warp
from ngeodesic.core.funnel_profile import (
    fit_radial_profile, analytic_core_template, priors_from_profile
)

# Denoising & guards
from ngeodesic.core.denoise import TemporalDenoiser, phantom_guard, snr_db

# Sidecar (micro-LLM hooks)
from ngeodesic.sidecar.hooks import ModelHooks   # Protocol interface
from ngeodesic.sidecar.runner import Runner      # Minimal loop runner
```

---

## Typical LLM pattern

1. **Encode** a prompt/context with `output_hidden_states=True` (e.g., GPT-2-medium).  
2. **Tap** a stable layer (e.g., −9), slice the last-k tokens (e.g., 160).  
3. **Detect** candidate primitives with `geodesic_parse_report(...)`.  
4. **Denoise/guard** to suppress phantoms and late flips.  
5. **Execute** confirmed steps in your domain layer (e.g., DeFiPy, ARC primitives, tool calls).

---


## Status & roadmap

- **Status:** pre-release `0.1.0a1` (APIs stabilizing; tests/docs in progress).  
- **Next:** finalize `parser.py` Stage-11 path, add DeFi synthetic generator, and ship examples.  
- **Micro-LLM research:** lives in a separate fast-iteration repo; hardened pieces will fold back here as `ngeodesic.micro.*`.

---

## License & notice

- **License:** Apache-2.0 (see `LICENSE`).  
- **Notice:** repository may reference methods described in the NGF provisional filings. Publishing the code under Apache-2.0 does **not** grant a patent license.

---

## Citation (optional)

If this work helps your research or product, please cite:

```
@software{ngeodesic,
  title   = {ngeodesic: Geodesic Reasoning Engine for LLMs},
  author  = {Moore, Ian C.},
  year    = {2025},
  url     = {https://pypi.org/project/ngeodesic/}
}
```

