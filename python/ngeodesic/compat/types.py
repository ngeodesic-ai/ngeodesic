# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Literal

# ------------------------------------------------------------
# Parser configuration (Stage-10/11 style)
# ------------------------------------------------------------
@dataclass(frozen=True)
class ParserConfig:
    # smoothing + template
    sigma: int = 7
    proto_width: int = 64

    # null model
    null_shifts: int = 600
    z: float = 2.2
    null_mode: Literal["perm", "circ"] = "perm"

    # gates
    rel_floor: float = 0.70
    area_floor: float = 6.0
    # kept for backward-compat with older scripts; not required by current parser
    area_floor_frac: float = 0.00
    margin_floor: float = 0.03
    allow_empty: bool = False

    def merge(self, **overrides) -> "ParserConfig":
        """Create a copy with selected fields overridden."""
        return replace(self, **{k: v for k, v in overrides.items() if hasattr(self, k)})

# ------------------------------------------------------------
# Denoiser configuration (EMA / median / hybrid)
# ------------------------------------------------------------
@dataclass(frozen=True)
class DenoiseConfig:
    method: Literal["ema", "median", "hybrid"] = "hybrid"
    # EMA
    ema_alpha: float = 0.20
    # median / hybrid windows (odd recommended)
    median_k: int = 5
    hybrid_k: int = 5
    # optional “phantom” guard (z-score clamp before smoothing)
    guard_z: float = 3.0

# ------------------------------------------------------------
# Funnel / well-fitting parameters for priors (Stage-11)
# ------------------------------------------------------------
@dataclass(frozen=True)
class WellParams:
    n_r: int = 220
    fit_quantile: float = 0.65
    rbf_bw: float = 0.30
    core_k: float = 0.18
    core_p: float = 1.70
    core_r0_frac: float = 0.14
    blend_core: float = 0.25
    template_D: float = 1.20
    template_p: float = 1.60
