from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ParserConfig:
    # smoothing + template
    sigma: int = 7
    proto_width: int = 64

    # null model
    null_shifts: int = 600
    z: float = 2.2
    null_mode: str = "perm"

    # gates
    rel_floor: float = 0.70
    area_floor: float = 6.0
    area_floor_frac: float = 0.10   # NEW
    margin_floor: float = 0.03
    corr_floor: float = 0.18        # NEW

@dataclass(frozen=True)
class DenoiseConfig:
    mode: Literal["ema", "median", "hybrid"] = "hybrid"
    ema_decay: float = 0.85
    median_k: int = 3
    conf_floor: float = 0.15       # confidence gate threshold
    jitter_J: int = 32
    jitter_eps: float = 0.01

@dataclass(frozen=True)
class WellParams:
    pca_k: int = 9                 # PCA target dim
    core_alpha: float = 0.35       # funnel/core blend
    core_k: float = 1.0            # core depth scale
    core_r0: float = 0.5           # core radius
    core_p: float = 2.0            # exponent

@dataclass(frozen=True)
class ParserConfig:
    sigma: int = 7
    proto_width: int = 64
    null_shifts: int = 600
    z: float = 2.2
    null_mode: str = "perm"   # "perm" | "circular"
    rel_floor: float = 0.70
    area_floor: float = 6.0
    margin_floor: float = 0.03
