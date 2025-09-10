from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ParserConfig:
    sigma: int = 9                 # smoothing width for envelopes
    proto_width: int = 64          # matched-filter template length
    tau_area: float = 10.0         # area floor (synthetic scale)
    tau_corr: float = 0.7          # nxcorr floor
    null_shifts: int = 64          # circular-shift samples for nulls

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
