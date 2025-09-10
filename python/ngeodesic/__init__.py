"""ngeodesic — Geometry-First AI Reasoning Engine."""
from .core.parser import stock_parse, geodesic_parse_report, geodesic_parse_with_prior
from .core.denoise import TemporalDenoiser, phantom_guard, snr_db

__all__ = [
    "stock_parse",
    "geodesic_parse_report",
    "geodesic_parse_with_prior",
    "TemporalDenoiser",
    "phantom_guard",
    "snr_db",
]
__version__ = "0.0.2a1"
