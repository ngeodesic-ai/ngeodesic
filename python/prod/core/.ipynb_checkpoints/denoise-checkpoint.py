import numpy as np
from .smoothing import ema, median_filter

class TemporalDenoiser:
    def __init__(self, mode="hybrid", ema_decay=0.85, median_k=3):
        self.mode, self.ema_decay, self.median_k = mode, ema_decay, median_k

    def smooth(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "ema": return ema(x, self.ema_decay)
        if self.mode == "median": return median_filter(x, self.median_k)
        return median_filter(ema(x, self.ema_decay), self.median_k)

def phantom_guard(direction_dot: float, floor: float = 0.0) -> bool:
    return direction_dot > floor

def snr_db(signal: np.ndarray, noise: np.ndarray, eps: float = 1e-9) -> float:
    num = np.linalg.norm(signal)
    den = np.linalg.norm(noise) + eps
    return 20.0 * np.log10(num / den + eps)

