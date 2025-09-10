import numpy as np

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")

def ema(x: np.ndarray, decay: float) -> np.ndarray:
    y = np.empty_like(x, dtype=float)
    a = float(decay)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a * y[i-1] + (1 - a) * x[i]
    return y

def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    r = k // 2
    y = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lo, hi = max(0, i - r), min(len(x), i + r + 1)
        y[i] = np.median(x[lo:hi])
    return y

