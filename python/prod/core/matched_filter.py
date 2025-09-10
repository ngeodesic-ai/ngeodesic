import numpy as np

def half_sine_proto(n: int) -> np.ndarray:
    t = np.linspace(0, np.pi, n, endpoint=False)
    return np.sin(t)

def nxcorr(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    x = (x - x.mean()) / (x.std() + 1e-12)
    q = (q - q.mean()) / (q.std() + 1e-12)
    # naive valid-mode correlation
    out = np.correlate(x, q, mode="same")
    return out / (len(q) + 1e-12)

def null_threshold(x: np.ndarray, q: np.ndarray, shifts: int = 64, z: float = 3.0) -> float:
    # simple circular-shift null
    N = len(x)
    mx = []
    for k in range(shifts):
        s = (k * (N // shifts)) % N
        xr = np.roll(x, s)
        mx.append(nxcorr(xr, q).max())
    mu, sd = np.mean(mx), np.std(mx) + 1e-12
    return mu + z * sd
