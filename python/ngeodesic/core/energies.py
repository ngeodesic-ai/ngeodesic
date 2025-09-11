# ngeodesic/core/energies.py

import numpy as np

__all__ = [
    # ... existing exports ...,
    "perpendicular_energy",
]

def perpendicular_energy(H: np.ndarray, axis: int = 0) -> float:
    """
    Simple proxy: energy orthogonal to a chosen axis in H (PCA space).
    H: (N, D) points in PCA space
    axis: which principal axis to treat as 'radial'; others are 'perpendicular'
    Returns scalar energy = mean squared norm of components orthogonal to axis.
    """
    H = np.asarray(H, dtype=float)
    if H.ndim != 2:
        raise ValueError("H must be (N, D)")
    D = H.shape[1]
    mask = np.ones(D, dtype=bool); 
    if 0 <= axis < D: mask[axis] = False
    perp = H[:, mask]
    return float(np.mean(np.sum(perp * perp, axis=1)))
