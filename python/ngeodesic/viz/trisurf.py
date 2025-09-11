from __future__ import annotations
import numpy as np

__all__ = ["plot_trisurf"]

def plot_trisurf(X3: np.ndarray, energy: np.ndarray | None = None,
                 elev: float = 22.0, azim: float = -60.0, s: int = 6):
    """
    Minimal 3D scatter as a stand-in for a real trisurf. Avoids SciPy dependency.
    Returns (fig, ax). If 'energy' provided, uses it for color; else uses PC3.
    """
    import matplotlib.pyplot as plt  # optional dependency
    X3 = np.asarray(X3, float)
    if X3.ndim != 2 or X3.shape[1] < 3:
        raise ValueError("X3 must be (N,3)+")
    c = energy if energy is not None else X3[:, 2]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=s, c=c)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    return fig, ax

