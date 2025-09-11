__all__ = ["render_pca_well", "render_pca_well", "collect_HE"]

try:
    from .well_features import collect_HE
except Exception:
    def collect_HE(*args, **kwargs):
        raise ImportError("viz: collect_HE unavailable; missing energies.perpendicular_energy?")

try:
    from .well_render import render_pca_well
except Exception:
    def render_pca_well(*args, **kwargs):
        raise ImportError("viz: render_pca_well unavailable; missing viz deps?")
