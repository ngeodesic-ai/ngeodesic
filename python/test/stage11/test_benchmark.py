# python/test/stage11/run_benchmark.py
from __future__ import annotations
import numpy as np
from ngeodesic.synth.arc_like import make_synthetic_traces
from ngeodesic.core.parser import geodesic_parse_report

def main():
    # make sure lobe_width == proto_width for a clean match
    T = 160
    proto_width = 64
    sigma = 7

    traces, truth = make_synthetic_traces(seed=42, which=(1, 2), T=T, lobe_width=proto_width)

    keep_keys, order_keys = geodesic_parse_report(
        traces,
        sigma=sigma,
        proto_width=proto_width,
        # allow_empty=False   # default; set True only for “no-signal” demos
    )

    order = list(map(int, order_keys))
    keep_mask = [(str(i) in set(keep_keys)) for i in range(len(traces))]

    print("truth order:", truth)
    print("pred  order:", order)
    print("keep mask  :", keep_mask)

    # tiny sanity check: true channels are kept and ordered first
    assert order[: len(truth)] == truth, "Truth sequence should appear first in prediction."

if __name__ == "__main__":
    main()
