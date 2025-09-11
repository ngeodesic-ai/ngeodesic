# -*- coding: utf-8 -*-
"""
# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Copyright 2025 Ian C. Moore (Provisional Patents #63/864,726, #63/865,437, #63/871,647 and #63/872,334)
# Email: ngeodesic@gmail.com
# Part of Noetic Geodesic Framework (NGF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

python3 python/test/stage11/test_benchmark.py

"""

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
