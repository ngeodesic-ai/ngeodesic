# python/test/stage11/test_parser.py
from ngeodesic.synth.arc_like import make_synthetic_traces
from ngeodesic.core.parser import geodesic_parse_report
from ngeodesic.compat.types import ParserConfig

CFG = ParserConfig()  # only using sigma/proto_width from here

def _bool_mask_from_keep(keep_keys, n):
    keep_set = set(keep_keys)
    return [str(i) in keep_set for i in range(n)]

def _ints(keys):
    return [int(k) for k in keys]

def test_active_two_channels_detected_and_ordered():
    traces, truth = make_synthetic_traces(seed=123, which=(1, 2), T=160, lobe_width=64)

    keep_keys, order_keys = geodesic_parse_report(
        traces,
        sigma=CFG.sigma,
        proto_width=CFG.proto_width,
        # allow_empty defaults to False (script’s behavior)
    )

    order = _ints(order_keys)
    keep_mask = _bool_mask_from_keep(keep_keys, len(traces))

    # Baseline parser can keep extra weak channels; assert the truth appears in order, in order.
    assert order[: len(truth)] == truth
    # And both true channels are kept.
    assert keep_mask[1] and keep_mask[2]

def test_no_active_channels_kept():
    traces, _ = make_synthetic_traces(seed=321, which=(), T=160, lobe_width=64)

    keep_keys, order_keys = geodesic_parse_report(
        traces,
        sigma=CFG.sigma,
        proto_width=CFG.proto_width,
        allow_empty=True,   # explicitly request empty when no channel clears null
    )

    assert order_keys == []
    assert keep_keys == []
