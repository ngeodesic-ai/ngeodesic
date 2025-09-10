from __future__ import annotations
from typing import Sequence, Iterable
from collections import Counter

def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    tp = sum(t == p for t, p in zip(y_true, y_pred))
    fp = sum(t != p for t, p in zip(y_true, y_pred))
    fn = fp  # single-label simplification
    return {"tp": tp, "fp": fp, "fn": fn, "n": len(y_true)}

def prf(hits: Iterable[tuple[bool, bool]]) -> dict:
    # hits = iterable of (is_present_truth, is_present_pred)
    t = f = m = 0
    for truth, pred in hits:
        if truth and pred: t += 1
        elif not truth and pred: f += 1
        elif truth and not pred: m += 1
    prec = t / (t + f) if (t + f) else 0.0
    rec = t / (t + m) if (t + m) else 0.0
    f1 = 2 * prec * rec / (prec + rec + 1e-12) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "hallucination": f / max(1, t + f), "omission": m / max(1, t + m)}

