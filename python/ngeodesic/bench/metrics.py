from __future__ import annotations
from typing import Dict, Iterable, List, Sequence, Tuple
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



def set_metrics(true_list: Sequence[str], pred_list: Sequence[str]) -> Dict[str, float]:
    """
    Set-style metrics for Stage-11 reporting.

    Args
    ----
    true_list : ground-truth task keys, e.g. ["1","2"] or ["flip_h","rotate"]
    pred_list : predicted kept task keys (order ignored here), e.g. ["1","2"]

    Returns
    -------
    dict with:
      - precision, recall, f1, jaccard
      - hallucination_rate = FP / max(1, |pred|)
      - omission_rate      = FN / max(1, |true|)
    """
    Tset, Pset = set(true_list), set(pred_list)
    tp = len(Tset & Pset)
    fp = len(Pset - Tset)
    fn = len(Tset - Pset)

    precision = tp / max(1, len(Pset))
    recall    = tp / max(1, len(Tset))
    f1        = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    jaccard   = tp / max(1, len(Tset | Pset))

    return dict(
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=jaccard,
        hallucination_rate=fp / max(1, len(Pset)),
        omission_rate=fn / max(1, len(Tset)),
    )

def prefix_exact(true_order: Sequence[str], pred_order: Sequence[str]) -> bool:
    """
    True if the predicted order exactly matches the full true order.
    (Useful for the 'accuracy_exact' you print in summaries.)
    """
    return list(pred_order) == list(true_order)


