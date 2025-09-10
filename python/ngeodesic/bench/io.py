from __future__ import annotations
import json, csv
from typing import Iterable, Mapping, Any, Sequence

def write_rows_csv(path: str, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        with open(path, "w", newline="") as f:
            pass
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_json(path: str, obj: Any, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)

def compare_json(a: Mapping[str, Any], b: Mapping[str, Any], keys: Sequence[str]) -> dict:
    """
    Return a small diff map for selected keys (useful for --compare in the CLI).
    """
    out = {}
    for k in keys:
        av, bv = a.get(k, None), b.get(k, None)
        out[k] = {"a": av, "b": bv, "eq": av == bv}
    return out
