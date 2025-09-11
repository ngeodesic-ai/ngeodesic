# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, json, os
from typing import Any, Dict, Iterable, List, Optional

__all__ = ["write_rows_csv", "write_json"]

def _ensure_dir(path: str) -> None:
    """Create parent directory for a file path, if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def write_rows_csv(
    path: str,
    rows: List[Dict[str, Any]],
    *,
    fieldnames: Optional[Iterable[str]] = None,
    newline: str = ""
) -> None:
    """
    Write a list of dict rows to CSV. If `fieldnames` is None, uses keys of the first row.
    Skips writing if `rows` is empty (matches prior behavior).
    """
    if not rows:
        return
    _ensure_dir(path)
    header = list(fieldnames) if fieldnames else list(rows[0].keys())
    with open(path, "w", newline=newline, encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

def write_json(
    path: str,
    obj: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> None:
    """
    JSON writer with safe directory creation and UTF-8 by default.
    """
    _ensure_dir(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
        f.write("\n")
    os.replace(tmp, path)
