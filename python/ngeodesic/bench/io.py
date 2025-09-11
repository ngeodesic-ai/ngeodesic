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
"""

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
