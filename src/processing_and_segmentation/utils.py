from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size == 0:
        return x.astype(np.float32, copy=True)
    half = int(k) // 2
    y = np.empty_like(x, dtype=np.float32)
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        y[i] = float(np.median(x[lo:hi]))
    return y


def sec_to_hhmmss(sec_since_midnight: float) -> str:
    sec_since_midnight = int(round(sec_since_midnight)) % 86400
    h = sec_since_midnight // 3600
    m = (sec_since_midnight % 3600) // 60
    s = sec_since_midnight % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_start_time_from_label_json(label_json: Path) -> tuple[datetime, float]:
    data = load_json(label_json)
    acts = data.get("activities", [])
    if not acts:
        raise ValueError(f"{label_json} has no activities")
    start_str = acts[0]["start"]
    start_dt = datetime.strptime(start_str, "%H:%M")
    midnight = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    start_sec = float((start_dt - midnight).total_seconds())
    return start_dt, start_sec


def rle(labels: List[str], min_frames: int = 20) -> List[Tuple[str, int, int]]:
    """Run-length encode frame labels; very short runs become UNKNOWN."""
    if not labels:
        return []
    out: list[tuple[str, int, int]] = []
    cur = labels[0]
    s = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            out.append(("UNKNOWN" if i - s < min_frames else cur, s, i))
            cur = labels[i]
            s = i
    out.append(("UNKNOWN" if len(labels) - s < min_frames else cur, s, len(labels)))
    return out


def merge_adjacent_same(segs: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    if not segs:
        return []
    out = [segs[0]]
    for lab, s, e in segs[1:]:
        prev_lab, prev_s, prev_e = out[-1]
        if lab == prev_lab:
            out[-1] = (prev_lab, prev_s, e)
        else:
            out.append((lab, s, e))
    return out
