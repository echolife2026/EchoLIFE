from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from .config import SegmentationConfig
from .utils import parse_start_time_from_label_json


def find_day_dirs(root_parent: Path) -> list[Path]:
    day_dirs = [p for p in root_parent.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(day_dirs, key=lambda p: int(p.name))


def load_delta_day(day_dir: Path, cfg: SegmentationConfig) -> dict:
    npz_path = day_dir / cfg.input_deltae_npz_name
    json_path = day_dir / cfg.input_json_name
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"Missing {npz_path} or {json_path}")

    delta_by_room: Dict[str, np.ndarray] = {}
    with np.load(npz_path) as npz:
        for room in cfg.rooms:
            if room not in npz.files:
                continue
            arr = np.asarray(npz[room])
            if arr.ndim != 2:
                raise ValueError(f"{npz_path}:{room} must be 2D [T, D], got shape={arr.shape}")
            delta_by_room[room] = arr.astype(np.float32, copy=False)

    if not delta_by_room:
        raise ValueError(f"No configured room keys {cfg.rooms} found in {npz_path}")

    start_dt, start_sec = parse_start_time_from_label_json(json_path)
    return {
        "day_dir": day_dir,
        "npz_path": npz_path,
        "json_path": json_path,
        "rooms": sorted(delta_by_room.keys()),
        "delta_by_room": delta_by_room,
        "start_dt": start_dt,
        "start_sec": start_sec,
    }
