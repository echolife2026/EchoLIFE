from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["day"])
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(records: List[Dict[str, Any]], path: Path, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def hhmmss_to_hour(t: str) -> float:
    hh, mm, ss = map(int, str(t).split(":"))
    return hh + mm / 60.0 + ss / 3600.0


def list_day_dirs(root: Path) -> List[Path]:
    out = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    out.sort(key=lambda p: int(p.name))
    return out


def load_optional_layout(layout_json_path: Path, room_types: List[str]) -> Dict[str, Any]:
    if layout_json_path.exists():
        return load_json(layout_json_path)
    return {"rooms": [{"room_type": r, "room_id": r} for r in room_types]}


def load_optional_profile_text(profile_path: Path) -> str:
    if profile_path.exists():
        return profile_path.read_text(encoding="utf-8").strip()
    return ""


def load_day_inputs(day_dir: Path, segment_output_dir: str, summaries_name: str, day_meta_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    seg_dir = day_dir / segment_output_dir
    summaries_path = seg_dir / summaries_name
    day_meta_path = seg_dir / day_meta_name
    if not summaries_path.exists():
        raise FileNotFoundError(f"Missing {summaries_path}")
    if not day_meta_path.exists():
        raise FileNotFoundError(f"Missing {day_meta_path}")

    raw = load_json(summaries_path)
    if isinstance(raw, dict):
        summaries = raw.get("segment_summaries", raw.get("summaries", raw.get("segments", [])))
    else:
        summaries = raw
    day_meta = load_json(day_meta_path)
    return summaries, day_meta


def build_exec_segment(summary: Dict[str, Any]) -> Dict[str, Any]:
    start_time = summary.get("start_time") or summary.get("Start_time") or summary.get("t0_str")
    room = summary.get("room") or summary.get("Room") or summary.get("dominant_room_type")
    sal = summary.get("Salience_mean", summary.get("S_mean", 0.0))
    act = summary.get("Active_ratio", summary.get("active_ratio", 0.0))
    cen = summary.get("Centroid_mean", summary.get("c_mean", 0.0))
    spr = summary.get("Spread_mean", summary.get("spread_mean", 0.0))
    return {
        "start_time": start_time,
        "hour": float(hhmmss_to_hour(start_time)),
        "duration_s": float(summary["duration_s"]),
        "dominant_room_type": room,
        "Salience_mean": float(sal),
        "Active_ratio": float(act),
        "Centroid_mean": float(cen),
        "Spread_mean": float(spr),
        "t0_sec": float(summary.get("t0_sec", summary.get("t0", 0.0))),
        "t1_sec": float(summary.get("t1_sec", summary.get("t1", 0.0))),
    }
