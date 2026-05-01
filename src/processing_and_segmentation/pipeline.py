from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import SegmentationConfig
from .io import find_day_dirs, load_delta_day
from .processing import (
    absorb_short_segments,
    assign_room_per_frame,
    bridge_short_unknown,
    frame_features_from_z,
    refine_within_room_segments,
    resolve_long_unknown,
    segment_summary,
    standardize_across_windows,
)
from .utils import merge_adjacent_same, rle, save_json


def save_intermediates(
    out_dir: Path,
    delta_by_room: Dict[str, np.ndarray],
    z_by_room: Dict[str, np.ndarray],
    feats_by_room: Dict[str, Dict[str, np.ndarray]],
    assign_diag: Dict[str, np.ndarray],
    win_motion: Dict[str, np.ndarray],
    score_store: Dict[str, np.ndarray],
) -> None:
    np.savez_compressed(out_dir / "delta_by_room.npz", **delta_by_room)
    np.savez_compressed(out_dir / "z_by_room.npz", **z_by_room)
    feature_payload = {}
    for room, feats in feats_by_room.items():
        feature_payload[f"{room}_S"] = feats["S"]
        feature_payload[f"{room}_c"] = feats["c"]
        feature_payload[f"{room}_spread"] = feats["spread"]
    np.savez_compressed(out_dir / "features_S_c_spread.npz", **feature_payload)
    np.savez_compressed(out_dir / "assignment_diag.npz", **assign_diag)
    np.savez_compressed(out_dir / "window_motion_flags.npz", **win_motion)
    np.savez_compressed(out_dir / "split_scores.npz", **score_store)


def process_one_day(day: dict, cfg: SegmentationConfig, prev_baseline: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    day_dir: Path = day["day_dir"]
    out_dir = day_dir / cfg.output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    delta_by_room = day["delta_by_room"]
    z_by_room, new_baseline, win_motion = standardize_across_windows(delta_by_room, cfg, prev_baseline)
    feats_by_room = {room: frame_features_from_z(z_by_room[room], cfg) for room in day["rooms"]}

    frame_labels, assign_diag = assign_room_per_frame(feats_by_room, cfg)
    segs0 = merge_adjacent_same(rle(frame_labels, cfg.rle_min_frames))
    segs1 = bridge_short_unknown(segs0, feats_by_room, cfg)
    segs2, breathing_debug = resolve_long_unknown(segs1, z_by_room, cfg)
    segs2 = absorb_short_segments(segs2, cfg)
    segs3, score_store = refine_within_room_segments(segs2, feats_by_room, cfg)

    summaries = [segment_summary(seg, feats_by_room, day["start_sec"], cfg) for seg in segs3]

    save_json({
        "day": day_dir.name,
        "start_time": day["start_dt"].strftime("%Y-%m-%d %H:%M"),
        "delta_npz": str(day["npz_path"]),
        "rooms": day["rooms"],
        "baseline_in": prev_baseline,
        "baseline_out": new_baseline,
    }, out_dir / "day_meta.json")
    save_json({"frame_labels": frame_labels}, out_dir / "frame_labels.json")
    save_json({
        "segments_before_bridge": [{"label": a, "t0_frame": b, "t1_frame": c} for a, b, c in segs0],
        "segments_after_bridge": [{"label": a, "t0_frame": b, "t1_frame": c} for a, b, c in segs1],
        "segments_after_breathing": [{"label": a, "t0_frame": b, "t1_frame": c} for a, b, c in segs2],
        "segments_final": [{"label": a, "t0_frame": b, "t1_frame": c} for a, b, c in segs3],
    }, out_dir / "segments.json")
    save_json(breathing_debug, out_dir / "breathing_debug.json")
    save_json(summaries, out_dir / "segment_summaries.json")

    if cfg.save_intermediates:
        save_intermediates(out_dir, delta_by_room, z_by_room, feats_by_room, assign_diag, win_motion, score_store)

    if cfg.save_plots:
        from .plotting import plot_feature_traces, plot_room_assignment, plot_z_heatmaps

        plot_z_heatmaps(z_by_room, out_dir / "z_heatmaps.png", f"day {day_dir.name} standardized Z")
        plot_feature_traces(feats_by_room, out_dir / "feature_traces.png", f"day {day_dir.name} frame features")
        plot_room_assignment(segs0, out_dir / "room_timeline_after_room_assignment.png", f"day {day_dir.name} after room assignment", cfg)
        plot_room_assignment(segs1, out_dir / "room_timeline_short_UNKNOWN.png", f"day {day_dir.name} after short UNKNOWN bridge", cfg)
        plot_room_assignment(segs2, out_dir / "room_timeline_after_breathing.png", f"day {day_dir.name} after breathing resolution", cfg)
        plot_room_assignment(segs3, out_dir / "room_timeline_final.png", f"day {day_dir.name} final segments", cfg)

    print(f"[DONE] day {day_dir.name} -> {out_dir}")
    return new_baseline


def run_pipeline(cfg: SegmentationConfig) -> None:
    root_parent = cfg.root_parent.expanduser().resolve()
    if not root_parent.exists():
        raise FileNotFoundError(f"ROOT_PARENT does not exist: {root_parent}")

    day_dirs = find_day_dirs(root_parent)
    if not day_dirs:
        raise RuntimeError(f"No numeric day folders found under {root_parent}")

    baseline_state: Optional[Dict[str, Dict[str, float]]] = None
    processed = 0
    for day_dir in day_dirs:
        delta_path = day_dir / cfg.input_deltae_npz_name
        label_path = day_dir / cfg.input_json_name
        if not delta_path.exists() or not label_path.exists():
            print(f"[SKIP] {day_dir} missing {cfg.input_deltae_npz_name} or {cfg.input_json_name}")
            continue
        day = load_delta_day(day_dir, cfg)
        baseline_state = process_one_day(day, cfg, baseline_state)
        processed += 1

    if baseline_state is not None:
        save_json(baseline_state, root_parent / "baseline_state_last.json")
        print(f"[SAVED] final baseline -> {root_parent / 'baseline_state_last.json'}")
    print(f"[DONE] processed {processed}/{len(day_dirs)} day folders")
