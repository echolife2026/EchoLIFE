from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class SegmentationConfig:
    # I/O
    root_parent: Path
    output_dir_name: str = "segmentation_out"
    input_deltae_npz_name: str = "deltae.npz"
    input_json_name: str = "label.json"
    rooms: Tuple[str, ...] = ("kitchen", "living", "bedroom", "bathroom", "study", "dining")

    # Signal geometry
    num_bins: int = 65
    frame_seconds: float = 0.1
    frame_rate_hz: int = 10
    range_min_m: float = 0.2
    range_max_m: float = 3.0

    # Adaptive standardization
    win_sec: float = 120.0
    hop_sec: float = 60.0
    baseline_clip_k: float = 4.0
    alpha_fast: float = 0.20
    alpha_slow: float = 0.02
    win_motion_thr: float = 3.0
    win_motion_min_sec: float = 2.0
    baseline_sigma_floor: float = 1e-3
    eps: float = 1e-5

    # Frame features
    salient_percentile: float = 85.0
    feature_median_win_frames: int = 9

    # Room assignment
    tau_on: float = 1.0
    tau_margin: float = 0.8
    rle_min_frames: int = 20

    # UNKNOWN resolution
    short_unknown_max_sec: float = 300.0
    min_segment_sec: float = 6.0
    unknown_max_c_diff_m: float = 0.35
    unknown_max_s_ratio: float = 0.50
    breath_band_hz: Tuple[float, float] = (0.15, 0.42)
    breath_top_pct: float = 0.15
    breath_ratio_thr: float = 5.0
    breath_min_sec: float = 60.0
    breath_max_sec: float = 3600.0

    # Within-room split
    split_min_len_sec: float = 90.0
    split_half_win_sec: float = 90.0
    split_stride_sec: float = 5.0
    split_score_thr: float = 2.5
    split_cooldown_sec: float = 60.0
    split_peak_search_sec: float = 15.0
    w_s: float = 1.2
    w_c: float = 2.0
    w_spread: float = 1.0

    # Outputs
    save_plots: bool = False
    save_intermediates: bool = True
