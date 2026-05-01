from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import SegmentationConfig
from .utils import median_filter_1d, merge_adjacent_same, sec_to_hhmmss


def range_bins_m(cfg: SegmentationConfig, num_bins: int) -> np.ndarray:
    return np.linspace(cfg.range_min_m, cfg.range_max_m, num_bins, endpoint=False, dtype=np.float32)


def robust_mu_sigma(x: np.ndarray, cfg: SegmentationConfig) -> tuple[float, float]:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = max(1.4826 * mad, cfg.baseline_sigma_floor)
    return med, sigma


def standardize_across_windows(
    delta_by_room: Dict[str, np.ndarray],
    cfg: SegmentationConfig,
    prev_baseline: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    rooms = sorted(delta_by_room.keys())
    T = min(v.shape[0] for v in delta_by_room.values())
    D = min(v.shape[1] for v in delta_by_room.values())

    win = max(1, int(round(cfg.win_sec * cfg.frame_rate_hz)))
    hop = max(1, int(round(cfg.hop_sec * cfg.frame_rate_hz)))
    motion_min_frames = max(1, int(round(cfg.win_motion_min_sec * cfg.frame_rate_hz)))

    baseline: Dict[str, Dict[str, float]] = {}
    for room in rooms:
        x = delta_by_room[room][:T, :D]
        if prev_baseline is not None and room in prev_baseline:
            baseline[room] = {
                "mu": float(prev_baseline[room]["mu"]),
                "sigma": max(float(prev_baseline[room]["sigma"]), cfg.baseline_sigma_floor),
            }
        else:
            init_slice = x[: min(win, len(x))]
            mu0, sigma0 = robust_mu_sigma(init_slice.reshape(-1), cfg)
            baseline[room] = {"mu": mu0, "sigma": sigma0}

    z_sum = {room: np.zeros((T, D), dtype=np.float32) for room in rooms}
    w_sum = {room: np.zeros((T, D), dtype=np.float32) for room in rooms}
    win_motion: Dict[str, list[int]] = {room: [] for room in rooms}

    starts = list(range(0, max(1, T - win + 1), hop))
    if starts[-1] != max(0, T - win):
        starts.append(max(0, T - win))

    for s in starts:
        e = min(T, s + win)
        active_here: dict[str, bool] = {}
        z_window: dict[str, np.ndarray] = {}

        for room in rooms:
            xw = delta_by_room[room][s:e, :D]
            mu = baseline[room]["mu"]
            sigma = baseline[room]["sigma"]
            zw = (xw - mu) / (sigma + cfg.eps)
            z_window[room] = zw.astype(np.float32)
            sal = np.mean(np.maximum(zw, 0.0), axis=1)
            active = np.sum(sal >= cfg.win_motion_thr) >= motion_min_frames
            active_here[room] = bool(active)
            win_motion[room].append(1 if active else 0)

        for room in rooms:
            z_sum[room][s:e] += z_window[room]
            w_sum[room][s:e] += 1.0

        for room in rooms:
            xw = delta_by_room[room][s:e, :D]
            mu = baseline[room]["mu"]
            sigma = baseline[room]["sigma"]
            xclip = np.clip(xw, mu - cfg.baseline_clip_k * sigma, mu + cfg.baseline_clip_k * sigma)
            mu_w, sigma_w = robust_mu_sigma(xclip.reshape(-1), cfg)
            confirmed_empty = (not active_here[room]) and any(active_here[r] for r in rooms if r != room)
            alpha = cfg.alpha_fast if confirmed_empty else cfg.alpha_slow
            baseline[room]["mu"] = (1.0 - alpha) * mu + alpha * mu_w
            baseline[room]["sigma"] = max((1.0 - alpha) * sigma + alpha * sigma_w, cfg.baseline_sigma_floor)

    z_by_room = {
        room: (z_sum[room] / np.maximum(w_sum[room], 1.0)).astype(np.float32)
        for room in rooms
    }
    win_motion_np = {room: np.asarray(win_motion[room], dtype=np.int32) for room in rooms}
    return z_by_room, baseline, win_motion_np


def frame_features_from_z(z: np.ndarray, cfg: SegmentationConfig) -> Dict[str, np.ndarray]:
    rb = range_bins_m(cfg, z.shape[1]).reshape(1, -1)
    theta = np.percentile(z, cfg.salient_percentile, axis=1, keepdims=True)
    ze = np.where((z > theta) & (z > 0), z, 0.0).astype(np.float32)

    S = np.mean(ze, axis=1).astype(np.float32)
    wsum = np.sum(ze, axis=1, keepdims=True) + cfg.eps
    c = (np.sum(ze * rb, axis=1, keepdims=True) / wsum).reshape(-1).astype(np.float32)
    spread = np.sqrt(np.sum(ze * (rb - c.reshape(-1, 1)) ** 2, axis=1) / (wsum.reshape(-1) + cfg.eps)).astype(np.float32)

    S = median_filter_1d(S, cfg.feature_median_win_frames)
    c = median_filter_1d(c, cfg.feature_median_win_frames)
    spread = median_filter_1d(spread, cfg.feature_median_win_frames)
    return {"S": S, "c": c, "spread": spread, "ze": ze}


def assign_room_per_frame(
    feats_by_room: Dict[str, Dict[str, np.ndarray]],
    cfg: SegmentationConfig,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    rooms = sorted(feats_by_room.keys())
    T = min(len(feats_by_room[r]["S"]) for r in rooms)
    S = np.stack([feats_by_room[r]["S"][:T] for r in rooms], axis=1)

    order = np.argsort(S, axis=1)
    top1_idx = order[:, -1]
    top2_idx = order[:, -2] if S.shape[1] >= 2 else order[:, -1]
    top1 = S[np.arange(T), top1_idx]
    top2 = S[np.arange(T), top2_idx]
    margin = (top1 - top2) / (top1 + cfg.eps)

    labels: list[str] = []
    for t in range(T):
        if top1[t] >= cfg.tau_on and margin[t] >= cfg.tau_margin:
            labels.append(rooms[int(top1_idx[t])])
        else:
            labels.append("UNKNOWN")
    diag = {"top1": top1.astype(np.float32), "top2": top2.astype(np.float32), "margin": margin.astype(np.float32)}
    return labels, diag


def seg_stats(room: str, s: int, e: int, feats_by_room: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
    f = feats_by_room[room]
    return {
        "S_mean": float(np.mean(f["S"][s:e])) if e > s else 0.0,
        "c_mean": float(np.mean(f["c"][s:e])) if e > s else 0.0,
        "spread_mean": float(np.mean(f["spread"][s:e])) if e > s else 0.0,
    }


def bridge_short_unknown(
    segs: List[Tuple[str, int, int]],
    feats_by_room: Dict[str, Dict[str, np.ndarray]],
    cfg: SegmentationConfig,
) -> List[Tuple[str, int, int]]:
    max_len = int(round(cfg.short_unknown_max_sec * cfg.frame_rate_hz))
    out: List[Tuple[str, int, int]] = []
    i = 0
    while i < len(segs):
        lab, s, e = segs[i]
        if lab != "UNKNOWN" or (e - s) > max_len or i == 0 or i == len(segs) - 1:
            out.append((lab, s, e))
            i += 1
            continue

        prev_lab, ps, pe = out[-1]
        next_lab, ns, ne = segs[i + 1]
        if prev_lab not in feats_by_room or next_lab not in feats_by_room:
            out.append((lab, s, e))
            i += 1
            continue

        if prev_lab == next_lab:
            out[-1] = (prev_lab, ps, ne)
            i += 2
            continue

        u_prev = seg_stats(prev_lab, s, e, feats_by_room)
        p_prev = seg_stats(prev_lab, ps, pe, feats_by_room)
        u_next = seg_stats(next_lab, s, e, feats_by_room)
        n_next = seg_stats(next_lab, ns, ne, feats_by_room)

        c_diff_prev = abs(u_prev["c_mean"] - p_prev["c_mean"])
        c_diff_next = abs(u_next["c_mean"] - n_next["c_mean"])
        s_ratio_prev = u_prev["S_mean"] / (p_prev["S_mean"] + cfg.eps)
        s_ratio_next = u_next["S_mean"] / (n_next["S_mean"] + cfg.eps)

        ok_prev = (c_diff_prev <= cfg.unknown_max_c_diff_m) and (s_ratio_prev <= 1.0 + cfg.unknown_max_s_ratio)
        ok_next = (c_diff_next <= cfg.unknown_max_c_diff_m) and (s_ratio_next <= 1.0 + cfg.unknown_max_s_ratio)

        if ok_prev and not ok_next:
            out[-1] = (prev_lab, ps, e)
        elif ok_next and not ok_prev:
            segs[i + 1] = (next_lab, s, ne)
        elif ok_prev and ok_next:
            if c_diff_prev <= c_diff_next:
                out[-1] = (prev_lab, ps, e)
            else:
                segs[i + 1] = (next_lab, s, ne)
        else:
            out.append((lab, s, e))
        i += 1
    return merge_adjacent_same(out)


def breathing_ratio_from_interval(z_seg: np.ndarray, cfg: SegmentationConfig) -> Dict[str, float]:
    T, D = z_seg.shape
    min_len = int(round(cfg.breath_min_sec * cfg.frame_rate_hz))
    if T < max(min_len, 8):
        return {"peak_hz": 0.0, "peak_ratio": 0.0}

    strength = np.mean(np.maximum(z_seg, 0.0), axis=0)
    k = max(1, int(round(D * cfg.breath_top_pct)))
    idx = np.argpartition(strength, -k)[-k:]

    win = np.hanning(T).astype(np.float32)
    freqs = np.fft.rfftfreq(T, d=1.0 / cfg.frame_rate_hz)
    spec_sum = np.zeros_like(freqs, dtype=np.float64)
    for d in idx:
        x = z_seg[:, d].astype(np.float32)
        x = x - np.mean(x)
        X = np.fft.rfft(x * win)
        spec_sum += np.abs(X) ** 2

    mask = (freqs >= cfg.breath_band_hz[0]) & (freqs <= cfg.breath_band_hz[1])
    if not np.any(mask):
        return {"peak_hz": 0.0, "peak_ratio": 0.0}

    band = spec_sum[mask]
    peak_i = int(np.argmax(band))
    peak_val = float(band[peak_i])
    peak_hz = float(freqs[mask][peak_i])
    med = float(np.median(band)) + cfg.eps
    return {"peak_hz": peak_hz, "peak_ratio": peak_val / med}


def resolve_long_unknown(
    segs: List[Tuple[str, int, int]],
    z_by_room: Dict[str, np.ndarray],
    cfg: SegmentationConfig,
) -> Tuple[List[Tuple[str, int, int]], List[Dict]]:
    out: list[tuple[str, int, int]] = []
    debug: list[dict] = []

    for i, (lab, s, e) in enumerate(segs):
        if lab != "UNKNOWN":
            out.append((lab, s, e))
            continue

        best_room: str | None = None
        best_ratio = -1.0
        best_peak = 0.0
        ratios: dict[str, dict[str, float]] = {}

        neighbor_rooms: list[str] = []
        if i > 0 and segs[i - 1][0] in z_by_room:
            neighbor_rooms.append(segs[i - 1][0])
        if i < len(segs) - 1 and segs[i + 1][0] in z_by_room:
            neighbor_rooms.append(segs[i + 1][0])

        for room in sorted(set(neighbor_rooms)):
            br = breathing_ratio_from_interval(z_by_room[room][s:e], cfg)
            ratios[room] = br
            if br["peak_ratio"] > best_ratio:
                best_ratio = br["peak_ratio"]
                best_room = room
                best_peak = br["peak_hz"]

        if e - s > int(round(cfg.breath_max_sec * cfg.frame_rate_hz)) and best_room is not None and best_ratio < cfg.breath_ratio_thr:
            decision = "UNKNOWN"
        elif best_room is not None and best_ratio >= cfg.breath_ratio_thr / 2:
            decision = best_room
        else:
            decision = "UNKNOWN"
        out.append((decision, s, e))
        debug.append({
            "t0_frame": s,
            "t1_frame": e,
            "decision": decision,
            "best_room": best_room,
            "best_ratio": float(best_ratio),
            "best_peak_hz": float(best_peak),
            "ratios": {k: {"peak_hz": float(v["peak_hz"]), "peak_ratio": float(v["peak_ratio"])} for k, v in ratios.items()},
        })
    return merge_adjacent_same(out), debug


def absorb_short_segments(segs: List[Tuple[str, int, int]], cfg: SegmentationConfig) -> List[Tuple[str, int, int]]:
    min_len = int(round(cfg.min_segment_sec * cfg.frame_rate_hz))
    if len(segs) <= 1:
        return segs
    out: List[Tuple[str, int, int]] = []
    for i, (lab, s, e) in enumerate(segs):
        if (e - s) >= min_len:
            out.append((lab, s, e))
            continue
        if out:
            prev_lab, prev_s, _ = out[-1]
            out[-1] = (prev_lab, prev_s, e)
        elif i + 1 < len(segs):
            next_lab, _, next_e = segs[i + 1]
            segs[i + 1] = (next_lab, s, next_e)
        else:
            out.append((lab, s, e))
    return merge_adjacent_same(out)


def change_score(feats: Dict[str, np.ndarray], t: int, w: int, cfg: SegmentationConfig) -> float:
    SL = feats["S"][t - w:t]
    SR = feats["S"][t:t + w]
    cL = feats["c"][t - w:t]
    cR = feats["c"][t:t + w]
    spL = feats["spread"][t - w:t]
    spR = feats["spread"][t:t + w]

    def norm_diff(a: np.ndarray, b: np.ndarray) -> float:
        ma, mb = float(np.mean(a)), float(np.mean(b))
        sa, sb = float(np.std(a)), float(np.std(b))
        return abs(mb - ma) / (sa + sb + cfg.eps)

    return cfg.w_s * norm_diff(SL, SR) + cfg.w_c * norm_diff(cL, cR) + cfg.w_spread * norm_diff(spL, spR)


def refine_within_room_segments(
    segs: List[Tuple[str, int, int]],
    feats_by_room: Dict[str, Dict[str, np.ndarray]],
    cfg: SegmentationConfig,
) -> Tuple[List[Tuple[str, int, int]], Dict[str, np.ndarray]]:
    w = int(round(cfg.split_half_win_sec * cfg.frame_rate_hz))
    stride = max(1, int(round(cfg.split_stride_sec * cfg.frame_rate_hz)))
    min_len = int(round(cfg.split_min_len_sec * cfg.frame_rate_hz))
    cooldown = int(round(cfg.split_cooldown_sec * cfg.frame_rate_hz))
    peak_search = int(round(cfg.split_peak_search_sec * cfg.frame_rate_hz))

    score_store = {room: np.full(len(feats_by_room[room]["S"]), np.nan, dtype=np.float32) for room in feats_by_room}
    out: list[tuple[str, int, int]] = []

    for lab, s, e in segs:
        if lab in ("UNKNOWN", "OUTSIDE") or lab not in feats_by_room or (e - s) < max(2 * w + 1, 2 * min_len):
            out.append((lab, s, e))
            continue

        feats = feats_by_room[lab]
        candidates: list[int] = []
        last_cut = -10**9
        t = s + w
        t_end = e - w
        while t <= t_end:
            if (t - s) < min_len or (e - t) < min_len:
                t += stride
                continue
            sc = change_score(feats, t, w, cfg)
            score_store[lab][t] = sc
            if sc >= cfg.split_score_thr and (t - last_cut) >= cooldown:
                lo = max(s + w, t - peak_search)
                hi = min(e - w, t + peak_search)
                best_t = t
                best_sc = sc
                for tt in range(lo, hi + 1, max(1, stride // 2)):
                    sc2 = change_score(feats, tt, w, cfg)
                    if sc2 > best_sc:
                        best_sc = sc2
                        best_t = tt
                candidates.append(best_t)
                last_cut = best_t
                t = best_t + cooldown
            else:
                t += stride

        if not candidates:
            out.append((lab, s, e))
            continue

        cur = s
        for cp in sorted(set(candidates)):
            if cp - cur >= min_len and e - cp >= min_len:
                out.append((lab, cur, cp))
                cur = cp
        out.append((lab, cur, e))
    return out, score_store


def segment_summary(
    seg: Tuple[str, int, int],
    feats_by_room: Dict[str, Dict[str, np.ndarray]],
    start_sec: float,
    cfg: SegmentationConfig,
) -> Dict:
    lab, s, e = seg
    t0_sec = start_sec + s / cfg.frame_rate_hz
    t1_sec = start_sec + e / cfg.frame_rate_hz
    duration_s = float((e - s) / cfg.frame_rate_hz)

    base = {
        "label": lab,
        "room": lab,
        "start_time": sec_to_hhmmss(t0_sec),
        "duration_s": duration_s,
        "t0_frame": s,
        "t1_frame": e,
        "t0_sec": float(t0_sec),
        "t1_sec": float(t1_sec),
    }
    if lab not in feats_by_room:
        return base | {"Salience_mean": 0.0, "Active_ratio": 0.0, "Centroid_mean": 0.0, "Spread_mean": 0.0}

    f = feats_by_room[lab]
    S = f["S"][s:e]
    c = f["c"][s:e]
    sp = f["spread"][s:e]
    activity_thr = float(np.percentile(S, 99) * 0.5) if len(S) else 0.0
    return base | {
        "Salience_mean": float(np.mean(S)) if len(S) else 0.0,
        "Active_ratio": float(np.mean(S > activity_thr)) if len(S) else 0.0,
        "Centroid_mean": float(np.mean(c)) if len(c) else 0.0,
        "Spread_mean": float(np.mean(sp)) if len(sp) else 0.0,
    }
