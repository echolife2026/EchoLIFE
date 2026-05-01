from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .config import RecognitionConfig
from .constants import ensure_activity_subset
from .io_utils import load_json, save_csv, save_json, list_day_dirs

UNKNOWN_LABEL = "Unknown/Other"
MISSING_LABEL = "__MISSING__"

GT_TO_CANONICAL = {
    "Unknown/Other": "Unknown/Other",
    "Entertainment/Relax": "Entertainment/Relax",
    "Exercise": "Exercise",
    "Bathing": "Bathing",
    "Washing/Brushing": "Washing/Brushing",
    "Rest/Sleeping": "Rest/Sleeping",
    "Toileting": "Toileting",
    "Cooking": "Cooking",
    "Eating": "Eating",
    "Working/Studying": "Working/Studying",
    "Cleaning": "Cleaning",
}


@dataclass
class EvalDayData:
    day: str
    gt_frame_labels: List[str]
    pred_top1_labels: List[str]
    pred_top3_labels: List[List[str]]
    gt_collapsed_from_segments: List[str]
    pred_collapsed_from_segments: List[str]
    gt_segments: List[Dict[str, Any]]
    pred_segments: List[Dict[str, Any]]


def normalize_gt_label(x: str) -> str:
    return GT_TO_CANONICAL.get(str(x), str(x))


def parse_hhmm_or_datetime_to_sec(s: str) -> float:
    s = str(s).strip()
    if " " in s:
        s = s.split(" ")[-1]
    parts = s.split(":")
    if len(parts) == 2:
        hh, mm = map(int, parts)
        ss = 0
    elif len(parts) == 3:
        hh, mm, ss = map(int, parts)
    else:
        raise ValueError(f"Unsupported time format: {s}")
    return float(hh * 3600 + mm * 60 + ss)


def safe_frame_end(item: Dict[str, Any]) -> int:
    if "frame_end_exclusive" in item:
        return int(item["frame_end_exclusive"])
    if "frame_end" in item:
        return int(item["frame_end"])
    if "duration_frames" in item and "frame_start" in item:
        return int(item["frame_start"]) + int(item["duration_frames"])
    raise ValueError(f"Cannot infer frame end from item: {item}")


def collapse_sequence(seq: List[str], ignore_unknown: bool = False) -> List[str]:
    out: List[str] = []
    prev = None
    for lab in seq:
        if ignore_unknown and lab == UNKNOWN_LABEL:
            continue
        if lab != prev:
            out.append(lab)
            prev = lab
    return out


def levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def ctc_sequence_metrics(pred_seq: List[str], gt_seq: List[str], ignore_unknown: bool = False) -> Dict[str, Any]:
    p = collapse_sequence(pred_seq, ignore_unknown=ignore_unknown)
    g = collapse_sequence(gt_seq, ignore_unknown=ignore_unknown)
    dist = levenshtein(p, g)
    denom = max(len(p), len(g), 1)
    acc = 1.0 - dist / denom
    return {
        "pred_sequence": p,
        "gt_sequence": g,
        "edit_distance": int(dist),
        "denominator": int(denom),
        "ctc_sequence_accuracy": float(acc),
        "exact_match": bool(p == g),
    }


def load_ground_truth(day_dir: Path, gt_name: str, allowed_labels: set[str]) -> Tuple[List[Dict[str, Any]], int]:
    data = load_json(day_dir / gt_name)
    items = data["activities"] if isinstance(data, dict) and "activities" in data else data
    segs: List[Dict[str, Any]] = []
    total_frames = 0
    for item in items:
        s = int(item["frame_start"])
        e = safe_frame_end(item)
        if e <= s:
            continue
        lab = normalize_gt_label(item["label"])
        if lab not in allowed_labels:
            lab = UNKNOWN_LABEL
        segs.append({"label": lab, "frame_start": s, "frame_end_exclusive": e})
        total_frames = max(total_frames, e)
    segs.sort(key=lambda x: x["frame_start"])
    return segs, total_frames


def load_predictions(pred_path: Path, frame_rate_hz: float, allowed_labels: set[str]) -> Tuple[List[Dict[str, Any]], float]:
    data = load_json(pred_path)
    results = data["results"]
    day_start_sec = parse_hhmm_or_datetime_to_sec(data["start_time"])
    pred_segs: List[Dict[str, Any]] = []
    for r in results:
        seg = r["segment"]
        t0_sec = float(seg["t0_sec"])
        t1_sec = float(seg["t1_sec"])
        s = int(round((t0_sec - day_start_sec) * frame_rate_hz))
        e = int(round((t1_sec - day_start_sec) * frame_rate_hz))
        if e <= s:
            continue
        top3 = r.get("final_top3", [])
        if not top3:
            final_label = normalize_gt_label(r.get("final_label", UNKNOWN_LABEL))
            if final_label not in allowed_labels:
                final_label = UNKNOWN_LABEL
            top3_labels = [final_label]
        else:
            top3_labels = [normalize_gt_label(x["label"]) for x in top3]
            top3_labels = [lab if lab in allowed_labels else UNKNOWN_LABEL for lab in top3_labels]
            final_label = top3_labels[0]
        pred_segs.append({
            "label": final_label,
            "top3_labels": top3_labels[:3],
            "frame_start": s,
            "frame_end_exclusive": e,
        })
    pred_segs.sort(key=lambda x: x["frame_start"])
    return pred_segs, day_start_sec


def rasterize_ground_truth(gt_segs: List[Dict[str, Any]], total_frames: int) -> List[str]:
    arr = [MISSING_LABEL] * total_frames
    for seg in gt_segs:
        s = max(0, int(seg["frame_start"]))
        e = min(total_frames, int(seg["frame_end_exclusive"]))
        for i in range(s, e):
            arr[i] = seg["label"]
    return arr


def rasterize_predictions(pred_segs: List[Dict[str, Any]], total_frames: int) -> Tuple[List[str], List[List[str]]]:
    top1 = [MISSING_LABEL] * total_frames
    top3 = [[] for _ in range(total_frames)]
    for seg in pred_segs:
        s = max(0, int(seg["frame_start"]))
        e = min(total_frames, int(seg["frame_end_exclusive"]))
        labs = list(seg.get("top3_labels", [])) or [seg["label"]]
        labs = labs[:3]
        for i in range(s, e):
            top1[i] = labs[0]
            top3[i] = labs
    return top1, top3


def merge_adjacent_same_label_segments(segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not segs:
        return []
    segs_sorted = sorted(segs, key=lambda x: x["frame_start"])
    out: List[Dict[str, Any]] = [dict(segs_sorted[0])]
    for seg in segs_sorted[1:]:
        cur = dict(seg)
        prev = out[-1]
        if cur["label"] == prev["label"]:
            prev["frame_end_exclusive"] = max(int(prev["frame_end_exclusive"]), int(cur["frame_end_exclusive"]))
            if "top3_labels" in prev or "top3_labels" in cur:
                prev["top3_labels"] = list(prev.get("top3_labels", cur.get("top3_labels", [prev["label"]])))
        else:
            out.append(cur)
    return out


def build_day_data(day_dir: Path, pred_path: Path, cfg: RecognitionConfig) -> EvalDayData:
    allowed_labels = set(ensure_activity_subset(cfg.activities))
    gt_segs, total_frames = load_ground_truth(day_dir, cfg.ground_truth_name, allowed_labels)
    pred_segs, _ = load_predictions(pred_path, cfg.frame_rate_hz, allowed_labels)
    gt_frame_labels = rasterize_ground_truth(gt_segs, total_frames)
    pred_top1_labels, pred_top3_labels = rasterize_predictions(pred_segs, total_frames)
    gt_collapsed = [seg["label"] for seg in gt_segs]
    pred_collapsed = [seg["label"] for seg in pred_segs]
    return EvalDayData(day=day_dir.name, gt_frame_labels=gt_frame_labels, pred_top1_labels=pred_top1_labels, pred_top3_labels=pred_top3_labels, gt_collapsed_from_segments=gt_collapsed, pred_collapsed_from_segments=pred_collapsed, gt_segments=gt_segs, pred_segments=pred_segs)


def frame_metrics(gt: List[str], pred_top1: List[str], pred_top3: List[List[str]]) -> Dict[str, Any]:
    if not (len(gt) == len(pred_top1) == len(pred_top3)):
        raise ValueError("Frame arrays must have the same length")
    valid_indices = [i for i, lab in enumerate(gt) if lab != MISSING_LABEL]
    if not valid_indices:
        return {"num_frames": 0, "top1_accuracy": 0.0, "top3_accuracy": 0.0, "macro_recall": 0.0, "weighted_recall": 0.0, "micro_recall": 0.0, "per_class_recall": {}, "labels_present": []}
    correct_top1 = 0
    correct_top3 = 0
    support: Dict[str, int] = {}
    tp: Dict[str, int] = {}
    for i in valid_indices:
        g = gt[i]
        p1 = pred_top1[i]
        p3 = pred_top3[i]
        support[g] = support.get(g, 0) + 1
        if p1 == g:
            correct_top1 += 1
            tp[g] = tp.get(g, 0) + 1
        if g in p3:
            correct_top3 += 1
    per_class_recall: Dict[str, float] = {}
    recalls = []
    weighted_sum = 0.0
    total_support = 0
    for lab in sorted(support.keys()):
        rec = tp.get(lab, 0) / max(support[lab], 1)
        per_class_recall[lab] = float(rec)
        recalls.append(rec)
        weighted_sum += rec * support[lab]
        total_support += support[lab]
    top1_acc = correct_top1 / len(valid_indices)
    top3_acc = correct_top3 / len(valid_indices)
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    weighted_recall = weighted_sum / max(total_support, 1)
    micro_recall = top1_acc
    return {"num_frames": int(len(valid_indices)), "top1_accuracy": float(top1_acc), "top3_accuracy": float(top3_acc), "macro_recall": float(macro_recall), "weighted_recall": float(weighted_recall), "micro_recall": float(micro_recall), "per_class_recall": per_class_recall, "labels_present": sorted(support.keys())}


def pooled_per_class_metrics(gt: List[str], pred_top1: List[str]) -> Dict[str, Any]:
    valid = [(g, p) for g, p in zip(gt, pred_top1) if g != MISSING_LABEL]
    n = len(valid)
    labels_present = sorted(set(g for g, _ in valid))
    per_class: Dict[str, Any] = {}
    macro_precisions: List[float] = []
    macro_recalls: List[float] = []
    macro_f1s: List[float] = []
    macro_accs: List[float] = []
    for lab in labels_present:
        tp = sum(1 for g, p in valid if g == lab and p == lab)
        fp = sum(1 for g, p in valid if g != lab and p == lab)
        fn = sum(1 for g, p in valid if g == lab and p != lab)
        tn = n - tp - fp - fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        one_vs_rest_accuracy = (tp + tn) / max(n, 1)
        support = tp + fn
        per_class[lab] = {"support": int(support), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "precision": float(precision), "recall": float(recall), "f1": float(f1), "one_vs_rest_accuracy": float(one_vs_rest_accuracy)}
        macro_precisions.append(precision)
        macro_recalls.append(recall)
        macro_f1s.append(f1)
        macro_accs.append(one_vs_rest_accuracy)
    return {"num_frames": int(n), "labels_present_in_gt": labels_present, "macro_precision": float(np.mean(macro_precisions)) if macro_precisions else 0.0, "macro_recall": float(np.mean(macro_recalls)) if macro_recalls else 0.0, "macro_f1": float(np.mean(macro_f1s)) if macro_f1s else 0.0, "macro_one_vs_rest_accuracy": float(np.mean(macro_accs)) if macro_accs else 0.0, "per_class": per_class}


def segment_boundaries_from_segments(segs: List[Dict[str, Any]]) -> List[int]:
    segs_sorted = sorted(segs, key=lambda x: x["frame_start"])
    return [int(seg["frame_start"]) for seg in segs_sorted[1:]]


def boundary_metrics(gt_segs: List[Dict[str, Any]], pred_segs: List[Dict[str, Any]], tol_frames: int, tol_sec: float) -> Dict[str, Any]:
    gt_bounds = segment_boundaries_from_segments(gt_segs)
    pred_bounds = segment_boundaries_from_segments(pred_segs)
    i = j = matches = 0
    matched_pairs: List[Tuple[int, int]] = []
    while i < len(gt_bounds) and j < len(pred_bounds):
        g = gt_bounds[i]
        p = pred_bounds[j]
        if abs(p - g) <= tol_frames:
            matches += 1
            matched_pairs.append((g, p))
            i += 1
            j += 1
        elif p < g - tol_frames:
            j += 1
        else:
            i += 1
    n_gt = len(gt_bounds)
    n_pred = len(pred_bounds)
    precision = 1.0 if n_pred == 0 and n_gt == 0 else (0.0 if n_pred == 0 else matches / n_pred)
    recall = 1.0 if n_gt == 0 else matches / n_gt
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"boundary_tolerance_sec": float(tol_sec), "boundary_tolerance_frames": int(tol_frames), "num_gt_boundaries": int(n_gt), "num_pred_boundaries": int(n_pred), "matched_boundaries": int(matches), "segmentation_accuracy": float(precision), "segmentation_precision": float(precision), "segmentation_recall": float(recall), "segmentation_f1": float(f1), "matched_pairs_preview": matched_pairs[:20]}


def summarize_day(day_data: EvalDayData, cfg: RecognitionConfig) -> Dict[str, Any]:
    fm = frame_metrics(day_data.gt_frame_labels, day_data.pred_top1_labels, day_data.pred_top3_labels)
    ctc_keep = ctc_sequence_metrics(day_data.pred_collapsed_from_segments, day_data.gt_collapsed_from_segments, ignore_unknown=False)
    ctc_ignore = ctc_sequence_metrics(day_data.pred_collapsed_from_segments, day_data.gt_collapsed_from_segments, ignore_unknown=True)
    tol_frames = int(round(cfg.boundary_tol_sec * cfg.frame_rate_hz))
    bm = boundary_metrics(day_data.gt_segments, day_data.pred_segments, tol_frames, cfg.boundary_tol_sec)
    gt_merged = merge_adjacent_same_label_segments(day_data.gt_segments)
    pred_merged = merge_adjacent_same_label_segments(day_data.pred_segments)
    bm_merged = boundary_metrics(gt_merged, pred_merged, tol_frames, cfg.boundary_tol_sec)
    return {"day": day_data.day, **fm, **bm, "segmentation_after_same_label_merge": {**bm_merged, "num_gt_segments": len(gt_merged), "num_pred_segments": len(pred_merged)}, "num_gt_segments": len(day_data.gt_segments), "num_pred_segments": len(day_data.pred_segments), "num_gt_segments_after_same_label_merge": len(gt_merged), "num_pred_segments_after_same_label_merge": len(pred_merged), "ctc_keep_unknown": {"ctc_sequence_accuracy": ctc_keep["ctc_sequence_accuracy"], "ctc_edit_distance": ctc_keep["edit_distance"], "ctc_denominator": ctc_keep["denominator"], "ctc_exact_match": ctc_keep["exact_match"], "gt_sequence": ctc_keep["gt_sequence"], "pred_sequence": ctc_keep["pred_sequence"]}, "ctc_ignore_unknown": {"ctc_sequence_accuracy": ctc_ignore["ctc_sequence_accuracy"], "ctc_edit_distance": ctc_ignore["edit_distance"], "ctc_denominator": ctc_ignore["denominator"], "ctc_exact_match": ctc_ignore["exact_match"], "gt_sequence": ctc_ignore["gt_sequence"], "pred_sequence": ctc_ignore["pred_sequence"]}}


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize_overall(days: List[EvalDayData], daily: List[Dict[str, Any]]) -> Dict[str, Any]:
    gt_all: List[str] = []
    pred1_all: List[str] = []
    pred3_all: List[List[str]] = []
    for d in days:
        gt_all.extend(d.gt_frame_labels)
        pred1_all.extend(d.pred_top1_labels)
        pred3_all.extend(d.pred_top3_labels)
    pooled_all = frame_metrics(gt_all, pred1_all, pred3_all)
    pooled_per_class = pooled_per_class_metrics(gt_all, pred1_all)
    if not daily:
        return {"num_days": 0, "pooled_all_days": pooled_all, "pooled_all_days_per_class": pooled_per_class, "mean_over_days": {}, "weighted_over_days": {}}
    mean_over_days = {
        "num_days": len(daily),
        "top1_accuracy": _mean([d["top1_accuracy"] for d in daily]),
        "top3_accuracy": _mean([d["top3_accuracy"] for d in daily]),
        "macro_recall": _mean([d["macro_recall"] for d in daily]),
        "weighted_recall": _mean([d["weighted_recall"] for d in daily]),
        "micro_recall": _mean([d["micro_recall"] for d in daily]),
        "segmentation_accuracy": _mean([d["segmentation_accuracy"] for d in daily]),
        "segmentation_precision": _mean([d["segmentation_precision"] for d in daily]),
        "segmentation_recall": _mean([d["segmentation_recall"] for d in daily]),
        "segmentation_f1": _mean([d["segmentation_f1"] for d in daily]),
        "segmentation_after_same_label_merge": {
            "segmentation_accuracy": _mean([d["segmentation_after_same_label_merge"]["segmentation_accuracy"] for d in daily]),
            "segmentation_precision": _mean([d["segmentation_after_same_label_merge"]["segmentation_precision"] for d in daily]),
            "segmentation_recall": _mean([d["segmentation_after_same_label_merge"]["segmentation_recall"] for d in daily]),
            "segmentation_f1": _mean([d["segmentation_after_same_label_merge"]["segmentation_f1"] for d in daily]),
        },
        "ctc_keep_unknown": {"ctc_sequence_accuracy": _mean([d["ctc_keep_unknown"]["ctc_sequence_accuracy"] for d in daily])},
        "ctc_ignore_unknown": {"ctc_sequence_accuracy": _mean([d["ctc_ignore_unknown"]["ctc_sequence_accuracy"] for d in daily])},
    }
    total_frames = sum(d["num_frames"] for d in daily) or 1
    def wavg(key: str) -> float:
        return float(sum(d[key] * d["num_frames"] for d in daily) / total_frames)
    total_pred_boundaries = sum(d["num_pred_boundaries"] for d in daily)
    total_gt_boundaries = sum(d["num_gt_boundaries"] for d in daily)
    total_matches = sum(d["matched_boundaries"] for d in daily)
    seg_precision = 1.0 if total_pred_boundaries == 0 and total_gt_boundaries == 0 else (0.0 if total_pred_boundaries == 0 else total_matches / total_pred_boundaries)
    seg_recall = 1.0 if total_gt_boundaries == 0 else total_matches / total_gt_boundaries
    seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall) if seg_precision + seg_recall > 0 else 0.0
    total_pred_boundaries_merged = sum(d["segmentation_after_same_label_merge"]["num_pred_boundaries"] for d in daily)
    total_gt_boundaries_merged = sum(d["segmentation_after_same_label_merge"]["num_gt_boundaries"] for d in daily)
    total_matches_merged = sum(d["segmentation_after_same_label_merge"]["matched_boundaries"] for d in daily)
    seg_precision_m = 1.0 if total_pred_boundaries_merged == 0 and total_gt_boundaries_merged == 0 else (0.0 if total_pred_boundaries_merged == 0 else total_matches_merged / total_pred_boundaries_merged)
    seg_recall_m = 1.0 if total_gt_boundaries_merged == 0 else total_matches_merged / total_gt_boundaries_merged
    seg_f1_m = 2 * seg_precision_m * seg_recall_m / (seg_precision_m + seg_recall_m) if seg_precision_m + seg_recall_m > 0 else 0.0
    total_ctc_keep_denom = sum(d["ctc_keep_unknown"]["ctc_denominator"] for d in daily) or 1
    total_ctc_ignore_denom = sum(d["ctc_ignore_unknown"]["ctc_denominator"] for d in daily) or 1
    weighted_over_days = {
        "num_days": len(daily),
        "total_frames": int(total_frames),
        "top1_accuracy": wavg("top1_accuracy"),
        "top3_accuracy": wavg("top3_accuracy"),
        "macro_recall": wavg("macro_recall"),
        "weighted_recall": wavg("weighted_recall"),
        "micro_recall": wavg("micro_recall"),
        "segmentation_accuracy": float(seg_precision),
        "segmentation_precision": float(seg_precision),
        "segmentation_recall": float(seg_recall),
        "segmentation_f1": float(seg_f1),
        "total_pred_boundaries": int(total_pred_boundaries),
        "total_gt_boundaries": int(total_gt_boundaries),
        "total_matched_boundaries": int(total_matches),
        "segmentation_after_same_label_merge": {
            "segmentation_accuracy": float(seg_precision_m),
            "segmentation_precision": float(seg_precision_m),
            "segmentation_recall": float(seg_recall_m),
            "segmentation_f1": float(seg_f1_m),
            "total_pred_boundaries": int(total_pred_boundaries_merged),
            "total_gt_boundaries": int(total_gt_boundaries_merged),
            "total_matched_boundaries": int(total_matches_merged),
        },
        "ctc_keep_unknown": {"ctc_sequence_accuracy": float(sum(d["ctc_keep_unknown"]["ctc_sequence_accuracy"] * d["ctc_keep_unknown"]["ctc_denominator"] for d in daily) / total_ctc_keep_denom), "total_denominator": int(total_ctc_keep_denom)},
        "ctc_ignore_unknown": {"ctc_sequence_accuracy": float(sum(d["ctc_ignore_unknown"]["ctc_sequence_accuracy"] * d["ctc_ignore_unknown"]["ctc_denominator"] for d in daily) / total_ctc_ignore_denom), "total_denominator": int(total_ctc_ignore_denom)},
    }
    return {"num_days": len(daily), "pooled_all_days": pooled_all, "pooled_all_days_per_class": pooled_per_class, "mean_over_days": mean_over_days, "weighted_over_days": weighted_over_days}


def evaluate_mode(cfg: RecognitionConfig, mode_name: str) -> Dict[str, Any]:
    day_dirs = list_day_dirs(cfg.root_parent)
    valid_days: List[EvalDayData] = []
    for day_dir in day_dirs:
        gt_path = day_dir / cfg.ground_truth_name
        pred_path = cfg.output_root / mode_name / day_dir.name / "recognition_results.json"
        if not gt_path.exists() or not pred_path.exists():
            continue
        valid_days.append(build_day_data(day_dir, pred_path, cfg))
    if not valid_days:
        summary = {"mode": mode_name, "error": "No valid days found for evaluation."}
        out_dir = cfg.output_root / mode_name / cfg.eval_dir_name
        save_json(summary, out_dir / cfg.eval_summary_name)
        return summary
    daily = [summarize_day(d, cfg) for d in valid_days]
    overall = summarize_overall(valid_days, daily)
    out_dir = cfg.output_root / mode_name / cfg.eval_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "root_parent": str(cfg.root_parent),
            "ground_truth_name": cfg.ground_truth_name,
            "frame_rate_hz": cfg.frame_rate_hz,
            "boundary_tolerance_sec": cfg.boundary_tol_sec,
            "canonical_labels": ensure_activity_subset(cfg.activities),
        },
        "overall": overall,
        "daily": daily,
    }
    save_json(summary, out_dir / cfg.eval_summary_name)
    csv_rows: List[Dict[str, Any]] = []
    csv_rows.extend([
        {
            "day": "POOLED",
            "num_frames": overall["pooled_all_days"]["num_frames"],
            "top1_accuracy": overall["pooled_all_days"]["top1_accuracy"],
            "top3_accuracy": overall["pooled_all_days"]["top3_accuracy"],
            "macro_recall": overall["pooled_all_days"]["macro_recall"],
            "weighted_recall": overall["pooled_all_days"]["weighted_recall"],
            "micro_recall": overall["pooled_all_days"]["micro_recall"],
            "segmentation_accuracy": "",
            "segmentation_recall": "",
            "segmentation_f1": "",
            "segmentation_accuracy_after_same_label_merge": "",
            "segmentation_recall_after_same_label_merge": "",
            "segmentation_f1_after_same_label_merge": "",
            "ctc_keep_unknown": "",
            "ctc_ignore_unknown": "",
            "pooled_macro_precision": overall["pooled_all_days_per_class"]["macro_precision"],
            "pooled_macro_recall": overall["pooled_all_days_per_class"]["macro_recall"],
            "pooled_macro_f1": overall["pooled_all_days_per_class"]["macro_f1"],
            "pooled_macro_one_vs_rest_accuracy": overall["pooled_all_days_per_class"]["macro_one_vs_rest_accuracy"],
        },
        {
            "day": "MEAN",
            "num_frames": "",
            "top1_accuracy": overall["mean_over_days"]["top1_accuracy"],
            "top3_accuracy": overall["mean_over_days"]["top3_accuracy"],
            "macro_recall": overall["mean_over_days"]["macro_recall"],
            "weighted_recall": overall["mean_over_days"]["weighted_recall"],
            "micro_recall": overall["mean_over_days"]["micro_recall"],
            "segmentation_accuracy": overall["mean_over_days"]["segmentation_accuracy"],
            "segmentation_recall": overall["mean_over_days"]["segmentation_recall"],
            "segmentation_f1": overall["mean_over_days"]["segmentation_f1"],
            "segmentation_accuracy_after_same_label_merge": overall["mean_over_days"]["segmentation_after_same_label_merge"]["segmentation_accuracy"],
            "segmentation_recall_after_same_label_merge": overall["mean_over_days"]["segmentation_after_same_label_merge"]["segmentation_recall"],
            "segmentation_f1_after_same_label_merge": overall["mean_over_days"]["segmentation_after_same_label_merge"]["segmentation_f1"],
            "ctc_keep_unknown": overall["mean_over_days"]["ctc_keep_unknown"]["ctc_sequence_accuracy"],
            "ctc_ignore_unknown": overall["mean_over_days"]["ctc_ignore_unknown"]["ctc_sequence_accuracy"],
            "pooled_macro_precision": "",
            "pooled_macro_recall": "",
            "pooled_macro_f1": "",
            "pooled_macro_one_vs_rest_accuracy": "",
        },
        {
            "day": "WEIGHTED",
            "num_frames": overall["weighted_over_days"]["total_frames"],
            "top1_accuracy": overall["weighted_over_days"]["top1_accuracy"],
            "top3_accuracy": overall["weighted_over_days"]["top3_accuracy"],
            "macro_recall": overall["weighted_over_days"]["macro_recall"],
            "weighted_recall": overall["weighted_over_days"]["weighted_recall"],
            "micro_recall": overall["weighted_over_days"]["micro_recall"],
            "segmentation_accuracy": overall["weighted_over_days"]["segmentation_accuracy"],
            "segmentation_recall": overall["weighted_over_days"]["segmentation_recall"],
            "segmentation_f1": overall["weighted_over_days"]["segmentation_f1"],
            "segmentation_accuracy_after_same_label_merge": overall["weighted_over_days"]["segmentation_after_same_label_merge"]["segmentation_accuracy"],
            "segmentation_recall_after_same_label_merge": overall["weighted_over_days"]["segmentation_after_same_label_merge"]["segmentation_recall"],
            "segmentation_f1_after_same_label_merge": overall["weighted_over_days"]["segmentation_after_same_label_merge"]["segmentation_f1"],
            "ctc_keep_unknown": overall["weighted_over_days"]["ctc_keep_unknown"]["ctc_sequence_accuracy"],
            "ctc_ignore_unknown": overall["weighted_over_days"]["ctc_ignore_unknown"]["ctc_sequence_accuracy"],
            "pooled_macro_precision": "",
            "pooled_macro_recall": "",
            "pooled_macro_f1": "",
            "pooled_macro_one_vs_rest_accuracy": "",
        },
    ])
    for d in daily:
        csv_rows.append({
            "day": d["day"],
            "num_frames": d["num_frames"],
            "top1_accuracy": d["top1_accuracy"],
            "top3_accuracy": d["top3_accuracy"],
            "macro_recall": d["macro_recall"],
            "weighted_recall": d["weighted_recall"],
            "micro_recall": d["micro_recall"],
            "segmentation_accuracy": d["segmentation_accuracy"],
            "segmentation_recall": d["segmentation_recall"],
            "segmentation_f1": d["segmentation_f1"],
            "segmentation_accuracy_after_same_label_merge": d["segmentation_after_same_label_merge"]["segmentation_accuracy"],
            "segmentation_recall_after_same_label_merge": d["segmentation_after_same_label_merge"]["segmentation_recall"],
            "segmentation_f1_after_same_label_merge": d["segmentation_after_same_label_merge"]["segmentation_f1"],
            "ctc_keep_unknown": d["ctc_keep_unknown"]["ctc_sequence_accuracy"],
            "ctc_ignore_unknown": d["ctc_ignore_unknown"]["ctc_sequence_accuracy"],
            "pooled_macro_precision": "",
            "pooled_macro_recall": "",
            "pooled_macro_f1": "",
            "pooled_macro_one_vs_rest_accuracy": "",
        })
    save_csv(csv_rows, out_dir / cfg.eval_daily_csv_name)
    return summary


def evaluate_all_modes(cfg: RecognitionConfig, mode_names: List[str]) -> Dict[str, Any]:
    all_summary: Dict[str, Any] = {}
    for mode_name in mode_names:
        all_summary[mode_name] = evaluate_mode(cfg, mode_name)
    save_json(all_summary, cfg.output_root / "all_modes_evaluation_summary.json")
    return all_summary
