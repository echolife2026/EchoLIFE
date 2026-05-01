from __future__ import annotations

from typing import Dict, Iterable, List

ALL_ACTIVITIES = [
    "Rest/Sleeping",
    "Toileting",
    "Bathing",
    "Cooking",
    "Eating",
    "Working/Studying",
    "Washing/Brushing",
    "Entertainment/Relax",
    "Exercise",
    "Cleaning",
    "Unknown/Other",
]

ALL_ACTIVITY_DESCRIPTIONS = {
    "Rest/Sleeping": "Long-duration, low-activity behavior, commonly occurring in the bedroom or living room. Long nighttime episodes are more likely to indicate sleeping, while medium-to-long daytime sedentary episodes are more likely to indicate resting.",
    "Toileting": "Usually occurs at the toilet area in the bathroom/restroom. Urination is typically short (from a few seconds to one or two minutes), while defecation usually lasts a moderate amount of time (several minutes). Activity level is generally low, and the room constraint is strong.",
    "Bathing": "Usually occurs in the bathing area of the bathroom. Duration is typically medium to long. The room constraint is strong, while activity level is not necessarily intense.",
    "Cooking": "Usually occurs in the kitchen. Duration is typically moderate (often longer than 10 minutes), with a relatively high activity ratio.",
    "Eating": "Usually occurs in the dining area, kitchen, or living room. Duration is typically moderate (often longer than 10 minutes), with low-to-moderate activity. It often occurs before or after cooking and is commonly aligned with breakfast, lunch, or dinner periods.",
    "Working/Studying": "Usually occurs in the study or living room. Duration is typically long, activity level is moderate, and position is relatively stable.",
    "Washing/Brushing": "Usually occurs at the sink in the bathroom. Duration is usually not very short and is often a few minutes. Its activity pattern and duration typically fall between toileting and bathing.",
    "Entertainment/Relax": "Usually occurs in the living room or bedroom. Duration is medium to long, activity level is relatively low, and position is relatively stable.",
    "Exercise": "Usually occurs in the living room. Activity is strong, Active_ratio is high, spatial variation is large, and duration is typically moderate.",
    "Cleaning": "Usually occurs in the kitchen or living room. Activity is medium to high, positional variation is relatively large, duration is moderate, and it often happens after eating.",
    "Unknown/Other": "Used when rule evidence is insufficient, room semantics do not support a clear label, or the segment does not match any well-defined activity. Especially when the room is unknown, it often corresponds either to very short episodes of about one minute or to being out of home for longer than one hour.",
}

FEATURE_SCHEMA = {
    "fields": {
        "start_time": "HH:MM:SS wall-clock string of segment start",
        "hour": "Float hour-of-day derived from start_time",
        "duration_s": "Segment duration in seconds",
        "dominant_room_type": "Semantic room type; here it is directly the room name or UNKNOWN",
        "Salience_mean": "Mean salient motion strength over the segment. This is the average activity intensity during the segment, typically ranging from 0 to several tens; targets that are closer usually produce higher intensity.",
        "Active_ratio": "Fraction of active frames within the segment; proportion of time with motion",
        "Centroid_mean": "Mean range centroid over the segment, in meters",
        "Spread_mean": "Mean spatial spread over the segment, in meters",
        "t0_sec": "Segment start time in seconds since midnight",
        "t1_sec": "Segment end time in seconds since midnight",
    },
    "note": "Only these fields may be referenced by the RuleBook and calibration logic.",
}


def ensure_activity_subset(requested: Iterable[str]) -> List[str]:
    requested_list = [str(x) for x in requested]
    out = [a for a in ALL_ACTIVITIES if a in requested_list]
    if "Unknown/Other" not in out:
        out.append("Unknown/Other")
    return out


def activity_descriptions_subset(requested: Iterable[str]) -> Dict[str, str]:
    acts = ensure_activity_subset(requested)
    return {a: ALL_ACTIVITY_DESCRIPTIONS[a] for a in acts}
