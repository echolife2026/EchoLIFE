from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .config import RecognitionConfig
from .constants import FEATURE_SCHEMA, activity_descriptions_subset
from .llm_client import get_llm, invoke_json
from .usage import UsageTracker

EPS = 1e-9


LLM_ONLY_OUTPUT_EXAMPLE = {
    "final_label": "<label>",
    "final_top3": [
        {
            "label": "<label>",
            "score": 0.0,
            "confidence": 0.0,
            "evidence": ["<brief reason>"]
        },
        {
            "label": "<label>",
            "score": 0.0,
            "confidence": 0.0,
            "evidence": ["<brief reason>"]
        },
        {
            "label": "<label>",
            "score": 0.0,
            "confidence": 0.0,
            "evidence": ["<brief reason>"]
        }
    ],
    "decision_basis": {
        "summary": "<short explanation>",
        "retrieved_examples": [],
        "recent_context": []
    },
    "explanation": "<one short sentence>"
}


CALIBRATION_OUTPUT_EXAMPLE = {
    "final_label": "<label>",
    "final_top3": [
        {
            "label": "<label>",
            "score": 0.0,
            "confidence": 0.0,
            "evidence": ["<brief reason after calibration>"]
        },
        {
            "label": "<label>",
            "score": 0.0,
            "confidence": 0.0,
            "evidence": ["<brief reason after calibration>"]
        },
        {
            "label": "<label>",
            "score": 0.0,
            "confidence": 0.0,
            "evidence": ["<brief reason after calibration>"]
        }
    ],
    "decision_basis": {
        "summary": "<short explanation of the revision>",
        "retrieved_examples": [],
        "recent_context": []
    },
    "explanation": "<one short sentence>"
}


def segment_to_vector(seg: Dict[str, Any]) -> np.ndarray:
    hour = float(seg["hour"])
    theta = 2.0 * math.pi * hour / 24.0
    vec = np.asarray([
        float(seg["duration_s"]) / 3600.0,
        float(seg["Salience_mean"]),
        float(seg["Active_ratio"]),
        float(seg["Centroid_mean"]),
        float(seg["Spread_mean"]),
        math.sin(theta),
        math.cos(theta),
    ], dtype=np.float32)
    n = np.linalg.norm(vec) + EPS
    return vec / n


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + EPS) * (np.linalg.norm(b) + EPS)))


def retrieve_similar_records(query_seg: Dict[str, Any], memory_records: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if not memory_records:
        return []
    qv = segment_to_vector(query_seg)
    same_room = [r for r in memory_records if r["segment"]["dominant_room_type"] == query_seg["dominant_room_type"]]
    pool = same_room if same_room else memory_records
    scored = []
    for r in pool:
        sim = cosine_sim(qv, np.asarray(r["vector"], dtype=np.float32))
        rr = dict(r)
        rr["retrieval_similarity"] = float(sim)
        scored.append(rr)
    scored.sort(key=lambda x: x["retrieval_similarity"], reverse=True)
    return scored[:top_k]


def build_recent_context(day_records: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    out = []
    for r in day_records[-n:]:
        out.append({
            "start_time": r["segment"]["start_time"],
            "duration_s": r["segment"]["duration_s"],
            "room": r["segment"]["dominant_room_type"],
            "label": r["final_label"],
            "confidence": r["final_top3"][0]["confidence"] if r.get("final_top3") else None,
            "margin": r.get("final_margin"),
        })
    return out


def _safe_llm_output(out: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(out, dict):
        return None
    if not isinstance(out.get("final_label"), str):
        return None
    top3 = out.get("final_top3")
    if not isinstance(top3, list) or not top3:
        return None

    cleaned = []
    for item in top3:
        if not isinstance(item, dict) or "label" not in item:
            continue
        cleaned.append({
            "label": item["label"],
            "score": float(item.get("score", 0.0)),
            "confidence": float(item.get("confidence", 0.0)),
            "evidence": item.get("evidence", []) if isinstance(item.get("evidence", []), list) else [],
        })
    if not cleaned:
        return None

    decision_basis = out.get("decision_basis", {})
    if not isinstance(decision_basis, dict):
        decision_basis = {}
    if not isinstance(decision_basis.get("retrieved_examples", []), list):
        decision_basis["retrieved_examples"] = []
    if not isinstance(decision_basis.get("recent_context", []), list):
        decision_basis["recent_context"] = []

    explanation = out.get("explanation", "")
    if not isinstance(explanation, str):
        explanation = ""

    return {
        "final_label": out["final_label"],
        "final_top3": cleaned,
        "decision_basis": decision_basis,
        "explanation": explanation,
    }


def llm_calibrate(
    cfg: RecognitionConfig,
    current_seg: Dict[str, Any],
    rule_top3: List[Dict[str, Any]],
    retrieved: List[Dict[str, Any]],
    recent_ctx: List[Dict[str, Any]],
    layout: Dict[str, Any],
    profile_text: str,
    usage: UsageTracker,
    day: str,
    seg_idx: int,
) -> Optional[Dict[str, Any]]:
    llm = get_llm(cfg)
    if llm is None or not cfg.use_llm_for_calibration:
        return None

    system_msg = """
You are a strict activity classification expert.
Your task is to infer the most plausible human activity based on numeric sensor features.
You must reason using physical constraints, time-of-day, duration, motion intensity, and distance evidence.
Choose labels only from the provided activity_inventory.
Return strict JSON with keys: final_label, final_top3, decision_basis, explanation.
Each final_top3 item must contain label, score, confidence, evidence.

You should:
1. Reason carefully about what activities are physically possible or impossible.
2. Use time-of-day, duration, motion evidence, and distance jointly.
3. For each candidate, explain why it is plausible or why evidence is weak.
4. Focus on his personal habits and room layout.
The output format example is:
{output_example}
Return JSON only.
""".strip()
    human_msg = """
activity_inventory:
{activity_inventory}

feature_schema:
{feature_schema}

layout:
{layout}

profile_text:
{profile_text}

current_segment:
{current_segment}

rulebook_top3:
{rulebook_top3}

retrieved_similar_segments:
{retrieved}

recent_context:
{recent_context}

Return JSON only.
""".strip()
    try:
        _, out = invoke_json(
            llm=llm,
            system_msg=system_msg,
            human_msg=human_msg,
            variables={
                "activity_inventory": activity_descriptions_subset(cfg.activities),
                "feature_schema": FEATURE_SCHEMA,
                "layout": layout,
                "profile_text": profile_text,
                "current_segment": current_seg,
                "rulebook_top3": rule_top3,
                "retrieved": retrieved,
                "recent_context": recent_ctx,
                "output_example": CALIBRATION_OUTPUT_EXAMPLE,
            },
            tracker=usage,
            stage="llm_calibration",
            day=day,
            extra={"segment_index": seg_idx},
        )
    except Exception:
        return None
    return _safe_llm_output(out)


def llm_classify(
    cfg: RecognitionConfig,
    current_seg: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
    recent_ctx: List[Dict[str, Any]],
    layout: Dict[str, Any],
    profile_text: str,
    usage: UsageTracker,
    day: str,
    seg_idx: int,
) -> Dict[str, Any]:
    llm = get_llm(cfg)
    fallback = {
        "final_label": "Unknown/Other",
        "final_top3": [{"label": "Unknown/Other", "score": 0.0, "confidence": 1.0, "evidence": ["LLM unavailable or invalid output"]}],
        "decision_basis": {"retrieved_examples": retrieved, "recent_context": recent_ctx},
        "explanation": "LLM unavailable or invalid output.",
    }
    if llm is None or not cfg.use_llm_for_llm_only:
        return fallback

    system_msg = """
You are a strict activity classification expert.
Your task is to infer the most plausible human activity based on numeric sensor features.
You must reason using physical constraints, time-of-day, duration, motion intensity, and distance evidence.
Choose labels only from the provided activity_inventory.
Return strict JSON with keys: final_label, final_top3, decision_basis, explanation.
Each final_top3 item must contain label, score, confidence, evidence.

You should:
1. Reason carefully about what activities are physically possible or impossible.
2. Use time-of-day, duration, motion evidence, and distance jointly.
3. For each candidate, explain why it is plausible or why evidence is weak.
4. Focus on his personal habits and room layout.
The output format example is:
{output_example}
Return JSON only.
""".strip()
    human_msg = """
activity_inventory:
{activity_inventory}

feature_schema:
{feature_schema}

layout:
{layout}

profile_text:
{profile_text}

current_segment:
{current_segment}

retrieved_similar_segments:
{retrieved}

recent_context:
{recent_context}

Return JSON only.
""".strip()
    try:
        _, out = invoke_json(
            llm=llm,
            system_msg=system_msg,
            human_msg=human_msg,
            variables={
                "activity_inventory": activity_descriptions_subset(cfg.activities),
                "feature_schema": FEATURE_SCHEMA,
                "layout": layout,
                "profile_text": profile_text,
                "current_segment": current_seg,
                "retrieved": retrieved,
                "recent_context": recent_ctx,
                "output_example": LLM_ONLY_OUTPUT_EXAMPLE,
            },
            tracker=usage,
            stage="llm_only_recognition",
            day=day,
            extra={"segment_index": seg_idx},
        )
    except Exception:
        return fallback

    safe = _safe_llm_output(out)
    return safe if safe is not None else fallback
