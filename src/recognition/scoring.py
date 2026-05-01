from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

EPS = 1e-9


def softmax_from_scores(scores: List[float]) -> List[float]:
    arr = np.asarray(scores, dtype=np.float32)
    arr = np.clip(arr, -50.0, 50.0)
    ex = np.exp(arr - np.max(arr))
    ex = ex / (np.sum(ex) + EPS)
    return [float(v) for v in ex]


def topk_with_ties(cands: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if not cands:
        return []
    cands = sorted(cands, key=lambda x: x["score"], reverse=True)
    if len(cands) <= k:
        return cands
    kth = cands[k - 1]["score"]
    out = [c for c in cands if c["score"] >= kth]
    return out if len(out) >= k else cands[:k]


def _eval_pred(seg: Dict[str, Any], pred: Any) -> bool:
    v = seg.get(pred.field)
    op = pred.op
    val = pred.value
    if op == "==":
        return v == val
    if op == "!=":
        return v != val
    if v is None:
        return False
    if op == ">":
        return float(v) > float(val)
    if op == ">=":
        return float(v) >= float(val)
    if op == "<":
        return float(v) < float(val)
    if op == "<=":
        return float(v) <= float(val)
    if op == "in":
        return v in val
    if op == "not_in":
        return v not in val
    return False


def _rule_fires(seg: Dict[str, Any], rule: Any) -> bool:
    for p in rule.when_all:
        if not _eval_pred(seg, p):
            return False
    if rule.when_any and not any(_eval_pred(seg, p) for p in rule.when_any):
        return False
    return True


def apply_constraints(score_map: Dict[str, float], seg: Dict[str, Any], rb: Any) -> None:
    room = seg.get("dominant_room_type")
    for c in rb.constraints:
        if c.type == "hard_room_type" and (room is None or room not in c.allowed_room_types):
            score_map[c.activity] = -1e9


def score_segment_with_rulebook(seg: Dict[str, Any], rb: Any, top_k: int) -> List[Dict[str, Any]]:
    score_map = {a: 0.0 for a in rb.activities}
    evidence_map = {a: [] for a in rb.activities}

    for rule in rb.rules:
        act = rule.activity
        if act not in score_map:
            continue
        if _rule_fires(seg, rule):
            score_map[act] += float(rule.weight)
            if rule.evidence_template:
                evidence_map[act].append(rule.evidence_template)

    apply_constraints(score_map, seg, rb)

    labels = list(rb.activities)
    scores = [float(score_map[a]) for a in labels]
    probs = softmax_from_scores(scores)

    cands: List[Dict[str, Any]] = []
    for i, a in enumerate(labels):
        cands.append({
            "label": a,
            "score": float(score_map[a]),
            "confidence": float(probs[i]),
            "evidence": evidence_map[a][:8],
        })
    return topk_with_ties(cands, top_k)


def margin_from_top2(cands: List[Dict[str, Any]]) -> float:
    if len(cands) < 2:
        return 1e9 if cands else 0.0
    return float(cands[0]["confidence"] - cands[1]["confidence"])
