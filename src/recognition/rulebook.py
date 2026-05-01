from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import ValidationError

from .config import RecognitionConfig
from .constants import FEATURE_SCHEMA, activity_descriptions_subset, ensure_activity_subset
from .llm_client import get_llm, invoke_json
from .schema import RuleBook
from .usage import UsageTracker


def validate_rulebook(rb: Any) -> RuleBook:
    if isinstance(rb, RuleBook):
        return rb
    if not isinstance(rb, dict):
        raise TypeError("RuleBook must be a dict or RuleBook instance")
    try:
        return RuleBook.model_validate(rb)
    except ValidationError as e:
        raise ValueError(f"Invalid RuleBook DSL: {e}") from e


def check_rulebook_quality(rb: RuleBook) -> None:
    total_rules = len(rb.rules)
    if total_rules == 0:
        raise ValueError("RuleBook quality check failed: rules must not be empty.")

    nonempty_any = sum(1 for r in rb.rules if len(r.when_any) > 0)
    if nonempty_any / max(total_rules, 1) < 0.40:
        raise ValueError("RuleBook quality check failed: too few rules use non-empty when_any.")

    by_activity: Dict[str, List[Any]] = {}
    for r in rb.rules:
        by_activity.setdefault(r.activity, []).append(r)

    for act in rb.activities:
        act_rules = by_activity.get(act, [])
        if not act_rules:
            raise ValueError(f"RuleBook quality check failed: activity '{act}' has no rules.")
        if not any(len(r.when_any) > 0 for r in act_rules):
            raise ValueError(f"RuleBook quality check failed: activity '{act}' has no rule with non-empty when_any.")
        for r in act_rules:
            if len(r.when_all) > 4:
                raise ValueError(f"RuleBook quality check failed: rule '{r.id}' is too conjunctive.")


def digest_high_conf_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_act: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_act.setdefault(r["final_label"], []).append(r)

    digest = {"activities": {}}
    for act, recs in by_act.items():
        rooms = [x["segment"]["dominant_room_type"] for x in recs]
        hours = [float(x["segment"]["hour"]) for x in recs]
        durs = [float(x["segment"]["duration_s"]) for x in recs]
        room_counts = {room: rooms.count(room) for room in sorted(set(rooms))}
        dominant_room = max(room_counts.items(), key=lambda kv: kv[1])[0]
        digest["activities"][act] = {
            "count": len(recs),
            "dominant_room": dominant_room,
            "hour_mean": float(np.mean(hours)) if hours else None,
            "hour_std": float(np.std(hours)) if hours else None,
            "duration_median_s": float(np.median(durs)) if durs else None,
        }
    return digest


def _default_rulebook_path() -> Path:
    return Path(__file__).resolve().parent / "default_rulebook.json"


def filter_rulebook_to_activities(rulebook: Dict[str, Any], activities: List[str]) -> Dict[str, Any]:
    allowed = set(ensure_activity_subset(activities))
    out = dict(rulebook)
    out["activities"] = [a for a in rulebook.get("activities", []) if a in allowed]
    out["constraints"] = [c for c in rulebook.get("constraints", []) if c.get("activity") in allowed]
    out["rules"] = [r for r in rulebook.get("rules", []) if r.get("activity") in allowed]
    return out


def normalize_rulebook_dict(rulebook: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(rulebook)
    for c in out.get("constraints", []):
        ctype = c.get("type")
        if ctype in {"hard", "hard_room", "hard-room", "room_hard_constraint", "hardroomtype"}:
            c["type"] = "hard_room_type"
        penalty = c.get("penalty")
        if penalty == float("-inf"):
            c["penalty"] = "-inf"
    return out


def default_rulebook(cfg: RecognitionConfig) -> RuleBook:
    data = json.loads(_default_rulebook_path().read_text(encoding="utf-8"))
    filtered = filter_rulebook_to_activities(data, cfg.activities)
    rb = validate_rulebook(filtered)
    check_rulebook_quality(rb)
    return rb


def _rulebook_system_prompt() -> str:
    return """
    You are generating an executable EchoLIFE RuleBook in strict JSON.

    The output MUST match this DSL exactly:
    - top-level keys: version, activities, constraints, rules, merge_rules, personalization_priors
    - constraints: a JSON array of hard constraints
    - each constraint must have exactly: type, activity, allowed_room_types, penalty
    - penalty MUST be the exact string "-inf" for every hard constraint
    - rules: a JSON array
    - each rule must have exactly: id, activity, weight, when_all, when_any, evidence_template
    - when_all and when_any must be JSON arrays of predicate objects
    - each predicate object must have exactly: field, op, value
    - only use fields from the provided feature schema

    """.strip()


def _rulebook_human_prompt(with_base_rulebook: bool) -> str:
    if not with_base_rulebook:
        return """
activity_inventory:
{activity_inventory}

feature_schema:
{feature_schema}

layout:
{layout}

profile_text:
{profile_text}

high_confidence_history_digest:
{previous_days_digest}

Design requirements:
- Focus on his personal habits and room layout.
- For each activity, generate 2 to 4 rules.
- Use constraints only for strong room-type restrictions.
- Use when_all for core necessary conditions only.
- Use when_any for supportive or alternative evidence.
- At least one rule per activity must have non-empty when_any.
- At least 40 percent of all rules should have non-empty when_any.
- For most rules, when_all should contain only 1-2 core concepts.
- A rule should usually remain satisfiable even if one plausible cue is missing.
- Prefer several weaker additive rules over one overly strict conjunctive rule.
- Do not overfit to tiny counts in the history digest.
- If the history digest is weak or sparse, stay close to the default semantics.
- Recommended hard constraints:
  * Toileting, Bathing, Washing/Brushing -> bathroom
  * Cooking -> kitchen
Example constraint:
{{
  "type": "hard_room_type",
  "activity": "<label>",
  "allowed_room_types": ["<room>"],
  "penalty": "-inf"
}}

- Field usage guidance:
  * dominant_room_type and Centroid_mean are more suitable as when_all core conditions
  * Active_ratio, Salience_mean, coarse duration ranges, Spread_mean, and softer time-of-day cues are usually better as when_any supportive cues

Good example:
{{
  "id": "XXX_rule_1",
  "activity": "<label>",
  "weight": 0.90,
  "when_all": [
    {{"field": "dominant_room_type", "op": "==", "value": "XXX"}},
    {{"field": "Centroid_mean", "op": ">=", "value": xx}},
    {{"field": "Centroid_mean", "op": "<=", "value": xx}}
  ],
  "when_any": [
    {{"field": "Active_ratio", "op": "<", "value": xx}},
    {{"field": "Spread_mean", "op": "<=", "value": xx}}
  ],
  "evidence_template": "<brief reason>"
}}

Bad style to avoid:
- putting room type + duration + activity ratio + time-of-day + spatial pattern all into when_all
- leaving when_any empty for almost every rule
- making all rules so strict that they rarely fire

Return JSON only.
""".strip()

    return """
activity_inventory:
{activity_inventory}

feature_schema:
{feature_schema}

layout:
{layout}

profile_text:
{profile_text}

base_rulebook:
{base_rulebook}

high_confidence_history_digest:
{previous_days_digest}

Design requirements:
- Focus on his personal habits and room layout.
- If there have a base_rulebook, start from the provided base_rulebook and revise it conservatively.
- For each activity, generate 2 to 4 rules.
- Use constraints only for strong room-type restrictions.
- Use when_all for core necessary conditions only.
- Use when_any for supportive or alternative evidence.
- At least one rule per activity must have non-empty when_any.
- At least 40 percent of all rules should have non-empty when_any.
- For most rules, when_all should contain only 1-2 core concepts.
- A rule should usually remain satisfiable even if one plausible cue is missing.
- Prefer several weaker additive rules over one overly strict conjunctive rule.
- Do not overfit to tiny counts in the history digest.
- If the history digest is weak or sparse, stay close to the default semantics.
- Recommended hard constraints:
  * Toileting, Bathing, Washing/Brushing -> bathroom
  * Cooking -> kitchen
Example constraint:
{{
  "type": "hard_room_type",
  "activity": "<label>",
  "allowed_room_types": ["<room>"],
  "penalty": "-inf"
}}

- Field usage guidance:
  * dominant_room_type and Centroid_mean are more suitable as when_all core conditions
  * Active_ratio, Salience_mean, coarse duration ranges, Spread_mean, and softer time-of-day cues are usually better as when_any supportive cues

Good example:
{{
  "id": "XXX_rule_1",
  "activity": "<label>",
  "weight": 0.90,
  "when_all": [
    {{"field": "dominant_room_type", "op": "==", "value": "XXX"}},
    {{"field": "Centroid_mean", "op": ">=", "value": xx}},
    {{"field": "Centroid_mean", "op": "<=", "value": xx}}
  ],
  "when_any": [
    {{"field": "Active_ratio", "op": "<", "value": xx}},
    {{"field": "Spread_mean", "op": "<=", "value": xx}}
  ],
  "evidence_template": "<brief reason>"
}}

Bad style to avoid:
- putting room type + duration + activity ratio + time-of-day + spatial pattern all into when_all
- leaving when_any empty for almost every rule
- making all rules so strict that they rarely fire

Return JSON only.
""".strip()


def _rulebook_repair_system_prompt() -> str:
    return """
You repair invalid EchoLIFE RuleBook JSON into valid strict JSON.
Return JSON only.
Requirements:
- top-level keys must be: version, activities, constraints, rules
- every hard room constraint must use exactly: "type": "hard_room_type"
- every hard room constraint penalty must be exactly "-inf"
- constraints must be a JSON array
- rules must be a JSON array
- each rule must have: id, activity, weight, when_all, when_any, evidence_template
- when_all and when_any must be arrays of predicate objects with field, op, value
- do not introduce activities outside activity_inventory
- keep the semantics conservative and valid
""".strip()


def _rulebook_repair_human_prompt() -> str:
    return """
activity_inventory:
{activity_inventory}

validation_or_quality_error:
{error_text}

invalid_rulebook_json:
{invalid_rulebook}

Return corrected JSON only.
""".strip()


def _save_rulebook_debug(cfg: RecognitionConfig, day: str, suffix: str, text: str) -> None:
    debug_dir = cfg.output_root / "_debug_rulebook"
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / f"day_{day}_{suffix}.txt").write_text(text, encoding="utf-8")


def _attempt_validate_rulebook(cfg: RecognitionConfig, rb_dict: Dict[str, Any]) -> RuleBook:
    normalized = normalize_rulebook_dict(rb_dict)
    filtered = filter_rulebook_to_activities(normalized, cfg.activities)
    rb = validate_rulebook(filtered)
    check_rulebook_quality(rb)
    return rb


def _llm_generate_rulebook(
    cfg: RecognitionConfig,
    layout: Dict[str, Any],
    profile_text: str,
    previous_days_digest: Dict[str, Any],
    usage: UsageTracker,
    day: str,
    base_rulebook: RuleBook | None = None,
) -> Tuple[RuleBook, str]:
    llm = get_llm(cfg)
    if llm is None or not cfg.use_llm_for_rulebook:
        return default_rulebook(cfg), "default"

    activity_inventory = activity_descriptions_subset(cfg.activities)
    system_msg = _rulebook_system_prompt()
    human_msg = _rulebook_human_prompt(with_base_rulebook=base_rulebook is not None)

    variables = {
        "activity_inventory": json.dumps(activity_inventory, ensure_ascii=False, indent=2),
        "feature_schema": json.dumps(FEATURE_SCHEMA, ensure_ascii=False, indent=2),
        "layout": json.dumps(layout, ensure_ascii=False, indent=2),
        "profile_text": profile_text,
        "previous_days_digest": json.dumps(previous_days_digest, ensure_ascii=False, indent=2),
    }
    if base_rulebook is not None:
        variables["base_rulebook"] = json.dumps(base_rulebook.model_dump(), ensure_ascii=False, indent=2)

    last_error = "unknown error"
    last_text = ""
    attempt_count = max(1, int(cfg.rulebook_max_retries))

    for attempt_idx in range(attempt_count):
        try:
            cleaned_text, rb_dict = invoke_json(
                llm=llm,
                system_msg=system_msg,
                human_msg=human_msg,
                variables=variables,
                tracker=usage,
                stage="rulebook_generation",
                day=day,
                extra={"attempt": attempt_idx + 1},
            )
            last_text = cleaned_text
            rb = _attempt_validate_rulebook(cfg, rb_dict)
            if attempt_idx > 0 and cfg.verbose:
                print(f"[rulebook] day {day}: regeneration succeeded on attempt {attempt_idx + 1}.")
            return rb, "llm"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if cfg.verbose:
                print(f"[rulebook] day {day}: generation attempt {attempt_idx + 1}/{attempt_count} failed.")
                print(f"[rulebook] error: {last_error}")
            if last_text:
                _save_rulebook_debug(cfg, day, f"attempt_{attempt_idx + 1}_raw", last_text)

        if attempt_idx == attempt_count - 1:
            break

        repair_variables = {
            "activity_inventory": json.dumps(activity_inventory, ensure_ascii=False, indent=2),
            "error_text": last_error,
            "invalid_rulebook": last_text or "<empty>",
        }
        try:
            repaired_text, repaired_dict = invoke_json(
                llm=llm,
                system_msg=_rulebook_repair_system_prompt(),
                human_msg=_rulebook_repair_human_prompt(),
                variables=repair_variables,
                tracker=usage,
                stage="rulebook_generation_retry",
                day=day,
                extra={"attempt": attempt_idx + 1},
            )
            last_text = repaired_text
            rb = _attempt_validate_rulebook(cfg, repaired_dict)
            if cfg.verbose:
                print(f"[rulebook] day {day}: repaired successfully after failed attempt {attempt_idx + 1}.")
            return rb, "llm_repaired"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if cfg.verbose:
                print(f"[rulebook] day {day}: repair attempt after generation failure also failed.")
                print(f"[rulebook] repair error: {last_error}")
            if last_text:
                _save_rulebook_debug(cfg, day, f"attempt_{attempt_idx + 1}_repair_raw", last_text)

    if cfg.verbose:
        print(f"[rulebook] day {day}: all generation attempts failed, fallback to default rulebook.")
        print(f"[rulebook] final error: {last_error}")
    return default_rulebook(cfg), "fallback_default"



def get_rulebook_for_day(
    cfg: RecognitionConfig,
    current_rulebook: RuleBook | None,
    layout: Dict[str, Any],
    profile_text: str,
    high_conf_history_records: List[Dict[str, Any]],
    usage: UsageTracker,
    day: str,
) -> Tuple[RuleBook, str]:
    strategy = cfg.rulebook_strategy
    if strategy == "fixed":
        return current_rulebook or default_rulebook(cfg), "fixed"
    if strategy == "llm_once":
        if current_rulebook is not None:
            return current_rulebook, "cached"
        return _llm_generate_rulebook(cfg, layout, profile_text, digest_high_conf_records(high_conf_history_records), usage, day, base_rulebook=None)
    if strategy == "llm_tune_from_history":
        if current_rulebook is None:
            current_rulebook = default_rulebook(cfg)
        digest = digest_high_conf_records(high_conf_history_records)
        if not digest.get("activities"):
            return current_rulebook, "cached"
        return _llm_generate_rulebook(cfg, layout, profile_text, digest, usage, day, base_rulebook=current_rulebook)
    return current_rulebook or default_rulebook(cfg), "fixed"
