from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .config import RecognitionConfig
from .io_utils import build_exec_segment, list_day_dirs, load_day_inputs, load_optional_layout, load_optional_profile_text, save_json
from .retrieval_and_llm import build_recent_context, llm_calibrate, llm_classify, retrieve_similar_records, segment_to_vector
from .rulebook import default_rulebook, get_rulebook_for_day
from .scoring import margin_from_top2, score_segment_with_rulebook, topk_with_ties
from .usage import UsageTracker

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class ModeState:
    name: str
    high_conf_records: List[Dict[str, Any]] = field(default_factory=list)
    memory_records: List[Dict[str, Any]] = field(default_factory=list)
    all_day_outputs: List[Dict[str, Any]] = field(default_factory=list)
    usage: UsageTracker | None = None
    current_rulebook: Any = None

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = UsageTracker(self.name)


def _write_line(msg: str) -> None:
    if tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg)


def _is_high_conf(cfg: RecognitionConfig, final_label: str, final_margin: float) -> bool:
    if final_label in cfg.exclude_memory_labels:
        return False
    return float(final_margin) >= cfg.high_conf_margin


def _format_record(day: str, idx: int, exec_seg: Dict[str, Any], final_label: str, final_top3: List[Dict[str, Any]], final_margin: float, explanation: str, rulebook_top3: List[Dict[str, Any]] | None = None, rule_margin: float | None = None, calibrated: bool = False, retrieved: List[Dict[str, Any]] | None = None, recent_context: List[Dict[str, Any]] | None = None, decision_basis: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "day": day,
        "index": idx,
        "segment": exec_seg,
        "rulebook_top3": rulebook_top3 or [],
        "rule_margin": float(rule_margin) if rule_margin is not None else None,
        "calibrated": bool(calibrated),
        "retrieved": retrieved or [],
        "recent_context": recent_context or [],
        "final_label": final_label,
        "final_top3": final_top3,
        "final_confidence": float(final_top3[0]["confidence"]) if final_top3 else 0.0,
        "final_margin": float(final_margin),
        "decision_basis": decision_basis or {},
        "explanation": explanation,
    }


def _memory_record(day: str, idx: int, exec_seg: Dict[str, Any], rulebook_top3: List[Dict[str, Any]], final_label: str, final_top3: List[Dict[str, Any]], final_margin: float) -> Dict[str, Any]:
    return {
        "day": day,
        "segment_index": idx,
        "segment": exec_seg,
        "rulebook_top3": rulebook_top3,
        "final_label": final_label,
        "final_top3": final_top3,
        "final_margin": float(final_margin),
        "vector": segment_to_vector(exec_seg).astype(float).tolist(),
    }


def _save_mode_day_outputs(cfg: RecognitionConfig, mode: str, day: str, day_meta: Dict[str, Any], rulebook_obj: Any, rulebook_source: str, results: List[Dict[str, Any]], usage_summary: Dict[str, Any]) -> None:
    out_dir = cfg.output_root / mode / day
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json({"source": rulebook_source, "rulebook": rulebook_obj}, out_dir / "rulebook.json")
    save_json({"day": day, "start_time": day_meta.get("start_time"), "results": results}, out_dir / "recognition_results.json")
    save_json(usage_summary, out_dir / "usage_metadata.json")


def _prepare_rulebook(cfg: RecognitionConfig, state: ModeState, day: str, layout: Dict[str, Any], profile_text: str):
    rb, source = get_rulebook_for_day(
        cfg=cfg,
        current_rulebook=state.current_rulebook,
        layout=layout,
        profile_text=profile_text,
        high_conf_history_records=state.high_conf_records,
        usage=state.usage,
        day=day,
    )
    state.current_rulebook = rb
    return rb, source


def _run_rulebook_only_day(cfg: RecognitionConfig, state: ModeState, day_dir, layout: Dict[str, Any], profile_text: str) -> Dict[str, Any]:
    summaries, day_meta = load_day_inputs(day_dir, cfg.segment_output_dir, cfg.input_summaries_name, cfg.input_day_meta_name)
    rb, rb_source = _prepare_rulebook(cfg, state, day_dir.name, layout, profile_text)
    results: List[Dict[str, Any]] = []
    current_high_conf: List[Dict[str, Any]] = []
    usage_before = len(state.usage.records)
    for idx, summary in enumerate(summaries):
        exec_seg = build_exec_segment(summary)
        rule_top3 = score_segment_with_rulebook(exec_seg, rb, cfg.top_k)
        final_top3 = topk_with_ties(rule_top3, cfg.top_k)
        final_label = final_top3[0]["label"] if final_top3 else "Unknown/Other"
        rule_margin = margin_from_top2(rule_top3)
        results.append(_format_record(day_dir.name, idx, exec_seg, final_label, final_top3, rule_margin, "RuleBook only.", rulebook_top3=rule_top3, rule_margin=rule_margin))
        if _is_high_conf(cfg, final_label, rule_margin):
            mem = _memory_record(day_dir.name, idx, exec_seg, rule_top3, final_label, final_top3, rule_margin)
            state.memory_records.append(mem)
            current_high_conf.append(results[-1])
    state.high_conf_records.extend(current_high_conf)
    state.all_day_outputs.append({"day": day_dir.name, "start_time": day_meta.get("start_time"), "activities": rb.activities, "results": results})
    if cfg.save_per_day_files:
        _save_mode_day_outputs(cfg, state.name, day_dir.name, day_meta, rb.model_dump(), rb_source, results, state.usage.summary())
    return {
        "day": day_dir.name,
        "num_segments": len(results),
        "num_high_conf": len(current_high_conf),
        "rulebook_source": rb_source,
        "llm_calls": len(state.usage.records) - usage_before,
    }


def _run_llm_only_day(cfg: RecognitionConfig, state: ModeState, day_dir, layout: Dict[str, Any], profile_text: str) -> Dict[str, Any]:
    summaries, day_meta = load_day_inputs(day_dir, cfg.segment_output_dir, cfg.input_summaries_name, cfg.input_day_meta_name)
    rb, rb_source = _prepare_rulebook(cfg, state, day_dir.name, layout, profile_text)
    results: List[Dict[str, Any]] = []
    current_high_conf: List[Dict[str, Any]] = []
    usage_before = len(state.usage.records)
    for idx, summary in enumerate(summaries):
        exec_seg = build_exec_segment(summary)
        retrieved = retrieve_similar_records(exec_seg, state.memory_records, cfg.retrieve_top_k)
        recent_ctx = build_recent_context(current_high_conf, cfg.recent_context_n)
        out = llm_classify(cfg, exec_seg, retrieved, recent_ctx, layout, profile_text, state.usage, day_dir.name, idx)
        final_top3 = topk_with_ties(out["final_top3"], cfg.top_k)
        final_label = out["final_label"]
        final_margin = margin_from_top2(final_top3)
        results.append(_format_record(day_dir.name, idx, exec_seg, final_label, final_top3, final_margin, out.get("explanation", "LLM only."), retrieved=retrieved, recent_context=recent_ctx, decision_basis=out.get("decision_basis", {})))
        if _is_high_conf(cfg, final_label, final_margin):
            mem = _memory_record(day_dir.name, idx, exec_seg, [], final_label, final_top3, final_margin)
            state.memory_records.append(mem)
            current_high_conf.append(results[-1])
    state.high_conf_records.extend(current_high_conf)
    state.all_day_outputs.append({"day": day_dir.name, "start_time": day_meta.get("start_time"), "activities": cfg.activities, "results": results})
    if cfg.save_per_day_files:
        _save_mode_day_outputs(cfg, state.name, day_dir.name, day_meta, rb.model_dump(), rb_source, results, state.usage.summary())
    return {
        "day": day_dir.name,
        "num_segments": len(results),
        "num_high_conf": len(current_high_conf),
        "rulebook_source": rb_source,
        "llm_calls": len(state.usage.records) - usage_before,
    }


def _run_rulebook_then_calibration_day(cfg: RecognitionConfig, state: ModeState, day_dir, layout: Dict[str, Any], profile_text: str) -> Dict[str, Any]:
    summaries, day_meta = load_day_inputs(day_dir, cfg.segment_output_dir, cfg.input_summaries_name, cfg.input_day_meta_name)
    rb, rb_source = _prepare_rulebook(cfg, state, day_dir.name, layout, profile_text)
    results: List[Dict[str, Any]] = []
    current_high_conf: List[Dict[str, Any]] = []
    usage_before = len(state.usage.records)
    for idx, summary in enumerate(summaries):
        exec_seg = build_exec_segment(summary)
        rule_top3 = score_segment_with_rulebook(exec_seg, rb, cfg.top_k)
        rule_margin = margin_from_top2(rule_top3)
        retrieved = retrieve_similar_records(exec_seg, state.memory_records, cfg.retrieve_top_k)
        recent_ctx = build_recent_context(current_high_conf, cfg.recent_context_n)
        if rule_margin < cfg.calibration_margin:
            out = llm_calibrate(cfg, exec_seg, rule_top3, retrieved, recent_ctx, layout, profile_text, state.usage, day_dir.name, idx)
            if out is not None:
                final_top3 = topk_with_ties(out["final_top3"], cfg.top_k)
                final_label = out["final_label"]
                final_margin = margin_from_top2(final_top3)
                calibrated = True
                explanation = out.get("explanation", "RuleBook + LLM calibration.")
                decision_basis = out.get("decision_basis", {})
            else:
                final_top3 = topk_with_ties(rule_top3, cfg.top_k)
                final_label = final_top3[0]["label"] if final_top3 else "Unknown/Other"
                final_margin = rule_margin
                calibrated = False
                explanation = "Calibration required but LLM was unavailable or returned invalid output; kept RuleBook result."
                decision_basis = {"rule_support": rule_top3, "retrieved_examples": [], "recent_context": []}
        else:
            final_top3 = topk_with_ties(rule_top3, cfg.top_k)
            final_label = final_top3[0]["label"] if final_top3 else "Unknown/Other"
            final_margin = rule_margin
            calibrated = False
            explanation = "High-margin rule decision; calibration skipped."
            decision_basis = {"rule_support": rule_top3, "retrieved_examples": [], "recent_context": []}
        results.append(_format_record(day_dir.name, idx, exec_seg, final_label, final_top3, final_margin, explanation, rulebook_top3=rule_top3, rule_margin=rule_margin, calibrated=calibrated, retrieved=retrieved, recent_context=recent_ctx, decision_basis=decision_basis))
        if _is_high_conf(cfg, final_label, final_margin):
            mem = _memory_record(day_dir.name, idx, exec_seg, rule_top3, final_label, final_top3, final_margin)
            state.memory_records.append(mem)
            current_high_conf.append(results[-1])
    state.high_conf_records.extend(current_high_conf)
    state.all_day_outputs.append({"day": day_dir.name, "start_time": day_meta.get("start_time"), "activities": rb.activities, "results": results})
    if cfg.save_per_day_files:
        _save_mode_day_outputs(cfg, state.name, day_dir.name, day_meta, rb.model_dump(), rb_source, results, state.usage.summary())
    return {
        "day": day_dir.name,
        "num_segments": len(results),
        "num_high_conf": len(current_high_conf),
        "rulebook_source": rb_source,
        "llm_calls": len(state.usage.records) - usage_before,
    }


def _run_single_mode(cfg: RecognitionConfig, mode_name: str, state: ModeState, day_dirs: List[Any], layout: Dict[str, Any], profile_text: str, position: int = 0) -> str:
    runner = {
        "rulebook_only": _run_rulebook_only_day,
        "llm_only": _run_llm_only_day,
        "rulebook_then_llm_calibration": _run_rulebook_then_calibration_day,
    }[mode_name]

    _write_line(f"[{mode_name}] start | days={len(day_dirs)} | rulebook_strategy={cfg.rulebook_strategy} | calibration_margin={cfg.calibration_margin} | high_conf_margin={cfg.high_conf_margin}")

    bar = None
    iterator = day_dirs
    if cfg.show_progress and tqdm is not None:
        bar = tqdm(day_dirs, total=len(day_dirs), desc=mode_name, position=position, leave=True, dynamic_ncols=True)
        iterator = bar

    for day_dir in iterator:
        info = runner(cfg, state, day_dir, layout, profile_text)
        postfix = {
            "day": info["day"],
            "segments": info["num_segments"],
            "high_conf": info["num_high_conf"],
            "llm_calls": info["llm_calls"],
            "rb": info["rulebook_source"],
        }
        if bar is not None:
            bar.set_postfix(postfix)
        if cfg.verbose:
            _write_line(f"[{mode_name}] day {info['day']} done | segments={info['num_segments']} | high_conf={info['num_high_conf']} | llm_calls={info['llm_calls']} | rulebook={info['rulebook_source']}")

    if bar is not None:
        bar.close()

    out_dir = cfg.output_root / mode_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json({"mode": mode_name, "days": state.all_day_outputs}, out_dir / f"{mode_name}_results.json")
    save_json(state.usage.summary(), out_dir / f"{mode_name}_usage_metadata.json")
    _write_line(f"[{mode_name}] finished | output={out_dir}")
    return mode_name


def run_all_modes(cfg: RecognitionConfig) -> List[str]:
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    layout = load_optional_layout(cfg.layout_json_path, cfg.room_types)
    profile_text = load_optional_profile_text(cfg.profile_txt_path)
    day_dirs = list_day_dirs(cfg.root_parent)
    if not day_dirs:
        raise RuntimeError(f"No day folders found under {cfg.root_parent}")

    valid_modes = {"rulebook_only", "llm_only", "rulebook_then_llm_calibration"}
    requested_modes = list(dict.fromkeys(cfg.modes or ["rulebook_only"]))
    bad_modes = [m for m in requested_modes if m not in valid_modes]
    if bad_modes:
        raise ValueError(f"Unsupported recognition modes: {bad_modes}. Valid modes: {sorted(valid_modes)}")

    modes = {
        mode_name: ModeState(mode_name, current_rulebook=default_rulebook(cfg) if cfg.rulebook_strategy == "fixed" else None)
        for mode_name in requested_modes
    }

    save_json({"config": cfg.to_dict(), "num_days": len(day_dirs), "days": [d.name for d in day_dirs]}, cfg.output_root / "run_config.json")

    mode_names = list(modes.keys())
    if cfg.parallel_modes:
        max_workers = min(cfg.max_mode_workers, len(mode_names))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_run_single_mode, cfg, mode_name, modes[mode_name], day_dirs, layout, profile_text, idx): mode_name
                for idx, mode_name in enumerate(mode_names)
            }
            done_names = []
            for fut in as_completed(futures):
                done_names.append(fut.result())
        return mode_names

    for idx, mode_name in enumerate(mode_names):
        _run_single_mode(cfg, mode_name, modes[mode_name], day_dirs, layout, profile_text, idx)
    return mode_names
