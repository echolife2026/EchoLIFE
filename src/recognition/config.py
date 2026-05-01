from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal


@dataclass
class RecognitionConfig:
    root_parent: Path = Path("../dataset/output/p1")
    segment_output_dir: str = "echolife_segmentation_out"
    input_summaries_name: str = "segment_summaries.json"
    input_day_meta_name: str = "day_meta.json"
    output_dir_name: str = "echolife_recognition_out"
    optional_layout_json_name: str = "layout.json"
    optional_profile_txt_name: str = "profile.txt"
    ground_truth_name: str = "label.json"
    eval_dir_name: str = "evaluation"
    eval_summary_name: str = "metrics_summary.json"
    eval_daily_csv_name: str = "metrics_daily.csv"

    activities: List[str] = field(default_factory=lambda: [
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
    ])
    room_types: List[str] = field(default_factory=lambda: [
        "kitchen", "living", "bedroom", "bathroom", "study", "dining"
    ])

    top_k: int = 3
    retrieve_top_k: int = 5
    recent_context_n: int = 5
    calibration_margin: float = 0.10
    high_conf_margin: float = 0.20
    exclude_memory_labels: List[str] = field(default_factory=lambda: [""])

    modes: List[str] = field(default_factory=lambda: ["rulebook_only"] )
    rulebook_strategy: Literal["fixed", "llm_once", "llm_tune_from_history"] = "fixed"
    use_llm_for_rulebook: bool = True
    use_llm_for_llm_only: bool = True
    use_llm_for_calibration: bool = True
    llm_model: str = os.environ.get("ECHOLIFE_LLM_MODEL")
    llm_temperature: float = 0.2
    rulebook_max_retries: int = 3
    openai_api_key: str | None = os.environ.get("OPENAI_API_KEY")
    openai_base_url: str | None = os.environ.get("OPENAI_API_BASE")

    save_per_day_files: bool = True
    run_evaluation: bool = False
    frame_rate_hz: float = 10.0
    boundary_tol_sec: float = 60.0



    parallel_modes: bool = True
    max_mode_workers: int = 3
    show_progress: bool = True
    verbose: bool = True



    def __post_init__(self) -> None:
        from .constants import ensure_activity_subset
        self.activities = ensure_activity_subset(self.activities)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_parent": str(self.root_parent),
            "segment_output_dir": self.segment_output_dir,
            "input_summaries_name": self.input_summaries_name,
            "input_day_meta_name": self.input_day_meta_name,
            "output_dir_name": self.output_dir_name,
            "optional_layout_json_name": self.optional_layout_json_name,
            "optional_profile_txt_name": self.optional_profile_txt_name,
            "ground_truth_name": self.ground_truth_name,
            "eval_dir_name": self.eval_dir_name,
            "eval_summary_name": self.eval_summary_name,
            "eval_daily_csv_name": self.eval_daily_csv_name,
            "activities": list(self.activities),
            "room_types": list(self.room_types),
            "top_k": self.top_k,
            "retrieve_top_k": self.retrieve_top_k,
            "recent_context_n": self.recent_context_n,
            "calibration_margin": self.calibration_margin,
            "high_conf_margin": self.high_conf_margin,
            "exclude_memory_labels": list(self.exclude_memory_labels),
            "modes": list(self.modes),
            "rulebook_strategy": self.rulebook_strategy,
            "use_llm_for_rulebook": self.use_llm_for_rulebook,
            "use_llm_for_llm_only": self.use_llm_for_llm_only,
            "use_llm_for_calibration": self.use_llm_for_calibration,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "rulebook_max_retries": self.rulebook_max_retries,
            "openai_base_url": self.openai_base_url,
            "save_per_day_files": self.save_per_day_files,
            "run_evaluation": self.run_evaluation,
            "frame_rate_hz": self.frame_rate_hz,
            "boundary_tol_sec": self.boundary_tol_sec,
            "parallel_modes": self.parallel_modes,
            "max_mode_workers": self.max_mode_workers,
            "show_progress": self.show_progress,
            "verbose": self.verbose,
        }

    @property
    def output_root(self) -> Path:
        return self.root_parent / self.output_dir_name

    @property
    def layout_json_path(self) -> Path:
        return self.root_parent / self.optional_layout_json_name

    @property
    def profile_txt_path(self) -> Path:
        return self.root_parent / self.optional_profile_txt_name
