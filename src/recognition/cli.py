from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pformat

from .config import RecognitionConfig
from .evaluator import evaluate_all_modes
from .pipeline import run_all_modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EchoLIFE activity recognition from segmentation feature summaries."
    )
    parser.add_argument("--root-parent", type=Path, required=True,
                        help="Person folder containing numeric day folders, e.g. ../dataset/p1")
    parser.add_argument("--segment-output-dir", default="echolife_segmentation_out",
                        help="Segmentation output folder inside each day folder")
    parser.add_argument("--summaries-name", default="segment_summaries.json")
    parser.add_argument("--day-meta-name", default="day_meta.json")
    parser.add_argument("--output-dir-name", default="echolife_recognition_out")
    parser.add_argument("--ground-truth-name", default="label.json")
    parser.add_argument("--eval", action="store_true", help="Evaluate against ground-truth labels")

    parser.add_argument("--modes", nargs="+", default=["rulebook_only"],
                        choices=["rulebook_only", "llm_only", "rulebook_then_llm_calibration"],
                        help="Recognition modes to run")
    parser.add_argument("--rulebook-strategy", default="fixed",
                        choices=["fixed", "llm_once", "llm_tune_from_history"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--activities", nargs="+", default=None,
                        help="Optional subset of activity labels. Unknown/Other is kept automatically.")
    parser.add_argument("--rooms", nargs="+", default=["kitchen", "living", "bedroom", "bathroom","study","dining"])
    parser.add_argument("--parallel-modes", action="store_true", help="Run modes in parallel")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RecognitionConfig:
    cfg_kwargs = {
        "root_parent": args.root_parent,
        "segment_output_dir": args.segment_output_dir,
        "input_summaries_name": args.summaries_name,
        "input_day_meta_name": args.day_meta_name,
        "output_dir_name": args.output_dir_name,
        "ground_truth_name": args.ground_truth_name,
        "run_evaluation": args.eval,
        "modes": args.modes,
        "rulebook_strategy": args.rulebook_strategy,
        "room_types": args.rooms,
        "parallel_modes": args.parallel_modes,
        "show_progress": not args.no_progress,
        "verbose": not args.quiet,
    }
    if args.activities is not None:
        cfg_kwargs["activities"] = args.activities
    if args.llm_model is not None:
        cfg_kwargs["llm_model"] = args.llm_model
    if args.openai_base_url is not None:
        cfg_kwargs["openai_base_url"] = args.openai_base_url
    return RecognitionConfig(**cfg_kwargs)


def print_banner(cfg: RecognitionConfig) -> None:
    summary = {
        "root_parent": str(cfg.root_parent),
        "segment_output_dir": cfg.segment_output_dir,
        "output_root": str(cfg.output_root),
        "modes": cfg.modes,
        "rulebook_strategy": cfg.rulebook_strategy,
        "llm_model": cfg.llm_model,
        "run_evaluation": cfg.run_evaluation,
        "activities": list(cfg.activities),
    }
    print("=" * 80)
    print("EchoLIFE recognition")
    print("=" * 80)
    print(pformat(summary, sort_dicts=False))
    print("=" * 80)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    print_banner(cfg)
    mode_names = run_all_modes(cfg)
    if cfg.run_evaluation:
        print("[evaluation] start")
        evaluate_all_modes(cfg, mode_names)
        print("[evaluation] done")


if __name__ == "__main__":
    main()
