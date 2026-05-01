from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from processing_and_segmentation.config import SegmentationConfig
from processing_and_segmentation.pipeline import run_pipeline as run_segmentation
from recognition.config import RecognitionConfig
from recognition.evaluator import evaluate_all_modes
from recognition.pipeline import run_all_modes as run_recognition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EchoLIFE segmentation/features and activity recognition from DeltaE [T, D] npz files."
    )
    parser.add_argument("--root-parent", type=Path, required=True,
                        help="Person folder containing numeric day folders, e.g. ../dataset/p1")

    # Segmentation / feature extraction
    parser.add_argument("--deltae-name", default="deltae.npz")
    parser.add_argument("--label-name", default="label.json")
    parser.add_argument("--segment-output-dir", default="echolife_segmentation_out")
    parser.add_argument("--rooms", nargs="+", default=["kitchen", "living", "bedroom", "bathroom", "study", "dining"])
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--no-intermediates", action="store_true")
    parser.add_argument("--skip-segmentation", action="store_true",
                        help="Reuse existing segment_summaries.json/day_meta.json and only run recognition")

    # Recognition
    parser.add_argument("--skip-recognition", action="store_true",
                        help="Only run segmentation/features")
    parser.add_argument("--recognition-output-dir", default="echolife_recognition_out")
    parser.add_argument("--ground-truth-name", default="label.json")
    parser.add_argument("--eval", action="store_true", help="Evaluate against ground-truth labels")
    parser.add_argument("--modes", nargs="+", default=["rulebook_only"],
                        choices=["rulebook_only", "llm_only", "rulebook_then_llm_calibration"])
    parser.add_argument("--rulebook-strategy", default="llm_once",
                        choices=["fixed", "llm_once", "llm_tune_from_history"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--activities", nargs="+", default=None)
    parser.add_argument("--parallel-modes", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_segmentation:
        seg_cfg = SegmentationConfig(
            root_parent=args.root_parent,
            input_deltae_npz_name=args.deltae_name,
            input_json_name=args.label_name,
            output_dir_name=args.segment_output_dir,
            rooms=tuple(args.rooms),
            save_plots=args.plots,
            save_intermediates=not args.no_intermediates,
        )
        print("=" * 80)
        print("Stage 1/2: DeltaE -> segmentation features")
        print("=" * 80)
        run_segmentation(seg_cfg)

    if not args.skip_recognition:
        recog_kwargs = {
            "root_parent": args.root_parent,
            "segment_output_dir": args.segment_output_dir,
            "output_dir_name": args.recognition_output_dir,
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
            recog_kwargs["activities"] = args.activities
        if args.llm_model is not None:
            recog_kwargs["llm_model"] = args.llm_model
        if args.openai_base_url is not None:
            recog_kwargs["openai_base_url"] = args.openai_base_url
        rec_cfg = RecognitionConfig(**recog_kwargs)
        print("=" * 80)
        print("Stage 2/2: segmentation features -> activity recognition")
        print("=" * 80)
        mode_names = run_recognition(rec_cfg)
        if rec_cfg.run_evaluation:
            print("[evaluation] start")
            evaluate_all_modes(rec_cfg, mode_names)
            print("[evaluation] done")


if __name__ == "__main__":
    main()
