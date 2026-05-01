from __future__ import annotations

import argparse
from pathlib import Path

from .config import SegmentationConfig
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EchoLIFE segmentation pipeline from precomputed DeltaE [T, D] npz files.")
    parser.add_argument("--root-parent", type=Path, required=True, help="Person folder containing numeric day folders, e.g. ../dataset/p1")
    parser.add_argument("--deltae-name", default="deltae.npz", help="DeltaE npz filename inside each day folder")
    parser.add_argument("--label-name", default="label.json", help="Label json filename inside each day folder")
    parser.add_argument("--output-dir-name", default="echolife_segmentation_out", help="Output folder name inside each day folder")
    parser.add_argument("--rooms", nargs="+", default=["kitchen", "living", "bedroom", "bathroom", "study", "dining"], help="Room keys to use from the DeltaE npz")
    parser.add_argument("--plots", action="store_true", help="Save diagnostic png plots")
    parser.add_argument("--no-intermediates", action="store_true", help="Do not save intermediate npz files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SegmentationConfig(
        root_parent=args.root_parent,
        input_deltae_npz_name=args.deltae_name,
        input_json_name=args.label_name,
        output_dir_name=args.output_dir_name,
        rooms=tuple(args.rooms),
        save_plots=args.plots,
        save_intermediates=not args.no_intermediates,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
