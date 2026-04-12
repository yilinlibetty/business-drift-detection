from __future__ import annotations

import argparse
import json
from pathlib import Path

from drift_detection.visualization import save_analysis_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export drift analysis figures from drift_analysis.json")
    parser.add_argument("--input", default="outputs/drift_analysis.json", help="Path to drift_analysis.json")
    parser.add_argument("--output-dir", default="outputs/figures", help="Directory for exported figures")
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=["png", "pdf", "svg"],
        help="Output format. Repeat for multiple formats. Default: png",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI for raster outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Analysis JSON not found: {input_path}")

    with input_path.open(encoding="utf-8") as handle:
        result = json.load(handle)

    saved_paths = save_analysis_figures(
        result,
        output_dir=args.output_dir,
        formats=tuple(args.formats or ["png"]),
        dpi=args.dpi,
    )

    print(f"Exported {len(saved_paths)} figure file(s) to {Path(args.output_dir).resolve()}")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
