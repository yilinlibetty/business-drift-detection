from __future__ import annotations

import argparse
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from drift_detection.llm_support import enrich_with_llm_diagnosis, load_llm_settings
from drift_detection.pipeline import serialize_value
from drift_detection.reporting import render_markdown_report


DEFAULT_INPUT = os.path.join(ROOT_DIR, "outputs", "drift_analysis.json")
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "outputs", "final_drift_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for the upgraded drift analysis pipeline")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-llm", action="store_true", help="Re-run per-drift LLM diagnosis before rendering")
    parser.add_argument("--no-llm", action="store_true", help="Render report without refreshing LLM diagnosis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input analysis JSON not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as handle:
        result = json.load(handle)

    if args.refresh_llm and not args.no_llm:
        settings = load_llm_settings(True)
        drift_points, llm_meta = enrich_with_llm_diagnosis(
            result.get("drift_points", []),
            result.get("global_summary", {}),
            settings,
        )
        result["drift_points"] = drift_points
        result["llm"] = llm_meta
        with open(args.input, "w", encoding="utf-8") as handle:
            json.dump(serialize_value(result), handle, indent=2, ensure_ascii=False)

    report = render_markdown_report(result)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
