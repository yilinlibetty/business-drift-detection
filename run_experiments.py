from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd

from drift_detection.pipeline import PipelineConfig
from run_full_pipeline import run_pipeline


EXPERIMENT_FIELDS = [
    "scenario",
    "seed",
    "score_profile",
    "drift_type",
    "segments",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "mean_detection_delay_cases",
    "taxonomy_hit_rate",
    "predicted_interval_count",
]


SCENARIOS = [
    {"scenario": "structure_1_segment", "drift_type": "structure", "segments": 1},
    {"scenario": "delay_1_segment", "drift_type": "delay", "segments": 1},
    {"scenario": "mixed_1_segment", "drift_type": "mixed", "segments": 1},
    {"scenario": "mixed_2_segments", "drift_type": "mixed", "segments": 2},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible drift-detection ablation experiments")
    parser.add_argument("--output-dir", default=os.path.join("outputs", "experiments"))
    parser.add_argument("--seeds", default="11,42,73", help="Comma-separated integer seeds")
    parser.add_argument(
        "--score-profiles",
        default="trace-duration,multi-view",
        help="Comma-separated scoring profiles: trace-duration,multi-view",
    )
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--mad-multiplier", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_csv_ints(args.seeds)
    score_profiles = _parse_csv_strings(args.score_profiles)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for scenario in SCENARIOS:
        for seed in seeds:
            for score_profile in score_profiles:
                config = _build_experiment_config(args, scenario, seed, score_profile)
                result = run_pipeline(config, evaluate_requested=True, verbose=False)
                rows.append(_result_row(result, scenario, seed, score_profile))

    summary = pd.DataFrame(rows, columns=EXPERIMENT_FIELDS)
    csv_path = os.path.join(args.output_dir, "experiment_summary.csv")
    md_path = os.path.join(args.output_dir, "experiment_summary.md")
    summary.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(_render_markdown_summary(summary))

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def _build_experiment_config(
    args: argparse.Namespace,
    scenario: dict[str, Any],
    seed: int,
    score_profile: str,
) -> PipelineConfig:
    config = PipelineConfig.from_env()
    config.inject_drift = True
    config.llm_enabled = False
    config.drift_type = scenario["drift_type"]
    config.detection_mode = _detection_mode_for(config.drift_type)
    config.drift_segments = int(scenario["segments"])
    config.drift_seed = seed
    config.score_profile = score_profile
    config.window_size = args.window_size
    config.step_size = args.step_size
    config.mad_multiplier = args.mad_multiplier
    config.output_dir = args.output_dir
    return config


def _detection_mode_for(drift_type: str) -> str:
    if drift_type == "delay":
        return "delay"
    if drift_type == "structure":
        return "structure"
    return "mixed"


def _result_row(
    result: dict[str, Any],
    scenario: dict[str, Any],
    seed: int,
    score_profile: str,
) -> dict[str, Any]:
    evaluation = result.get("evaluation") or {}
    return {
        "scenario": scenario["scenario"],
        "seed": seed,
        "score_profile": score_profile,
        "drift_type": scenario["drift_type"],
        "segments": scenario["segments"],
        "precision": evaluation.get("interval_level_precision"),
        "recall": evaluation.get("interval_level_recall"),
        "f1": evaluation.get("interval_level_f1"),
        "false_positive_rate": evaluation.get("false_positive_rate"),
        "mean_detection_delay_cases": evaluation.get("mean_detection_delay_cases"),
        "taxonomy_hit_rate": evaluation.get("cause_taxonomy_hit_rate"),
        "predicted_interval_count": evaluation.get("predicted_interval_count"),
    }


def _render_markdown_summary(summary: pd.DataFrame) -> str:
    table_lines = [
        "| " + " | ".join(EXPERIMENT_FIELDS) + " |",
        "| " + " | ".join("---" for _ in EXPERIMENT_FIELDS) + " |",
    ]
    for row in summary.to_dict(orient="records"):
        table_lines.append("| " + " | ".join(str(row.get(field, "")) for field in EXPERIMENT_FIELDS) + " |")
    lines = [
        "# Drift Detection Experiment Summary",
        "",
        "This table compares `trace-duration` and `multi-view` score profiles on reproducible synthetic drift scenarios.",
        "",
        *table_lines,
        "",
    ]
    return "\n".join(lines)


def _parse_csv_ints(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--seeds must contain at least one integer seed")
    return [int(item) for item in values]


def _parse_csv_strings(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"trace-duration", "multi-view"}
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(f"Unsupported score profile(s): {', '.join(invalid)}")
    return values


if __name__ == "__main__":
    main()
