from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from drift_detection.evidence import build_evidence_pack, derive_rule_based_tags
from drift_detection.llm_support import enrich_with_llm_diagnosis, load_llm_settings
from drift_detection.pipeline import (
    PipelineConfig,
    augment_evaluation_with_evidence_fidelity,
    build_case_table,
    build_global_summary,
    build_legacy_summary,
    build_legacy_timeline,
    build_score_timeline,
    detect_drift_points,
    evaluate_detection,
    finalize_truth_intervals,
    inject_synthetic_drift,
    load_preprocessed_event_log,
    resolve_threshold,
    serialize_value,
)
from drift_detection.reporting import render_human_review_rubric, render_markdown_report


def parse_args() -> argparse.Namespace:
    defaults = PipelineConfig.from_env()
    parser = argparse.ArgumentParser(description="Run timeline-based business drift detection and reporting")
    parser.add_argument("--analysis-mode", choices=["timeline", "legacy-half-split"], default=defaults.analysis_mode)
    parser.add_argument("--legacy-half-split", action="store_true", help="Use the legacy first-half vs second-half comparison")
    parser.add_argument("--window-size", type=int, default=defaults.window_size)
    parser.add_argument("--step-size", type=int, default=defaults.step_size)
    parser.add_argument("--threshold", type=float, default=defaults.threshold)
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument("--auto-threshold", dest="auto_threshold", action="store_true", default=defaults.auto_threshold)
    threshold_group.add_argument("--fixed-threshold", dest="auto_threshold", action="store_false")
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--drift-metric", choices=["tv", "l1"], default=defaults.drift_metric)
    parser.add_argument("--detection-mode", choices=["structure", "delay", "mixed", "auto"], default=None)
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--llm-enabled", dest="llm_enabled", action="store_true", default=defaults.llm_enabled)
    llm_group.add_argument("--no-llm", dest="llm_enabled", action="store_false")
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--evaluate", action="store_true", help="Compute evaluation metrics when ground truth exists")
    parser.add_argument("--inject-drift", dest="inject_drift", action="store_true", default=defaults.inject_drift)
    parser.add_argument("--no-inject-drift", dest="inject_drift", action="store_false")
    parser.add_argument("--drift-type", choices=["structure", "delay", "mixed"], default=defaults.drift_type)
    parser.add_argument("--drift-segments", type=int, default=defaults.drift_segments)
    parser.add_argument("--drift-segment-ratio", type=float, default=defaults.drift_segment_ratio)
    parser.add_argument("--target-activity", default=defaults.target_activity)
    parser.add_argument("--drift-seed", type=int, default=defaults.drift_seed)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig.from_env()
    config.analysis_mode = "legacy-half-split" if args.legacy_half_split else args.analysis_mode
    config.window_size = args.window_size
    config.step_size = args.step_size
    config.threshold = args.threshold
    config.auto_threshold = args.auto_threshold
    config.top_k = args.top_k
    config.drift_metric = args.drift_metric
    config.llm_enabled = args.llm_enabled
    config.output_dir = args.output_dir
    config.inject_drift = args.inject_drift
    config.drift_type = args.drift_type
    if args.detection_mode is None:
        if config.inject_drift and config.drift_type == "delay":
            config.detection_mode = "delay"
        elif config.inject_drift and config.drift_type == "structure":
            config.detection_mode = "structure"
        else:
            config.detection_mode = config.detection_mode or "mixed"
    else:
        config.detection_mode = "mixed" if args.detection_mode == "auto" else args.detection_mode
    config.drift_segments = max(1, args.drift_segments)
    config.drift_segment_ratio = args.drift_segment_ratio
    config.target_activity = args.target_activity
    config.drift_seed = args.drift_seed
    return config


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serialize_value(payload), handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"[1/6] Loading event log from {config.file_path}")
    df_events = load_preprocessed_event_log(config)
    base_cases = build_case_table(df_events, config)
    print(f"      -> loaded {len(df_events)} events across {len(base_cases)} cases")

    ground_truth_annotations = []
    df_analysis = df_events
    if config.inject_drift:
        print(f"[2/6] Injecting synthetic drift: type={config.drift_type}, segments={config.drift_segments}")
        df_analysis, ground_truth_annotations = inject_synthetic_drift(df_events, base_cases, config)
    else:
        print("[2/6] No synthetic drift injection requested")

    cases = build_case_table(df_analysis, config)
    truth_intervals = finalize_truth_intervals(ground_truth_annotations, cases)

    print(f"[3/6] Building score timeline in {config.analysis_mode} mode")
    if config.analysis_mode == "legacy-half-split":
        timeline = build_legacy_timeline(cases, config)
    else:
        timeline = build_score_timeline(cases, config)

    if timeline:
        resolved_window_size = timeline[0]["reference_end_index"] - timeline[0]["reference_start_index"] + 1
        config.window_size = resolved_window_size
        if len(timeline) > 1:
            config.step_size = timeline[1]["reference_start_index"] - timeline[0]["reference_start_index"]
        elif config.step_size is None:
            config.step_size = resolved_window_size

    threshold, threshold_details = resolve_threshold(timeline, config.threshold, config.auto_threshold)
    drift_points = detect_drift_points(timeline, threshold)
    print(f"      -> detected {len(drift_points)} drift point(s) with threshold {threshold:.4f}")

    print("[4/6] Extracting evidence and rule-based tags")
    for point in drift_points:
        point["evidence"] = build_evidence_pack(point, cases, df_analysis, config)
        point["rule_based_tags"] = derive_rule_based_tags(point, point["evidence"])
        if config.detection_mode == "delay":
            point["rule_based_tags"] = [
                tag for tag in point["rule_based_tags"] if tag["tag"] in {"delay_increase", "loop_increase"}
            ]
        elif config.detection_mode == "structure":
            point["rule_based_tags"] = [
                tag for tag in point["rule_based_tags"] if tag["tag"] != "delay_increase"
            ]

    print("[5/6] Running LLM/fallback diagnosis")
    global_summary = build_global_summary(timeline, drift_points, threshold, threshold_details, config)
    llm_settings = load_llm_settings(config.llm_enabled)
    drift_points, llm_meta = enrich_with_llm_diagnosis(drift_points, global_summary, llm_settings)

    evaluation = None
    if args.evaluate or truth_intervals:
        evaluation = evaluate_detection(timeline, drift_points, truth_intervals, threshold)
        evaluation = augment_evaluation_with_evidence_fidelity(evaluation, drift_points)

    result = {
        "run_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_log": config.file_path,
            "event_count": len(df_analysis),
            "case_count": len(cases),
        },
        "config": serialize_value(config.to_dict()),
        "global_summary": global_summary,
        "score_timeline": timeline,
        "drift_points": drift_points,
        "ground_truth_intervals": truth_intervals,
        "evaluation": evaluation,
        "llm": llm_meta,
    }

    print("[6/6] Writing outputs")
    analysis_json_path = os.path.join(config.output_dir, "drift_analysis.json")
    timeline_csv_path = os.path.join(config.output_dir, "drift_score_timeline.csv")
    report_md_path = os.path.join(config.output_dir, "final_drift_report.md")
    rubric_md_path = os.path.join(config.output_dir, "human_review_rubric.md")
    legacy_json_path = os.path.join(config.output_dir, "legacy_final_report_for_azure.json")

    write_json(analysis_json_path, result)
    timeline_frame = pd.DataFrame(serialize_value(timeline))
    if not timeline_frame.empty:
        timeline_frame["threshold"] = threshold
        timeline_frame["is_drift_window"] = timeline_frame["final_score"] > threshold
    timeline_frame.to_csv(timeline_csv_path, index=False)

    with open(report_md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown_report(result))
    with open(rubric_md_path, "w", encoding="utf-8") as handle:
        handle.write(render_human_review_rubric(result))
    write_json(legacy_json_path, build_legacy_summary(global_summary, drift_points, config))

    print(f"      -> {analysis_json_path}")
    print(f"      -> {timeline_csv_path}")
    print(f"      -> {report_md_path}")
    print(f"      -> {rubric_md_path}")
    print(f"      -> {legacy_json_path}")
    print("Pipeline finished.")


if __name__ == "__main__":
    main()
