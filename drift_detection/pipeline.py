from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import os
import random
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from convert_data import load_event_log


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class PipelineConfig:
    file_path: str
    col_case_id: str
    col_activity: str
    col_timestamp: str
    keep_only_complete: bool = True
    drift_metric: str = "tv"
    detection_mode: str = "mixed"
    top_k: int = 10
    threshold: float = 0.05
    auto_threshold: bool = True
    analysis_mode: str = "timeline"
    window_size: int | None = None
    step_size: int | None = None
    llm_enabled: bool = True
    output_dir: str = "outputs"
    inject_drift: bool = False
    drift_type: str = "structure"
    drift_seed: int = 42
    target_activity: str = "Live Chat"
    drift_segments: int = 1
    drift_segment_ratio: float = 0.12
    legacy_half_split: bool = False

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return cls(
            file_path=os.getenv("EVENT_LOG_PATH", os.path.join(base_dir, "datasets", "finale.csv")),
            col_case_id=os.getenv("COL_CASE_ID", "Case ID"),
            col_activity=os.getenv("COL_ACTIVITY", "Activity"),
            col_timestamp=os.getenv("COL_TIMESTAMP", "Complete Timestamp"),
            keep_only_complete=_bool_env("KEEP_ONLY_COMPLETE", True),
            drift_metric=os.getenv("DRIFT_METRIC", "tv").lower(),
            detection_mode=os.getenv("DETECTION_MODE", os.getenv("DRIFT_TYPE", "mixed")).lower(),
            top_k=int(os.getenv("TOP_K_TRACES", "10")),
            threshold=float(os.getenv("DRIFT_THRESHOLD", "0.05")),
            auto_threshold=not _bool_env("FIXED_THRESHOLD", False),
            analysis_mode=os.getenv("ANALYSIS_MODE", "timeline").lower(),
            window_size=int(os.getenv("WINDOW_SIZE")) if os.getenv("WINDOW_SIZE") else None,
            step_size=int(os.getenv("STEP_SIZE")) if os.getenv("STEP_SIZE") else None,
            llm_enabled=not _bool_env("NO_LLM", False),
            output_dir=os.getenv("OUTPUT_DIR", os.path.join(base_dir, "outputs")),
            inject_drift=_bool_env("INJECT_DRIFT", False),
            drift_type=os.getenv("DRIFT_TYPE", "structure").lower(),
            drift_seed=int(os.getenv("DRIFT_SEED", "42")),
            target_activity=os.getenv("TARGET_ACTIVITY", "Live Chat"),
            drift_segments=int(os.getenv("DRIFT_SEGMENTS", "1")),
            drift_segment_ratio=float(os.getenv("DRIFT_SEGMENT_RATIO", "0.12")),
            legacy_half_split=_bool_env("LEGACY_HALF_SPLIT", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def serialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def isoformat_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def compute_distribution_distance(vec_a, vec_b, metric: str = "tv") -> float:
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    if metric == "l1":
        return float(np.sum(np.abs(a - b)))
    return float(0.5 * np.sum(np.abs(a - b)))


def compute_duration_drift(reference_cases: pd.DataFrame, current_cases: pd.DataFrame) -> tuple[float, float]:
    reference = reference_cases["Duration"].dropna().to_numpy(dtype=float)
    current = current_cases["Duration"].dropna().to_numpy(dtype=float)
    if reference.size == 0 or current.size == 0:
        return 0.0, 0.0
    raw = float(wasserstein_distance(reference, current))
    scale = float(np.median(reference)) if float(np.median(reference)) > 0 else float(np.mean(reference))
    if scale <= 0:
        return raw, raw
    return raw / scale, raw


def combine_drift_scores(trace_score: float, duration_score: float, mode: str = "mixed") -> float:
    if mode == "structure":
        return trace_score
    if mode == "delay":
        return duration_score
    return max(trace_score, duration_score)


def default_window_size(total_cases: int) -> int:
    recommended = min(300, max(100, round(total_cases * 0.1)))
    feasible = max(5, total_cases // 2)
    return min(recommended, feasible)


def default_step_size(window_size: int, total_cases: int) -> int:
    recommended = max(25, window_size // 4)
    feasible = max(1, total_cases - (2 * window_size) + 1)
    return min(recommended, feasible)


def load_preprocessed_event_log(config: PipelineConfig) -> pd.DataFrame:
    df = load_event_log(
        config.file_path,
        config.col_case_id,
        config.col_activity,
        config.col_timestamp,
        keep_only_complete=config.keep_only_complete,
    )
    missing = [col for col in [config.col_case_id, config.col_activity, config.col_timestamp] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    working = df.copy()
    working[config.col_case_id] = working[config.col_case_id].astype(str)
    working[config.col_activity] = working[config.col_activity].astype(str)
    working[config.col_timestamp] = pd.to_datetime(working[config.col_timestamp], errors="coerce")
    working = working.dropna(subset=[config.col_timestamp])
    working = working.sort_values([config.col_case_id, config.col_timestamp]).reset_index(drop=True)
    return working


def build_case_table(df_events: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    grouped = (
        df_events.sort_values([config.col_case_id, config.col_timestamp])
        .groupby(config.col_case_id, sort=False)
        .agg(
            Activities=(config.col_activity, lambda x: tuple(map(str, x))),
            StartTime=(config.col_timestamp, "min"),
            EndTime=(config.col_timestamp, "max"),
            EventCount=(config.col_activity, "size"),
        )
        .reset_index()
    )
    grouped = grouped.rename(columns={config.col_case_id: "CaseID"})
    grouped["Trace"] = grouped["Activities"].apply(lambda items: " -> ".join(items))
    grouped["Duration"] = (grouped["EndTime"] - grouped["StartTime"]).dt.total_seconds() / 60.0
    grouped["RepeatedActivityCount"] = grouped["Activities"].apply(_count_repeated_activities)
    grouped["HasLoop"] = grouped["RepeatedActivityCount"] > 0
    grouped = grouped.sort_values("EndTime").reset_index(drop=True)
    grouped["CaseIndex"] = grouped.index
    return grouped


def _count_repeated_activities(activities: tuple[str, ...]) -> int:
    seen: set[str] = set()
    repeated = 0
    for activity in activities:
        if activity in seen:
            repeated += 1
        else:
            seen.add(activity)
    return repeated


def expected_tags_for_drift_type(drift_type: str) -> list[str]:
    if drift_type == "delay":
        return ["delay_increase"]
    if drift_type == "mixed":
        return ["delay_increase", "path_added", "path_removed_or_skipped_step"]
    return ["path_added", "path_removed_or_skipped_step"]


def _choose_segment_starts(total_cases: int, segment_count: int, segment_length: int) -> list[int]:
    if segment_count <= 1:
        return [max(0, int(total_cases * 0.55) - segment_length // 2)]
    centers = np.linspace(0.45, 0.9, num=segment_count)
    starts = []
    min_gap = max(5, segment_length // 2)
    for center in centers:
        start = int(total_cases * center) - segment_length // 2
        start = max(0, min(start, total_cases - segment_length))
        if starts and start < starts[-1] + min_gap:
            start = min(total_cases - segment_length, starts[-1] + min_gap)
        starts.append(max(0, start))
    return starts


def inject_synthetic_drift(
    df_events: pd.DataFrame,
    cases: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not config.inject_drift:
        return df_events.copy(), []

    random.seed(config.drift_seed)
    np.random.seed(config.drift_seed)

    working = df_events.copy()
    total_cases = len(cases)
    segment_count = max(1, config.drift_segments)
    segment_length = max(20, int(round(total_cases * config.drift_segment_ratio)))
    segment_length = min(segment_length, max(10, total_cases // max(2, segment_count + 1)))
    starts = _choose_segment_starts(total_cases, segment_count, segment_length)

    truth_annotations: list[dict[str, Any]] = []
    for index, start in enumerate(starts, start=1):
        end = min(total_cases, start + segment_length)
        case_slice = cases.iloc[start:end]
        affected_case_ids = case_slice["CaseID"].tolist()
        if not affected_case_ids:
            continue

        annotation = {
            "segment_id": f"GT{index:02d}",
            "drift_type": config.drift_type,
            "expected_tags": expected_tags_for_drift_type(config.drift_type),
            "affected_case_ids": affected_case_ids,
            "start_case_index_original": int(case_slice["CaseIndex"].min()),
            "end_case_index_original": int(case_slice["CaseIndex"].max()),
        }

        if config.drift_type in {"delay", "mixed"}:
            _inject_delay_drift(working, affected_case_ids, config)
        if config.drift_type in {"structure", "mixed"}:
            _inject_structure_drift(working, affected_case_ids, config)

        truth_annotations.append(annotation)

    working = working.sort_values([config.col_case_id, config.col_timestamp]).reset_index(drop=True)
    return working, truth_annotations


def _inject_delay_drift(df_events: pd.DataFrame, affected_case_ids: list[str], config: PipelineConfig) -> None:
    for case_id in affected_case_ids:
        case_mask = df_events[config.col_case_id] == case_id
        case_rows = df_events.loc[case_mask].sort_values(config.col_timestamp)
        if case_rows.empty:
            continue
        target_mask = case_rows[config.col_activity] == config.target_activity
        if target_mask.any():
            shift_start = case_rows.loc[target_mask, config.col_timestamp].min()
        else:
            midpoint = max(0, len(case_rows) // 2 - 1)
            shift_start = case_rows.iloc[midpoint][config.col_timestamp]
        shift_mask = case_mask & (df_events[config.col_timestamp] >= shift_start)
        case_duration_minutes = max(
            1.0,
            float((case_rows[config.col_timestamp].max() - case_rows[config.col_timestamp].min()).total_seconds() / 60.0),
        )
        delay_cap = 4320
        min_delay = min(max(180, int(case_duration_minutes * 0.25)), delay_cap)
        max_delay = min(max(min_delay + 60, int(case_duration_minutes * 0.75)), delay_cap)
        if max_delay <= min_delay:
            delay_minutes = int(min_delay)
        else:
            delay_minutes = int(np.random.randint(min_delay, max_delay + 1))
        df_events.loc[shift_mask, config.col_timestamp] = (
            df_events.loc[shift_mask, config.col_timestamp] + pd.to_timedelta(delay_minutes, unit="m")
        )


def _inject_structure_drift(df_events: pd.DataFrame, affected_case_ids: list[str], config: PipelineConfig) -> None:
    drop_indices: list[int] = []
    for case_id in affected_case_ids:
        case_rows = df_events.loc[df_events[config.col_case_id] == case_id].sort_values(config.col_timestamp)
        if len(case_rows) <= 1:
            continue
        candidate_indices = list(case_rows.index[1:-1]) or list(case_rows.index[:1])
        if not candidate_indices:
            continue
        drop_indices.append(int(np.random.choice(candidate_indices)))
    if drop_indices:
        df_events.drop(index=drop_indices, inplace=True)


def finalize_truth_intervals(
    annotations: list[dict[str, Any]],
    cases: pd.DataFrame,
) -> list[dict[str, Any]]:
    if not annotations:
        return []

    case_index_map = {row["CaseID"]: int(row["CaseIndex"]) for _, row in cases.iterrows()}
    finalized: list[dict[str, Any]] = []
    for annotation in annotations:
        indices = [case_index_map[case_id] for case_id in annotation["affected_case_ids"] if case_id in case_index_map]
        if not indices:
            continue
        start_index = min(indices)
        end_index = max(indices)
        finalized.append(
            {
                "segment_id": annotation["segment_id"],
                "drift_type": annotation["drift_type"],
                "expected_tags": annotation["expected_tags"],
                "start_case_index": start_index,
                "end_case_index": end_index,
                "start_time": isoformat_or_none(cases.iloc[start_index]["StartTime"]),
                "end_time": isoformat_or_none(cases.iloc[end_index]["EndTime"]),
                "affected_case_count": len(indices),
            }
        )
    return finalized


def build_score_timeline(cases: pd.DataFrame, config: PipelineConfig) -> list[dict[str, Any]]:
    total_cases = len(cases)
    if total_cases < 2:
        return []

    window_size = config.window_size or default_window_size(total_cases)
    step_size = config.step_size or default_step_size(window_size, total_cases)
    if total_cases < 2 * window_size:
        window_size = max(5, total_cases // 2)
        step_size = max(1, min(step_size, max(1, window_size // 2)))

    score_rows: list[dict[str, Any]] = []
    window_id = 1
    for start in range(0, total_cases - 2 * window_size + 1, step_size):
        reference = cases.iloc[start:start + window_size]
        current = cases.iloc[start + window_size:start + 2 * window_size]
        trace_score = _compute_trace_score(reference, current, config.drift_metric)
        duration_score, duration_score_raw = compute_duration_drift(reference, current)
        final_score_raw = combine_drift_scores(trace_score, duration_score, config.detection_mode)
        score_rows.append(
            {
                "window_id": f"W{window_id:04d}",
                "window_index": window_id - 1,
                "reference_start_index": int(reference.iloc[0]["CaseIndex"]),
                "reference_end_index": int(reference.iloc[-1]["CaseIndex"]),
                "current_start_index": int(current.iloc[0]["CaseIndex"]),
                "current_end_index": int(current.iloc[-1]["CaseIndex"]),
                "reference_start_time": isoformat_or_none(reference.iloc[0]["StartTime"]),
                "reference_end_time": isoformat_or_none(reference.iloc[-1]["EndTime"]),
                "current_start_time": isoformat_or_none(current.iloc[0]["StartTime"]),
                "current_end_time": isoformat_or_none(current.iloc[-1]["EndTime"]),
                "trace_score": float(trace_score),
                "duration_score": float(duration_score),
                "duration_score_raw": float(duration_score_raw),
                "final_score_raw": float(final_score_raw),
            }
        )
        window_id += 1

    smoothed_scores = _median_smooth([row["final_score_raw"] for row in score_rows])
    for row, score in zip(score_rows, smoothed_scores):
        row["final_score"] = float(score)
    return score_rows


def _compute_trace_score(reference: pd.DataFrame, current: pd.DataFrame, metric: str) -> float:
    all_traces = sorted(set(reference["Trace"].unique()) | set(current["Trace"].unique()))
    if not all_traces:
        return 0.0
    trace_map = {trace: idx for idx, trace in enumerate(all_traces)}

    def get_vector(cases_df: pd.DataFrame) -> np.ndarray:
        counts = cases_df["Trace"].value_counts(normalize=True)
        vector = np.zeros(len(all_traces))
        for trace, frequency in counts.items():
            vector[trace_map[trace]] = frequency
        return vector

    return compute_distribution_distance(get_vector(reference), get_vector(current), metric=metric)


def _median_smooth(values: list[float]) -> list[float]:
    if not values:
        return []
    smoothed: list[float] = []
    for idx in range(len(values)):
        left = max(0, idx - 1)
        right = min(len(values), idx + 2)
        smoothed.append(float(np.median(values[left:right])))
    return smoothed


def build_legacy_timeline(cases: pd.DataFrame, config: PipelineConfig) -> list[dict[str, Any]]:
    if len(cases) < 2:
        return []
    mid = len(cases) // 2
    reference = cases.iloc[:mid]
    current = cases.iloc[mid:]
    if reference.empty or current.empty:
        return []
    trace_score = _compute_trace_score(reference, current, config.drift_metric)
    duration_score, duration_score_raw = compute_duration_drift(reference, current)
    final_score = combine_drift_scores(trace_score, duration_score, config.detection_mode)
    return [
        {
            "window_id": "W0001",
            "window_index": 0,
            "reference_start_index": int(reference.iloc[0]["CaseIndex"]),
            "reference_end_index": int(reference.iloc[-1]["CaseIndex"]),
            "current_start_index": int(current.iloc[0]["CaseIndex"]),
            "current_end_index": int(current.iloc[-1]["CaseIndex"]),
            "reference_start_time": isoformat_or_none(reference.iloc[0]["StartTime"]),
            "reference_end_time": isoformat_or_none(reference.iloc[-1]["EndTime"]),
            "current_start_time": isoformat_or_none(current.iloc[0]["StartTime"]),
            "current_end_time": isoformat_or_none(current.iloc[-1]["EndTime"]),
            "trace_score": float(trace_score),
            "duration_score": float(duration_score),
            "duration_score_raw": float(duration_score_raw),
            "final_score_raw": float(final_score),
            "final_score": float(final_score),
        }
    ]


def resolve_threshold(
    timeline: list[dict[str, Any]],
    base_threshold: float,
    auto_threshold: bool,
) -> tuple[float, dict[str, Any]]:
    if not timeline:
        return base_threshold, {"source": "configured", "configured_threshold": base_threshold}

    scores = np.asarray([row["final_score"] for row in timeline], dtype=float)
    if not auto_threshold:
        return base_threshold, {"source": "configured", "configured_threshold": base_threshold}

    median_score = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median_score)))
    auto_candidate = median_score + 3.0 * mad
    threshold = max(base_threshold, auto_candidate)
    return threshold, {
        "source": "auto",
        "configured_threshold": base_threshold,
        "median_score": median_score,
        "mad_score": mad,
        "auto_candidate": auto_candidate,
    }


def detect_drift_points(
    timeline: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    if not timeline:
        return []

    active_indices = [idx for idx, row in enumerate(timeline) if row["final_score"] > threshold]
    if not active_indices:
        return []

    groups: list[list[int]] = [[active_indices[0]]]
    for idx in active_indices[1:]:
        if idx - groups[-1][-1] <= 2:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    merge_gap_cases = _resolve_interval_merge_gap_cases(timeline)
    merged_groups: list[list[int]] = [groups[0][:]]
    for group in groups[1:]:
        previous_group = merged_groups[-1]
        previous_end = timeline[previous_group[-1]]["current_end_index"]
        next_start = timeline[group[0]]["current_start_index"]
        gap_cases = next_start - previous_end
        if gap_cases <= merge_gap_cases:
            merged_groups[-1].extend(group)
        else:
            merged_groups.append(group[:])

    drift_points: list[dict[str, Any]] = []
    for point_index, group in enumerate(merged_groups, start=1):
        windows = [timeline[idx] for idx in group]
        peak_window = max(windows, key=lambda item: (item["final_score"], item["window_index"]))
        first_window = windows[0]
        last_window = windows[-1]
        drift_points.append(
            {
                "id": f"DP{point_index:02d}",
                "interval_start_time": first_window["current_start_time"],
                "interval_end_time": last_window["current_end_time"],
                "interval_start_case_index": first_window["current_start_index"],
                "interval_end_case_index": last_window["current_end_index"],
                "peak_time": peak_window["current_end_time"],
                "peak_score": round(float(peak_window["final_score"]), 4),
                "trace_score": round(float(peak_window["trace_score"]), 4),
                "duration_score": round(float(peak_window["duration_score"]), 4),
                "duration_score_raw": round(float(peak_window["duration_score_raw"]), 4),
                "threshold_excess": round(float(peak_window["final_score"] - threshold), 4),
                "reference_window": {
                    "start_case_index": peak_window["reference_start_index"],
                    "end_case_index": peak_window["reference_end_index"],
                    "start_time": peak_window["reference_start_time"],
                    "end_time": peak_window["reference_end_time"],
                },
                "current_window": {
                    "start_case_index": peak_window["current_start_index"],
                    "end_case_index": peak_window["current_end_index"],
                    "start_time": peak_window["current_start_time"],
                    "end_time": peak_window["current_end_time"],
                },
                "detection_delay_proxy": {
                    "cases_to_peak": int(peak_window["current_start_index"] - first_window["current_start_index"]),
                    "hours_to_peak": _hours_between(first_window["current_start_time"], peak_window["current_end_time"]),
                },
                "window_ids": [window["window_id"] for window in windows],
                "evidence": {},
                "rule_based_tags": [],
                "llm_diagnosis": None,
            }
        )
    return drift_points


def _resolve_interval_merge_gap_cases(timeline: list[dict[str, Any]]) -> int:
    if not timeline:
        return 0
    window_size = int(timeline[0]["current_end_index"] - timeline[0]["current_start_index"] + 1)
    if len(timeline) > 1:
        step_size = int(timeline[1]["current_start_index"] - timeline[0]["current_start_index"])
    else:
        step_size = window_size
    return max((window_size // 2) + 1, (step_size * 2) + 1)


def _hours_between(start_time: str | None, end_time: str | None) -> float | None:
    if not start_time or not end_time:
        return None
    start = pd.Timestamp(start_time)
    end = pd.Timestamp(end_time)
    return round(float((end - start).total_seconds() / 3600.0), 2)


def build_global_summary(
    timeline: list[dict[str, Any]],
    drift_points: list[dict[str, Any]],
    threshold: float,
    threshold_details: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    scores = [row["final_score"] for row in timeline]
    peak_score = max(scores) if scores else 0.0
    return {
        "status": "DRIFT DETECTED" if drift_points else "STABLE",
        "analysis_mode": config.analysis_mode,
        "window_count": len(timeline),
        "drift_point_count": len(drift_points),
        "threshold": round(float(threshold), 4),
        "threshold_details": serialize_value(threshold_details),
        "peak_score": round(float(peak_score), 4),
        "mean_score": round(float(np.mean(scores)) if scores else 0.0, 4),
        "std_score": round(float(np.std(scores)) if scores else 0.0, 4),
        "detection_mode": config.detection_mode,
        "drift_metric": config.drift_metric,
    }


def overlap_exists(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def evaluate_detection(
    timeline: list[dict[str, Any]],
    drift_points: list[dict[str, Any]],
    truth_intervals: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any] | None:
    if not truth_intervals:
        return None

    tp = fp = fn = tn = 0
    for row in timeline:
        predicted = row["final_score"] > threshold
        truth = any(
            overlap_exists(
                row["current_start_index"],
                row["current_end_index"],
                truth_interval["start_case_index"],
                truth_interval["end_case_index"],
            )
            for truth_interval in truth_intervals
        )
        if predicted and truth:
            tp += 1
        elif predicted and not truth:
            fp += 1
        elif not predicted and truth:
            fn += 1
        else:
            tn += 1

    matched_truth: set[int] = set()
    matched_pred: set[int] = set()
    detection_delays: list[int] = []
    taxonomy_hits = 0
    candidate_pairs: list[tuple[int, int, int, int]] = []

    for pred_idx, point in enumerate(drift_points):
        for truth_idx, truth_interval in enumerate(truth_intervals):
            if overlap_exists(
                point["interval_start_case_index"],
                point["interval_end_case_index"],
                truth_interval["start_case_index"],
                truth_interval["end_case_index"],
            ):
                overlap_size = min(point["interval_end_case_index"], truth_interval["end_case_index"]) - max(
                    point["interval_start_case_index"], truth_interval["start_case_index"]
                ) + 1
                start_distance = abs(point["interval_start_case_index"] - truth_interval["start_case_index"])
                candidate_pairs.append((pred_idx, truth_idx, overlap_size, start_distance))

    candidate_pairs.sort(key=lambda item: (-item[2], item[3], item[0], item[1]))
    for pred_idx, truth_idx, _overlap_size, _start_distance in candidate_pairs:
        if pred_idx in matched_pred or truth_idx in matched_truth:
            continue
        matched_pred.add(pred_idx)
        matched_truth.add(truth_idx)
        point = drift_points[pred_idx]
        truth_interval = truth_intervals[truth_idx]
        detection_delays.append(max(0, point["interval_start_case_index"] - truth_interval["start_case_index"]))
        predicted_tags = {tag["tag"] for tag in point.get("rule_based_tags", [])}
        if predicted_tags.intersection(set(truth_interval.get("expected_tags", []))):
            taxonomy_hits += 1

    interval_tp = len(matched_truth)
    interval_fp = len(drift_points) - len(matched_pred)
    interval_fn = len(truth_intervals) - len(matched_truth)

    precision = interval_tp / (interval_tp + interval_fp) if (interval_tp + interval_fp) else 0.0
    recall = interval_tp / (interval_tp + interval_fn) if (interval_tp + interval_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "window_level_confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "interval_level_precision": round(float(precision), 4),
        "interval_level_recall": round(float(recall), 4),
        "interval_level_f1": round(float(f1), 4),
        "false_positive_rate": round(float(false_positive_rate), 4),
        "mean_detection_delay_cases": round(float(np.mean(detection_delays)) if detection_delays else 0.0, 2),
        "cause_taxonomy_hit_rate": round(taxonomy_hits / len(truth_intervals), 4) if truth_intervals else None,
        "truth_interval_count": len(truth_intervals),
        "predicted_interval_count": len(drift_points),
    }


def augment_evaluation_with_evidence_fidelity(
    evaluation: dict[str, Any] | None,
    drift_points: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if evaluation is None:
        return None

    total_references = 0
    valid_references = 0
    cause_entries = 0
    cause_entries_with_evidence = 0

    for point in drift_points:
        valid_ids = set(point.get("evidence", {}).get("evidence_ids", []))
        diagnosis = point.get("llm_diagnosis") or {}
        for cause in diagnosis.get("candidate_causes", []):
            cause_entries += 1
            evidence_ids = cause.get("evidence_ids", []) or []
            if evidence_ids:
                cause_entries_with_evidence += 1
            total_references += len(evidence_ids)
            valid_references += sum(1 for evidence_id in evidence_ids if evidence_id in valid_ids)

    evaluation["evidence_fidelity"] = {
        "valid_reference_ratio": round(valid_references / total_references, 4) if total_references else None,
        "cause_entries_with_evidence_ratio": (
            round(cause_entries_with_evidence / cause_entries, 4) if cause_entries else None
        ),
    }
    return evaluation


def build_legacy_summary(
    global_summary: dict[str, Any],
    drift_points: list[dict[str, Any]],
    config: PipelineConfig,
) -> dict[str, Any]:
    strongest = drift_points[0] if drift_points else None
    return {
        "status": global_summary["status"],
        "drift_score": strongest["peak_score"] if strongest else 0.0,
        "trace_drift_score": strongest["trace_score"] if strongest else 0.0,
        "duration_drift_score": strongest["duration_score"] if strongest else 0.0,
        "duration_drift_score_raw": strongest["duration_score_raw"] if strongest else 0.0,
        "drift_metric": config.drift_metric,
        "detection_mode": config.detection_mode,
        "detection_threshold": global_summary["threshold"],
        "analysis": {
            "drift_point_count": len(drift_points),
            "strongest_drift_point": strongest["id"] if strongest else None,
        },
    }
