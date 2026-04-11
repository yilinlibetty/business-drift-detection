from __future__ import annotations

import pathlib
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .pipeline import ATTRIBUTE_SCORE_CANDIDATES, PipelineConfig, _extract_transitions, isoformat_or_none

_RULES_PATH = pathlib.Path(__file__).parent.parent / "config" / "tagging_rules.yaml"
_REQUIRED_CONFIDENCE_KEYS = {
    "path_added",
    "path_removed_with_shortening",
    "path_removed_without_shortening",
    "delay_increase",
    "loop_increase",
    "handoff_or_escalation_increase",
    "case_mix_shift_attribute",
    "case_mix_shift_fallback",
}
_REQUIRED_THRESHOLD_KEYS = {
    "path_added_min_delta",
    "path_added_strong_delta",
    "path_removed_min_delta",
    "delay_median_min_minutes",
    "delay_p90_min_minutes",
    "duration_score_signal_min",
    "loop_rate_min_delta",
    "loop_repeated_activity_min_delta",
    "escalation_min_delta",
    "attribute_shift_min_delta",
}


def _load_tagging_rules(path: pathlib.Path = _RULES_PATH) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Tagging rule config not found: {path}") from exc
    _validate_tagging_rules(payload, str(path))
    return payload


def _validate_tagging_rules(payload: Any, source: str = "tagging rules") -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must be a mapping.")

    required_sections = {"confidence", "thresholds", "escalation_keywords"}
    missing_sections = sorted(required_sections - set(payload))
    if missing_sections:
        raise ValueError(f"{source} missing required section(s): {', '.join(missing_sections)}")

    if not isinstance(payload["confidence"], dict):
        raise ValueError(f"{source} section 'confidence' must be a mapping.")
    if not isinstance(payload["thresholds"], dict):
        raise ValueError(f"{source} section 'thresholds' must be a mapping.")
    if not isinstance(payload["escalation_keywords"], list):
        raise ValueError(f"{source} section 'escalation_keywords' must be a list.")

    missing_confidence = sorted(_REQUIRED_CONFIDENCE_KEYS - set(payload["confidence"]))
    if missing_confidence:
        raise ValueError(f"{source} missing confidence key(s): {', '.join(missing_confidence)}")

    missing_thresholds = sorted(_REQUIRED_THRESHOLD_KEYS - set(payload["thresholds"]))
    if missing_thresholds:
        raise ValueError(f"{source} missing threshold key(s): {', '.join(missing_thresholds)}")


_TAGGING_RULES: dict = _load_tagging_rules()
_CONF = _TAGGING_RULES["confidence"]
_THR = _TAGGING_RULES["thresholds"]
_ESCALATION_KEYWORDS: set[str] = set(_TAGGING_RULES["escalation_keywords"])




def build_evidence_pack(
    drift_point: dict[str, Any],
    cases: pd.DataFrame,
    df_events: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, Any]:
    reference_cases = _slice_case_window(cases, drift_point["reference_window"])
    current_cases = _slice_case_window(cases, drift_point["current_window"])
    reference_case_ids = set(reference_cases["CaseID"].tolist())
    current_case_ids = set(current_cases["CaseID"].tolist())
    reference_events = df_events[df_events[config.col_case_id].isin(reference_case_ids)].copy()
    current_events = df_events[df_events[config.col_case_id].isin(current_case_ids)].copy()

    evidence_ids: list[str] = []
    evidence_index: dict[str, str] = {}
    counter = 1

    def register(payload: dict[str, Any], label: str) -> dict[str, Any]:
        nonlocal counter
        evidence_id = f"{drift_point['id']}-E{counter:02d}"
        counter += 1
        payload["evidence_id"] = evidence_id
        evidence_ids.append(evidence_id)
        evidence_index[evidence_id] = label
        return payload

    top_increased_traces, top_decreased_traces = _trace_deltas(reference_cases, current_cases, config.top_k, register)
    top_changed_transitions = _transition_deltas(reference_cases, current_cases, config.top_k, register)
    activity_frequency_deltas = _activity_deltas(reference_cases, current_cases, config.top_k, register)
    rework_or_loop_rate_delta = _loop_delta(reference_cases, current_cases, register)
    duration_stats_delta = _duration_delta(reference_cases, current_cases, register)
    attribute_distribution_deltas = _attribute_deltas(reference_events, current_events, config.top_k, register)
    score_contribution = _score_contribution(drift_point)

    return {
        "case_count": {"reference": len(reference_cases), "current": len(current_cases)},
        "window_time_range": {
            "reference": {
                "start_time": isoformat_or_none(reference_cases.iloc[0]["StartTime"]) if not reference_cases.empty else None,
                "end_time": isoformat_or_none(reference_cases.iloc[-1]["EndTime"]) if not reference_cases.empty else None,
            },
            "current": {
                "start_time": isoformat_or_none(current_cases.iloc[0]["StartTime"]) if not current_cases.empty else None,
                "end_time": isoformat_or_none(current_cases.iloc[-1]["EndTime"]) if not current_cases.empty else None,
            },
        },
        "top_increased_traces": top_increased_traces,
        "top_decreased_traces": top_decreased_traces,
        "top_changed_transitions": top_changed_transitions,
        "activity_frequency_deltas": activity_frequency_deltas,
        "rework_or_loop_rate_delta": rework_or_loop_rate_delta,
        "duration_stats_delta": duration_stats_delta,
        "attribute_distribution_deltas": attribute_distribution_deltas,
        "score_contribution": score_contribution,
        "evidence_ids": evidence_ids,
        "evidence_index": evidence_index,
    }


def _slice_case_window(cases: pd.DataFrame, window_meta: dict[str, Any]) -> pd.DataFrame:
    start_idx = int(window_meta["start_case_index"])
    end_idx = int(window_meta["end_case_index"])
    return cases.iloc[start_idx:end_idx + 1].copy()


def _score_contribution(drift_point: dict[str, Any]) -> dict[str, Any]:
    return {
        "score_profile": drift_point.get("score_profile"),
        "dominant_signal": drift_point.get("dominant_signal"),
        "core_score": drift_point.get("core_score"),
        "trace_score": drift_point.get("trace_score"),
        "transition_score": drift_point.get("transition_score"),
        "duration_score": drift_point.get("duration_score"),
        "loop_score": drift_point.get("loop_score"),
        "attribute_score": drift_point.get("attribute_score"),
    }


def _trace_deltas(
    reference_cases: pd.DataFrame,
    current_cases: pd.DataFrame,
    top_k: int,
    register,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _distribution_delta_rows(
        reference_cases["Trace"].value_counts(),
        current_cases["Trace"].value_counts(),
        "trace",
    )
    increased = []
    decreased = []
    for row in rows:
        entry = register(
            row,
            f"Trace {row['trace']}: {row['reference_freq']:.3f} -> {row['current_freq']:.3f} ({row['delta']:+.3f})",
        )
        if row["delta"] > 0 and len(increased) < top_k:
            increased.append(entry)
        elif row["delta"] < 0 and len(decreased) < top_k:
            decreased.append(entry)
        if len(increased) >= top_k and len(decreased) >= top_k:
            break
    return increased, decreased


def _transition_deltas(
    reference_cases: pd.DataFrame,
    current_cases: pd.DataFrame,
    top_k: int,
    register,
) -> list[dict[str, Any]]:
    reference_counter = Counter(_extract_transitions(reference_cases))
    current_counter = Counter(_extract_transitions(current_cases))
    rows = _distribution_delta_rows(pd.Series(reference_counter), pd.Series(current_counter), "transition", sort_abs=True)
    results = []
    for row in rows[:top_k]:
        results.append(
            register(
                row,
                f"Transition {row['transition']}: {row['reference_freq']:.3f} -> {row['current_freq']:.3f} ({row['delta']:+.3f})",
            )
        )
    return results


def _activity_deltas(
    reference_cases: pd.DataFrame,
    current_cases: pd.DataFrame,
    top_k: int,
    register,
) -> list[dict[str, Any]]:
    reference_counter = Counter(activity for activities in reference_cases["Activities"] for activity in activities)
    current_counter = Counter(activity for activities in current_cases["Activities"] for activity in activities)
    rows = _distribution_delta_rows(pd.Series(reference_counter), pd.Series(current_counter), "activity", sort_abs=True)
    return [
        register(
            row,
            f"Activity {row['activity']}: {row['reference_freq']:.3f} -> {row['current_freq']:.3f} ({row['delta']:+.3f})",
        )
        for row in rows[:top_k]
    ]


def _loop_delta(reference_cases: pd.DataFrame, current_cases: pd.DataFrame, register) -> dict[str, Any]:
    reference_loop_rate = float(reference_cases["HasLoop"].mean()) if len(reference_cases) else 0.0
    current_loop_rate = float(current_cases["HasLoop"].mean()) if len(current_cases) else 0.0
    reference_repeat_avg = float(reference_cases["RepeatedActivityCount"].mean()) if len(reference_cases) else 0.0
    current_repeat_avg = float(current_cases["RepeatedActivityCount"].mean()) if len(current_cases) else 0.0
    payload = {
        "reference_loop_rate": round(reference_loop_rate, 4),
        "current_loop_rate": round(current_loop_rate, 4),
        "loop_rate_delta": round(current_loop_rate - reference_loop_rate, 4),
        "reference_avg_repeated_activities": round(reference_repeat_avg, 4),
        "current_avg_repeated_activities": round(current_repeat_avg, 4),
        "avg_repeated_activities_delta": round(current_repeat_avg - reference_repeat_avg, 4),
    }
    return register(
        payload,
        "Loop rate {:.3f} -> {:.3f}, repeated activities {:.3f} -> {:.3f}".format(
            reference_loop_rate,
            current_loop_rate,
            reference_repeat_avg,
            current_repeat_avg,
        ),
    )


def _duration_delta(reference_cases: pd.DataFrame, current_cases: pd.DataFrame, register) -> dict[str, Any]:
    reference_values = reference_cases["Duration"].to_numpy(dtype=float) if len(reference_cases) else np.asarray([])
    current_values = current_cases["Duration"].to_numpy(dtype=float) if len(current_cases) else np.asarray([])
    max_samples = 500

    def stats(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0}
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
        }

    reference_stats = stats(reference_values)
    current_stats = stats(current_values)
    payload = {
        "reference": {key: round(value, 4) for key, value in reference_stats.items()},
        "current": {key: round(value, 4) for key, value in current_stats.items()},
        "delta": {
            key: round(current_stats[key] - reference_stats[key], 4)
            for key in reference_stats
        },
        "samples": {
            "reference": [float(value) for value in reference_values[:max_samples]],
            "current": [float(value) for value in current_values[:max_samples]],
            "max_samples": max_samples,
            "truncated": bool(reference_values.size > max_samples or current_values.size > max_samples),
        },
    }
    return register(
        payload,
        "Duration median {:.2f} -> {:.2f}, p90 {:.2f} -> {:.2f}".format(
            reference_stats["median"],
            current_stats["median"],
            reference_stats["p90"],
            current_stats["p90"],
        ),
    )


def _attribute_deltas(
    reference_events: pd.DataFrame,
    current_events: pd.DataFrame,
    top_k: int,
    register,
) -> list[dict[str, Any]]:
    results = []
    normalized_columns = {column.lower(): column for column in set(reference_events.columns).union(current_events.columns)}
    for logical_name, candidates in ATTRIBUTE_SCORE_CANDIDATES.items():
        column_name = next((normalized_columns[candidate] for candidate in candidates if candidate in normalized_columns), None)
        if not column_name:
            continue
        rows = _distribution_delta_rows(
            reference_events[column_name].astype(str).value_counts(),
            current_events[column_name].astype(str).value_counts(),
            "value",
            sort_abs=True,
        )
        for row in rows[:top_k]:
            row["attribute"] = logical_name
            results.append(
                register(
                    row,
                    f"Attribute {logical_name}={row['value']}: {row['reference_freq']:.3f} -> {row['current_freq']:.3f} ({row['delta']:+.3f})",
                )
            )
    return results


def _distribution_delta_rows(
    reference_counts: pd.Series,
    current_counts: pd.Series,
    key_name: str,
    sort_abs: bool = False,
) -> list[dict[str, Any]]:
    reference_counts = reference_counts.astype(float) if not reference_counts.empty else pd.Series(dtype=float)
    current_counts = current_counts.astype(float) if not current_counts.empty else pd.Series(dtype=float)
    total_reference = float(reference_counts.sum()) or 1.0
    total_current = float(current_counts.sum()) or 1.0
    keys = sorted(set(reference_counts.index).union(set(current_counts.index)))
    rows = []
    for key in keys:
        reference_count = float(reference_counts.get(key, 0.0))
        current_count = float(current_counts.get(key, 0.0))
        reference_freq = reference_count / total_reference
        current_freq = current_count / total_current
        rows.append(
            {
                key_name: str(key),
                "reference_count": int(reference_count),
                "current_count": int(current_count),
                "reference_freq": round(reference_freq, 4),
                "current_freq": round(current_freq, 4),
                "delta": round(current_freq - reference_freq, 4),
            }
        )
    if sort_abs:
        return sorted(rows, key=lambda row: abs(row["delta"]), reverse=True)
    return sorted(rows, key=lambda row: row["delta"], reverse=True)


def derive_rule_based_tags(drift_point: dict[str, Any], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    tags: list[dict[str, Any]] = []
    increased = evidence.get("top_increased_traces", [])
    decreased = evidence.get("top_decreased_traces", [])
    transitions = evidence.get("top_changed_transitions", [])
    activities = evidence.get("activity_frequency_deltas", [])
    loop_delta = evidence.get("rework_or_loop_rate_delta", {})
    duration_delta = evidence.get("duration_stats_delta", {})
    attribute_deltas = evidence.get("attribute_distribution_deltas", [])

    significant_added = [
        item for item in increased[:5]
        if item["delta"] >= _THR["path_added_min_delta"]
        and (item["reference_count"] == 0 or item["delta"] >= _THR["path_added_strong_delta"])
    ]
    if significant_added:
        top = max(significant_added, key=lambda item: item["delta"])
        tags.append(_build_tag("path_added", _CONF["path_added"], [top["evidence_id"]], "Current window contains newly prominent paths."))

    decreased_lengths = [_trace_length(item.get("trace", "")) for item in decreased[:5] if item.get("trace")]
    shortened_paths = []
    for item in increased[:5]:
        if item["delta"] < _THR["path_removed_min_delta"]:
            continue
        trace_length = _trace_length(item.get("trace", ""))
        if decreased_lengths and trace_length + 1 <= max(decreased_lengths):
            shortened_paths.append(item)
        elif trace_length <= 3 and "Closed" in item.get("trace", ""):
            shortened_paths.append(item)

    if any(item["delta"] <= -_THR["path_removed_min_delta"] for item in decreased[:3]) or shortened_paths:
        evidence_ids = []
        if decreased[:3]:
            evidence_ids.append(min(decreased[:3], key=lambda item: item["delta"])["evidence_id"])
        if shortened_paths:
            evidence_ids.append(shortened_paths[0]["evidence_id"])
        tags.append(
            _build_tag(
                "path_removed_or_skipped_step",
                _CONF["path_removed_with_shortening"] if shortened_paths else _CONF["path_removed_without_shortening"],
                evidence_ids,
                "Previously common paths weakened or current paths became shorter, consistent with skipped steps or path removal.",
            )
        )

    duration_delta_stats = duration_delta.get("delta", {})
    if duration_delta_stats and (
        duration_delta_stats.get("median", 0.0) >= _THR["delay_median_min_minutes"]
        or duration_delta_stats.get("p90", 0.0) >= _THR["delay_p90_min_minutes"]
        or drift_point.get("duration_score", 0.0) >= max(_THR["duration_score_signal_min"], drift_point.get("trace_score", 0.0))
    ):
        tags.append(
            _build_tag(
                "delay_increase",
                _CONF["delay_increase"],
                [duration_delta.get("evidence_id")],
                "Case duration shifted upward, especially in median or high-percentile latency.",
            )
        )

    if loop_delta and (
        loop_delta.get("loop_rate_delta", 0.0) >= _THR["loop_rate_min_delta"]
        or loop_delta.get("avg_repeated_activities_delta", 0.0) >= _THR["loop_repeated_activity_min_delta"]
    ):
        tags.append(
            _build_tag(
                "loop_increase",
                _CONF["loop_increase"],
                [loop_delta.get("evidence_id")],
                "Repeated activities became more common in the current window.",
            )
        )

    escalation_evidence_ids = []
    for item in increased[:5] + transitions[:5] + activities[:5]:
        label = " ".join(str(value).lower() for key, value in item.items() if key in {"trace", "transition", "activity"})
        if any(keyword in label for keyword in _ESCALATION_KEYWORDS) and abs(item.get("delta", 0.0)) >= _THR["escalation_min_delta"]:
            escalation_evidence_ids.append(item["evidence_id"])
    if escalation_evidence_ids:
        tags.append(
            _build_tag(
                "handoff_or_escalation_increase",
                _CONF["handoff_or_escalation_increase"],
                escalation_evidence_ids[:3],
                "Escalation or handoff-like transitions increased in the current window.",
            )
        )

    if any(abs(item.get("delta", 0.0)) >= _THR["attribute_shift_min_delta"] for item in attribute_deltas[:5]):
        tags.append(
            _build_tag(
                "case_mix_shift",
                _CONF["case_mix_shift_attribute"],
                [item["evidence_id"] for item in attribute_deltas[:3]],
                "Optional business attributes shifted, indicating case mix changes.",
            )
        )
    elif not tags and (increased or decreased or activities):
        fallback_evidence = []
        if increased:
            fallback_evidence.append(increased[0]["evidence_id"])
        if activities:
            fallback_evidence.append(activities[0]["evidence_id"])
        tags.append(
            _build_tag(
                "case_mix_shift",
                _CONF["case_mix_shift_fallback"],
                fallback_evidence,
                "Multiple behavior changes are present but do not map cleanly to a single stronger taxonomy.",
            )
        )

    return tags


def _trace_length(trace: str) -> int:
    return len([segment for segment in trace.split(" -> ") if segment])


def _build_tag(tag: str, confidence: float, evidence_ids: list[str | None], rationale: str) -> dict[str, Any]:
    return {
        "tag": tag,
        "confidence": round(confidence, 2),
        "evidence_ids": [evidence_id for evidence_id in evidence_ids if evidence_id],
        "rationale": rationale,
    }
