from __future__ import annotations

import math
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .pipeline import detect_drift_points, resolve_threshold

DEFAULT_SENSITIVITY_MULTIPLIERS = tuple(round(1.0 + 0.5 * idx, 1) for idx in range(9))


def plot_score_timeline(result: dict[str, Any]) -> Figure:
    timeline = result.get("score_timeline", []) or []
    if not timeline:
        return _empty_figure("Drift Score Timeline", "No score timeline is available.")

    frame = pd.DataFrame(timeline)
    x_values = _series_as_float(frame.get("current_end_index", frame.get("window_index")))
    if not x_values:
        x_values = list(range(len(frame)))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    _plot_score_series(ax, x_values, frame, "final_score", "Final score", "#1f4e5f", "-", 2.3)
    _plot_score_series(ax, x_values, frame, "trace_score", "Trace score", "#2a9d8f", "--", 1.4)
    _plot_score_series(ax, x_values, frame, "duration_score", "Duration score", "#e76f51", ":", 1.8)
    if frame.get("core_score") is not None and frame["core_score"].notna().any():
        _plot_score_series(ax, x_values, frame, "core_score", "Core score", "#8ab17d", "-.", 1.2)

    threshold = _safe_float((result.get("global_summary") or {}).get("threshold"), default=None)
    if threshold is not None:
        ax.axhline(threshold, color="#b00020", linestyle="--", linewidth=1.2, label=f"Threshold {threshold:.3f}")

    max_score = max(_series_as_float(frame.get("final_score")), default=0.0)
    for point in result.get("drift_points", []) or []:
        start = _safe_float(point.get("interval_start_case_index"), default=None)
        end = _safe_float(point.get("interval_end_case_index"), default=None)
        if start is not None and end is not None:
            ax.axvspan(start, end, color="#f4a261", alpha=0.12)
            marker_x = end
        else:
            marker_x = _safe_float((point.get("current_window") or {}).get("end_case_index"), default=None)
        if marker_x is None:
            continue
        peak = _safe_float(point.get("peak_score"), default=max_score) or max_score
        ax.axvline(marker_x, color="#f4a261", linestyle="-", linewidth=0.9, alpha=0.7)
        label = str(point.get("id", "drift"))
        dominant = point.get("dominant_signal")
        if dominant:
            label = f"{label}\n{dominant}"
        ax.text(marker_x, peak, label, fontsize=8, rotation=0, ha="center", va="bottom", color="#7a3e00")

    ax.set_title("Drift Score Timeline")
    ax.set_xlabel("Case index")
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_trace_distribution(point: dict[str, Any], top_k: int = 10) -> Figure:
    evidence = point.get("evidence", {}) or {}
    rows = list(evidence.get("top_increased_traces", []) or []) + list(evidence.get("top_decreased_traces", []) or [])
    rows = sorted(rows, key=lambda row: abs(_safe_float(row.get("delta")) or 0.0), reverse=True)[:top_k]
    if not rows:
        return _empty_figure("Trace Distribution Change", "No trace distribution evidence is available.")

    labels = [_truncate_label(str(row.get("trace", "N/A"))) for row in rows]
    y_pos = np.arange(len(rows))
    reference = [_safe_float(row.get("reference_freq")) or 0.0 for row in rows]
    current = [_safe_float(row.get("current_freq")) or 0.0 for row in rows]

    fig, ax = plt.subplots(figsize=(10, max(3.6, len(rows) * 0.42)))
    height = 0.36
    ax.barh(y_pos - height / 2, reference, height=height, color="#3d5a80", label="Reference")
    ax.barh(y_pos + height / 2, current, height=height, color="#ee6c4d", label="Current")
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top-{len(rows)} Trace Distribution Change")
    ax.grid(True, axis="x", alpha=0.22)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_activity_delta(point: dict[str, Any], top_k: int = 12) -> Figure:
    evidence = point.get("evidence", {}) or {}
    rows = sorted(
        evidence.get("activity_frequency_deltas", []) or [],
        key=lambda row: abs(_safe_float(row.get("delta")) or 0.0),
        reverse=True,
    )[:top_k]
    if not rows:
        return _empty_figure("Activity Frequency Delta", "No activity frequency delta evidence is available.")

    labels = [_truncate_label(str(row.get("activity", "N/A")), limit=38) for row in rows]
    values = [_safe_float(row.get("delta")) or 0.0 for row in rows]
    colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in values]

    fig, ax = plt.subplots(figsize=(9, max(3.6, len(rows) * 0.38)))
    ax.barh(np.arange(len(rows)), values, color=colors)
    ax.set_yticks(np.arange(len(rows)), labels)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("Current frequency - reference frequency")
    ax.set_title("Activity Frequency Delta")
    ax.grid(True, axis="x", alpha=0.22)
    fig.tight_layout()
    return fig


def plot_threshold_sensitivity(
    result: dict[str, Any],
    multipliers: Iterable[float] = DEFAULT_SENSITIVITY_MULTIPLIERS,
) -> Figure:
    timeline = result.get("score_timeline", []) or []
    if not timeline:
        return _empty_figure("Threshold Sensitivity", "No score timeline is available.")

    config = result.get("config", {}) or {}
    global_summary = result.get("global_summary", {}) or {}
    threshold_details = global_summary.get("threshold_details", {}) or {}
    base_threshold = _safe_float(
        threshold_details.get(
            "configured_threshold",
            config.get("threshold", global_summary.get("threshold", 0.05)),
        ),
        default=0.05,
    )
    auto_threshold = bool(config.get("auto_threshold", True))

    multiplier_values = [float(multiplier) for multiplier in multipliers]
    counts: list[int] = []
    thresholds: list[float] = []
    for multiplier in multiplier_values:
        threshold, _ = resolve_threshold(
            timeline,
            base_threshold or 0.05,
            auto_threshold=auto_threshold,
            mad_multiplier=multiplier,
        )
        thresholds.append(threshold)
        counts.append(len(detect_drift_points(timeline, threshold)))

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(multiplier_values, counts, marker="o", color="#1f4e5f", linewidth=2.0)
    for x_value, count, threshold in zip(multiplier_values, counts, thresholds):
        ax.annotate(f"{count}", (x_value, count), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        ax.plot([], [], label=f"{x_value:g}: threshold={threshold:.3f}", alpha=0)
    ax.set_title("Threshold Sensitivity")
    ax.set_xlabel("MAD multiplier")
    ax.set_ylabel("Detected drift point count")
    ax.set_xticks(multiplier_values)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.threshold_sensitivity_counts = counts  # type: ignore[attr-defined]
    fig.threshold_sensitivity_thresholds = thresholds  # type: ignore[attr-defined]
    return fig


def plot_duration_comparison(point: dict[str, Any]) -> Figure:
    evidence = point.get("evidence", {}) or {}
    duration = evidence.get("duration_stats_delta", {}) or {}
    samples = duration.get("samples", {}) or {}
    reference_samples = _float_list(samples.get("reference", []))
    current_samples = _float_list(samples.get("current", []))

    if reference_samples or current_samples:
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        data = [
            reference_samples if reference_samples else [0.0],
            current_samples if current_samples else [0.0],
        ]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks([1, 2], ["Reference", "Current"])
        ax.set_title("Duration Distribution Comparison")
        ax.set_ylabel("Duration (minutes)")
        ax.grid(True, axis="y", alpha=0.22)
        fig.tight_layout()
        return fig

    reference = duration.get("reference", {}) or {}
    current = duration.get("current", {}) or {}
    metrics = [metric for metric in ["median", "p90", "mean"] if metric in reference or metric in current]
    if not metrics:
        return _empty_figure("Duration Comparison", "No duration evidence is available.")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x_pos = np.arange(len(metrics))
    width = 0.34
    ax.bar(x_pos - width / 2, [_safe_float(reference.get(metric)) or 0.0 for metric in metrics], width, label="Reference", color="#3d5a80")
    ax.bar(x_pos + width / 2, [_safe_float(current.get(metric)) or 0.0 for metric in metrics], width, label="Current", color="#ee6c4d")
    ax.set_xticks(x_pos, metrics)
    ax.set_title("Duration Summary Comparison")
    ax.set_ylabel("Duration (minutes)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_multiview_radar(point: dict[str, Any]) -> Figure:
    evidence = point.get("evidence", {}) or {}
    contribution = evidence.get("score_contribution", {}) or {}
    if not contribution:
        contribution = point

    labels = ["trace", "transition", "duration", "loop", "attribute", "core"]
    values = [_safe_float(contribution.get(f"{label}_score")) or 0.0 for label in labels]
    if not any(values):
        return _empty_figure("Multi-view Score Radar", "No multi-view score contribution is available.")

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    closed_values = values + values[:1]
    closed_angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(5.8, 5.8), subplot_kw={"polar": True})
    ax.plot(closed_angles, closed_values, color="#1f4e5f", linewidth=2.0)
    ax.fill(closed_angles, closed_values, color="#2a9d8f", alpha=0.22)
    ax.set_xticks(angles, labels)
    ax.set_ylim(0.0, max(1.0, max(values) * 1.15))
    ax.set_title("Multi-view Score Contribution", pad=18)
    ax.grid(True, alpha=0.32)
    fig.tight_layout()
    return fig


def _plot_score_series(
    ax: Any,
    x_values: list[float],
    frame: pd.DataFrame,
    column: str,
    label: str,
    color: str,
    linestyle: str,
    linewidth: float,
) -> None:
    if column not in frame.columns:
        return
    values = _series_as_float(frame[column])
    if not values:
        return
    ax.plot(x_values[:len(values)], values, label=label, color=color, linestyle=linestyle, linewidth=linewidth)


def _series_as_float(series: Any) -> list[float]:
    if series is None:
        return []
    values = []
    for value in list(series):
        converted = _safe_float(value, default=None)
        if converted is not None:
            values.append(converted)
    return values


def _float_list(values: Any) -> list[float]:
    result = []
    for value in values or []:
        converted = _safe_float(value, default=None)
        if converted is not None:
            result.append(converted)
    return result


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _truncate_label(value: str, limit: int = 54) -> str:
    return value if len(value) <= limit else f"{value[:limit - 1]}..."


def _empty_figure(title: str, message: str) -> Figure:
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color="#555555")
    ax.set_axis_off()
    fig.tight_layout()
    return fig
