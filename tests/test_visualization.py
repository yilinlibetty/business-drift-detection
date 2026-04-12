from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from drift_detection.visualization import (
    figures_to_zip_bytes,
    plot_activity_delta,
    plot_attribute_delta,
    plot_dominant_signal_distribution,
    plot_drift_point_score_breakdown,
    plot_duration_comparison,
    plot_multiview_radar,
    plot_score_timeline,
    plot_score_component_heatmap,
    plot_threshold_sensitivity,
    plot_transition_delta,
    plot_trace_distribution,
    save_analysis_figures,
)


def _make_window(index: int, score: float, window_size: int = 5, step: int = 20) -> dict:
    current_start = index * step
    current_end = current_start + window_size - 1
    reference_start = max(0, current_start - window_size)
    reference_end = max(0, current_start - 1)
    return {
        "window_id": f"W{index:04d}",
        "window_index": index,
        "reference_start_index": reference_start,
        "reference_end_index": reference_end,
        "current_start_index": current_start,
        "current_end_index": current_end,
        "reference_start_time": None,
        "reference_end_time": None,
        "current_start_time": None,
        "current_end_time": None,
        "score_profile": "multi-view",
        "core_score": score,
        "trace_score": score,
        "transition_score": score / 2,
        "duration_score": score / 3,
        "duration_score_raw": score / 3,
        "loop_score": 0.0,
        "attribute_score": 0.0,
        "dominant_signal": "trace",
        "final_score_raw": score,
        "final_score": score,
    }


def _make_point() -> dict:
    return {
        "id": "DP01",
        "interval_start_case_index": 40,
        "interval_end_case_index": 44,
        "interval_start_time": None,
        "interval_end_time": None,
        "peak_score": 0.9,
        "trace_score": 0.9,
        "transition_score": 0.45,
        "duration_score": 0.3,
        "loop_score": 0.1,
        "attribute_score": 0.2,
        "core_score": 0.9,
        "dominant_signal": "trace",
        "threshold_excess": 0.4,
        "evidence": {
            "top_increased_traces": [
                {
                    "trace": "A -> C",
                    "reference_freq": 0.1,
                    "current_freq": 0.4,
                    "delta": 0.3,
                }
            ],
            "top_decreased_traces": [
                {
                    "trace": "A -> B",
                    "reference_freq": 0.7,
                    "current_freq": 0.3,
                    "delta": -0.4,
                }
            ],
            "activity_frequency_deltas": [
                {"activity": "C", "delta": 0.25},
                {"activity": "B", "delta": -0.20},
            ],
            "top_changed_transitions": [
                {"transition": "A -> C", "delta": 0.30},
                {"transition": "A -> B", "delta": -0.35},
            ],
            "attribute_distribution_deltas": [
                {"attribute": "resource", "value": "R2", "delta": 0.40},
                {"attribute": "resource", "value": "R1", "delta": -0.40},
            ],
            "duration_stats_delta": {
                "reference": {"mean": 10.0, "median": 10.0, "p90": 12.0},
                "current": {"mean": 22.0, "median": 21.0, "p90": 26.0},
                "delta": {"mean": 12.0, "median": 11.0, "p90": 14.0},
                "samples": {"reference": [9.0, 10.0, 12.0], "current": [18.0, 21.0, 27.0]},
            },
            "score_contribution": {
                "trace_score": 0.9,
                "transition_score": 0.45,
                "duration_score": 0.3,
                "loop_score": 0.1,
                "attribute_score": 0.2,
                "core_score": 0.9,
            },
        },
    }


def _make_result() -> dict:
    timeline = [_make_window(index, 0.1) for index in range(10)]
    timeline[2] = _make_window(2, 0.9)
    timeline[8] = _make_window(8, 0.8)
    return {
        "score_timeline": timeline,
        "drift_points": [_make_point()],
        "global_summary": {
            "threshold": 0.5,
            "threshold_details": {"source": "configured", "configured_threshold": 0.5},
        },
        "config": {"auto_threshold": False, "threshold": 0.5},
    }


def test_plot_functions_return_matplotlib_figures():
    result = _make_result()
    point = result["drift_points"][0]
    figures = [
        plot_score_timeline(result),
        plot_trace_distribution(point),
        plot_activity_delta(point),
        plot_transition_delta(point),
        plot_attribute_delta(point),
        plot_threshold_sensitivity(result),
        plot_duration_comparison(point),
        plot_multiview_radar(point),
        plot_drift_point_score_breakdown(point),
        plot_score_component_heatmap(result),
        plot_dominant_signal_distribution(result),
    ]

    try:
        assert all(isinstance(fig, Figure) for fig in figures)
    finally:
        for fig in figures:
            plt.close(fig)


def test_plot_functions_handle_missing_data():
    empty_point = {"id": "DP01", "evidence": {}}
    empty_result = {"score_timeline": [], "drift_points": [], "global_summary": {}, "config": {}}
    figures = [
        plot_score_timeline(empty_result),
        plot_trace_distribution(empty_point),
        plot_activity_delta(empty_point),
        plot_transition_delta(empty_point),
        plot_attribute_delta(empty_point),
        plot_threshold_sensitivity(empty_result),
        plot_duration_comparison(empty_point),
        plot_multiview_radar(empty_point),
        plot_drift_point_score_breakdown(empty_point),
        plot_score_component_heatmap(empty_result),
        plot_dominant_signal_distribution(empty_result),
    ]

    try:
        assert all(isinstance(fig, Figure) for fig in figures)
    finally:
        for fig in figures:
            plt.close(fig)


def test_threshold_sensitivity_counts_detected_points():
    fig = plot_threshold_sensitivity(_make_result(), multipliers=(1.0, 2.0, 5.0))
    try:
        assert fig.threshold_sensitivity_counts == [2, 2, 2]
    finally:
        plt.close(fig)


def test_save_analysis_figures_exports_pngs(tmp_path):
    saved_paths = save_analysis_figures(_make_result(), tmp_path)

    assert saved_paths
    assert all(path.endswith(".png") for path in saved_paths)
    assert all(Path(path).exists() for path in saved_paths)
    assert any("figure_01_score_timeline" in path for path in saved_paths)
    assert any("figure_09_transition_delta" in path for path in saved_paths)


def test_figures_to_zip_bytes_contains_expected_names():
    import zipfile
    from io import BytesIO

    payload = figures_to_zip_bytes(_make_result())

    assert payload
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        names = archive.namelist()
    assert "figure_01_score_timeline.png" in names
    assert "dp01_figure_10_attribute_delta.png" in names
