from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_detection.pipeline import detect_drift_points, resolve_threshold


def _make_window(index: int, score: float, window_size: int = 50) -> dict:
    start = index * 10
    end = start + window_size - 1
    return {
        "window_index": index,
        "window_id": f"W{index:03d}",
        "final_score": score,
        "trace_score": score,
        "duration_score": 0.0,
        "duration_score_raw": 0.0,
        "reference_start_index": max(0, start - window_size),
        "reference_end_index": max(0, start - 1),
        "reference_start_time": None,
        "reference_end_time": None,
        "current_start_index": start,
        "current_end_index": end,
        "current_start_time": None,
        "current_end_time": None,
    }


# ── resolve_threshold ────────────────────────────────────────────────────────

class TestResolveThreshold:
    def test_empty_timeline_returns_base(self):
        threshold, meta = resolve_threshold([], 0.10, auto_threshold=True)
        assert threshold == 0.10
        assert meta["source"] == "configured"

    def test_fixed_threshold_ignores_scores(self):
        timeline = [_make_window(i, 0.9) for i in range(20)]
        threshold, meta = resolve_threshold(timeline, 0.10, auto_threshold=False)
        assert threshold == 0.10
        assert meta["source"] == "configured"

    def test_auto_threshold_at_least_base(self):
        timeline = [_make_window(i, 0.01) for i in range(20)]
        threshold, _ = resolve_threshold(timeline, 0.10, auto_threshold=True)
        assert threshold >= 0.10

    def test_auto_threshold_rises_with_high_scores(self):
        low = [_make_window(i, 0.01) for i in range(10)]
        high = [_make_window(i + 10, 0.90) for i in range(10)]
        threshold_low, _ = resolve_threshold(low, 0.05, auto_threshold=True)
        threshold_high, _ = resolve_threshold(low + high, 0.05, auto_threshold=True)
        assert threshold_high > threshold_low

    def test_mad_multiplier_affects_threshold(self):
        timeline = [_make_window(i, float(i) / 20) for i in range(20)]
        t1, _ = resolve_threshold(timeline, 0.05, auto_threshold=True, mad_multiplier=1.0)
        t2, _ = resolve_threshold(timeline, 0.05, auto_threshold=True, mad_multiplier=5.0)
        assert t2 > t1

    def test_auto_meta_includes_mad_multiplier(self):
        timeline = [_make_window(i, 0.1) for i in range(10)]
        _, meta = resolve_threshold(timeline, 0.05, auto_threshold=True, mad_multiplier=2.5)
        assert meta.get("mad_multiplier") == 2.5


# ── detect_drift_points ──────────────────────────────────────────────────────

class TestDetectDriftPoints:
    def test_empty_timeline_returns_empty(self):
        assert detect_drift_points([], threshold=0.10) == []

    def test_no_window_above_threshold(self):
        timeline = [_make_window(i, 0.01) for i in range(10)]
        assert detect_drift_points(timeline, threshold=0.50) == []

    def test_single_active_window_yields_one_point(self):
        timeline = [_make_window(i, 0.01) for i in range(10)]
        timeline[5] = _make_window(5, 0.90)
        points = detect_drift_points(timeline, threshold=0.50)
        assert len(points) == 1
        assert points[0]["id"] == "DP01"
        assert points[0]["peak_score"] == 0.90

    def test_adjacent_windows_merged_into_one_point(self):
        timeline = [_make_window(i, 0.01) for i in range(20)]
        # Two consecutive high-score windows — should merge
        timeline[5] = _make_window(5, 0.80)
        timeline[6] = _make_window(6, 0.85)
        points = detect_drift_points(timeline, threshold=0.50)
        assert len(points) == 1

    def test_well_separated_windows_yield_two_points(self):
        timeline = [_make_window(i, 0.01, window_size=10) for i in range(60)]
        timeline[5] = _make_window(5, 0.90, window_size=10)
        timeline[50] = _make_window(50, 0.90, window_size=10)
        points = detect_drift_points(timeline, threshold=0.50)
        assert len(points) == 2

    def test_drift_point_ids_are_sequential(self):
        timeline = [_make_window(i, 0.01, window_size=10) for i in range(60)]
        timeline[5] = _make_window(5, 0.90, window_size=10)
        timeline[50] = _make_window(50, 0.90, window_size=10)
        points = detect_drift_points(timeline, threshold=0.50)
        assert [p["id"] for p in points] == ["DP01", "DP02"]

    def test_scores_nonnegative(self):
        import pandas as pd
        import numpy as np
        from drift_detection.pipeline import PipelineConfig, build_case_table, build_score_timeline

        n = 200
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Case ID": [f"C{i:04d}" for i in range(n)],
            "Activity": rng.choice(["A", "B", "C", "D"], size=n).tolist(),
            "Complete Timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
        })
        config = PipelineConfig(
            file_path="dummy",
            col_case_id="Case ID",
            col_activity="Activity",
            col_timestamp="Complete Timestamp",
            keep_only_complete=False,
        )
        cases = build_case_table(df, config)
        timeline = build_score_timeline(cases, config)
        for row in timeline:
            assert row["trace_score"] >= 0.0
            assert row["duration_score"] >= 0.0
            assert row["final_score"] >= 0.0

    def test_multi_view_scores_are_nonnegative_and_dominant_signal_matches_max(self):
        import pandas as pd
        from drift_detection.pipeline import PipelineConfig, build_case_table, build_score_timeline

        rows = []
        for i in range(60):
            rows.extend([
                {"Case ID": f"C{i:04d}", "Activity": "A", "Resource": "R1", "Complete Timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 3)},
                {"Case ID": f"C{i:04d}", "Activity": "B", "Resource": "R1", "Complete Timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 3 + 1)},
            ])
        for i in range(60, 120):
            rows.extend([
                {"Case ID": f"C{i:04d}", "Activity": "A", "Resource": "R2", "Complete Timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 3)},
                {"Case ID": f"C{i:04d}", "Activity": "C", "Resource": "R2", "Complete Timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 3 + 1)},
                {"Case ID": f"C{i:04d}", "Activity": "C", "Resource": "R2", "Complete Timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 3 + 2)},
            ])
        df = pd.DataFrame(rows)
        config = PipelineConfig(
            file_path="dummy",
            col_case_id="Case ID",
            col_activity="Activity",
            col_timestamp="Complete Timestamp",
            keep_only_complete=False,
            score_profile="multi-view",
            window_size=30,
            step_size=30,
        )
        cases = build_case_table(df, config)
        timeline = build_score_timeline(cases, config)
        row = timeline[1]

        signal_scores = {
            "core": row["core_score"],
            "trace": row["trace_score"],
            "transition": row["transition_score"],
            "duration": row["duration_score"],
            "loop": row["loop_score"],
            "attribute": row["attribute_score"],
        }
        assert all(score >= 0.0 for score in signal_scores.values())
        assert row["dominant_signal"] == max(signal_scores.items(), key=lambda item: (item[1], item[0]))[0]
