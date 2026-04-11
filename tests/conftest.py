from __future__ import annotations

import pytest


@pytest.fixture
def minimal_drift_point() -> dict:
    return {
        "id": "DP01",
        "peak_score": 0.25,
        "trace_score": 0.20,
        "duration_score": 0.10,
        "duration_score_raw": 0.10,
        "threshold_excess": 0.10,
        "reference_window": {"start_case_index": 0, "end_case_index": 49},
        "current_window": {"start_case_index": 50, "end_case_index": 99},
        "evidence": {"evidence_ids": [], "attribute_distribution_deltas": []},
        "rule_based_tags": [],
        "llm_diagnosis": None,
    }


@pytest.fixture
def evidence_with_path_added() -> dict:
    return {
        "top_increased_traces": [
            {
                "trace": "A -> B -> C -> D",
                "delta": 0.15,
                "reference_count": 0,
                "current_count": 12,
                "evidence_id": "DP01-E01",
            }
        ],
        "top_decreased_traces": [],
        "top_changed_transitions": [],
        "activity_frequency_deltas": [],
        "rework_or_loop_rate_delta": {},
        "duration_stats_delta": {},
        "attribute_distribution_deltas": [],
        "evidence_ids": ["DP01-E01"],
        "evidence_index": {"DP01-E01": "top_increased_traces[0]"},
    }


@pytest.fixture
def evidence_with_delay() -> dict:
    return {
        "top_increased_traces": [],
        "top_decreased_traces": [],
        "top_changed_transitions": [],
        "activity_frequency_deltas": [],
        "rework_or_loop_rate_delta": {},
        "duration_stats_delta": {
            "evidence_id": "DP01-E02",
            "delta": {"mean": 30.0, "median": 20.0, "p90": 45.0},
        },
        "attribute_distribution_deltas": [],
        "evidence_ids": ["DP01-E02"],
        "evidence_index": {"DP01-E02": "duration_stats_delta"},
    }


@pytest.fixture
def evidence_with_loop() -> dict:
    return {
        "top_increased_traces": [],
        "top_decreased_traces": [],
        "top_changed_transitions": [],
        "activity_frequency_deltas": [],
        "rework_or_loop_rate_delta": {
            "evidence_id": "DP01-E03",
            "loop_rate_delta": 0.08,
            "avg_repeated_activities_delta": 0.1,
        },
        "duration_stats_delta": {},
        "attribute_distribution_deltas": [],
        "evidence_ids": ["DP01-E03"],
        "evidence_index": {"DP01-E03": "rework_or_loop_rate_delta"},
    }
