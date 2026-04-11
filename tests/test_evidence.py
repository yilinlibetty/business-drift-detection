from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from drift_detection.evidence import _validate_tagging_rules, build_evidence_pack, derive_rule_based_tags
from drift_detection.pipeline import PipelineConfig


class TestPathAddedTag:
    def test_large_positive_delta_triggers_path_added(self, minimal_drift_point, evidence_with_path_added):
        tags = derive_rule_based_tags(minimal_drift_point, evidence_with_path_added)
        tag_names = [t["tag"] for t in tags]
        assert "path_added" in tag_names

    def test_small_delta_does_not_trigger_path_added(self, minimal_drift_point):
        evidence = {
            "top_increased_traces": [
                {"trace": "A -> B", "delta": 0.02, "reference_count": 5, "current_count": 6, "evidence_id": "DP01-E01"}
            ],
            "top_decreased_traces": [],
            "top_changed_transitions": [],
            "activity_frequency_deltas": [],
            "rework_or_loop_rate_delta": {},
            "duration_stats_delta": {},
            "attribute_distribution_deltas": [],
            "evidence_ids": ["DP01-E01"],
            "evidence_index": {},
        }
        tags = derive_rule_based_tags(minimal_drift_point, evidence)
        tag_names = [t["tag"] for t in tags]
        assert "path_added" not in tag_names


class TestDelayIncreaseTag:
    def test_high_median_duration_triggers_delay(self, minimal_drift_point, evidence_with_delay):
        tags = derive_rule_based_tags(minimal_drift_point, evidence_with_delay)
        tag_names = [t["tag"] for t in tags]
        assert "delay_increase" in tag_names

    def test_small_duration_delta_does_not_trigger_delay(self, minimal_drift_point):
        evidence = {
            "top_increased_traces": [],
            "top_decreased_traces": [],
            "top_changed_transitions": [],
            "activity_frequency_deltas": [],
            "rework_or_loop_rate_delta": {},
            "duration_stats_delta": {
                "evidence_id": "DP01-E02",
                "delta": {"mean": 1.0, "median": 2.0, "p90": 5.0},
            },
            "attribute_distribution_deltas": [],
            "evidence_ids": ["DP01-E02"],
            "evidence_index": {},
        }
        # Ensure drift point has no high duration_score either
        point = dict(minimal_drift_point)
        point["duration_score"] = 0.05
        point["trace_score"] = 0.20
        tags = derive_rule_based_tags(point, evidence)
        tag_names = [t["tag"] for t in tags]
        assert "delay_increase" not in tag_names


class TestLoopIncreaseTag:
    def test_high_loop_rate_delta_triggers_loop_increase(self, minimal_drift_point, evidence_with_loop):
        tags = derive_rule_based_tags(minimal_drift_point, evidence_with_loop)
        tag_names = [t["tag"] for t in tags]
        assert "loop_increase" in tag_names

    def test_low_loop_rate_does_not_trigger(self, minimal_drift_point):
        evidence = {
            "top_increased_traces": [],
            "top_decreased_traces": [],
            "top_changed_transitions": [],
            "activity_frequency_deltas": [],
            "rework_or_loop_rate_delta": {
                "evidence_id": "DP01-E03",
                "loop_rate_delta": 0.01,
                "avg_repeated_activities_delta": 0.01,
            },
            "duration_stats_delta": {},
            "attribute_distribution_deltas": [],
            "evidence_ids": ["DP01-E03"],
            "evidence_index": {},
        }
        tags = derive_rule_based_tags(minimal_drift_point, evidence)
        tag_names = [t["tag"] for t in tags]
        assert "loop_increase" not in tag_names


class TestFallbackTag:
    def test_no_strong_evidence_produces_case_mix_shift_fallback(self, minimal_drift_point):
        evidence = {
            "top_increased_traces": [
                {"trace": "A -> B", "delta": 0.02, "reference_count": 5, "current_count": 6, "evidence_id": "DP01-E01"}
            ],
            "top_decreased_traces": [],
            "top_changed_transitions": [],
            "activity_frequency_deltas": [
                {"activity": "A", "delta": 0.02, "evidence_id": "DP01-E02"}
            ],
            "rework_or_loop_rate_delta": {},
            "duration_stats_delta": {},
            "attribute_distribution_deltas": [],
            "evidence_ids": ["DP01-E01", "DP01-E02"],
            "evidence_index": {},
        }
        tags = derive_rule_based_tags(minimal_drift_point, evidence)
        assert len(tags) == 1
        assert tags[0]["tag"] == "case_mix_shift"
        assert tags[0]["confidence"] < 0.50  # fallback confidence

    def test_empty_evidence_produces_no_tags(self, minimal_drift_point):
        empty_evidence = {
            "top_increased_traces": [],
            "top_decreased_traces": [],
            "top_changed_transitions": [],
            "activity_frequency_deltas": [],
            "rework_or_loop_rate_delta": {},
            "duration_stats_delta": {},
            "attribute_distribution_deltas": [],
            "evidence_ids": [],
            "evidence_index": {},
        }
        tags = derive_rule_based_tags(minimal_drift_point, empty_evidence)
        assert tags == []


class TestConfidenceBounds:
    def test_all_tag_confidences_in_range(self, minimal_drift_point, evidence_with_path_added, evidence_with_delay, evidence_with_loop):
        for evidence in [evidence_with_path_added, evidence_with_delay, evidence_with_loop]:
            tags = derive_rule_based_tags(minimal_drift_point, evidence)
            for tag in tags:
                assert 0.0 <= tag["confidence"] <= 1.0, (
                    f"Tag {tag['tag']} has out-of-range confidence {tag['confidence']}"
                )


class TestTaggingRulesValidation:
    def test_missing_required_threshold_key_has_clear_error(self):
        payload = {
            "confidence": {
                "path_added": 0.8,
                "path_removed_with_shortening": 0.84,
                "path_removed_without_shortening": 0.74,
                "delay_increase": 0.82,
                "loop_increase": 0.76,
                "handoff_or_escalation_increase": 0.68,
                "case_mix_shift_attribute": 0.72,
                "case_mix_shift_fallback": 0.45,
            },
            "thresholds": {
                "path_added_min_delta": 0.05,
                "path_added_strong_delta": 0.08,
                "path_removed_min_delta": 0.03,
                "delay_median_min_minutes": 15.0,
                "delay_p90_min_minutes": 30.0,
                "loop_rate_min_delta": 0.05,
                "loop_repeated_activity_min_delta": 0.20,
                "escalation_min_delta": 0.03,
            },
            "escalation_keywords": ["upgrade"],
        }

        import pytest

        with pytest.raises(ValueError, match="attribute_shift_min_delta"):
            _validate_tagging_rules(payload, "test-rules")


class TestDurationSamples:
    def test_evidence_pack_includes_bounded_duration_samples(self):
        cases = pd.DataFrame(
            [
                {
                    "CaseID": f"C{idx:02d}",
                    "Activities": ("A", "B"),
                    "Trace": "A -> B",
                    "Duration": float(10 + idx),
                    "RepeatedActivityCount": 0,
                    "HasLoop": False,
                    "StartTime": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=idx),
                    "EndTime": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=idx, minutes=10 + idx),
                    "CaseIndex": idx,
                }
                for idx in range(10)
            ]
        )
        df_events = pd.DataFrame(
            [
                {
                    "Case ID": f"C{idx:02d}",
                    "Activity": activity,
                    "Complete Timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=idx, minutes=offset),
                }
                for idx in range(10)
                for offset, activity in [(0, "A"), (10 + idx, "B")]
            ]
        )
        config = PipelineConfig(
            file_path="dummy.csv",
            col_case_id="Case ID",
            col_activity="Activity",
            col_timestamp="Complete Timestamp",
            keep_only_complete=False,
            top_k=3,
        )
        drift_point = {
            "id": "DP01",
            "reference_window": {"start_case_index": 0, "end_case_index": 4},
            "current_window": {"start_case_index": 5, "end_case_index": 9},
            "score_profile": "trace-duration",
            "dominant_signal": None,
            "core_score": 0.5,
            "trace_score": 0.1,
            "transition_score": None,
            "duration_score": 0.5,
            "loop_score": None,
            "attribute_score": None,
        }

        evidence = build_evidence_pack(drift_point, cases, df_events, config)
        samples = evidence["duration_stats_delta"]["samples"]

        assert samples["reference"] == [10.0, 11.0, 12.0, 13.0, 14.0]
        assert samples["current"] == [15.0, 16.0, 17.0, 18.0, 19.0]
        assert samples["truncated"] is False
        assert samples["max_samples"] == 500
