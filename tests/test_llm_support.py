from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_detection.llm_support import enrich_with_llm_diagnosis, fallback_diagnosis


class TestFallbackDiagnosis:
    def test_with_tags_generates_candidate_causes(self, minimal_drift_point):
        point = dict(minimal_drift_point)
        point["rule_based_tags"] = [
            {
                "tag": "path_added",
                "confidence": 0.80,
                "rationale": "New paths appeared.",
                "evidence_ids": ["DP01-E01"],
            }
        ]
        result = fallback_diagnosis(point)
        assert result["source"] == "fallback"
        assert len(result["candidate_causes"]) == 1
        assert result["candidate_causes"][0]["confidence"] == 0.80

    def test_no_tags_generates_generic_cause(self, minimal_drift_point):
        result = fallback_diagnosis(minimal_drift_point)
        assert result["source"] == "fallback"
        assert len(result["candidate_causes"]) == 1
        assert len(result["recommendations"]) == 1

    def test_error_message_appears_in_missing_data(self, minimal_drift_point):
        result = fallback_diagnosis(minimal_drift_point, error_message="timeout")
        assert any("timeout" in item for item in result["missing_data"])

    def test_confidence_is_bounded(self, minimal_drift_point):
        point = dict(minimal_drift_point)
        point["rule_based_tags"] = [
            {"tag": "delay_increase", "confidence": 1.5, "rationale": "x", "evidence_ids": []}
        ]
        result = fallback_diagnosis(point)
        assert result["confidence"] <= 1.0


class TestEnrichWithLLMDiagnosis:
    def _settings(self, enabled=True, api_key=None):
        return {
            "enabled": enabled,
            "api_key": api_key,
            "base_url": None,
            "model": "gpt-test",
        }

    def test_disabled_llm_uses_fallback_for_all_points(self, minimal_drift_point):
        points = [dict(minimal_drift_point), dict(minimal_drift_point)]
        points[1]["id"] = "DP02"
        result_points, meta = enrich_with_llm_diagnosis(points, {}, self._settings(enabled=False))
        assert meta["used_llm"] is False
        assert any("disabled" in str(r).lower() for r in meta["fallback_reasons"])
        for p in result_points:
            assert p["llm_diagnosis"]["source"] == "fallback"

    def test_missing_api_key_uses_fallback(self, minimal_drift_point):
        points = [dict(minimal_drift_point)]
        result_points, meta = enrich_with_llm_diagnosis(points, {}, self._settings(enabled=True, api_key=None))
        assert meta["used_llm"] is False
        assert any("OPENAI_API_KEY" in str(r) for r in meta["fallback_reasons"])
        assert result_points[0]["llm_diagnosis"]["source"] == "fallback"

    def test_per_point_failure_tracked_separately(self, minimal_drift_point):
        """Each failing drift point should add its own entry to fallback_reasons."""
        import unittest.mock as mock

        point_a = dict(minimal_drift_point)
        point_a["id"] = "DP01"
        point_b = dict(minimal_drift_point)
        point_b["id"] = "DP02"

        settings = self._settings(enabled=True, api_key="fake-key")

        with mock.patch(
            "drift_detection.llm_support.diagnose_drift_point",
            side_effect=RuntimeError("network error"),
        ):
            result_points, meta = enrich_with_llm_diagnosis([point_a, point_b], {}, settings)

        assert meta["used_llm"] is False
        assert len(meta["fallback_reasons"]) == 2
        ids_in_reasons = {r["point_id"] for r in meta["fallback_reasons"]}
        assert ids_in_reasons == {"DP01", "DP02"}
        for p in result_points:
            assert p["llm_diagnosis"]["source"] == "fallback"
