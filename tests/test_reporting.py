from __future__ import annotations

from drift_detection.reporting import render_markdown_report


def test_report_renders_fallback_reasons_and_mad_multiplier():
    result = {
        "global_summary": {
            "status": "STABLE",
            "drift_point_count": 0,
            "window_count": 1,
            "peak_score": 0.1,
            "threshold": 0.2,
            "score_profile": "multi-view",
            "threshold_details": {
                "source": "auto",
                "median_score": 0.1,
                "mad_score": 0.02,
                "mad_multiplier": 2.5,
                "auto_candidate": 0.15,
            },
        },
        "config": {
            "analysis_mode": "timeline",
            "score_profile": "multi-view",
            "drift_metric": "tv",
            "detection_mode": "mixed",
            "window_size": 50,
            "step_size": 10,
        },
        "drift_points": [],
        "evaluation": None,
        "llm": {
            "enabled": True,
            "used_llm": False,
            "fallback_reasons": [
                "OPENAI_API_KEY is not configured.",
                {"point_id": "DP01", "error": "network error"},
            ],
        },
    }

    rendered = render_markdown_report(result)
    assert "OPENAI_API_KEY is not configured." in rendered
    assert "DP01: network error" in rendered
    assert "MAD multiplier=`2.5`" in rendered
    assert "评分配置：`multi-view`" in rendered
