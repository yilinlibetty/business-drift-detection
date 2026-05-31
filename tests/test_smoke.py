"""Smoke tests — confirm every drift.* module imports cleanly.

Each subsequent phase will add a real test file (test_injection.py, test_metrics.py, ...).
"""

import importlib

import pytest

MODULES = [
    "drift",
    "drift.io",
    "drift.injection",
    "drift.metrics",
    "drift.significance",
    "drift.localization",
    "drift.ot_attribution",
    "drift.evaluation",
    "drift.viz",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_importable(module_name):
    importlib.import_module(module_name)


def test_load_event_log_reexported_from_convert_data():
    """convert_data.load_event_log must still work after the move (back-compat shim)."""
    import convert_data
    from drift.io import load_event_log

    assert convert_data.load_event_log is load_event_log


def test_build_cases_dataframe_basic():
    """Sanity: build_cases_dataframe produces tuple Trace + Duration in minutes."""
    import pandas as pd

    from drift.io import build_cases_dataframe

    df = pd.DataFrame({
        "case": ["c1", "c1", "c1", "c2", "c2"],
        "act": ["A", "B", "C", "A", "C"],
        "ts": pd.to_datetime([
            "2020-01-01 00:00:00",
            "2020-01-01 00:01:00",
            "2020-01-01 00:02:00",
            "2020-01-02 00:00:00",
            "2020-01-02 00:00:30",
        ]),
    })
    cases = build_cases_dataframe(df, "case", "act", "ts")

    assert list(cases["CaseID"]) == ["c1", "c2"]
    assert cases.loc[0, "Trace"] == ("A", "B", "C")
    assert cases.loc[1, "Trace"] == ("A", "C")
    assert cases.loc[0, "TraceStr"] == "A -> B -> C"
    assert cases.loc[0, "Duration"] == pytest.approx(2.0)
    assert cases.loc[1, "Duration"] == pytest.approx(0.5)
    assert cases.loc[0, "EventCount"] == 3
