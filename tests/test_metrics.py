"""Tests for drift.metrics (Phase 2 / M1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drift.injection import inject_deletion, inject_insertion
from drift.metrics import (
    activity_frequency_dist,
    align,
    dfg_dist,
    jsd,
    multi_scale_drift,
    trace_variant_dist,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_log(n_cases: int = 60, seed: int = 0) -> pd.DataFrame:
    """ABCD plus optional Wait between B and C in ~50% of cases."""
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(n_cases):
        seq = ["A", "B", "C", "D"]
        if rng.random() < 0.5:
            seq = ["A", "B", "Wait", "C", "D"]
        for j, act in enumerate(seq):
            rows.append({
                "Case ID": f"c{i:03d}",
                "Activity": act,
                "Complete Timestamp": base + pd.Timedelta(minutes=10 * i + j),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def log():
    return _make_log()


# ---------------------------------------------------------------------------
# Distribution basics
# ---------------------------------------------------------------------------


def test_activity_frequency_sums_to_one(log):
    p = activity_frequency_dist(log)
    assert p.sum() == pytest.approx(1.0)
    assert (p >= 0).all()


def test_dfg_sums_to_one(log):
    p = dfg_dist(log)
    assert p.sum() == pytest.approx(1.0)
    assert (p >= 0).all()
    # Every pair should be from the activity alphabet
    activities = set(log["Activity"].unique())
    for src, dst in p.index:
        assert src in activities
        assert dst in activities


def test_dfg_does_not_cross_cases():
    # Two cases: c1=[A,B], c2=[C,D]. The only pairs allowed are (A,B) and (C,D).
    df = pd.DataFrame({
        "Case ID":           ["c1", "c1", "c2", "c2"],
        "Activity":          ["A",  "B",  "C",  "D"],
        "Complete Timestamp": pd.to_datetime([
            "2024-01-01 00:00:00", "2024-01-01 00:01:00",
            "2024-01-01 00:02:00", "2024-01-01 00:03:00",
        ]),
    })
    p = dfg_dist(df)
    assert set(p.index) == {("A", "B"), ("C", "D")}
    assert p[("A", "B")] == pytest.approx(0.5)
    assert p[("C", "D")] == pytest.approx(0.5)


def test_trace_variant_sums_to_one(log):
    p = trace_variant_dist(log)
    assert p.sum() == pytest.approx(1.0)
    # Variants should be tuples of activity strings
    for v in p.index:
        assert isinstance(v, tuple)
        assert all(isinstance(a, str) for a in v)


# ---------------------------------------------------------------------------
# JSD properties
# ---------------------------------------------------------------------------


def test_jsd_identity_is_zero(log):
    p = activity_frequency_dist(log)
    a, b = align(p, p)
    assert jsd(a, b) == pytest.approx(0.0, abs=1e-12)


def test_jsd_bounded_in_unit_interval():
    # extreme: disjoint supports
    p = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 0.0, 0.0])
    val = jsd(p, q, base=2)
    assert 0.0 <= val <= 1.0
    assert val == pytest.approx(1.0, abs=1e-12)


def test_jsd_symmetry():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(alpha=np.ones(8))
    q = rng.dirichlet(alpha=np.ones(8))
    assert jsd(p, q) == pytest.approx(jsd(q, p), abs=1e-12)


def test_jsd_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        jsd(np.array([0.5, 0.5]), np.array([0.5, 0.3, 0.2]))


def test_jsd_handles_empty_or_zero_inputs():
    assert jsd(np.zeros(0), np.zeros(0)) == 0.0
    assert jsd(np.zeros(3), np.zeros(3)) == 0.0


# ---------------------------------------------------------------------------
# Multi-scale aggregator
# ---------------------------------------------------------------------------


def test_multi_scale_identical_halves_yield_zero(log):
    # split log into two halves of cases that have IDENTICAL composition by
    # construction -- alternate case-id assignment.
    case_ids = sorted(log["Case ID"].unique())
    a_ids = set(case_ids[::2])
    b_ids = set(case_ids[1::2])
    df_a = log[log["Case ID"].isin(a_ids)]
    df_b = log[log["Case ID"].isin(b_ids)]
    d = multi_scale_drift(df_a, df_b)
    # Not strictly zero (random alternation has noise) but should be tiny.
    assert d["activity_jsd"] < 0.05
    assert d["dfg_jsd"]      < 0.05
    assert d["trace_jsd"]    < 0.10


def test_multi_scale_self_vs_self_is_zero(log):
    d = multi_scale_drift(log, log)
    for k, v in d.items():
        assert v == pytest.approx(0.0, abs=1e-12), f"{k}={v}"


def test_multi_scale_in_unit_interval(log):
    df_inj, _ = inject_deletion(log, target_activity="Wait", fraction=1.0, seed=42)
    d = multi_scale_drift(log, df_inj)
    for k, v in d.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


def test_insertion_dfg_more_sensitive_than_activity_freq():
    """The Plan-agent's key validation: inserting an activity should move
    DFG-JSD more than activity-frequency-JSD, because each insertion creates
    two brand-new direct-follows edges but only adds to one activity bin.
    """
    log = _make_log(n_cases=200, seed=0)
    df_inj, gt = inject_insertion(
        log, after_activity="B", new_activity="NewlyInsertedStep",
        fraction=1.0, seed=0,
    )
    d = multi_scale_drift(log, df_inj)
    assert d["dfg_jsd"] > d["activity_jsd"], (
        f"DFG-JSD should exceed activity-freq-JSD after insertion drift; "
        f"got dfg={d['dfg_jsd']:.4f} act={d['activity_jsd']:.4f}"
    )
    # And both should detect the drift (i.e. be nontrivial)
    assert d["dfg_jsd"] > 0.01
    assert d["trace_jsd"] > 0.01


def test_multi_scale_keys_are_floats(log):
    df_inj, _ = inject_deletion(log, target_activity="Wait", fraction=0.5, seed=0)
    d = multi_scale_drift(log, df_inj)
    assert set(d.keys()) == {"activity_jsd", "dfg_jsd", "trace_jsd"}
    for v in d.values():
        assert isinstance(v, float)
