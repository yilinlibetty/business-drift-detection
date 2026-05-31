"""Tests for drift.ot_attribution (Phase 5 / M2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drift.injection import inject_insertion
from drift.io import build_cases_dataframe
from drift.ot_attribution import (
    attach_case_samples,
    attribution_report,
    edit_distance_matrix,
    joint_support,
    top_k_flows,
    top_variant_changes,
    transport_plan,
    variant_distribution,
    w1_distance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _toy_cases(traces: list[tuple], multiplicities: list[int]) -> pd.DataFrame:
    """Build a cases DataFrame with the given variants and counts."""
    rows = []
    n = 0
    for t, m in zip(traces, multiplicities):
        for _ in range(m):
            rows.append({"CaseID": f"c{n:03d}", "Trace": t})
            n += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# variant_distribution
# ---------------------------------------------------------------------------


def test_variant_distribution_sums_to_one_and_returns_ids():
    cases = _toy_cases([("A", "B"), ("A", "X", "B"), ("A", "B")], [3, 2, 1])
    variants, probs, ids = variant_distribution(cases)
    # variant_distribution does not preserve insertion order -- only that probs sum to 1.
    assert probs.sum() == pytest.approx(1.0)
    assert set(variants) == {("A", "B"), ("A", "X", "B")}
    # IDs are bucketed correctly
    assert len(ids[("A", "B")]) == 4
    assert len(ids[("A", "X", "B")]) == 2


def test_variant_distribution_accepts_legacy_string_trace_format():
    # If a trace column comes in as a " -> ".join() string, it should still parse.
    df = pd.DataFrame({"CaseID": ["c0", "c1"], "Trace": ["A -> B", "A -> X -> B"]})
    variants, probs, _ = variant_distribution(df)
    assert set(variants) == {("A", "B"), ("A", "X", "B")}
    assert probs.sum() == pytest.approx(1.0)


def test_variant_distribution_canonical_ordering_deterministic():
    cases = _toy_cases([("A", "B"), ("A", "X", "B"), ("C", "D")], [1, 1, 1])
    variants_a, _, _ = variant_distribution(cases)
    variants_b, _, _ = variant_distribution(cases.sample(frac=1.0, random_state=42))
    assert variants_a == variants_b


# ---------------------------------------------------------------------------
# joint_support
# ---------------------------------------------------------------------------


def test_joint_support_aligns_on_union():
    va, pa = [("A", "B"), ("X",)], np.array([0.5, 0.5])
    vb, pb = [("X",), ("Y",)], np.array([0.6, 0.4])
    union, pa_j, pb_j = joint_support(va, pa, vb, pb)
    assert set(union) == {("A", "B"), ("X",), ("Y",)}
    assert pa_j.sum() == pytest.approx(1.0)
    assert pb_j.sum() == pytest.approx(1.0)
    idx_x = union.index(("X",))
    assert pa_j[idx_x] == pytest.approx(0.5)
    assert pb_j[idx_x] == pytest.approx(0.6)
    idx_ab = union.index(("A", "B"))
    assert pa_j[idx_ab] == pytest.approx(0.5)
    assert pb_j[idx_ab] == 0.0


# ---------------------------------------------------------------------------
# edit_distance_matrix
# ---------------------------------------------------------------------------


def test_edit_distance_matrix_diagonal_zero_when_same_support():
    variants = [("A", "B"), ("A", "X", "B"), ("C", "D")]
    M = edit_distance_matrix(variants, variants)
    assert M.shape == (3, 3)
    np.testing.assert_array_equal(np.diag(M), [0, 0, 0])


def test_edit_distance_matrix_normalised_to_unit_interval():
    variants = [("A", "B", "C", "D"), ("A", "X", "Y", "Z"), ("A", "B", "C", "D", "E")]
    M = edit_distance_matrix(variants, variants)
    assert (M >= 0).all() and (M <= 1).all()


def test_edit_distance_matrix_off_diagonal_known():
    # ("A","B","C") -> ("A","X","C") = 1 edit / 3 length = 0.333..
    variants = [("A", "B", "C"), ("A", "X", "C")]
    M = edit_distance_matrix(variants, variants)
    assert M[0, 1] == pytest.approx(1.0 / 3.0)
    assert M[1, 0] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# w1_distance + transport_plan
# ---------------------------------------------------------------------------


def test_w1_zero_on_identical_distributions():
    p = np.array([0.4, 0.3, 0.3])
    M = np.array([[0, 0.5, 1], [0.5, 0, 0.5], [1, 0.5, 0]])
    assert w1_distance(p, p, M) == pytest.approx(0.0, abs=1e-12)


def test_transport_plan_diagonal_on_identical_distributions():
    p = np.array([0.4, 0.3, 0.3])
    M = np.array([[0, 0.5, 1], [0.5, 0, 0.5], [1, 0.5, 0]])
    plan = transport_plan(p, p, M)
    np.testing.assert_allclose(np.diag(plan), p)
    np.testing.assert_allclose(plan - np.diag(np.diag(plan)), 0, atol=1e-12)


def test_w1_hand_computed_two_atom_transport():
    """All mass moves from variant 0 to variant 1 at cost 0.2."""
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    M = np.array([[0, 0.2], [0.2, 0]])
    assert w1_distance(p, q, M) == pytest.approx(0.2, abs=1e-12)
    plan = transport_plan(p, q, M)
    expected = np.array([[0, 1.0], [0, 0]])
    np.testing.assert_allclose(plan, expected, atol=1e-12)


def test_transport_plan_marginals():
    """Row-sums and column-sums must match the input marginals exactly."""
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.4, 0.4])
    M = np.array([[0, 0.3, 0.7], [0.3, 0, 0.4], [0.7, 0.4, 0]])
    plan = transport_plan(p, q, M)
    np.testing.assert_allclose(plan.sum(axis=1), p, atol=1e-12)
    np.testing.assert_allclose(plan.sum(axis=0), q, atol=1e-12)


def test_w1_validates_shapes():
    with pytest.raises(ValueError, match="cost shape"):
        w1_distance(np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([[0, 1, 0], [1, 0, 0]]))


def test_w1_rejects_non_simplex():
    with pytest.raises(ValueError, match="sum to 1"):
        w1_distance(np.array([0.5, 0.3]), np.array([0.5, 0.5]), np.array([[0, 1], [1, 0]]))


# ---------------------------------------------------------------------------
# top_k_flows
# ---------------------------------------------------------------------------


def test_top_k_flows_sorted_desc_and_excludes_diagonal():
    plan = np.array([
        [0.10, 0.30, 0.05],   # diagonal 0.10 should be excluded by default
        [0.20, 0.05, 0.15],   # diagonal 0.05 excluded
        [0.00, 0.10, 0.05],   # diagonal 0.05 excluded
    ])
    cost = np.array([[0, 0.5, 1.0], [0.5, 0, 0.5], [1.0, 0.5, 0]])
    variants = [("A",), ("B",), ("C",)]
    flows = top_k_flows(plan, variants, cost=cost, k=10)
    # excludes the three diagonal entries
    expected_n = (plan > 0).sum() - 3
    assert len(flows) == expected_n
    masses = [f["mass"] for f in flows]
    assert masses == sorted(masses, reverse=True)
    # all flows include edit_distance because we passed cost
    assert all("edit_distance" in f for f in flows)


def test_top_k_flows_respects_k():
    plan = np.full((4, 4), 0.05)
    np.fill_diagonal(plan, 0.0)
    variants = [("A",), ("B",), ("C",), ("D",)]
    flows = top_k_flows(plan, variants, k=3)
    assert len(flows) == 3


# ---------------------------------------------------------------------------
# top_variant_changes
# ---------------------------------------------------------------------------


def test_top_variant_changes_lost_and_gained():
    variants = [("A", "B"), ("A", "X", "B"), ("C",)]
    pa = np.array([0.7, 0.2, 0.1])
    pb = np.array([0.3, 0.6, 0.1])
    ids_a = {v: [f"a{i}" for i in range(int(p * 100))] for v, p in zip(variants, pa)}
    ids_b = {v: [f"b{i}" for i in range(int(p * 100))] for v, p in zip(variants, pb)}
    changes = top_variant_changes(variants, pa, pb, ids_a, ids_b, k=3)
    assert changes["lost"][0]["variant"] == ("A", "B")
    assert changes["lost"][0]["delta"] == pytest.approx(-0.4)
    assert changes["gained"][0]["variant"] == ("A", "X", "B")
    assert changes["gained"][0]["delta"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# attach_case_samples
# ---------------------------------------------------------------------------


def test_attach_case_samples_deterministic_and_sized():
    variants = [("A",), ("B",)]
    plan = np.array([[0.0, 0.5], [0.5, 0.0]])
    cost = np.array([[0, 1.0], [1.0, 0]])
    ids_a = {("A",): [f"a{i}" for i in range(20)], ("B",): []}
    ids_b = {("B",): [f"b{i}" for i in range(20)], ("A",): []}
    flows1 = top_k_flows(plan, variants, cost=cost, k=10)
    flows2 = top_k_flows(plan, variants, cost=cost, k=10)
    attach_case_samples(flows1, ids_a, ids_b, variants, sample_size=3, seed=42)
    attach_case_samples(flows2, ids_a, ids_b, variants, sample_size=3, seed=42)
    assert flows1 == flows2
    f = flows1[0]
    assert len(f["from_case_ids_sample"]) <= 3
    assert len(f["to_case_ids_sample"]) <= 3


# ---------------------------------------------------------------------------
# attribution_report end-to-end
# ---------------------------------------------------------------------------


def _build_log(n=120, drift=False, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(n):
        if drift and i >= n // 2:
            seq = ["A", "B", "Wait", "C", "D"]
        elif rng.random() < 0.5:
            seq = ["A", "B", "C", "D"]
        else:
            seq = ["A", "B", "Wait", "C", "D"]
        for j, act in enumerate(seq):
            rows.append({
                "Case ID": f"c{i:04d}",
                "Activity": act,
                "Complete Timestamp": base + pd.Timedelta(minutes=10 * i + j),
            })
    return pd.DataFrame(rows)


def test_attribution_report_identical_inputs_gives_zero_w1():
    events = _build_log(60, drift=False, seed=0)
    cases = build_cases_dataframe(events, "Case ID", "Activity", "Complete Timestamp")
    report = attribution_report(cases, cases)
    assert report["w1"] == pytest.approx(0.0, abs=1e-12)
    # no nontrivial off-diagonal flows
    assert report["top_transport_flows"] == []
    # neither list contains any entries since every delta is exactly 0
    assert report["top_lost_variants"] == []
    assert report["top_gained_variants"] == []


def test_attribution_report_under_insertion_drift_finds_meaningful_flow():
    events = _build_log(120, drift=False, seed=0)
    inj, gt = inject_insertion(
        events, after_activity="B", new_activity="Review",
        fraction=1.0, seed=0,
    )
    cases_base = build_cases_dataframe(events, "Case ID", "Activity", "Complete Timestamp")
    cases_curr = build_cases_dataframe(inj,    "Case ID", "Activity", "Complete Timestamp")
    report = attribution_report(cases_base, cases_curr, k_flows=5)

    assert report["w1"] > 0.0
    assert report["n_variants_union"] > report["n_variants_baseline"]
    # The top flow's destination variant must contain 'Review' somewhere.
    top = report["top_transport_flows"][0]
    assert "Review" in top["to_variant"], (
        f"top flow target variant should contain 'Review', got {top['to_variant']}"
    )
    # The lost variants on baseline should not contain 'Review'; the gained on
    # current should.
    lost_has_review = any("Review" in v["variant"] for v in report["top_lost_variants"])
    gained_has_review = any("Review" in v["variant"] for v in report["top_gained_variants"])
    assert not lost_has_review
    assert gained_has_review


def test_attribution_report_case_id_samples_populated():
    events = _build_log(80, drift=False, seed=0)
    inj, _ = inject_insertion(events, after_activity="B", new_activity="Review",
                              fraction=1.0, seed=0)
    cases_base = build_cases_dataframe(events, "Case ID", "Activity", "Complete Timestamp")
    cases_curr = build_cases_dataframe(inj,    "Case ID", "Activity", "Complete Timestamp")
    report = attribution_report(cases_base, cases_curr, k_flows=3, sample_size=4)
    for flow in report["top_transport_flows"]:
        # Either side could be empty (variant only on the other side); we
        # only require the sample fields exist.
        assert "from_case_ids_sample" in flow
        assert "to_case_ids_sample" in flow
