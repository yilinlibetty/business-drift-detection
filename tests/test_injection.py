"""Tests for drift.injection (Phase 1 / M5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drift.injection import (
    GroundTruth,
    inject,
    inject_deletion,
    inject_insertion,
    inject_loop,
    inject_substitution,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _make_log(n_cases: int = 40, seed: int = 0) -> pd.DataFrame:
    """Small synthetic event log: 40 cases, each follows A->B->C->D, half also have a Wait."""
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(n_cases):
        case_id = f"c{i:03d}"
        seq = ["A", "B", "C", "D"]
        if rng.random() < 0.5:
            # insert Wait between B and C in half the cases (so Wait is candidate-rich)
            seq = ["A", "B", "Wait", "C", "D"]
        for j, act in enumerate(seq):
            rows.append({
                "Case ID": case_id,
                "Activity": act,
                "Complete Timestamp": base + pd.Timedelta(minutes=10 * i + j),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def log():
    return _make_log()


# ---------------------------------------------------------------------------
# Event-count delta correctness
# ---------------------------------------------------------------------------


def test_insertion_event_count_delta(log):
    n_before = len(log)
    df_out, gt = inject_insertion(log, after_activity="B", new_activity="Review",
                                  fraction=0.5, seed=42)
    n_after = len(df_out)
    assert gt.pattern == "insertion"
    assert gt.target_activity == "B"
    assert gt.secondary_activity == "Review"
    # every affected case has exactly one B, so n_events_changed == n_affected_cases
    assert n_after - n_before == gt.n_events_changed
    assert gt.n_events_changed == gt.n_affected_cases


def test_deletion_event_count_delta(log):
    n_before = len(log)
    df_out, gt = inject_deletion(log, target_activity="Wait", fraction=1.0, seed=7)
    n_after = len(df_out)
    assert gt.pattern == "deletion"
    assert n_before - n_after == gt.n_events_changed
    # after full-fraction deletion no 'Wait' rows should survive in affected cases
    survivors = df_out[df_out["Case ID"].isin(gt.affected_case_ids)]
    assert (survivors["Activity"] == "Wait").sum() == 0


def test_substitution_event_count_preserved(log):
    n_before = len(log)
    df_out, gt = inject_substitution(log, src_activity="Wait", dst_activity="QueuedWait",
                                     fraction=0.6, seed=1)
    assert gt.pattern == "substitution"
    assert len(df_out) == n_before  # event count unchanged
    relabelled = df_out[df_out["Case ID"].isin(gt.affected_case_ids)]
    assert (relabelled["Activity"] == "Wait").sum() == 0
    assert (relabelled["Activity"] == "QueuedWait").sum() == gt.n_events_changed


def test_loop_event_count_delta(log):
    n_before = len(log)
    df_out, gt = inject_loop(log, target_activity="C", fraction=1.0,
                             repeat_range=(2, 2), seed=3)
    n_after = len(df_out)
    assert gt.pattern == "loop"
    # every C in every affected case is duplicated exactly 2 extra times
    n_c_per_case = 1  # by construction
    expected = gt.n_affected_cases * n_c_per_case * 2
    assert gt.n_events_changed == expected
    assert n_after - n_before == expected


# ---------------------------------------------------------------------------
# Affected-case selection
# ---------------------------------------------------------------------------


def test_affected_cases_are_subset_of_candidates(log):
    candidates = set(log.loc[log["Activity"] == "Wait", "Case ID"])
    _, gt = inject_deletion(log, target_activity="Wait", fraction=0.5, seed=11)
    assert set(gt.affected_case_ids) <= candidates


def test_affected_count_matches_fraction(log):
    candidates = log.loc[log["Activity"] == "Wait", "Case ID"].drop_duplicates().tolist()
    _, gt = inject_deletion(log, target_activity="Wait", fraction=0.4, seed=11)
    assert gt.n_affected_cases == max(1, round(len(candidates) * 0.4))


def test_no_candidates_is_safe(log):
    df_out, gt = inject_insertion(log, after_activity="NotPresent", new_activity="X",
                                  fraction=1.0, seed=0)
    assert gt.n_affected_cases == 0
    assert gt.affected_case_ids == []
    # df is structurally unchanged (same length, same activities)
    assert len(df_out) == len(log)


# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern,kwargs", [
    ("insertion",    dict(after_activity="B", new_activity="R", fraction=0.5)),
    ("deletion",     dict(target_activity="Wait", fraction=0.6)),
    ("substitution", dict(src_activity="Wait", dst_activity="QueuedWait", fraction=0.7)),
    ("loop",         dict(target_activity="C", fraction=0.8)),
])
def test_seed_determinism(log, pattern, kwargs):
    df_a, gt_a = inject(log, pattern, seed=2024, **kwargs)
    df_b, gt_b = inject(log, pattern, seed=2024, **kwargs)
    pd.testing.assert_frame_equal(df_a.reset_index(drop=True), df_b.reset_index(drop=True))
    assert gt_a.affected_case_ids == gt_b.affected_case_ids
    assert gt_a.n_events_changed == gt_b.n_events_changed


def test_different_seeds_give_different_selection(log):
    _, gt_a = inject_deletion(log, target_activity="Wait", fraction=0.5, seed=0)
    _, gt_b = inject_deletion(log, target_activity="Wait", fraction=0.5, seed=1)
    # not strictly required, but at this size collisions should be vanishingly rare
    assert gt_a.affected_case_ids != gt_b.affected_case_ids


# ---------------------------------------------------------------------------
# Ground-truth dict serialization (M6 will consume this)
# ---------------------------------------------------------------------------


def test_ground_truth_to_dict_is_json_safe(log):
    _, gt = inject_insertion(log, after_activity="B", new_activity="Review",
                             fraction=0.5, seed=42)
    d = gt.to_dict()
    import json
    payload = json.dumps(d)
    reloaded = json.loads(payload)
    assert reloaded["pattern"] == "insertion"
    assert reloaded["target_activity"] == "B"
    assert reloaded["secondary_activity"] == "Review"
    assert reloaded["n_affected_cases"] == gt.n_affected_cases
    assert reloaded["affected_case_ids"] == [str(c) for c in gt.affected_case_ids]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_unknown_pattern_raises(log):
    with pytest.raises(ValueError, match="unknown injection pattern"):
        inject(log, "parallelization", fraction=0.5, seed=0)


def test_dispatcher_routes_correctly(log):
    df_a, gt_a = inject(log, "deletion", target_activity="Wait", fraction=0.5, seed=42)
    df_b, gt_b = inject_deletion(log, target_activity="Wait", fraction=0.5, seed=42)
    pd.testing.assert_frame_equal(df_a.reset_index(drop=True), df_b.reset_index(drop=True))
    assert gt_a.affected_case_ids == gt_b.affected_case_ids


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_fraction_bounds(log):
    with pytest.raises(ValueError):
        inject_insertion(log, after_activity="B", new_activity="R", fraction=0.0, seed=0)
    with pytest.raises(ValueError):
        inject_insertion(log, after_activity="B", new_activity="R", fraction=1.5, seed=0)


def test_timestamps_remain_sorted_per_case_after_injection(log):
    df_out, _ = inject_insertion(log, after_activity="B", new_activity="Review",
                                 fraction=1.0, seed=0)
    for case_id, sub in df_out.groupby("Case ID"):
        ts = sub["Complete Timestamp"].tolist()
        assert ts == sorted(ts), f"case {case_id} timestamps not monotonic after insertion"


def test_loop_invalid_range_raises(log):
    with pytest.raises(ValueError):
        inject_loop(log, target_activity="C", fraction=0.5, repeat_range=(0, 3), seed=0)
    with pytest.raises(ValueError):
        inject_loop(log, target_activity="C", fraction=0.5, repeat_range=(3, 1), seed=0)
