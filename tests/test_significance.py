"""Tests for drift.significance (Phase 3 / M4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest

from drift.injection import inject_insertion
from drift.metrics import (
    activity_frequency_dist,
    align,
    jsd,
    multi_scale_drift,
)
from drift.significance import permutation_pvalue


# ---------------------------------------------------------------------------
# Statistics used by the tests
# ---------------------------------------------------------------------------


def _activity_jsd_stat(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    pa = activity_frequency_dist(df_a)
    pb = activity_frequency_dist(df_b)
    return jsd(*align(pa, pb))


def _aggregate_drift_stat(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    d = multi_scale_drift(df_a, df_b)
    return max(d.values())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_homogeneous_log(n_cases: int = 80, seed: int = 0) -> pd.DataFrame:
    """All cases drawn from the SAME generative process.

    Four trace variants are used (not two) so the resulting statistic
    distribution has enough granularity to avoid pathological ties when used as
    the H0 sample for the uniformity test. With only two variants and an
    activity-only statistic, permutation values collapse onto a coarse grid
    and the conservative Phipson-Smyth ">=" estimator biases p-values upward.
    """
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(n_cases):
        r = rng.random()
        if r < 0.4:
            seq = ["A", "B", "C", "D"]
        elif r < 0.7:
            seq = ["A", "B", "Wait", "C", "D"]
        elif r < 0.9:
            seq = ["A", "X", "C", "D"]
        else:
            seq = ["A", "B", "C", "Y", "D"]
        for j, act in enumerate(seq):
            rows.append({
                "Case ID": f"c{i:04d}",
                "Activity": act,
                "Complete Timestamp": base + pd.Timedelta(minutes=10 * i + j),
            })
    return pd.DataFrame(rows)


def _random_split(df: pd.DataFrame, seed: int, case_id_col: str = "Case ID"):
    rng = np.random.default_rng(seed)
    case_ids = df[case_id_col].drop_duplicates().to_numpy()
    rng.shuffle(case_ids)
    mid = len(case_ids) // 2
    a_ids, b_ids = set(case_ids[:mid]), set(case_ids[mid:])
    return df[df[case_id_col].isin(a_ids)], df[df[case_id_col].isin(b_ids)]


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


def test_pvalue_in_unit_interval():
    log = _make_homogeneous_log(40, seed=0)
    a, b = _random_split(log, seed=0)
    p = permutation_pvalue(a, b, _activity_jsd_stat, B=50, seed=0)
    assert 0.0 < p <= 1.0


def test_pvalue_is_reproducible():
    log = _make_homogeneous_log(40, seed=0)
    a, b = _random_split(log, seed=0)
    p1 = permutation_pvalue(a, b, _activity_jsd_stat, B=50, seed=42)
    p2 = permutation_pvalue(a, b, _activity_jsd_stat, B=50, seed=42)
    assert p1 == p2


def test_pvalue_unbiased_floor():
    """With B=99 the minimum possible p-value is 1/100."""
    log = _make_homogeneous_log(40, seed=0)
    a, b = _random_split(log, seed=0)
    p = permutation_pvalue(a, b, _activity_jsd_stat, B=99, seed=0)
    assert p >= 1 / 100


def test_empty_input_raises():
    log = _make_homogeneous_log(20)
    a, _ = _random_split(log, seed=0)
    empty = a.iloc[0:0]
    with pytest.raises(ValueError, match="no cases"):
        permutation_pvalue(empty, a, _activity_jsd_stat, B=10, seed=0)


def test_overlapping_cases_raise():
    log = _make_homogeneous_log(20)
    # Same DF on both sides -- case ids overlap.
    with pytest.raises(ValueError, match="not disjoint"):
        permutation_pvalue(log, log, _activity_jsd_stat, B=10, seed=0)


# ---------------------------------------------------------------------------
# H1 detection: real drift -> small p-value
# ---------------------------------------------------------------------------


def test_real_drift_yields_small_pvalue():
    log = _make_homogeneous_log(120, seed=0)
    base, curr = _random_split(log, seed=0)
    curr_inj, _ = inject_insertion(curr, after_activity="B", new_activity="Review",
                                   fraction=1.0, seed=0)
    p = permutation_pvalue(base, curr_inj, _aggregate_drift_stat, B=200, seed=0)
    # With full-fraction insertion of a brand-new activity into half the log,
    # the observed stat dominates every plausible permutation.
    assert p < 0.05


# ---------------------------------------------------------------------------
# H0 uniformity: p-values are approximately Uniform(0,1)
# ---------------------------------------------------------------------------


def test_h0_pvalues_uniform_under_null():
    """Under H0 (no drift), permutation p-values should be roughly Uniform(0,1).

    We run 40 independent splits of a homogeneous log and use the multi-scale
    aggregate (richer statistic -- avoids tie pathologies of a single-activity
    JSD). The resulting p-values must not be rejected as non-uniform by a KS
    test at alpha=0.01.
    """
    rng = np.random.default_rng(2024)
    log = _make_homogeneous_log(140, seed=0)
    pvals = []
    for _ in range(40):
        split_seed = int(rng.integers(0, 10_000))
        perm_seed = int(rng.integers(0, 10_000))
        a, b = _random_split(log, seed=split_seed)
        pvals.append(permutation_pvalue(a, b, _aggregate_drift_stat, B=49, seed=perm_seed))
    pvals = np.array(pvals)

    ks_stat, ks_p = kstest(pvals, "uniform")
    assert ks_p > 0.01, (
        f"H0 p-values not uniform: KS={ks_stat:.3f} (p={ks_p:.4f}). "
        f"Mean={pvals.mean():.3f} (target ~0.5)."
    )
    # also a coarse sanity bound on the empirical mean
    assert 0.30 < pvals.mean() < 0.70, f"H0 p-value mean off: {pvals.mean():.3f}"
