"""Tests for drift.localization (Phase 4 / M3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drift.localization import (
    bootstrap_change_point_ci,
    compute_drift_signal,
    detect_change_points,
    signal_index_to_case_position,
)


# ---------------------------------------------------------------------------
# Synthetic signals
# ---------------------------------------------------------------------------


def _step_signal(n: int = 200, step_at: int = 100, low: float = 0.05, high: float = 0.40, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.where(np.arange(n) < step_at, low, high)
    return base + rng.normal(scale=noise, size=n)


def _flat_signal(n: int = 200, level: float = 0.10, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.full(n, level) + rng.normal(scale=noise, size=n)


# ---------------------------------------------------------------------------
# detect_change_points
# ---------------------------------------------------------------------------


def test_step_signal_detects_one_cp_near_step():
    s = _step_signal(n=200, step_at=100, seed=0)
    cps = detect_change_points(s)
    assert len(cps) >= 1
    nearest = min(cps, key=lambda c: abs(c - 100))
    assert abs(nearest - 100) <= 5, f"expected CP near 100, got {cps}"


def test_flat_signal_returns_no_cps():
    s = _flat_signal(n=200, seed=0)
    cps = detect_change_points(s)
    assert cps == [], f"flat signal should produce no CPs, got {cps}"


def test_signal_with_two_steps():
    rng = np.random.default_rng(0)
    n = 300
    base = np.zeros(n)
    base[100:200] = 0.3
    base[200:] = 0.1
    s = base + rng.normal(scale=0.02, size=n)
    cps = detect_change_points(s)
    # We expect at least two CPs near 100 and 200.
    assert len(cps) >= 2
    # Each true CP must have a detected CP within +-5.
    for true_cp in (100, 200):
        nearest = min(cps, key=lambda c: abs(c - true_cp))
        assert abs(nearest - true_cp) <= 5, f"CP near {true_cp} missing; got {cps}"


def test_multivariate_signal_supported():
    rng = np.random.default_rng(0)
    n = 200
    a = np.where(np.arange(n) < 100, 0.05, 0.40) + rng.normal(scale=0.02, size=n)
    b = np.where(np.arange(n) < 100, 0.10, 0.20) + rng.normal(scale=0.02, size=n)
    sig = np.column_stack([a, b])
    cps = detect_change_points(sig)
    assert len(cps) >= 1
    assert abs(min(cps, key=lambda c: abs(c - 100)) - 100) <= 5


def test_too_short_signal_returns_empty():
    assert detect_change_points(np.array([1.0])) == []
    assert detect_change_points(np.array([1.0, 2.0])) == []


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_contains_observed_cp():
    s = _step_signal(n=200, step_at=100, seed=0)
    cps = detect_change_points(s)
    cis = bootstrap_change_point_ci(s, cps, B=40, seed=0)
    assert len(cis) == len(cps)
    for ci, c in zip(cis, cps):
        assert ci["cp_index"] == c
        assert ci["ci_lo"] <= c <= ci["ci_hi"]


def test_bootstrap_ci_is_reproducible():
    s = _step_signal(n=200, step_at=100, seed=0)
    cps = detect_change_points(s)
    ci_a = bootstrap_change_point_ci(s, cps, B=20, seed=42)
    ci_b = bootstrap_change_point_ci(s, cps, B=20, seed=42)
    assert ci_a == ci_b


def test_bootstrap_empty_cps_is_safe():
    s = _flat_signal(n=100, seed=0)
    assert bootstrap_change_point_ci(s, [], B=10, seed=0) == []


# ---------------------------------------------------------------------------
# compute_drift_signal on a small synthetic log
# ---------------------------------------------------------------------------


def _build_two_phase_log(n_per_phase: int = 80, drift_at_phase: bool = True, seed: int = 0) -> pd.DataFrame:
    """Cases 0..n_per_phase-1 follow A-B-C-D; cases n_per_phase..2n follow
    A-B-Wait-C-D (if drift_at_phase) or also A-B-C-D (no drift).
    """
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2024-01-01")
    for i in range(2 * n_per_phase):
        if i < n_per_phase:
            seq = ["A", "B", "C", "D"]
        else:
            seq = ["A", "B", "Wait", "C", "D"] if drift_at_phase else ["A", "B", "C", "D"]
        for j, act in enumerate(seq):
            rows.append({
                "Case ID": f"c{i:04d}",
                "Activity": act,
                "Complete Timestamp": base + pd.Timedelta(minutes=10 * i + j),
            })
    return pd.DataFrame(rows)


def test_compute_drift_signal_shapes():
    log = _build_two_phase_log(n_per_phase=200, drift_at_phase=True, seed=0)
    out = compute_drift_signal(log, window=80, step=10)
    n_windows = (400 - 80) // 10 + 1
    assert out["signal"].shape == (n_windows, 3)
    assert out["case_positions"].shape == (n_windows,)
    assert out["scale_names"] == ("activity_jsd", "dfg_jsd", "trace_jsd")
    assert out["n_cases"] == 400


def test_compute_drift_signal_first_window_is_zero():
    log = _build_two_phase_log(n_per_phase=200, drift_at_phase=True, seed=0)
    out = compute_drift_signal(log, window=80, step=10)
    # The first window IS the baseline, so its drift must be exactly 0.
    assert np.allclose(out["signal"][0], 0.0)


def test_compute_drift_signal_detects_phase_change():
    log = _build_two_phase_log(n_per_phase=200, drift_at_phase=True, seed=0)
    out = compute_drift_signal(log, window=80, step=10)
    cps = detect_change_points(out["signal"])
    assert len(cps) >= 1
    # True drift onset = case position ~200. Granularity of step=10.
    cp_case_positions = [signal_index_to_case_position(c, out) for c in cps]
    nearest = min(cp_case_positions, key=lambda p: abs(p - 200))
    # Window-step granularity is 10 cases; allow some slack from window-mixing.
    assert abs(nearest - 200) <= 80, f"CP at case pos {nearest}, expected near 200; cps={cp_case_positions}"


def test_compute_drift_signal_no_drift_yields_quiet_signal():
    log = _build_two_phase_log(n_per_phase=200, drift_at_phase=False, seed=0)
    out = compute_drift_signal(log, window=80, step=10)
    # All multi-scale components should stay small (< 0.10).
    assert (out["signal"] < 0.10).all(), f"unexpected drift in no-drift log: max={out['signal'].max():.3f}"


def test_compute_drift_signal_validates_window_step():
    log = _build_two_phase_log(n_per_phase=20, drift_at_phase=True, seed=0)
    with pytest.raises(ValueError, match="window"):
        compute_drift_signal(log, window=1, step=5)
    with pytest.raises(ValueError, match="step"):
        compute_drift_signal(log, window=10, step=0)
    with pytest.raises(ValueError, match="at least"):
        compute_drift_signal(log, window=200, step=10)
