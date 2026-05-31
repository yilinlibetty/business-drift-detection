"""M3 -- Sliding-window drift signal + PELT change-point localization.

Replaces the previous 50/50 case split (which assumed drift falls exactly at
the midpoint) with:

    1. A sliding-window drift signal indexed by case-completion order.
       Each window is compared against the *first* window so the signal
       represents "distance from initial state" over time.
    2. PELT change-point detection (Killick et al. 2012) on that signal,
       implemented via the ``ruptures`` package with an rbf kernel cost
       (handles multivariate signals).
    3. Optional bootstrap CIs around each detected change point using a
       noise-perturbation scheme that preserves temporal structure.

Public API:
    compute_drift_signal(df, window, step, ...) -> dict
    detect_change_points(signal, pen=None)      -> list[int]
    bootstrap_change_point_ci(signal, cps, B=100, seed=0, pen=None) -> list[dict]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt

from drift.metrics import multi_scale_drift


# ---------------------------------------------------------------------------
# Sliding-window drift signal
# ---------------------------------------------------------------------------


def compute_drift_signal(
    df: pd.DataFrame,
    window: int = 100,
    step: int = 20,
    case_id_col: str = "Case ID",
    activity_col: str = "Activity",
    timestamp_col: str = "Complete Timestamp",
) -> dict:
    """Sliding-window multi-scale drift signal, baseline = first window.

    Cases are ordered by their completion timestamp; the *case position* (1-based
    in completion order) is what change-point indices later refer to.

    Returns dict:
        case_positions : np.ndarray of int    (window-END case position)
        signal         : np.ndarray (n_windows, 3)  (activity_jsd, dfg_jsd, trace_jsd)
        scale_names    : tuple[str, str, str]
        window         : int
        step           : int
        n_cases        : int
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    for c in (case_id_col, activity_col, timestamp_col):
        if c not in df.columns:
            raise KeyError(f"missing column {c!r}")

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Case-completion-time order
    case_end = (
        df.groupby(case_id_col)[timestamp_col]
        .max()
        .sort_values()
    )
    ordered_case_ids = case_end.index.to_list()
    n_cases = len(ordered_case_ids)

    if n_cases < window:
        raise ValueError(
            f"need at least window={window} cases; log has only {n_cases}"
        )

    # Pre-index events by case for O(1) per-window slicing.
    events_by_case = {cid: sub for cid, sub in df.groupby(case_id_col, sort=False)}

    def _events_for(case_id_subset):
        return pd.concat(
            [events_by_case[c] for c in case_id_subset], ignore_index=True
        )

    baseline_cases = ordered_case_ids[:window]
    baseline_events = _events_for(baseline_cases)

    case_positions = []
    rows = []
    for start in range(0, n_cases - window + 1, step):
        end = start + window
        window_cases = ordered_case_ids[start:end]
        window_events = _events_for(window_cases)
        d = multi_scale_drift(
            baseline_events,
            window_events,
            case_id_col=case_id_col,
            activity_col=activity_col,
            timestamp_col=timestamp_col,
        )
        rows.append([d["activity_jsd"], d["dfg_jsd"], d["trace_jsd"]])
        case_positions.append(end)

    signal = np.asarray(rows, dtype=float)
    return {
        "case_positions": np.asarray(case_positions, dtype=int),
        "signal": signal,
        "scale_names": ("activity_jsd", "dfg_jsd", "trace_jsd"),
        "window": window,
        "step": step,
        "n_cases": n_cases,
    }


# ---------------------------------------------------------------------------
# Change-point detection
# ---------------------------------------------------------------------------


def _bic_penalty(signal: np.ndarray) -> float:
    """SIC-style penalty: d * log(N).

    The rbf kernel cost in ruptures is scale-invariant (bandwidth chosen by
    median pairwise distance), so the penalty does NOT need to depend on the
    signal's variance.  Empirically this constant suppresses noise-driven
    false CPs on flat signals (verified up to sigma=0.2 across 1-5 dims) while
    still detecting ramp-shaped transitions on short multivariate signals.

    For a multivariate signal of shape (n, d), the penalty scales with d so
    the per-feature evidence threshold is held constant.
    """
    if signal.ndim == 1:
        n, d = signal.shape[0], 1
    else:
        n, d = signal.shape
    return float(d * np.log(max(n, 2)))


def detect_change_points(
    signal: np.ndarray,
    pen: float | None = None,
    model: str = "rbf",
    min_size: int = 2,
) -> list[int]:
    """PELT change-point detection on a (possibly multivariate) signal.

    Returns a list of internal change-point indices (i.e. excludes the trivial
    boundary at ``len(signal)`` that ``ruptures`` appends to every prediction).
    Indices index INTO ``signal``, not into the original case sequence.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n = signal.shape[0]
    if n < 2 * min_size:
        return []

    if pen is None:
        pen = _bic_penalty(signal)

    algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
    bkps = algo.predict(pen=pen)
    # ruptures appends the trivial end boundary; strip it
    return [int(b) for b in bkps if b < n]


def signal_index_to_case_position(idx: int, drift_signal: dict) -> int:
    """Map a signal index (0..n_windows-1) to the case-end position it represents."""
    pos = drift_signal["case_positions"]
    if not 0 <= idx < len(pos):
        raise IndexError(f"signal index {idx} out of range [0, {len(pos)})")
    return int(pos[idx])


# ---------------------------------------------------------------------------
# Bootstrap CIs around detected change points
# ---------------------------------------------------------------------------


def bootstrap_change_point_ci(
    signal: np.ndarray,
    cps: list[int],
    B: int = 100,
    seed: int = 0,
    pen: float | None = None,
    model: str = "rbf",
    min_size: int = 2,
    match_window: int | None = None,
) -> list[dict]:
    """95% bootstrap CIs around each detected change point.

    Bootstrap scheme: additive Gaussian noise scaled by the empirical per-dim
    std of the original signal. This preserves temporal structure (no shuffling
    or block-resampling distortion) and is appropriate for short signals where
    classical resampling has too few samples per block.

    For each canonical change point ``c`` in ``cps``, the CI is the empirical
    2.5/97.5 percentile of the *nearest* bootstrap change point within
    ``match_window`` of ``c`` (default: 2 * mean inter-cp spacing).  Bootstrap
    iterations that have no neighbour within the match window are dropped from
    that CP's CI estimate.

    Returns one dict per element of ``cps``:
        {cp_index, ci_lo, ci_hi, n_matched, B}
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n = signal.shape[0]
    if not cps:
        return []
    if match_window is None:
        if len(cps) >= 2:
            match_window = int(np.mean(np.diff([0] + list(cps) + [n])))
        else:
            match_window = max(2, n // 4)

    rng = np.random.default_rng(seed)
    noise_scale = signal.std(axis=0)
    # avoid zero-std dims contributing nothing or NaN
    noise_scale = np.where(noise_scale > 0, noise_scale, 1e-9)

    if pen is None:
        pen = _bic_penalty(signal)

    bootstrap_cps_per_iter: list[list[int]] = []
    for _ in range(B):
        noisy = signal + rng.normal(scale=noise_scale, size=signal.shape)
        bs_cps = detect_change_points(noisy, pen=pen, model=model, min_size=min_size)
        bootstrap_cps_per_iter.append(bs_cps)

    out = []
    for c in cps:
        nearest = []
        for bs_cps in bootstrap_cps_per_iter:
            if not bs_cps:
                continue
            j = min(bs_cps, key=lambda x: abs(x - c))
            if abs(j - c) <= match_window:
                nearest.append(j)
        if nearest:
            lo = int(np.percentile(nearest, 2.5))
            hi = int(np.percentile(nearest, 97.5))
        else:
            lo = hi = int(c)
        out.append({
            "cp_index": int(c),
            "ci_lo": lo,
            "ci_hi": hi,
            "n_matched": len(nearest),
            "B": B,
        })
    return out
