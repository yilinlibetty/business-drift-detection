"""M1 -- Multi-scale Jensen-Shannon decomposition of process drift.

Replaces the previous ad-hoc ``max(TV(traces), W(durations)/median)`` aggregator
with a principled three-component vector that captures different *scales* of
control-flow change:

    activity-frequency scale  P(a)        global "what activities happen"
    direct-follows scale      P(a->b)     local "what activity transitions happen"
    trace-variant scale       P(sigma)    end-to-end "what whole paths happen"

The three components live in [0, 1] (because we use squared scipy
jensenshannon with base=2) so they are directly comparable to each other and
to any later [0, 1]-normalised optimal-transport score.

Public API:
    activity_frequency_dist(df, activity_col)              -> pd.Series
    dfg_dist(df, case_id_col, activity_col, timestamp_col) -> pd.Series  (MultiIndex (a, b))
    trace_variant_dist(df, ...)                            -> pd.Series  (tuple-of-str index)
    jsd(p, q)                                              -> float in [0, 1]
    align(s1, s2)                                          -> (np.ndarray, np.ndarray)
    multi_scale_drift(df_a, df_b, ...)                     -> dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def activity_frequency_dist(df: pd.DataFrame, activity_col: str = "Activity") -> pd.Series:
    """Empirical P(a) over individual events."""
    if activity_col not in df.columns:
        raise KeyError(f"missing column {activity_col!r}")
    counts = df[activity_col].value_counts(normalize=True)
    counts.index.name = "activity"
    counts.name = "p"
    return counts


def dfg_dist(
    df: pd.DataFrame,
    case_id_col: str = "Case ID",
    activity_col: str = "Activity",
    timestamp_col: str = "Complete Timestamp",
) -> pd.Series:
    """Empirical P(a -> b) over directly-follows pairs WITHIN each case.

    The probability mass for each pair is normalised over all directly-follows
    pairs in the log (a case with k events contributes k-1 pairs).  Cases with
    a single event contribute no mass.
    """
    for c in (case_id_col, activity_col, timestamp_col):
        if c not in df.columns:
            raise KeyError(f"missing column {c!r}")

    df = df.sort_values([case_id_col, timestamp_col])
    acts = df[activity_col].astype(str).values
    cases = df[case_id_col].astype(object).values

    # follow pairs only WITHIN the same case
    same_case = cases[1:] == cases[:-1]
    src = acts[:-1][same_case]
    dst = acts[1:][same_case]

    if src.size == 0:
        return pd.Series(dtype=float, name="p")

    pairs = pd.MultiIndex.from_arrays([src, dst], names=("from", "to"))
    counts = pd.Series(1, index=pairs).groupby(level=("from", "to")).sum()
    return (counts / counts.sum()).rename("p")


def trace_variant_dist(
    df: pd.DataFrame,
    case_id_col: str = "Case ID",
    activity_col: str = "Activity",
    timestamp_col: str = "Complete Timestamp",
) -> pd.Series:
    """Empirical P(sigma) over distinct trace variants (tuple-of-activity)."""
    for c in (case_id_col, activity_col, timestamp_col):
        if c not in df.columns:
            raise KeyError(f"missing column {c!r}")

    df = df.sort_values([case_id_col, timestamp_col])
    variants = (
        df.groupby(case_id_col, sort=False)[activity_col]
        .apply(lambda s: tuple(map(str, s.tolist())))
    )
    counts = variants.value_counts(normalize=True)
    counts.name = "p"
    return counts


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------


def align(s1: pd.Series, s2: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Reindex two probability Series onto the union of their supports with 0-fill.

    Returns two same-length float arrays. If either input is empty the
    returned arrays are empty and downstream `jsd` returns 0.
    """
    if len(s1) == 0 and len(s2) == 0:
        return np.zeros(0), np.zeros(0)
    union = s1.index.union(s2.index)
    a = s1.reindex(union, fill_value=0.0).to_numpy(dtype=float)
    b = s2.reindex(union, fill_value=0.0).to_numpy(dtype=float)
    return a, b


def jsd(p, q, base: float = 2.0) -> float:
    """Jensen-Shannon **divergence** (not the metric).

    Uses scipy's ``jensenshannon`` (which returns sqrt(JSD)) and squares the
    result so the return value is the divergence itself, in [0, log_{base}(2)].
    With ``base=2`` the range is [0, 1].
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.size == 0 or q.size == 0 or p.sum() == 0 or q.sum() == 0:
        return 0.0
    if p.shape != q.shape:
        raise ValueError(
            f"jsd: shape mismatch {p.shape} vs {q.shape}; align distributions first."
        )
    # scipy returns sqrt of divergence
    d = jensenshannon(p, q, base=base)
    if np.isnan(d):
        return 0.0
    return float(d * d)


# ---------------------------------------------------------------------------
# Top-level aggregator
# ---------------------------------------------------------------------------


def multi_scale_drift(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    case_id_col: str = "Case ID",
    activity_col: str = "Activity",
    timestamp_col: str = "Complete Timestamp",
) -> dict:
    """Compute the three-scale drift vector between two event-log slices.

    Returns dict with keys: ``activity_jsd``, ``dfg_jsd``, ``trace_jsd``.
    Each value is a Python float in [0, 1].
    """
    pa = activity_frequency_dist(df_a, activity_col)
    pb = activity_frequency_dist(df_b, activity_col)
    dfg_a = dfg_dist(df_a, case_id_col, activity_col, timestamp_col)
    dfg_b = dfg_dist(df_b, case_id_col, activity_col, timestamp_col)
    tv_a = trace_variant_dist(df_a, case_id_col, activity_col, timestamp_col)
    tv_b = trace_variant_dist(df_b, case_id_col, activity_col, timestamp_col)

    return {
        "activity_jsd": jsd(*align(pa, pb)),
        "dfg_jsd":      jsd(*align(dfg_a, dfg_b)),
        "trace_jsd":    jsd(*align(tv_a, tv_b)),
    }
