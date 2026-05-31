"""M4 -- Permutation-test p-values for any drift statistic.

Replaces the hand-picked detection threshold (was 0.05 on TV) with a
distribution-free significance test:

    H0 : the case populations on the two sides come from the same distribution.
    Reject when an unusually large value of statistic_fn(a, b) is observed
    relative to the distribution obtained by reshuffling case-to-side labels
    while keeping events within each case intact.

The permutation unit is the **case**, not the event -- shuffling events would
destroy trace structure and trivialise every structural statistic.

Public API:
    permutation_pvalue(events_a, events_b, statistic_fn, B=200, seed=0,
                       case_id_col="Case ID") -> float in (0, 1]
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def permutation_pvalue(
    events_a: pd.DataFrame,
    events_b: pd.DataFrame,
    statistic_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    B: int = 200,
    seed: int = 0,
    case_id_col: str = "Case ID",
) -> float:
    """One-sided permutation p-value.

    Parameters
    ----------
    events_a, events_b : DataFrame
        Two flat event-log slices. They must share the same schema and the
        case-id column ``case_id_col``. Cases must be disjoint between the two
        sides (this is the normal "baseline vs current" split convention).
    statistic_fn : callable
        ``statistic_fn(df_a, df_b) -> float``. Larger values must indicate
        more drift; a permutation that yields ``>= observed`` counts as
        "as extreme as observed".
    B : int
        Number of permutations. 200 is sufficient for FYP-scale precision and
        keeps the runtime under a few seconds for the multi-scale aggregator.
    seed : int
        RNG seed for reproducibility.
    case_id_col : str
        Name of the case-id column used as the permutation unit.

    Returns
    -------
    p : float in (1/(B+1), 1]
        ``(#{permuted >= observed} + 1) / (B + 1)``  --  the "+1 on both sides"
        guard is the standard unbiased estimator (see Phipson & Smyth, 2010)
        and bounds the p-value strictly above zero.
    """
    if case_id_col not in events_a.columns or case_id_col not in events_b.columns:
        raise KeyError(f"both event frames must contain column {case_id_col!r}")
    if B < 1:
        raise ValueError(f"B must be >= 1, got {B}")

    case_ids_a = events_a[case_id_col].drop_duplicates().tolist()
    case_ids_b = events_b[case_id_col].drop_duplicates().tolist()
    n_a = len(case_ids_a)
    n_b = len(case_ids_b)
    if n_a == 0 or n_b == 0:
        raise ValueError("permutation_pvalue: one of the event frames has no cases")

    # Index pooled events by case id once -- avoids re-filtering pooled DF per draw.
    pooled = pd.concat([events_a, events_b], ignore_index=True)
    case_to_events = {cid: sub for cid, sub in pooled.groupby(case_id_col, sort=False)}
    all_case_ids = np.array(list(case_to_events.keys()), dtype=object)
    if len(all_case_ids) != n_a + n_b:
        raise ValueError(
            "permutation_pvalue: cases are not disjoint between the two sides; "
            "found overlapping case ids in pooled events."
        )

    observed = float(statistic_fn(events_a, events_b))
    rng = np.random.default_rng(seed)

    n_extreme = 1  # +1 for the observed; Phipson-Smyth unbiased estimator
    for _ in range(B):
        perm = rng.permutation(len(all_case_ids))
        side_a_ids = all_case_ids[perm[:n_a]]
        side_b_ids = all_case_ids[perm[n_a:]]
        df_a = pd.concat([case_to_events[c] for c in side_a_ids], ignore_index=True)
        df_b = pd.concat([case_to_events[c] for c in side_b_ids], ignore_index=True)
        stat = float(statistic_fn(df_a, df_b))
        if stat >= observed:
            n_extreme += 1

    return n_extreme / (B + 1)
