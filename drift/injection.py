"""M5 -- Bose-style controlled drift injection on a flat event-log DataFrame.

The four patterns implemented are a pragmatic subset of Bose et al. (2014)
control-flow drift patterns:

    insertion       -- insert a new activity after every occurrence of an anchor
                       activity in a randomly chosen fraction of cases. Models
                       a newly added step (e.g. an approval gate that did not
                       exist before).
    deletion        -- remove every occurrence of a target activity in a
                       randomly chosen fraction of cases. Models the "skip" of
                       a previously mandatory step.
    substitution    -- replace every occurrence of src activity with dst
                       activity in a randomly chosen fraction of cases. Models
                       the rename / re-routing of a step.
    loop            -- duplicate every occurrence of a target activity
                       repeat_count times (drawn uniformly per affected case
                       from the supplied range) in a randomly chosen fraction
                       of cases. Models the appearance of rework loops.

`parallelization` is intentionally omitted -- it is fiddly to realize on real
logs without a known parallel block, and substitution already covers the
structural-drift narrative.

Every function returns ``(df_modified, ground_truth)`` where ``ground_truth``
captures the information M6 (LLM evaluation) needs to grade an analyst's
report against what was actually injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ground-truth payload
# ---------------------------------------------------------------------------


@dataclass
class GroundTruth:
    """Schema-stable container for what an injection actually did.

    The fields are pre-grouped into "what the analyst should be able to claim"
    (pattern, target_activity, secondary_activity) and "what happened
    quantitatively" (fraction, n_affected_cases, n_events_changed, seed).
    """

    pattern: str
    target_activity: str
    secondary_activity: str | None
    fraction: float
    seed: int
    affected_case_ids: list[Any] = field(default_factory=list)
    n_affected_cases: int = 0
    n_events_changed: int = 0
    description: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON does not like sets / tuples / numpy ints
        d["affected_case_ids"] = [
            _jsonify_case_id(cid) for cid in self.affected_case_ids
        ]
        return d


def _jsonify_case_id(cid):
    if isinstance(cid, (np.integer,)):
        return int(cid)
    return cid


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _select_affected_cases(
    df: pd.DataFrame,
    case_id_col: str,
    candidate_mask: pd.Series,
    fraction: float,
    seed: int,
) -> list:
    """Choose, deterministically from ``seed``, which case IDs are affected.

    Candidates are the cases for which ``candidate_mask`` is True on at least
    one of their events.  The result is a sorted list (for reproducibility
    across pandas hash-table randomization).
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
    candidate_case_ids = (
        df.loc[candidate_mask, case_id_col].drop_duplicates().tolist()
    )
    candidate_case_ids = sorted(candidate_case_ids, key=str)
    if not candidate_case_ids:
        return []
    rng = np.random.default_rng(seed)
    n_select = max(1, int(round(len(candidate_case_ids) * fraction)))
    n_select = min(n_select, len(candidate_case_ids))
    chosen_idx = rng.choice(len(candidate_case_ids), size=n_select, replace=False)
    return [candidate_case_ids[i] for i in sorted(chosen_idx)]


def _resolve_columns(
    df: pd.DataFrame,
    case_id_col: str,
    activity_col: str,
    timestamp_col: str,
) -> None:
    missing = [c for c in (case_id_col, activity_col, timestamp_col) if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing required columns: {missing}")


def _midpoint_timestamp(
    case_rows: pd.DataFrame,
    insertion_row_pos: int,
    timestamp_col: str,
    fallback_seconds: int = 1,
) -> pd.Timestamp:
    """Pick a timestamp midway between event ``insertion_row_pos`` and the next.

    If the anchor event is the last one in the case, return anchor_ts +
    ``fallback_seconds``.
    """
    anchor_ts = case_rows.iloc[insertion_row_pos][timestamp_col]
    if insertion_row_pos + 1 < len(case_rows):
        next_ts = case_rows.iloc[insertion_row_pos + 1][timestamp_col]
        return anchor_ts + (next_ts - anchor_ts) / 2
    return anchor_ts + pd.Timedelta(seconds=fallback_seconds)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


def inject_insertion(
    df: pd.DataFrame,
    after_activity: str,
    new_activity: str,
    fraction: float,
    seed: int = 0,
    *,
    case_id_col: str = "Case ID",
    activity_col: str = "Activity",
    timestamp_col: str = "Complete Timestamp",
) -> tuple[pd.DataFrame, GroundTruth]:
    """Insert ``new_activity`` immediately after every occurrence of
    ``after_activity`` in a randomly chosen fraction of cases.

    The new event's timestamp is the midpoint between the anchor event and the
    next event in the case (or anchor_ts + 1s if the anchor is the last event).
    """
    _resolve_columns(df, case_id_col, activity_col, timestamp_col)
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    candidate_mask = df[activity_col] == after_activity
    affected = _select_affected_cases(df, case_id_col, candidate_mask, fraction, seed)

    if not affected:
        gt = GroundTruth(
            pattern="insertion",
            target_activity=after_activity,
            secondary_activity=new_activity,
            fraction=fraction,
            seed=seed,
            description=f"Insert '{new_activity}' after '{after_activity}' (no candidate cases)",
        )
        return df.sort_values([case_id_col, timestamp_col]).reset_index(drop=True), gt

    affected_set = set(affected)
    new_rows = []
    template_row = df.iloc[0].to_dict()  # for filling non-required columns

    for case_id in affected:
        case_rows = (
            df[df[case_id_col] == case_id]
            .sort_values(timestamp_col)
            .reset_index(drop=True)
        )
        # itertuples() mangles non-identifier column names (e.g. 'concept:name')
        # to internal _N positions; iterate by position + column lookup instead.
        activity_col_values = case_rows[activity_col].to_numpy()
        for pos in np.where(activity_col_values == after_activity)[0]:
            new_ts = _midpoint_timestamp(case_rows, pos, timestamp_col)
            new_row = template_row.copy()
            new_row[case_id_col] = case_id
            new_row[activity_col] = new_activity
            new_row[timestamp_col] = new_ts
            new_rows.append(new_row)

    df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df_aug = df_aug.sort_values([case_id_col, timestamp_col]).reset_index(drop=True)

    gt = GroundTruth(
        pattern="insertion",
        target_activity=after_activity,
        secondary_activity=new_activity,
        fraction=fraction,
        seed=seed,
        affected_case_ids=list(affected),
        n_affected_cases=len(affected_set),
        n_events_changed=len(new_rows),
        description=(
            f"Insert '{new_activity}' after every occurrence of '{after_activity}' "
            f"in {len(affected_set)} cases ({len(new_rows)} new events)."
        ),
    )
    return df_aug, gt


def inject_deletion(
    df: pd.DataFrame,
    target_activity: str,
    fraction: float,
    seed: int = 0,
    *,
    case_id_col: str = "Case ID",
    activity_col: str = "Activity",
    timestamp_col: str = "Complete Timestamp",
) -> tuple[pd.DataFrame, GroundTruth]:
    """Remove every occurrence of ``target_activity`` from a randomly chosen
    fraction of cases (a 'skip-step' drift).
    """
    _resolve_columns(df, case_id_col, activity_col, timestamp_col)
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    candidate_mask = df[activity_col] == target_activity
    affected = _select_affected_cases(df, case_id_col, candidate_mask, fraction, seed)

    if not affected:
        gt = GroundTruth(
            pattern="deletion",
            target_activity=target_activity,
            secondary_activity=None,
            fraction=fraction,
            seed=seed,
            description=f"Delete '{target_activity}' (no candidate cases)",
        )
        return df.sort_values([case_id_col, timestamp_col]).reset_index(drop=True), gt

    affected_set = set(affected)
    drop_mask = df[case_id_col].isin(affected_set) & (df[activity_col] == target_activity)
    n_dropped = int(drop_mask.sum())
    df_out = df.loc[~drop_mask].copy()
    df_out = df_out.sort_values([case_id_col, timestamp_col]).reset_index(drop=True)

    gt = GroundTruth(
        pattern="deletion",
        target_activity=target_activity,
        secondary_activity=None,
        fraction=fraction,
        seed=seed,
        affected_case_ids=list(affected),
        n_affected_cases=len(affected_set),
        n_events_changed=n_dropped,
        description=(
            f"Remove every occurrence of '{target_activity}' from {len(affected_set)} "
            f"cases ({n_dropped} events dropped)."
        ),
    )
    return df_out, gt


def inject_substitution(
    df: pd.DataFrame,
    src_activity: str,
    dst_activity: str,
    fraction: float,
    seed: int = 0,
    *,
    case_id_col: str = "Case ID",
    activity_col: str = "Activity",
    timestamp_col: str = "Complete Timestamp",
) -> tuple[pd.DataFrame, GroundTruth]:
    """Replace every occurrence of ``src_activity`` with ``dst_activity`` in a
    randomly chosen fraction of cases. Event count is preserved.
    """
    _resolve_columns(df, case_id_col, activity_col, timestamp_col)
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    candidate_mask = df[activity_col] == src_activity
    affected = _select_affected_cases(df, case_id_col, candidate_mask, fraction, seed)

    if not affected:
        gt = GroundTruth(
            pattern="substitution",
            target_activity=src_activity,
            secondary_activity=dst_activity,
            fraction=fraction,
            seed=seed,
            description=f"Replace '{src_activity}' -> '{dst_activity}' (no candidate cases)",
        )
        return df.sort_values([case_id_col, timestamp_col]).reset_index(drop=True), gt

    affected_set = set(affected)
    sub_mask = df[case_id_col].isin(affected_set) & (df[activity_col] == src_activity)
    n_changed = int(sub_mask.sum())
    df.loc[sub_mask, activity_col] = dst_activity
    df_out = df.sort_values([case_id_col, timestamp_col]).reset_index(drop=True)

    gt = GroundTruth(
        pattern="substitution",
        target_activity=src_activity,
        secondary_activity=dst_activity,
        fraction=fraction,
        seed=seed,
        affected_case_ids=list(affected),
        n_affected_cases=len(affected_set),
        n_events_changed=n_changed,
        description=(
            f"Replace every '{src_activity}' with '{dst_activity}' in "
            f"{len(affected_set)} cases ({n_changed} events relabelled)."
        ),
    )
    return df_out, gt


def inject_loop(
    df: pd.DataFrame,
    target_activity: str,
    fraction: float,
    seed: int = 0,
    *,
    repeat_range: tuple[int, int] = (1, 3),
    case_id_col: str = "Case ID",
    activity_col: str = "Activity",
    timestamp_col: str = "Complete Timestamp",
) -> tuple[pd.DataFrame, GroundTruth]:
    """Duplicate every occurrence of ``target_activity`` ``r`` extra times,
    where ``r`` is drawn uniformly per affected case from
    ``[repeat_range[0], repeat_range[1]]``.

    Each duplicate timestamp is anchor_ts + (k+1) seconds so order is preserved.
    """
    if repeat_range[0] < 1 or repeat_range[1] < repeat_range[0]:
        raise ValueError(f"repeat_range must satisfy 1 <= lo <= hi, got {repeat_range!r}")
    _resolve_columns(df, case_id_col, activity_col, timestamp_col)
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    candidate_mask = df[activity_col] == target_activity
    affected = _select_affected_cases(df, case_id_col, candidate_mask, fraction, seed)

    if not affected:
        gt = GroundTruth(
            pattern="loop",
            target_activity=target_activity,
            secondary_activity=None,
            fraction=fraction,
            seed=seed,
            description=f"Loop '{target_activity}' (no candidate cases)",
        )
        return df.sort_values([case_id_col, timestamp_col]).reset_index(drop=True), gt

    rng = np.random.default_rng(seed + 1)  # decouple from case-selection RNG
    affected_set = set(affected)
    template_row = df.iloc[0].to_dict()
    new_rows = []

    for case_id in affected:
        r = int(rng.integers(low=repeat_range[0], high=repeat_range[1] + 1))
        case_mask = (df[case_id_col] == case_id) & (df[activity_col] == target_activity)
        anchor_rows = df.loc[case_mask, [timestamp_col]]
        for ts in anchor_rows[timestamp_col]:
            for k in range(r):
                new_row = template_row.copy()
                new_row[case_id_col] = case_id
                new_row[activity_col] = target_activity
                new_row[timestamp_col] = ts + pd.Timedelta(seconds=k + 1)
                new_rows.append(new_row)

    df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df_aug = df_aug.sort_values([case_id_col, timestamp_col]).reset_index(drop=True)

    gt = GroundTruth(
        pattern="loop",
        target_activity=target_activity,
        secondary_activity=None,
        fraction=fraction,
        seed=seed,
        affected_case_ids=list(affected),
        n_affected_cases=len(affected_set),
        n_events_changed=len(new_rows),
        description=(
            f"Duplicate '{target_activity}' 1..{repeat_range[1]} extra times in "
            f"{len(affected_set)} cases ({len(new_rows)} new events)."
        ),
    )
    return df_aug, gt


# ---------------------------------------------------------------------------
# Dispatcher (helpful for the run_full_pipeline.py rewrite in Phase 6)
# ---------------------------------------------------------------------------


_DISPATCH = {
    "insertion": inject_insertion,
    "deletion": inject_deletion,
    "substitution": inject_substitution,
    "loop": inject_loop,
}


def inject(
    df: pd.DataFrame,
    pattern: str,
    *,
    fraction: float,
    seed: int = 0,
    **kwargs,
) -> tuple[pd.DataFrame, GroundTruth]:
    """Dispatch by ``pattern`` name. Extra kwargs are forwarded verbatim."""
    if pattern not in _DISPATCH:
        raise ValueError(
            f"unknown injection pattern {pattern!r}; valid: {sorted(_DISPATCH)}"
        )
    fn = _DISPATCH[pattern]
    return fn(df, fraction=fraction, seed=seed, **kwargs)


def _safe_attr(col: str) -> str:
    """Best-effort tuple-attribute name for itertuples()."""
    # itertuples replaces non-identifier chars with '_'; keep simple here.
    out = []
    for ch in col:
        out.append(ch if ch.isalnum() or ch == "_" else "_")
    name = "".join(out)
    if name and name[0].isdigit():
        name = "_" + name
    return name
