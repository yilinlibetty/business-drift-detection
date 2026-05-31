"""M2 -- Optimal Transport on trace variants with Levenshtein ground metric.

This is the headline novelty: we lift drift detection into Wasserstein space
where each atomic mass is a *trace variant* (a distinct activity sequence) and
the ground metric is the normalised Levenshtein distance between variants.

Why this is more informative than TV/JSD on traces:
  * TV/JSD treat traces as bare symbols -- ``A->B->C`` and ``A->X->C`` are
    "different" with no notion of how different.
  * Wasserstein uses an underlying metric (edit distance) so traces that
    differ by a single activity insertion are *close*, and the resulting
    transport plan gives a principled per-flow attribution at no extra cost:
    "which baseline variants morphed into which current variants, and by how
    much mass."
  * The transport plan back-projects to case IDs in O(|V|) time, giving
    case-level explanations consumable by the M6 LLM evaluation harness.

Public API:
    variant_distribution(cases_df, ...)         -> (variants, probs, ids_per_variant)
    joint_support(va, pa, vb, pb)               -> (union, pa_joint, pb_joint)
    edit_distance_matrix(variants_a, variants_b) -> np.ndarray   (normalised, [0,1])
    w1_distance(probs_a, probs_b, cost)         -> float
    transport_plan(probs_a, probs_b, cost)      -> np.ndarray
    top_k_flows(plan, variants, cost=None, k=10, exclude_diagonal=True) -> list[dict]
    top_variant_changes(variants, probs_a, probs_b, ids_a, ids_b, k=10) -> dict
    attach_case_samples(flows, ids_a, ids_b, variants, sample_size=5, seed=0) -> list[dict]
    attribution_report(cases_base, cases_curr, ...) -> dict      (schema-v2-ready)
"""

from __future__ import annotations

import editdistance
import numpy as np
import ot
import pandas as pd


# ---------------------------------------------------------------------------
# Variant distributions
# ---------------------------------------------------------------------------


def variant_distribution(
    cases_df: pd.DataFrame,
    trace_col: str = "Trace",
    case_id_col: str = "CaseID",
) -> tuple[list[tuple], np.ndarray, dict[tuple, list]]:
    """Empirical distribution over distinct trace variants.

    Returns
    -------
    variants : list[tuple[str, ...]]
        Canonical-ordered list of unique trace tuples (sort by (length, tuple)
        for cross-run determinism).
    probs : np.ndarray of float (sums to 1.0)
        Variant probabilities, in the same order as ``variants``.
    case_ids_per_variant : dict[tuple, list]
        Map variant -> list of case IDs that exhibit it.
    """
    if trace_col not in cases_df.columns:
        raise KeyError(f"missing trace column {trace_col!r}")
    if case_id_col not in cases_df.columns:
        raise KeyError(f"missing case-id column {case_id_col!r}")
    if len(cases_df) == 0:
        return [], np.zeros(0), {}

    # Trace column may contain tuples already (preferred) or strings.
    # We normalise into tuples for hashability.
    def _as_tuple(t):
        if isinstance(t, tuple):
            return t
        if isinstance(t, (list, np.ndarray)):
            return tuple(t)
        if isinstance(t, str):
            # legacy " -> ".join() format
            return tuple(s.strip() for s in t.split("->"))
        raise TypeError(f"cannot interpret trace value {t!r}")

    trace_tuples = cases_df[trace_col].map(_as_tuple)
    case_ids_per_variant: dict[tuple, list] = {}
    for v, cid in zip(trace_tuples, cases_df[case_id_col]):
        case_ids_per_variant.setdefault(v, []).append(cid)

    variants = sorted(case_ids_per_variant.keys(), key=lambda t: (len(t), t))
    counts = np.array([len(case_ids_per_variant[v]) for v in variants], dtype=float)
    probs = counts / counts.sum()
    return variants, probs, case_ids_per_variant


def joint_support(
    va: list[tuple], pa: np.ndarray,
    vb: list[tuple], pb: np.ndarray,
) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """Reindex two variant distributions onto their union support.

    Variants only in one side get mass 0 on the other.  The returned
    ``union`` list uses the same canonical ordering as
    ``variant_distribution``.
    """
    pa_map = dict(zip(va, pa))
    pb_map = dict(zip(vb, pb))
    union = sorted(set(va) | set(vb), key=lambda t: (len(t), t))
    pa_joint = np.array([pa_map.get(v, 0.0) for v in union], dtype=float)
    pb_joint = np.array([pb_map.get(v, 0.0) for v in union], dtype=float)
    return union, pa_joint, pb_joint


# ---------------------------------------------------------------------------
# Ground metric
# ---------------------------------------------------------------------------


def edit_distance_matrix(
    variants_a: list[tuple],
    variants_b: list[tuple],
) -> np.ndarray:
    """Normalised Levenshtein distance matrix in [0, 1].

    ``M[i, j] = editdistance(variants_a[i], variants_b[j]) /
                max(len(variants_a[i]), len(variants_b[j]))``

    When both sides come from ``joint_support`` (so they are equal lists), the
    matrix is square symmetric with a 0 diagonal -- the natural cost structure
    that lets the transport plan reveal "preserved mass" on the diagonal and
    "moved mass" off-diagonal.
    """
    n_a, n_b = len(variants_a), len(variants_b)
    M = np.empty((n_a, n_b), dtype=float)
    for i, va in enumerate(variants_a):
        for j, vb in enumerate(variants_b):
            denom = max(len(va), len(vb), 1)
            M[i, j] = editdistance.eval(va, vb) / denom
    return M


# ---------------------------------------------------------------------------
# Wasserstein-1
# ---------------------------------------------------------------------------


def _validate_simplex(p: np.ndarray, name: str):
    if p.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {p.shape}")
    if (p < -1e-9).any():
        raise ValueError(f"{name} has negative entries")
    s = float(p.sum())
    if s > 0 and not np.isclose(s, 1.0, atol=1e-6):
        raise ValueError(f"{name} does not sum to 1 (sum={s:.6g})")


def w1_distance(probs_a: np.ndarray, probs_b: np.ndarray, cost: np.ndarray) -> float:
    """Exact Wasserstein-1 via ``ot.emd2`` (LP)."""
    probs_a = np.asarray(probs_a, dtype=float)
    probs_b = np.asarray(probs_b, dtype=float)
    cost = np.asarray(cost, dtype=float)
    _validate_simplex(probs_a, "probs_a")
    _validate_simplex(probs_b, "probs_b")
    if cost.shape != (probs_a.size, probs_b.size):
        raise ValueError(
            f"cost shape {cost.shape} != ({probs_a.size}, {probs_b.size})"
        )
    if probs_a.size == 0:
        return 0.0
    return float(ot.emd2(probs_a, probs_b, cost))


def transport_plan(probs_a: np.ndarray, probs_b: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Exact transport plan via ``ot.emd`` (LP).

    Returns a (|a|, |b|) matrix π with row-sums == probs_a and column-sums == probs_b.
    """
    probs_a = np.asarray(probs_a, dtype=float)
    probs_b = np.asarray(probs_b, dtype=float)
    cost = np.asarray(cost, dtype=float)
    _validate_simplex(probs_a, "probs_a")
    _validate_simplex(probs_b, "probs_b")
    if cost.shape != (probs_a.size, probs_b.size):
        raise ValueError(
            f"cost shape {cost.shape} != ({probs_a.size}, {probs_b.size})"
        )
    if probs_a.size == 0:
        return np.zeros((0, 0))
    return np.asarray(ot.emd(probs_a, probs_b, cost), dtype=float)


# ---------------------------------------------------------------------------
# Plan summarisation
# ---------------------------------------------------------------------------


def top_k_flows(
    plan: np.ndarray,
    variants: list[tuple],
    cost: np.ndarray | None = None,
    k: int = 10,
    exclude_diagonal: bool = True,
    mass_floor: float = 1e-9,
) -> list[dict]:
    """Return the K largest mass flows in the transport plan.

    ``variants`` must be the *joint* support so that ``plan[i, j]`` is mass
    moving from variant i to variant j (both indexed into the same list).

    Flows where i == j (mass that stayed put) are excluded by default --
    they carry no attribution information.  Pass ``exclude_diagonal=False``
    to include them.
    """
    plan = np.asarray(plan, dtype=float)
    if cost is not None and plan.shape != cost.shape:
        raise ValueError(f"plan shape {plan.shape} != cost shape {cost.shape}")

    flows = []
    for i in range(plan.shape[0]):
        for j in range(plan.shape[1]):
            if exclude_diagonal and i == j:
                continue
            mass = float(plan[i, j])
            if mass <= mass_floor:
                continue
            entry = {
                "from_variant_idx": i,
                "to_variant_idx": j,
                "from_variant": variants[i] if i < len(variants) else None,
                "to_variant": variants[j] if j < len(variants) else None,
                "mass": mass,
            }
            if cost is not None:
                entry["edit_distance"] = float(cost[i, j])
            flows.append(entry)
    flows.sort(key=lambda f: f["mass"], reverse=True)
    return flows[:k]


def top_variant_changes(
    variants: list[tuple],
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    case_ids_a: dict[tuple, list],
    case_ids_b: dict[tuple, list],
    k: int = 10,
    delta_floor: float = 1e-12,
) -> dict:
    """Top variants that gained or lost the most marginal mass.

    The two lists are filtered by sign of delta so ``lost`` only contains
    variants whose mass strictly decreased and ``gained`` only contains
    variants whose mass strictly increased. Variants with delta == 0 are
    excluded from both. This avoids the K > |truly-changed| padding bug
    where unchanged or oppositely-signed variants leak into a list.
    """
    rows = []
    for i, v in enumerate(variants):
        rows.append({
            "variant_idx": i,
            "variant": v,
            "mass_baseline": float(probs_a[i]),
            "mass_current": float(probs_b[i]),
            "delta": float(probs_b[i] - probs_a[i]),
            "n_cases_baseline": len(case_ids_a.get(v, [])),
            "n_cases_current": len(case_ids_b.get(v, [])),
        })
    lost = sorted(
        (r for r in rows if r["delta"] < -delta_floor),
        key=lambda d: d["delta"],
    )[:k]
    gained = sorted(
        (r for r in rows if r["delta"] > delta_floor),
        key=lambda d: -d["delta"],
    )[:k]
    return {"lost": lost, "gained": gained}


def attach_case_samples(
    flows: list[dict],
    case_ids_a: dict[tuple, list],
    case_ids_b: dict[tuple, list],
    variants: list[tuple],
    sample_size: int = 5,
    seed: int = 0,
) -> list[dict]:
    """Mutates each flow dict in-place to add ``from_case_ids_sample`` and
    ``to_case_ids_sample``, drawn deterministically from ``seed``.
    """
    rng = np.random.default_rng(seed)
    for flow in flows:
        v_from = variants[flow["from_variant_idx"]]
        v_to = variants[flow["to_variant_idx"]]
        candidates_from = case_ids_a.get(v_from, [])
        candidates_to = case_ids_b.get(v_to, [])
        n_from = min(sample_size, len(candidates_from))
        n_to = min(sample_size, len(candidates_to))
        if n_from > 0:
            idx = rng.choice(len(candidates_from), size=n_from, replace=False)
            flow["from_case_ids_sample"] = [candidates_from[k] for k in sorted(idx)]
        else:
            flow["from_case_ids_sample"] = []
        if n_to > 0:
            idx = rng.choice(len(candidates_to), size=n_to, replace=False)
            flow["to_case_ids_sample"] = [candidates_to[k] for k in sorted(idx)]
        else:
            flow["to_case_ids_sample"] = []
    return flows


# ---------------------------------------------------------------------------
# Schema-v2-ready convenience
# ---------------------------------------------------------------------------


def attribution_report(
    cases_base: pd.DataFrame,
    cases_curr: pd.DataFrame,
    *,
    trace_col: str = "Trace",
    case_id_col: str = "CaseID",
    k_flows: int = 10,
    k_changes: int = 10,
    sample_size: int = 5,
    seed: int = 0,
) -> dict:
    """Compose the full OT attribution payload consumed by schema v2.

    Returns dict with:
      w1                          float, the Wasserstein-1 distance
      n_variants_baseline / current / union
      top_transport_flows         list of {from/to_variant + mass + edit_distance + case_id_samples}
      top_lost_variants           list of variants whose marginal mass shrank most
      top_gained_variants         list of variants whose marginal mass grew most
    """
    va, pa, ids_a = variant_distribution(cases_base, trace_col, case_id_col)
    vb, pb, ids_b = variant_distribution(cases_curr, trace_col, case_id_col)
    if not va or not vb:
        return {
            "w1": 0.0,
            "n_variants_baseline": len(va),
            "n_variants_current": len(vb),
            "n_variants_union": len(set(va) | set(vb)),
            "top_transport_flows": [],
            "top_lost_variants": [],
            "top_gained_variants": [],
        }

    union, pa_j, pb_j = joint_support(va, pa, vb, pb)
    M = edit_distance_matrix(union, union)
    w1 = w1_distance(pa_j, pb_j, M)
    plan = transport_plan(pa_j, pb_j, M)

    flows = top_k_flows(plan, union, cost=M, k=k_flows, exclude_diagonal=True)
    attach_case_samples(flows, ids_a, ids_b, union, sample_size=sample_size, seed=seed)
    changes = top_variant_changes(union, pa_j, pb_j, ids_a, ids_b, k=k_changes)

    return {
        "w1": w1,
        "n_variants_baseline": len(va),
        "n_variants_current": len(vb),
        "n_variants_union": len(union),
        "top_transport_flows": flows,
        "top_lost_variants": changes["lost"],
        "top_gained_variants": changes["gained"],
    }
