"""Thin orchestrator for the multi-scale OT-based drift detection pipeline.

Replaces the previous 506-line monolith with a ~200-line composition over the
``drift/*`` modules.  Same single-command entrypoint, richer output JSON
(schema v2 -- see ``datasets/final_report_for_azure.json`` after running).

Environment overrides (preserved from v1 for back-compat):
    EVENT_LOG_PATH        default: datasets/finale.csv
    COL_CASE_ID           default: "Case ID"
    COL_ACTIVITY          default: "Activity"
    COL_TIMESTAMP         default: "Complete Timestamp"
    KEEP_ONLY_COMPLETE    default: true
    INJECT_DRIFT          default: false
    DRIFT_PATTERN         default: insertion       (insertion|deletion|substitution|loop)
    DRIFT_TARGET          default: "Live Chat"     (anchor/target activity)
    DRIFT_SECONDARY       default: "AutoReview"    (only used by insertion/substitution)
    DRIFT_FRACTION        default: 0.5
    DRIFT_SEED            default: 42
    SPLIT_METHOD          default: cpd             (cpd|midpoint)
    PERMUTATION_B         default: 100

CLI flags (extend, not replace, env vars):
    --legacy            run the v1 pipeline path only (max(TV,W) on midpoint)
    --no-perm           skip permutation p-values (faster but less honest)
    --no-cpd            same as SPLIT_METHOD=midpoint
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd

from drift.io import build_cases_dataframe, load_event_log
from drift.injection import inject as inject_drift
from drift.localization import (
    bootstrap_change_point_ci,
    compute_drift_signal,
    detect_change_points,
    signal_index_to_case_position,
)
from drift.metrics import (
    activity_frequency_dist,
    align,
    dfg_dist,
    jsd,
    multi_scale_drift,
)
from drift.ot_attribution import (
    attribution_report,
    edit_distance_matrix,
    joint_support,
    variant_distribution,
    w1_distance,
)
from drift.significance import permutation_pvalue


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes"}


HERE = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.getenv("EVENT_LOG_PATH", os.path.join(HERE, "datasets", "finale.csv"))
COL_CASE_ID = os.getenv("COL_CASE_ID", "Case ID")
COL_ACTIVITY = os.getenv("COL_ACTIVITY", "Activity")
COL_TIMESTAMP = os.getenv("COL_TIMESTAMP", "Complete Timestamp")
KEEP_ONLY_COMPLETE = _env_bool("KEEP_ONLY_COMPLETE", True)

INJECT_DRIFT = _env_bool("INJECT_DRIFT", False)
DRIFT_PATTERN = os.getenv("DRIFT_PATTERN", "insertion").lower()
DRIFT_TARGET = os.getenv("DRIFT_TARGET", "Live Chat")
DRIFT_SECONDARY = os.getenv("DRIFT_SECONDARY", "AutoReview")
DRIFT_FRACTION = float(os.getenv("DRIFT_FRACTION", "0.5"))
DRIFT_SEED = int(os.getenv("DRIFT_SEED", "42"))
SPLIT_METHOD = os.getenv("SPLIT_METHOD", "cpd").lower()
PERMUTATION_B = int(os.getenv("PERMUTATION_B", "100"))

# CPD signal hyperparams: scale to dataset size in main()
CPD_DEFAULT_WINDOW = 200
CPD_DEFAULT_STEP = 50

JSON_OUTPUT_PATH = os.path.join(HERE, "datasets", "final_report_for_azure.json")
PROMPT_OUTPUT_PATH = os.path.join(HERE, "final_llm_input_prompt.txt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print(msg: str = ""):
    print(msg, flush=True)


def _load_events() -> pd.DataFrame:
    _print(f"[1] Loading {FILE_PATH}")
    df = load_event_log(
        FILE_PATH, COL_CASE_ID, COL_ACTIVITY, COL_TIMESTAMP,
        keep_only_complete=KEEP_ONLY_COMPLETE,
    )
    missing = [c for c in (COL_CASE_ID, COL_ACTIVITY, COL_TIMESTAMP) if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing columns: {missing}. Available: {list(df.columns)}.\n"
            "Set COL_CASE_ID / COL_ACTIVITY / COL_TIMESTAMP env vars."
        )
    # ISO8601 covers most XES-exported logs; fall back to mixed inference if it fails.
    try:
        df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], format="ISO8601")
    except (ValueError, TypeError):
        df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], format="mixed")
    df = df.sort_values([COL_CASE_ID, COL_TIMESTAMP]).reset_index(drop=True)
    _print(f"    loaded {len(df)} events across {df[COL_CASE_ID].nunique()} cases")
    return df


def _maybe_inject(df_curr: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict | None]:
    if not INJECT_DRIFT:
        return df_curr, None
    _print(f"[2] Injecting drift: {DRIFT_PATTERN} on '{DRIFT_TARGET}' "
           f"(fraction={DRIFT_FRACTION}, seed={seed})")
    kwargs = dict(
        fraction=DRIFT_FRACTION, seed=seed,
        case_id_col=COL_CASE_ID, activity_col=COL_ACTIVITY, timestamp_col=COL_TIMESTAMP,
    )
    if DRIFT_PATTERN == "insertion":
        df_out, gt = inject_drift(df_curr, "insertion",
                                  after_activity=DRIFT_TARGET, new_activity=DRIFT_SECONDARY, **kwargs)
    elif DRIFT_PATTERN == "deletion":
        df_out, gt = inject_drift(df_curr, "deletion",
                                  target_activity=DRIFT_TARGET, **kwargs)
    elif DRIFT_PATTERN == "substitution":
        df_out, gt = inject_drift(df_curr, "substitution",
                                  src_activity=DRIFT_TARGET, dst_activity=DRIFT_SECONDARY, **kwargs)
    elif DRIFT_PATTERN == "loop":
        df_out, gt = inject_drift(df_curr, "loop",
                                  target_activity=DRIFT_TARGET, **kwargs)
    else:
        raise SystemExit(f"Unknown DRIFT_PATTERN={DRIFT_PATTERN!r}")
    _print(f"    affected {gt.n_affected_cases} cases, {gt.n_events_changed} events changed")
    return df_out, gt.to_dict()


def _split_by_case_time(df: pd.DataFrame, split_position: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split events into two halves by case completion-time rank.

    Cases with completion-time rank < ``split_position`` go to baseline; the
    rest to current. Returns ``(events_baseline, events_current)``.
    """
    case_end = (
        df.groupby(COL_CASE_ID)[COL_TIMESTAMP]
        .max()
        .sort_values()
        .reset_index(drop=False)
    )
    base_ids = set(case_end.iloc[:split_position][COL_CASE_ID])
    curr_ids = set(case_end.iloc[split_position:][COL_CASE_ID])
    df_base = df[df[COL_CASE_ID].isin(base_ids)].copy()
    df_curr = df[df[COL_CASE_ID].isin(curr_ids)].copy()
    return df_base, df_curr


def _cpd_split(df: pd.DataFrame, window: int, step: int) -> tuple[int, dict]:
    """Run CPD; return (split_position, cpd_report_dict).

    If CPD finds no change points, fall back to midpoint.
    """
    n_cases = df[COL_CASE_ID].nunique()
    # adapt window/step to dataset
    window = min(window, n_cases // 3)
    step = max(1, min(step, window // 4))
    drift_signal = compute_drift_signal(
        df, window=window, step=step,
        case_id_col=COL_CASE_ID, activity_col=COL_ACTIVITY, timestamp_col=COL_TIMESTAMP,
    )
    cps = detect_change_points(drift_signal["signal"])
    if not cps:
        _print(f"    no change points detected; falling back to midpoint")
        return n_cases // 2, {
            "method": "cpd_fallback_midpoint",
            "window": window, "step": step,
            "change_points": [], "ci_95": [],
        }
    cis = bootstrap_change_point_ci(drift_signal["signal"], cps, B=50, seed=DRIFT_SEED)
    # use the FIRST (earliest) detected CP as the split
    first_cp_idx = cps[0]
    split_pos = signal_index_to_case_position(first_cp_idx, drift_signal)
    cp_case_positions = [signal_index_to_case_position(c, drift_signal) for c in cps]
    ci_case_positions = [
        [signal_index_to_case_position(ci["ci_lo"], drift_signal),
         signal_index_to_case_position(ci["ci_hi"], drift_signal)]
        for ci in cis
    ]
    _print(f"    CPD found {len(cps)} CP(s) at case positions {cp_case_positions}; "
           f"splitting at {split_pos}")
    return split_pos, {
        "method": "cpd",
        "window": window, "step": step,
        "change_points": cp_case_positions,
        "ci_95": ci_case_positions,
    }


def _compute_drift_vector(df_base: pd.DataFrame, df_curr: pd.DataFrame,
                          run_permutation: bool) -> dict:
    _print(f"[4] Multi-scale drift + Wasserstein...")
    msd = multi_scale_drift(df_base, df_curr,
                            case_id_col=COL_CASE_ID, activity_col=COL_ACTIVITY,
                            timestamp_col=COL_TIMESTAMP)
    # variant-level OT for the trace scale
    cases_base = build_cases_dataframe(df_base, COL_CASE_ID, COL_ACTIVITY, COL_TIMESTAMP)
    cases_curr = build_cases_dataframe(df_curr, COL_CASE_ID, COL_ACTIVITY, COL_TIMESTAMP)
    va, pa, _ = variant_distribution(cases_base)
    vb, pb, _ = variant_distribution(cases_curr)
    union, pa_j, pb_j = joint_support(va, pa, vb, pb)
    M = edit_distance_matrix(union, union)
    w1 = w1_distance(pa_j, pb_j, M) if union else 0.0

    out = {
        "activity_jsd": msd["activity_jsd"],
        "dfg_jsd": msd["dfg_jsd"],
        "trace_jsd": msd["trace_jsd"],
        "trace_w1": w1,
    }
    _print(f"    activity_JSD={out['activity_jsd']:.4f}  "
           f"dfg_JSD={out['dfg_jsd']:.4f}  "
           f"trace_JSD={out['trace_jsd']:.4f}  "
           f"trace_W1={out['trace_w1']:.4f}")

    if not run_permutation:
        return out

    _print(f"    permutation p-values (B={PERMUTATION_B})...")

    def _act_stat(a, b):
        pa = activity_frequency_dist(a, COL_ACTIVITY)
        pb = activity_frequency_dist(b, COL_ACTIVITY)
        return jsd(*align(pa, pb))

    def _dfg_stat(a, b):
        pa = dfg_dist(a, COL_CASE_ID, COL_ACTIVITY, COL_TIMESTAMP)
        pb = dfg_dist(b, COL_CASE_ID, COL_ACTIVITY, COL_TIMESTAMP)
        return jsd(*align(pa, pb))

    def _agg_stat(a, b):
        d = multi_scale_drift(a, b, COL_CASE_ID, COL_ACTIVITY, COL_TIMESTAMP)
        return max(d.values())

    out["activity_pvalue"] = permutation_pvalue(
        df_base, df_curr, _act_stat, B=PERMUTATION_B, seed=DRIFT_SEED, case_id_col=COL_CASE_ID,
    )
    out["dfg_pvalue"] = permutation_pvalue(
        df_base, df_curr, _dfg_stat, B=PERMUTATION_B, seed=DRIFT_SEED, case_id_col=COL_CASE_ID,
    )
    out["aggregate_pvalue"] = permutation_pvalue(
        df_base, df_curr, _agg_stat, B=PERMUTATION_B, seed=DRIFT_SEED, case_id_col=COL_CASE_ID,
    )
    _print(f"    p-values: activity={out['activity_pvalue']:.4f} "
           f"dfg={out['dfg_pvalue']:.4f} aggregate={out['aggregate_pvalue']:.4f}")
    return out


def _legacy_block(df_base: pd.DataFrame, df_curr: pd.DataFrame, top_k: int = 10) -> dict:
    """Compute the v1 scalar score + top-K freq dicts so legacy LLM scripts keep working."""
    from scipy.stats import wasserstein_distance as _wd

    # legacy trace-string variant
    def _trace_str(df):
        df = df.sort_values([COL_CASE_ID, COL_TIMESTAMP])
        return (
            df.groupby(COL_CASE_ID, sort=False)[COL_ACTIVITY]
            .apply(lambda s: " -> ".join(map(str, s.tolist())))
        )

    traces_b = _trace_str(df_base)
    traces_c = _trace_str(df_curr)
    all_traces = sorted(set(traces_b) | set(traces_c))
    idx = {t: i for i, t in enumerate(all_traces)}
    vec_b = np.zeros(len(all_traces))
    vec_c = np.zeros(len(all_traces))
    for t, freq in traces_b.value_counts(normalize=True).items():
        vec_b[idx[t]] = freq
    for t, freq in traces_c.value_counts(normalize=True).items():
        vec_c[idx[t]] = freq
    tv = float(0.5 * np.sum(np.abs(vec_b - vec_c)))

    # legacy duration
    def _durations(df):
        g = df.groupby(COL_CASE_ID)[COL_TIMESTAMP]
        return ((g.max() - g.min()).dt.total_seconds() / 60.0).to_numpy()

    d_b = _durations(df_base)
    d_c = _durations(df_curr)
    if d_b.size and d_c.size:
        w_raw = float(_wd(d_b, d_c))
        scale = float(np.median(d_b)) if np.median(d_b) > 0 else float(np.mean(d_b))
        w_norm = w_raw / scale if scale > 0 else 0.0
    else:
        w_raw = w_norm = 0.0

    drift_score = max(tv, w_norm)
    top_b = traces_b.value_counts(normalize=True).head(top_k).to_dict()
    top_c = traces_c.value_counts(normalize=True).head(top_k).to_dict()
    return {
        "drift_score": round(drift_score, 4),
        "trace_drift_score": round(tv, 4),
        "duration_drift_score": round(w_norm, 4),
        "duration_drift_score_raw": round(w_raw, 4),
        "drift_metric": "tv",
        "detection_threshold": 0.05,
        "status": "DRIFT DETECTED" if drift_score > 0.05 else "STABLE",
        "top_baseline_process_freq": top_b,
        "top_current_process_freq": top_c,
        "baseline_count": len(traces_b),
        "current_count": len(traces_c),
    }


def _build_report(
    df: pd.DataFrame,
    df_base: pd.DataFrame,
    df_curr: pd.DataFrame,
    split_info: dict,
    drift_vector: dict,
    attribution: dict,
    ground_truth: dict | None,
) -> dict:
    n_variants_union = attribution.get("n_variants_union", 0)
    primary_p = drift_vector.get("aggregate_pvalue", None)
    if primary_p is None:
        status = "DRIFT DETECTED" if max(
            drift_vector["activity_jsd"], drift_vector["dfg_jsd"],
            drift_vector["trace_jsd"], drift_vector["trace_w1"],
        ) > 0.05 else "STABLE"
    else:
        status = "DRIFT DETECTED" if primary_p < 0.05 else "STABLE"

    legacy = _legacy_block(df_base, df_curr)
    legacy["status"] = status  # synchronise with v2

    return {
        "schema_version": "2.0",
        "status": status,
        "dataset": {
            "path": FILE_PATH,
            "name": os.path.splitext(os.path.basename(FILE_PATH))[0],
            "n_events": int(len(df)),
            "n_cases": int(df[COL_CASE_ID].nunique()),
            "n_activities": int(df[COL_ACTIVITY].nunique()),
            "n_variants_baseline": attribution.get("n_variants_baseline", 0),
            "n_variants_current": attribution.get("n_variants_current", 0),
            "n_variants_union": n_variants_union,
        },
        "split": split_info,
        "drift_vector": drift_vector,
        "attribution": {
            "w1": attribution.get("w1", 0.0),
            "top_transport_flows": attribution.get("top_transport_flows", []),
            "top_lost_variants": attribution.get("top_lost_variants", []),
            "top_gained_variants": attribution.get("top_gained_variants", []),
        },
        "ground_truth": ground_truth,
        "legacy": legacy,
    }


def _generate_prompt(report: dict) -> str:
    """Compact Markdown-ready prompt that fits the schema-v2 attribution block."""
    dv = report["drift_vector"]
    attr = report["attribution"]
    gt = report.get("ground_truth")
    top_flows = attr["top_transport_flows"][:5]
    top_lost = attr["top_lost_variants"][:5]
    top_gained = attr["top_gained_variants"][:5]

    def _fmt_variant(v):
        return " -> ".join(v) if isinstance(v, (list, tuple)) else str(v)

    p_lines = [
        "[System Analysis Report — Schema v2]",
        f"Dataset: {report['dataset']['name']} | events={report['dataset']['n_events']} | "
        f"cases={report['dataset']['n_cases']} | activities={report['dataset']['n_activities']}",
        f"Split: method={report['split']['method']} change_points={report['split'].get('change_points', [])}",
        "",
        "## Multi-scale drift",
        f"  activity_jsd : {dv.get('activity_jsd', 'n/a'):.4f}  p={dv.get('activity_pvalue', 'n/a')}",
        f"  dfg_jsd      : {dv.get('dfg_jsd', 'n/a'):.4f}  p={dv.get('dfg_pvalue', 'n/a')}",
        f"  trace_jsd    : {dv.get('trace_jsd', 'n/a'):.4f}",
        f"  trace_w1     : {dv.get('trace_w1', 'n/a'):.4f}",
        f"  aggregate p  : {dv.get('aggregate_pvalue', 'n/a')}",
        "",
        f"## Top {len(top_flows)} transport flows (baseline -> current)",
    ]
    for i, f in enumerate(top_flows, 1):
        p_lines.append(
            f"  {i}. mass={f['mass']:.4f}  edit_distance={f.get('edit_distance', 0):.3f}"
        )
        p_lines.append(f"     from: {_fmt_variant(f['from_variant'])}")
        p_lines.append(f"     to:   {_fmt_variant(f['to_variant'])}")
    p_lines += ["", f"## Top {len(top_lost)} lost variants (baseline -> current)"]
    for v in top_lost:
        p_lines.append(
            f"  delta={v['delta']:+.4f}  ({v['n_cases_baseline']} -> {v['n_cases_current']} cases)  "
            f"{_fmt_variant(v['variant'])}"
        )
    p_lines += ["", f"## Top {len(top_gained)} gained variants"]
    for v in top_gained:
        p_lines.append(
            f"  delta={v['delta']:+.4f}  ({v['n_cases_baseline']} -> {v['n_cases_current']} cases)  "
            f"{_fmt_variant(v['variant'])}"
        )
    if gt:
        p_lines += ["", "## Injected ground truth (for evaluation only)",
                    f"  pattern={gt['pattern']}  target={gt['target_activity']}  "
                    f"secondary={gt['secondary_activity']}  "
                    f"affected_cases={gt['n_affected_cases']}"]
    p_lines += [
        "",
        "[Task] 输出中文 Markdown 报告，固定结构: 总览 / 关键变化 / 根因推断 / 改进建议。",
        "必须引用上面的具体数值（drift_vector, 流量百分比, 受影响 case 数）。"
    ]
    return "\n".join(p_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_legacy_pipeline():
    """Fallback to the v1 max(TV, W) pipeline for back-compat demos."""
    _print("[legacy mode] running v1 max(TV, Wasserstein) midpoint pipeline...")
    df = _load_events()
    case_end = df.groupby(COL_CASE_ID)[COL_TIMESTAMP].max().sort_values()
    mid = len(case_end) // 2
    base_ids = set(case_end.iloc[:mid].index)
    df_base = df[df[COL_CASE_ID].isin(base_ids)]
    df_curr_raw = df[~df[COL_CASE_ID].isin(base_ids)]
    df_curr, _ = _maybe_inject(df_curr_raw, DRIFT_SEED)
    legacy = _legacy_block(df_base, df_curr)
    _print(f"    drift_score={legacy['drift_score']}  status={legacy['status']}")
    report = {"schema_version": "1.0_legacy", **legacy}
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _print(f"    wrote {JSON_OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Multi-scale OT-based drift detection (schema v2).")
    parser.add_argument("--legacy", action="store_true",
                        help="Run the v1 max(TV, W) midpoint pipeline only.")
    parser.add_argument("--no-perm", action="store_true",
                        help="Skip permutation p-values.")
    parser.add_argument("--no-cpd", action="store_true",
                        help="Force midpoint split (skip change-point detection).")
    args = parser.parse_args()

    if args.legacy:
        run_legacy_pipeline()
        return

    random.seed(DRIFT_SEED)
    np.random.seed(DRIFT_SEED)

    df = _load_events()

    # Apply injection on the WHOLE log first if requested so CPD can localise it.
    if INJECT_DRIFT:
        # split current = back half (post-midpoint cases by completion time) so
        # injected drift is concentrated in time
        case_end = df.groupby(COL_CASE_ID)[COL_TIMESTAMP].max().sort_values()
        late_ids = set(case_end.iloc[len(case_end) // 2:].index)
        df_late = df[df[COL_CASE_ID].isin(late_ids)]
        df_early = df[~df[COL_CASE_ID].isin(late_ids)]
        df_late_inj, gt = _maybe_inject(df_late, DRIFT_SEED)
        df_full = pd.concat([df_early, df_late_inj], ignore_index=True)
    else:
        df_full = df
        gt = None

    _print(f"[3] Split method: {SPLIT_METHOD}{' (forced midpoint via --no-cpd)' if args.no_cpd else ''}")
    if args.no_cpd or SPLIT_METHOD == "midpoint":
        n_cases = df_full[COL_CASE_ID].nunique()
        split_pos = n_cases // 2
        split_info = {"method": "midpoint", "change_points": [split_pos], "ci_95": []}
    else:
        split_pos, split_info = _cpd_split(df_full, CPD_DEFAULT_WINDOW, CPD_DEFAULT_STEP)

    df_base, df_curr = _split_by_case_time(df_full, split_pos)
    _print(f"    baseline: {len(df_base)} events / {df_base[COL_CASE_ID].nunique()} cases; "
           f"current: {len(df_curr)} events / {df_curr[COL_CASE_ID].nunique()} cases")

    drift_vector = _compute_drift_vector(df_base, df_curr, run_permutation=not args.no_perm)

    _print("[5] OT attribution (variant-level)...")
    cases_base = build_cases_dataframe(df_base, COL_CASE_ID, COL_ACTIVITY, COL_TIMESTAMP)
    cases_curr = build_cases_dataframe(df_curr, COL_CASE_ID, COL_ACTIVITY, COL_TIMESTAMP)
    attribution = attribution_report(cases_base, cases_curr,
                                     k_flows=10, k_changes=10, sample_size=5, seed=DRIFT_SEED)
    _print(f"    W1={attribution['w1']:.4f}  "
           f"|V|: base={attribution['n_variants_baseline']} curr={attribution['n_variants_current']} "
           f"union={attribution['n_variants_union']}")

    _print("[6] Writing schema v2 report + LLM prompt...")
    report = _build_report(df_full, df_base, df_curr, split_info, drift_vector, attribution, gt)
    os.makedirs(os.path.dirname(JSON_OUTPUT_PATH), exist_ok=True)
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    _print(f"    wrote {JSON_OUTPUT_PATH}")

    prompt = _generate_prompt(report)
    with open(PROMPT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(prompt)
    _print(f"    wrote {PROMPT_OUTPUT_PATH}")
    _print(f"\nstatus: {report['status']}")


if __name__ == "__main__":
    sys.exit(main())
