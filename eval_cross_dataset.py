"""Cross-dataset signature comparison for the experiments-section appendix.

Runs the multi-scale + OT pipeline on both Help Desk (default) and BPI 2017
(loan-application log), under all four injection patterns × N seeds,
collecting:
  - drift_vector (activity_jsd, dfg_jsd, trace_jsd, trace_w1)
  - case counts
  - W1 value

Output: a Markdown table (paper-ready) and a CSV.

Usage:
    python eval_cross_dataset.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from drift.injection import inject as inject_drift
from drift.io import build_cases_dataframe, load_event_log
from drift.metrics import multi_scale_drift
from drift.ot_attribution import attribution_report


HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DatasetSpec:
    name: str
    path: str
    case_col: str
    activity_col: str
    timestamp_col: str
    scenarios: dict  # pattern -> kwargs (target, secondary if any, fraction)


HELPDESK = DatasetSpec(
    name="Help Desk",
    path=os.path.join(HERE, "datasets", "finale.csv"),
    case_col="Case ID",
    activity_col="Activity",
    timestamp_col="Complete Timestamp",
    scenarios={
        "insertion":    dict(after_activity="Take in charge ticket", new_activity="AutoReview", fraction=0.6),
        "deletion":     dict(target_activity="Wait", fraction=0.8),
        "substitution": dict(src_activity="Wait", dst_activity="QueuedWait", fraction=0.7),
        "loop":         dict(target_activity="Resolve ticket", fraction=0.5, repeat_range=(2, 2)),
    },
)

BPI2017 = DatasetSpec(
    name="BPI 2017",
    path=os.path.join(HERE, "datasets", "frequency-log.csv"),
    case_col="case:concept:name",
    activity_col="concept:name",
    timestamp_col="time:timestamp",
    scenarios={
        "insertion":    dict(after_activity="A_Create Application", new_activity="A_AutoReview", fraction=0.6),
        "deletion":     dict(target_activity="W_Validate application", fraction=0.6),
        "substitution": dict(src_activity="W_Validate application", dst_activity="W_Q_Validate", fraction=0.6),
        "loop":         dict(target_activity="A_Create Application", fraction=0.5, repeat_range=(2, 2)),
    },
)


def _load_and_split(spec: DatasetSpec):
    df = load_event_log(
        spec.path, spec.case_col, spec.activity_col, spec.timestamp_col,
        keep_only_complete=True,
    )
    try:
        df[spec.timestamp_col] = pd.to_datetime(df[spec.timestamp_col], format="ISO8601")
    except (ValueError, TypeError):
        df[spec.timestamp_col] = pd.to_datetime(df[spec.timestamp_col], format="mixed")
    df = df.dropna(subset=[spec.timestamp_col])
    df = df.sort_values([spec.case_col, spec.timestamp_col]).reset_index(drop=True)
    case_end = df.groupby(spec.case_col)[spec.timestamp_col].max().sort_values()
    mid = len(case_end) // 2
    late_ids = set(case_end.iloc[mid:].index)
    df_early = df[~df[spec.case_col].isin(late_ids)]
    df_late = df[df[spec.case_col].isin(late_ids)]
    return df, df_early, df_late


def _compute_row(df_b, df_c, spec):
    msd = multi_scale_drift(
        df_b, df_c,
        case_id_col=spec.case_col, activity_col=spec.activity_col,
        timestamp_col=spec.timestamp_col,
    )
    cb = build_cases_dataframe(df_b, spec.case_col, spec.activity_col, spec.timestamp_col)
    cc = build_cases_dataframe(df_c, spec.case_col, spec.activity_col, spec.timestamp_col)
    attr = attribution_report(cb, cc, k_flows=1, k_changes=1)
    return {
        "activity_jsd": float(msd["activity_jsd"]),
        "dfg_jsd":      float(msd["dfg_jsd"]),
        "trace_jsd":    float(msd["trace_jsd"]),
        "trace_w1":     float(attr["w1"]),
        "n_variants_baseline": int(attr["n_variants_baseline"]),
        "n_variants_current":  int(attr["n_variants_current"]),
        "n_variants_union":    int(attr["n_variants_union"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output-md", default=os.path.join(HERE, "outputs", "cross_dataset.md"))
    parser.add_argument("--output-csv", default=os.path.join(HERE, "outputs", "cross_dataset.csv"))
    args = parser.parse_args()

    rows = []
    for spec in (HELPDESK, BPI2017):
        print(f"\n=== {spec.name} ===")
        df, df_early, df_late = _load_and_split(spec)
        print(f"  {len(df)} events, {df[spec.case_col].nunique()} cases, "
              f"{df[spec.activity_col].nunique()} activities")

        # NULL run (no injection): early vs late
        t0 = time.time()
        null_metrics = _compute_row(df_early, df_late, spec)
        print(f"  null: {null_metrics} ({time.time()-t0:.1f}s)")
        for k, v in null_metrics.items():
            if isinstance(v, float):
                pass
        rows.append({
            "dataset": spec.name, "pattern": "null", "seed": -1,
            "n_affected_cases": 0, **null_metrics,
        })

        for pattern, kw in spec.scenarios.items():
            for seed in range(args.seeds):
                t0 = time.time()
                df_late_inj, gt = inject_drift(
                    df_late, pattern, seed=seed,
                    case_id_col=spec.case_col, activity_col=spec.activity_col,
                    timestamp_col=spec.timestamp_col, **kw,
                )
                m = _compute_row(df_early, df_late_inj, spec)
                rows.append({
                    "dataset": spec.name, "pattern": pattern, "seed": seed,
                    "n_affected_cases": gt.n_affected_cases, **m,
                })
                print(f"  {pattern:13s} seed={seed} n_affected={gt.n_affected_cases:>4d}  "
                      f"act={m['activity_jsd']:.3f} dfg={m['dfg_jsd']:.3f} trc={m['trace_jsd']:.3f} W1={m['trace_w1']:.3f}  "
                      f"({time.time()-t0:.1f}s)")

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(args.output_csv, index=False)

    # Build the paper-ready Markdown table: aggregate by (dataset, pattern)
    agg = df_rows.groupby(["dataset", "pattern"]).agg(
        n_affected=("n_affected_cases", "mean"),
        act_jsd_mean=("activity_jsd", "mean"),
        act_jsd_std=("activity_jsd", "std"),
        dfg_jsd_mean=("dfg_jsd", "mean"),
        dfg_jsd_std=("dfg_jsd", "std"),
        trace_jsd_mean=("trace_jsd", "mean"),
        trace_jsd_std=("trace_jsd", "std"),
        w1_mean=("trace_w1", "mean"),
        w1_std=("trace_w1", "std"),
    ).reset_index()

    # 重排：null 在最上，然后按数据集 → 模式
    pattern_order = ["null", "insertion", "deletion", "substitution", "loop"]
    agg["pat_rank"] = agg["pattern"].map({p: i for i, p in enumerate(pattern_order)})
    agg = agg.sort_values(["dataset", "pat_rank"]).drop(columns="pat_rank")

    def _fmt(m, s):
        if pd.isna(s) or s == 0:
            return f"{m:.3f}"
        return f"{m:.3f} ± {s:.3f}"

    md_lines = [
        "| Dataset | Pattern | n_affected | act_JSD | DFG_JSD | trace_JSD | W₁ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        md_lines.append(
            f"| {r['dataset']} | {r['pattern']} | {int(r['n_affected'])} "
            f"| {_fmt(r['act_jsd_mean'], r['act_jsd_std'])} "
            f"| {_fmt(r['dfg_jsd_mean'], r['dfg_jsd_std'])} "
            f"| {_fmt(r['trace_jsd_mean'], r['trace_jsd_std'])} "
            f"| {_fmt(r['w1_mean'], r['w1_std'])} |"
        )

    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(f"# Cross-dataset drift-vector summary (n_seeds = {args.seeds})\n\n")
        f.write("Mean ± std over injection seeds; `null` row is single (no injection).\n\n")
        f.write("\n".join(md_lines))
        f.write("\n")

    print(f"\nwrote {args.output_md}")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    sys.exit(main())
