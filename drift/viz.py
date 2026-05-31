"""M7 -- Paper-figure style + per-figure builders.

Style policy (set once via ``set_paper_style``):

  * Times serif font + matplotlib rcParams (no seaborn theme — finer control)
  * Ocean Dusk palette: colorblind-safe, professional
  * proposed-method always renders in OUR_COLOR (coral); baselines in BASELINE_COLOR (cool grey)
  * confidence bands shaded with alpha 0.15
  * **NO titles or suptitles inside the figure** — titles go in LaTeX captions only
  * Both PDF (vector, for LaTeX) and PNG (300 dpi, for previews) are written

Each ``fig_*`` function is self-sufficient: it computes (or accepts) data,
draws, returns the matplotlib Figure, and writes a PDF to ``figures/``.

The functions are dispatched from ``make_figures.py`` via the --figure flag.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; required for PDF-only output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")


# ---------------------------------------------------------------------------
# Ocean Dusk palette (colorblind-safe, professional)
# ---------------------------------------------------------------------------

# Semantic colors (semantic name preserved for backwards-compat with existing builders;
# values updated to Ocean Dusk).
OUR_COLOR      = "#E76F51"   # coral — the proposed method (was WARM_ORANGE)
BASELINE_COLOR = "#B0BEC5"   # cool grey — baselines that should recede
ACCENT_TEAL    = "#2A9D8F"   # teal — secondary highlight
DARK_TEXT      = "#2C2C2C"   # near-black for axis labels

# Legacy aliases (still used by existing builders):
WARM_ORANGE = OUR_COLOR
COOL_GREY   = "#5E6B7A"      # darker grey for line/edge work (recedes less than BASELINE_COLOR)

# Full Ocean Dusk palette for multi-series charts (5 distinct colors + grey)
PALETTE = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#5E6B7A"]


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def set_paper_style():
    """Idempotent publication style.

    Override rcParams directly (instead of sns.set_theme) for finer control.
    Removes ALL titles from figures — captions live in LaTeX, not in the figure.
    """
    plt.rcParams.update({
        # Typography — Times serif (industry-standard for ML conferences)
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "serif"],
        "font.size":         9.5,
        "axes.titlesize":    10.5,
        "axes.titleweight":  "regular",
        "axes.labelsize":    9.5,
        "axes.labelweight":  "regular",
        "axes.labelcolor":   DARK_TEXT,
        "axes.titlecolor":   DARK_TEXT,
        "axes.edgecolor":    DARK_TEXT,
        "axes.linewidth":    0.8,
        "xtick.labelsize":   8.5,
        "ytick.labelsize":   8.5,
        "xtick.color":       DARK_TEXT,
        "ytick.color":       DARK_TEXT,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.size":  3.0,
        "ytick.major.size":  3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "text.color":        DARK_TEXT,

        # Spines — only left and bottom (cleaner look)
        "axes.spines.top":   False,
        "axes.spines.right": False,

        # Grid — very subtle
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.alpha":        0.18,
        "grid.linewidth":    0.5,
        "grid.linestyle":    "-",
        "grid.color":        "#9CA3AF",

        # Lines & markers
        "lines.linewidth":   1.8,
        "lines.markersize":  4.5,
        "lines.markeredgewidth": 0.6,

        # Legend
        "legend.fontsize":   8.0,
        "legend.frameon":    True,
        "legend.framealpha": 0.92,
        "legend.edgecolor":  "#D1D5DB",
        "legend.fancybox":   False,

        # Saving — vector text, 300 dpi
        "figure.dpi":        120,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype":      42,   # TrueType, no Type 3 bitmaps
        "ps.fonttype":       42,
    })


def _savefig(fig, name: str) -> str:
    """Save both PDF (vector, for LaTeX \\includegraphics) and PNG (300 dpi preview).

    Returns the PDF path.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf_path = os.path.join(FIGURES_DIR, f"{name}.pdf")
    png_path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path


# ---------------------------------------------------------------------------
# Shared data plumbing
# ---------------------------------------------------------------------------


_DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "datasets", "finale.csv")
_COLS = ("Case ID", "Activity", "Complete Timestamp")


def _load_dataset(path: str | None = None):
    from drift.io import load_event_log
    path = path or _DEFAULT_DATASET
    df = load_event_log(path, *_COLS, keep_only_complete=True)
    df["Complete Timestamp"] = pd.to_datetime(df["Complete Timestamp"])
    df = df.sort_values(["Case ID", "Complete Timestamp"]).reset_index(drop=True)
    return df


def _inject_on_back_half(df: pd.DataFrame, pattern: str, **kwargs):
    """Apply injection to the back half of the log (by case completion time)."""
    from drift.injection import inject as inject_drift
    case_end = df.groupby("Case ID")["Complete Timestamp"].max().sort_values()
    mid = len(case_end) // 2
    late_ids = set(case_end.iloc[mid:].index)
    df_late = df[df["Case ID"].isin(late_ids)]
    df_early = df[~df["Case ID"].isin(late_ids)]
    df_late_inj, gt = inject_drift(
        df_late, pattern,
        case_id_col="Case ID", activity_col="Activity",
        timestamp_col="Complete Timestamp", **kwargs,
    )
    return pd.concat([df_early, df_late_inj], ignore_index=True), gt, mid


# ---------------------------------------------------------------------------
# Figure 1 -- Teaser
# ---------------------------------------------------------------------------


def fig_teaser(out_name: str = "fig1_teaser",
               dataset_path: str | None = None,
               pattern: str = "insertion",
               target: str = "Take in charge ticket",
               secondary: str = "AutoReview",
               fraction: float = 0.7,
               seed: int = 42) -> str:
    """Three-panel teaser: drift signal + CPD + top transport flows."""
    set_paper_style()
    from drift.localization import (
        bootstrap_change_point_ci, compute_drift_signal, detect_change_points,
        signal_index_to_case_position,
    )
    from drift.io import build_cases_dataframe
    from drift.ot_attribution import attribution_report

    df = _load_dataset(dataset_path)
    df_inj, gt, true_split = _inject_on_back_half(
        df, pattern, after_activity=target, new_activity=secondary,
        fraction=fraction, seed=seed,
    )

    sig = compute_drift_signal(df_inj, window=200, step=50,
                                case_id_col="Case ID", activity_col="Activity",
                                timestamp_col="Complete Timestamp")
    cps = detect_change_points(sig["signal"])
    cis = bootstrap_change_point_ci(sig["signal"], cps, B=50, seed=seed)

    # Split at the first detected CP for the attribution panel
    if cps:
        split_pos = signal_index_to_case_position(cps[0], sig)
    else:
        split_pos = true_split
    case_end = df_inj.groupby("Case ID")["Complete Timestamp"].max().sort_values()
    base_ids = set(case_end.iloc[:split_pos].index)
    df_base = df_inj[df_inj["Case ID"].isin(base_ids)]
    df_curr = df_inj[~df_inj["Case ID"].isin(base_ids)]
    cases_base = build_cases_dataframe(df_base, *_COLS)
    cases_curr = build_cases_dataframe(df_curr, *_COLS)
    attr = attribution_report(cases_base, cases_curr, k_flows=5, k_changes=5)

    fig = plt.figure(figsize=(11, 4.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0])
    ax_signal = fig.add_subplot(gs[0, 0])
    ax_flows = fig.add_subplot(gs[0, 1])

    # --- left: drift signal + CPD ---
    scale_names = sig["scale_names"]
    scale_colors = [COOL_GREY, ACCENT_TEAL, WARM_ORANGE]
    scale_labels = ["activity JSD", "DFG JSD", "trace JSD"]
    for col, color, label in zip(range(3), scale_colors, scale_labels):
        ax_signal.plot(sig["case_positions"], sig["signal"][:, col],
                       color=color, label=label, linewidth=1.6)
    # ground-truth drift onset
    ax_signal.axvspan(true_split, sig["case_positions"][-1], color=WARM_ORANGE,
                      alpha=0.07, label="injected drift region")
    ax_signal.axvline(true_split, color=WARM_ORANGE, linestyle=":", linewidth=1.4,
                      label=f"true onset @ case {true_split}")
    # detected change points + 95% CI
    for cp, ci in zip(cps, cis):
        cp_case = signal_index_to_case_position(cp, sig)
        ci_lo = signal_index_to_case_position(ci["ci_lo"], sig)
        ci_hi = signal_index_to_case_position(ci["ci_hi"], sig)
        ax_signal.axvspan(ci_lo, ci_hi, color=DARK_TEXT, alpha=0.12)
        ax_signal.axvline(cp_case, color=DARK_TEXT, linestyle="--", linewidth=1.4,
                          label=f"detected CP @ {cp_case} (CI [{ci_lo}, {ci_hi}])")
    ax_signal.set_xlabel("case position (completion order)")
    ax_signal.set_ylabel("multi-scale JSD")
    ax_signal.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.95)

    # --- right: top transport flows ---
    # Compact activity name → shorter label for readability
    _ABBREV = {
        "Assign seriousness": "Assign",
        "Take in charge ticket": "Take",
        "Resolve ticket": "Resolve",
        "Closed": "Closed",
        "Wait": "Wait",
        "Require upgrade": "Upgrade",
        "AutoReview": "AutoReview",
        "QueuedWait": "QueuedWait",
        "Create SW anomaly": "CreateSW",
        "Resolve SW anomaly": "ResolveSW",
        "Schedule intervention": "Schedule",
    }
    def _abbr_seq(seq):
        return " → ".join(_ABBREV.get(a, a) for a in seq)

    flows = attr["top_transport_flows"][:5]
    labels = []
    for f in flows:
        from_str = _abbr_seq(f["from_variant"])
        to_str = _abbr_seq(f["to_variant"])
        labels.append(f"{from_str}\n→ {to_str}")
    masses = [f["mass"] for f in flows]
    y = np.arange(len(masses))
    ax_flows.barh(y, masses, color=WARM_ORANGE, edgecolor=DARK_TEXT, linewidth=0.6)
    ax_flows.set_yticks(y)
    ax_flows.set_yticklabels(labels, fontsize=7)
    ax_flows.invert_yaxis()
    ax_flows.set_xlabel(f"transport mass  ($W_1$ = {attr['w1']:.3f})")

    return _savefig(fig, out_name)


# ---------------------------------------------------------------------------
# Figure 2 -- Method overview (pipeline diagram)
# ---------------------------------------------------------------------------


def fig_method_overview(out_name: str = "fig2_method_overview") -> str:
    """Method-overview pipeline diagram. No data --- pure architecture.

    Layout (3 horizontal bands):
      Top    : Event log L (single source).
      Middle : Detection grid (4 modules M1--M4 in parallel) on the LEFT;
               M5 controlled injection on the RIGHT, producing oracle G.
      Bottom : M6 LLM evaluation pipeline (Analyst -> Extract -> Metrics -> Judge),
               consuming R(L) from the detection grid above and G from M5 on the right.
    """
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.patches import FancyArrowPatch

    set_paper_style()
    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Color roles ----------------------------------------------------
    DATA   = "#264653"           # dark teal — log / R(L) / G / E artifacts
    DETECT = ACCENT_TEAL         # teal — detection modules (M1, M3, M4)
    OT     = WARM_ORANGE         # coral — M2 (core technical contribution)
    INJECT = "#E9C46A"           # sand — M5 injection oracle path
    EVAL   = WARM_ORANGE         # coral — M6 pipeline (core methodological contribution)
    GROUP_FILL = "#F4F2EE"       # very soft warm grey for group panels
    GROUP_EDGE = "#C8C8C2"

    def box(x, y, w, h, label, *, face, edge=DARK_TEXT, text="white", fs=9,
            weight="bold", lw=1.0, alpha=0.92, radius=1.8):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.02,rounding_size={radius}",
            facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=text, weight=weight, linespacing=1.25)

    def group(x, y, w, h, title):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=2.2",
            facecolor=GROUP_FILL, edgecolor=GROUP_EDGE, linewidth=0.8,
        ))
        # title at the TOP-CENTER, well above any content
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top",
                fontsize=9, color=DARK_TEXT, weight="bold", style="italic")

    def arrow(x0, y0, x1, y1, *, color=DARK_TEXT, lw=1.1, style="-|>",
              connectionstyle="arc3,rad=0", label=None, label_pos=0.5,
              label_offset=(0, 1.6), label_fs=8):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle=style, mutation_scale=12,
            color=color, linewidth=lw,
            connectionstyle=connectionstyle, shrinkA=0, shrinkB=0,
        ))
        if label is not None:
            lx = x0 + (x1 - x0) * label_pos + label_offset[0]
            ly = y0 + (y1 - y0) * label_pos + label_offset[1]
            ax.text(lx, ly, label, ha="center", va="center",
                    fontsize=label_fs, color=DARK_TEXT, style="italic",
                    bbox=dict(boxstyle="round,pad=0.20",
                              facecolor="white", edgecolor="none", alpha=0.96))

    # =====================================================================
    # TOP BAND --- Event log L (wide box, well centered)
    # =====================================================================
    box(28, 88, 44, 9.5,
        "Event log  $L = \\{(c,\\, a,\\, t)\\}$\n"
        "$N$ cases   ·   $|\\mathcal{A}|$ activities   ·   $T$ events",
        face=DATA, fs=10)

    # =====================================================================
    # MIDDLE BAND --- Detection (left)  |  Injection (right)
    # =====================================================================
    # Detection group panel
    group(2, 38, 62, 44, "Detection  ·  produces structured report  $\\mathcal{R}(L)$")

    # Sub-label for the split
    ax.text(33, 73.5, "case-completion split",
            fontsize=8, color=DARK_TEXT, style="italic", ha="center")
    box(7, 65.5, 25, 6, "Baseline  $L_b$",
        face="white", edge=DATA, text=DATA, fs=9, weight="bold", radius=1.2, alpha=1.0)
    box(34, 65.5, 25, 6, "Current  $L_c$",
        face="white", edge=DATA, text=DATA, fs=9, weight="bold", radius=1.2, alpha=1.0)

    # 2x2 grid of modules M1..M4
    box(7,  52, 25, 9, "M1  ·  Multi-scale JSD\n$\\boldsymbol{d} = (d^{\\mathrm{act}}, d^{\\mathrm{dfg}}, d^{\\mathrm{var}})$",
        face=DETECT, fs=8.5)
    box(34, 52, 25, 9, "M2  ·  Variant-level OT\n$W_1$,  plan  $\\pi$,  top-$K$ flows",
        face=OT, fs=8.5)
    box(7,  41, 25, 8, "M3  ·  PELT change-point\nlocation + 95% CI",
        face=DETECT, fs=8.5)
    box(34, 41, 25, 8, "M4  ·  Permutation test\np-values  (case unit)",
        face=DETECT, fs=8.5)

    # Injection panel (right)
    group(68, 38, 30, 44, "Oracle branch  ·  M5 injection")
    box(72, 65, 22, 8, "M5  ·  Bose-style injection\ninsertion / deletion\nsubst. / loop",
        face=INJECT, text=DARK_TEXT, fs=8.5)
    box(72, 50, 22, 9, "Oracle  $\\mathcal{G}$\npattern,  $a^\\star$,  $a^+$,\n$\\mathcal{C}_{\\mathrm{affected}}$",
        face="white", edge=INJECT, text=DARK_TEXT, fs=8.5, weight="bold", lw=1.4, radius=1.2)

    # R(L) collector annotation
    ax.text(33, 35.5, "$\\mathcal{R}(L)$  ·  structured report",
            ha="center", va="center", fontsize=9, color=DATA,
            weight="bold", style="italic")

    # =====================================================================
    # BOTTOM BAND --- M6 LLM evaluation pipeline
    # =====================================================================
    group(2, 4, 96, 27, "M6  ·  LLM evaluation protocol  (extract  ·  metrics  ·  judge)")

    box(5,  14, 17, 9, "Analyst LLM\n$\\mathcal{R}(L) \\rightarrow \\mathcal{E}$",
        face=EVAL, fs=8.5)
    box(26, 14, 15, 9, "Extract\n$\\mathcal{E} \\rightarrow$ claim",
        face=EVAL, fs=8.5)
    box(45, 14, 18, 9, "Metrics\nF1,  Jaccard,\npattern\\_match",
        face=EVAL, fs=8.5)
    box(67, 14, 14, 9, "Judge LLM\nLikert 1–5",
        face=EVAL, fs=8.5)
    box(85, 12.5, 12, 12, "8 quantitative\nmetrics",
        face="white", edge=EVAL, text=EVAL, fs=9, weight="bold", lw=1.4, radius=1.2)

    # =====================================================================
    # ARROWS
    # =====================================================================
    # Event log split into baseline/current
    arrow(44, 88, 19.5, 71.5)
    arrow(56, 88, 46.5, 71.5)
    # Event log into M5 injection
    arrow(72, 92, 83, 73, label="optional", label_offset=(2.5, 1.6))

    # Baseline & Current into modules
    arrow(19.5, 65.5, 19.5, 61)
    arrow(46.5, 65.5, 46.5, 61)
    arrow(19.5, 52, 19.5, 49, color="#888888", lw=0.7)
    arrow(46.5, 52, 46.5, 49, color="#888888", lw=0.7)

    # Modules converge into R(L)
    arrow(19.5, 41, 31, 37, color=DATA, lw=1.1)
    arrow(46.5, 41, 35, 37, color=DATA, lw=1.1)

    # R(L) into Analyst LLM (curved down-left)
    arrow(28, 34, 14, 23.2, color=DATA, lw=1.2,
          connectionstyle="arc3,rad=0.20",
          label="$\\mathcal{R}(L)$", label_pos=0.45, label_offset=(-2.8, 0.6))

    # M5 -> Oracle G
    arrow(83, 65, 83, 59, color="#B89740")
    # Oracle G -> Metrics
    arrow(83, 50, 54, 23.2, color="#B89740", lw=1.1,
          connectionstyle="arc3,rad=0.22",
          label="$\\mathcal{G}$", label_pos=0.55, label_offset=(2.4, -0.3))
    # Oracle G -> Judge
    arrow(83, 50, 74, 23.2, color="#B89740", lw=1.1,
          connectionstyle="arc3,rad=-0.18",
          label="$\\mathcal{G}$", label_pos=0.6, label_offset=(2.6, 0.6))

    # Pipeline within M6
    arrow(22, 18.5, 26, 18.5)
    arrow(41, 18.5, 45, 18.5)
    arrow(63, 18.5, 67, 18.5)
    arrow(81, 18.5, 85, 18.5)

    # E annotation on Analyst -> Extract
    ax.text(24, 21.5, "$\\mathcal{E}$",
            ha="center", va="center", fontsize=9.5, color=DARK_TEXT,
            style="italic")

    # =====================================================================
    # LEGEND (top-right of the figure, doesn't compete with group titles)
    # =====================================================================
    leg_x, leg_y = 2, 84.5
    legend_items = [
        (DETECT, "detection"),
        (OT,     "core OT"),
        (INJECT, "oracle"),
        (EVAL,   "LLM eval"),
        (DATA,   "data / R(L)"),
    ]
    for i, (c, lbl) in enumerate(legend_items):
        cx = leg_x
        cy = leg_y - i * 2.4
        ax.add_patch(FancyBboxPatch(
            (cx, cy), 1.6, 1.6,
            boxstyle="round,pad=0.02,rounding_size=0.45",
            facecolor=c, edgecolor=DARK_TEXT, linewidth=0.5,
        ))
        ax.text(cx + 2.4, cy + 0.8, lbl, ha="left", va="center",
                fontsize=8, color=DARK_TEXT)

    return _savefig(fig, out_name)


# ---------------------------------------------------------------------------
# Figure 3 -- Multi-scale signature across 4 injection patterns
# ---------------------------------------------------------------------------


def fig_multi_scale(out_name: str = "fig3_multi_scale",
                    dataset_path: str | None = None) -> str:
    """4 grouped-bar panels: drift vector per injection pattern."""
    set_paper_style()
    from drift.metrics import multi_scale_drift
    from drift.ot_attribution import (
        attribution_report,
    )
    from drift.io import build_cases_dataframe

    df = _load_dataset(dataset_path)

    scenarios = [
        ("insertion",    dict(after_activity="Take in charge ticket", new_activity="AutoReview", fraction=0.6)),
        ("deletion",     dict(target_activity="Wait", fraction=0.8)),
        ("substitution", dict(src_activity="Wait", dst_activity="QueuedWait", fraction=0.7)),
        ("loop",         dict(target_activity="Resolve ticket", fraction=0.5, repeat_range=(2, 2))),
    ]

    # NULL run (no injection) — split by midpoint
    case_end = df.groupby("Case ID")["Complete Timestamp"].max().sort_values()
    mid = len(case_end) // 2
    base_ids = set(case_end.iloc[:mid].index)
    df_base_null = df[df["Case ID"].isin(base_ids)]
    df_curr_null = df[~df["Case ID"].isin(base_ids)]
    null_scores = multi_scale_drift(df_base_null, df_curr_null,
                                     case_id_col="Case ID", activity_col="Activity",
                                     timestamp_col="Complete Timestamp")
    cases_b_null = build_cases_dataframe(df_base_null, *_COLS)
    cases_c_null = build_cases_dataframe(df_curr_null, *_COLS)
    null_w1 = attribution_report(cases_b_null, cases_c_null, k_flows=1, k_changes=1)["w1"]

    rows = []
    for pat, kw in scenarios:
        df_full, gt, _ = _inject_on_back_half(df, pat, seed=42, **kw)
        case_end_f = df_full.groupby("Case ID")["Complete Timestamp"].max().sort_values()
        mid_f = len(case_end_f) // 2
        base_ids_f = set(case_end_f.iloc[:mid_f].index)
        df_b = df_full[df_full["Case ID"].isin(base_ids_f)]
        df_c = df_full[~df_full["Case ID"].isin(base_ids_f)]
        msd = multi_scale_drift(df_b, df_c,
                                 case_id_col="Case ID", activity_col="Activity",
                                 timestamp_col="Complete Timestamp")
        cb = build_cases_dataframe(df_b, *_COLS)
        cc = build_cases_dataframe(df_c, *_COLS)
        w1 = attribution_report(cb, cc, k_flows=1, k_changes=1)["w1"]
        rows.append({
            "pattern": pat,
            "activity_jsd": msd["activity_jsd"],
            "dfg_jsd":      msd["dfg_jsd"],
            "trace_jsd":    msd["trace_jsd"],
            "trace_w1":     w1,
        })

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True, constrained_layout=True)
    scales = ["activity_jsd", "dfg_jsd", "trace_jsd", "trace_w1"]
    scale_labels = ["activity\nJSD", "DFG\nJSD", "trace\nJSD", "trace\n$W_1$"]
    bar_colors = [COOL_GREY, ACCENT_TEAL, WARM_ORANGE, "#A04060"]

    for ax, row in zip(axes, rows):
        values = [row[s] for s in scales]
        x = np.arange(len(scales))
        ax.bar(x, values, color=bar_colors, edgecolor=DARK_TEXT, linewidth=0.6)
        # NULL baseline overlaid as light grey horizontal markers
        null_vals = [null_scores["activity_jsd"], null_scores["dfg_jsd"],
                     null_scores["trace_jsd"], null_w1]
        for xi, vi in zip(x, null_vals):
            ax.hlines(vi, xi - 0.4, xi + 0.4, colors=DARK_TEXT, linestyles="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(scale_labels, fontsize=8)
        ax.set_title(row["pattern"], fontsize=10)
        ax.set_ylim(0, max(0.5, max(values) * 1.15))
    axes[0].set_ylabel("divergence / distance")
    return _savefig(fig, out_name)


# ---------------------------------------------------------------------------
# Figure 4 -- Detection ROC: proposed vs 3 baselines, with bootstrap CI
# ---------------------------------------------------------------------------


def fig_detection_roc(out_name: str = "fig4_detection_roc",
                       n_seeds: int = 6,
                       n_bootstrap: int = 200,
                       dataset_path: str | None = None) -> str:
    """ROC for 4 detectors with bootstrap 95% CI envelopes.

    Methods:
        proposed       : variant-level W1 with edit-distance ground metric
        TV-traces      : TV on the same variant distributions (ablates OT geometry)
        legacy max(TV,W): max(TV_trace, W_duration/median) -- the v1 baseline
        chi2-activities: chi-squared on activity frequency table (no structure)
    """
    set_paper_style()
    from drift.metrics import activity_frequency_dist, align, dfg_dist, jsd, multi_scale_drift
    from drift.io import build_cases_dataframe
    from drift.ot_attribution import (
        attribution_report, edit_distance_matrix, joint_support, variant_distribution,
    )
    from drift.injection import inject as inject_drift
    from scipy.stats import chi2_contingency
    from scipy.stats import wasserstein_distance as _wd

    df_orig = _load_dataset(dataset_path)
    case_end = df_orig.groupby("Case ID")["Complete Timestamp"].max().sort_values()
    mid = len(case_end) // 2
    base_ids = set(case_end.iloc[:mid].index)
    df_base = df_orig[df_orig["Case ID"].isin(base_ids)]
    df_curr_raw = df_orig[~df_orig["Case ID"].isin(base_ids)]

    # ------------------------------------------------------------------ scoring fns

    def _variant_tv_score(df_b, df_c):
        """TV on variant distributions (same support as proposed, no ground metric)."""
        cb = build_cases_dataframe(df_b, *_COLS)
        cc = build_cases_dataframe(df_c, *_COLS)
        if cb.empty or cc.empty:
            return 0.0
        va, pa, _ = variant_distribution(cb)
        vb_, pb, _ = variant_distribution(cc)
        _, pa_j, pb_j = joint_support(va, pa, vb_, pb)
        return float(0.5 * np.sum(np.abs(pa_j - pb_j)))

    def _proposed_score(df_b, df_c):
        cb = build_cases_dataframe(df_b, *_COLS)
        cc = build_cases_dataframe(df_c, *_COLS)
        if cb.empty or cc.empty:
            return 0.0
        attr = attribution_report(cb, cc, k_flows=1, k_changes=1)
        return float(attr["w1"])

    def _legacy_score(df_b, df_c):
        def _to_str(df):
            return df.sort_values(["Case ID", "Complete Timestamp"]).groupby(
                "Case ID", sort=False)["Activity"].apply(lambda s: " -> ".join(map(str, s.tolist())))
        traces_b = _to_str(df_b); traces_c = _to_str(df_c)
        all_t = sorted(set(traces_b) | set(traces_c))
        if not all_t:
            return 0.0
        idx = {t: i for i, t in enumerate(all_t)}
        vb = np.zeros(len(all_t)); vc = np.zeros(len(all_t))
        for t, f in traces_b.value_counts(normalize=True).items(): vb[idx[t]] = f
        for t, f in traces_c.value_counts(normalize=True).items(): vc[idx[t]] = f
        tv = float(0.5 * np.sum(np.abs(vb - vc)))
        d_b = (df_b.groupby("Case ID")["Complete Timestamp"].max()
               - df_b.groupby("Case ID")["Complete Timestamp"].min()).dt.total_seconds().to_numpy() / 60.0
        d_c = (df_c.groupby("Case ID")["Complete Timestamp"].max()
               - df_c.groupby("Case ID")["Complete Timestamp"].min()).dt.total_seconds().to_numpy() / 60.0
        if d_b.size and d_c.size:
            w = float(_wd(d_b, d_c))
            sc = float(np.median(d_b)) if np.median(d_b) > 0 else float(np.mean(d_b))
            wn = w / sc if sc > 0 else 0.0
        else:
            wn = 0.0
        return max(tv, wn)

    def _chi2_score(df_b, df_c):
        cnt_b = df_b["Activity"].value_counts()
        cnt_c = df_c["Activity"].value_counts()
        all_acts = sorted(set(cnt_b.index) | set(cnt_c.index))
        if not all_acts:
            return 0.0
        obs = np.array([
            [int(cnt_b.get(a, 0)) for a in all_acts],
            [int(cnt_c.get(a, 0)) for a in all_acts],
        ], dtype=float)
        # chi2_contingency requires no zero columns; drop them
        col_sums = obs.sum(axis=0)
        keep = col_sums > 0
        obs = obs[:, keep]
        if obs.shape[1] < 2:
            return 0.0
        try:
            chi2, _, _, _ = chi2_contingency(obs)
            return float(chi2)
        except (ValueError, ZeroDivisionError):
            return 0.0

    methods = [
        ("proposed $W_1$(traces, edit)",     _proposed_score,    WARM_ORANGE, 2.4),
        ("TV(traces)",                       _variant_tv_score,  ACCENT_TEAL, 1.8),
        ("legacy max(TV, W_dur/median)",     _legacy_score,      COOL_GREY,   1.6),
        ("chi²(activities)",                 _chi2_score,        "#B7B7B7",   1.4),
    ]

    # ------------------------------------------------------------------ scoring grid

    patterns = ["insertion", "deletion", "substitution", "loop"]
    pattern_kwargs = {
        "insertion":    dict(after_activity="Take in charge ticket", new_activity="AutoReview", fraction=0.6),
        "deletion":     dict(target_activity="Wait", fraction=0.8),
        "substitution": dict(src_activity="Wait", dst_activity="QueuedWait", fraction=0.7),
        "loop":         dict(target_activity="Resolve ticket", fraction=0.5, repeat_range=(2, 2)),
    }

    method_scores = {name: [] for name, _, _, _ in methods}
    labels = []  # parallel list

    # NULL samples: random splits of the entire log (no injection -> label 0)
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        cids = df_orig["Case ID"].drop_duplicates().to_numpy()
        rng.shuffle(cids)
        m = len(cids) // 2
        a_ids = set(cids[:m]); b_ids = set(cids[m:])
        df_a = df_orig[df_orig["Case ID"].isin(a_ids)]
        df_b = df_orig[df_orig["Case ID"].isin(b_ids)]
        for name, fn, _, _ in methods:
            method_scores[name].append(fn(df_a, df_b))
        labels.append(0)

    # INJECTED samples (4 patterns × n_seeds, label = 1)
    for pat in patterns:
        for seed in range(n_seeds):
            df_curr_inj, _ = inject_drift(
                df_curr_raw, pat, seed=seed,
                case_id_col="Case ID", activity_col="Activity",
                timestamp_col="Complete Timestamp", **pattern_kwargs[pat],
            )
            for name, fn, _, _ in methods:
                method_scores[name].append(fn(df_base, df_curr_inj))
            labels.append(1)

    labels = np.asarray(labels)

    # ------------------------------------------------------------------ ROC + bootstrap

    def _roc_curve(scores, lab):
        order = np.argsort(-scores)
        lab = lab[order]
        tp = np.cumsum(lab == 1)
        fp = np.cumsum(lab == 0)
        n_pos = max(1, int((lab == 1).sum()))
        n_neg = max(1, int((lab == 0).sum()))
        tpr = np.concatenate([[0], tp / n_pos, [1]])
        fpr = np.concatenate([[0], fp / n_neg, [1]])
        # ensure monotone in fpr for interpolation
        order_f = np.argsort(fpr)
        fpr, tpr = fpr[order_f], tpr[order_f]
        return fpr, tpr, float(np.trapezoid(tpr, fpr))

    def _bootstrap_ci(scores, lab, B, seed):
        rng = np.random.default_rng(seed)
        n = len(scores)
        grid = np.linspace(0, 1, 100)
        tprs = np.empty((B, grid.size))
        aucs = np.empty(B)
        for b in range(B):
            idx = rng.choice(n, size=n, replace=True)
            f, t, auc = _roc_curve(scores[idx], lab[idx])
            tprs[b] = np.interp(grid, f, t)
            aucs[b] = auc
        return grid, tprs, aucs

    # ------------------------------------------------------------------ plot

    fig, ax = plt.subplots(figsize=(5.6, 4.3), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="lightgrey", linestyle=":", linewidth=1.0,
            label="chance")
    for (name, _, color, lw) in methods:
        s = np.asarray(method_scores[name])
        f_obs, t_obs, auc_obs = _roc_curve(s, labels)
        grid, tprs_bs, aucs_bs = _bootstrap_ci(s, labels, n_bootstrap, seed=0)
        lo, hi = np.percentile(tprs_bs, [2.5, 97.5], axis=0)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.12, linewidth=0)
        auc_lo, auc_hi = np.percentile(aucs_bs, [2.5, 97.5])
        ax.plot(grid, np.interp(grid, f_obs, t_obs), color=color, linewidth=lw,
                label=f"{name} — AUC={auc_obs:.2f} [{auc_lo:.2f},{auc_hi:.2f}]")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.95)
    return _savefig(fig, out_name)


# ---------------------------------------------------------------------------
# Figure 5 -- Localization accuracy scatter
# ---------------------------------------------------------------------------


def fig_localization(out_name: str = "fig5_localization",
                     n_seeds: int = 10,
                     dataset_path: str | None = None) -> str:
    """Scatter true vs detected change-point across many seeds & patterns."""
    set_paper_style()
    from drift.localization import (
        compute_drift_signal, detect_change_points, signal_index_to_case_position,
    )
    from drift.injection import inject as inject_drift

    df = _load_dataset(dataset_path)
    case_end = df.groupby("Case ID")["Complete Timestamp"].max().sort_values()
    n = len(case_end)

    patterns_kwargs = {
        "insertion":    dict(after_activity="Take in charge ticket", new_activity="AutoReview", fraction=0.6),
        "deletion":     dict(target_activity="Wait", fraction=0.8),
        "substitution": dict(src_activity="Wait", dst_activity="QueuedWait", fraction=0.7),
        "loop":         dict(target_activity="Resolve ticket", fraction=0.5, repeat_range=(2, 2)),
    }

    rows = []
    # vary the true onset across seeds: fraction in {0.3, 0.45, 0.6} of cases
    for pat, kw in patterns_kwargs.items():
        for seed in range(n_seeds):
            onset_frac = 0.3 + (seed % 3) * 0.15
            true_split = int(n * onset_frac)
            late_ids = set(case_end.iloc[true_split:].index)
            df_late = df[df["Case ID"].isin(late_ids)]
            df_early = df[~df["Case ID"].isin(late_ids)]
            df_late_inj, _ = inject_drift(df_late, pat, seed=seed,
                                          case_id_col="Case ID",
                                          activity_col="Activity",
                                          timestamp_col="Complete Timestamp", **kw)
            df_full = pd.concat([df_early, df_late_inj], ignore_index=True)
            sig = compute_drift_signal(df_full, window=200, step=50,
                                       case_id_col="Case ID", activity_col="Activity",
                                       timestamp_col="Complete Timestamp")
            cps = detect_change_points(sig["signal"])
            if cps:
                detected = signal_index_to_case_position(cps[0], sig)
            else:
                detected = None
            rows.append({"pattern": pat, "seed": seed, "true": true_split, "detected": detected})

    df_r = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                              gridspec_kw={"width_ratios": [1.3, 0.7]})
    ax_scat = axes[0]
    markers = {"insertion": "o", "deletion": "s", "substitution": "^", "loop": "D"}
    colors  = {"insertion": WARM_ORANGE, "deletion": COOL_GREY,
               "substitution": ACCENT_TEAL, "loop": "#A04060"}
    detected_mask = df_r["detected"].notna()
    for pat in patterns_kwargs:
        sub = df_r[(df_r["pattern"] == pat) & detected_mask]
        ax_scat.scatter(sub["true"], sub["detected"], s=42,
                        marker=markers[pat], facecolor=colors[pat],
                        edgecolor=DARK_TEXT, alpha=0.8, label=pat)
    # missed detections plotted along the bottom
    missed = df_r[~detected_mask]
    if len(missed) > 0:
        ax_scat.scatter(missed["true"], [0] * len(missed), marker="x",
                        s=42, color=DARK_TEXT, label="undetected")
    lim_min = min(df_r["true"].min(), df_r[detected_mask]["detected"].min()) - 50 if detected_mask.any() else 0
    lim_max = max(df_r["true"].max(), df_r[detected_mask]["detected"].max()) + 50 if detected_mask.any() else n
    ax_scat.plot([lim_min, lim_max], [lim_min, lim_max], color="lightgrey",
                 linestyle=":", linewidth=1.0)
    # ±200 case tolerance band
    ax_scat.fill_between([lim_min, lim_max],
                          [lim_min - 200, lim_max - 200],
                          [lim_min + 200, lim_max + 200],
                          color="grey", alpha=0.08, label="±200 case tol.")
    ax_scat.set_xlabel("true drift onset (case position)")
    ax_scat.set_ylabel("detected CP (case position)")
    ax_scat.legend(loc="lower right", fontsize=8)

    # delay boxplot (right panel)
    df_r["delay"] = df_r["detected"] - df_r["true"]
    df_box = df_r[df_r["delay"].notna()]
    sns.boxplot(data=df_box, x="pattern", y="delay", ax=axes[1],
                palette=[colors[p] for p in patterns_kwargs])
    axes[1].axhline(0, color=DARK_TEXT, linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("detected − true (cases)")
    for label in axes[1].get_xticklabels():
        label.set_rotation(30)
    fig.tight_layout()
    return _savefig(fig, out_name)


# ---------------------------------------------------------------------------
# Figure 6 -- LLM rubric scores (needs run_llm_evaluation.py output)
# ---------------------------------------------------------------------------


def fig_llm_rubric(out_name: str = "fig6_llm_rubric",
                   grid_path: str | None = None) -> str:
    """Three-panel comparison of analyst framings across 4 injection patterns.

    Left panel: activity F1 (auto metric). Middle: judge accuracy (1-5).
    Right: judge completeness (1-5). For each panel, 4 patterns x 3 framings
    grouped bars with std-dev error bars.

    Reads ``outputs/llm_evaluation_grid.json`` (with framings dimension)
    produced by ``run_llm_evaluation.py``.
    """
    set_paper_style()
    grid_path = grid_path or os.path.join(PROJECT_ROOT, "outputs", "llm_evaluation_grid.json")

    if not os.path.exists(grid_path):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.55, "fig6 placeholder", ha="center",
                fontsize=14, weight="bold", color=DARK_TEXT)
        ax.text(0.5, 0.30,
                f"run `python run_llm_evaluation.py` first.\n"
                f"expected: {os.path.relpath(grid_path, PROJECT_ROOT)}",
                ha="center", fontsize=10, color=DARK_TEXT)
        return _savefig(fig, out_name)

    with open(grid_path, "r", encoding="utf-8") as f:
        grid = json.load(f)
    aggregates = grid.get("aggregates", {})

    # Detect "framing-pattern" labels vs just "pattern"
    framings, patterns = [], []
    for k in aggregates:
        if "-" in k:
            framing, pattern = k.split("-", 1)
        else:
            framing, pattern = "all", k
        if framing not in framings:
            framings.append(framing)
        if pattern not in patterns:
            patterns.append(pattern)

    # Canonical order
    framing_order = ["proposed", "legacy", "raw", "all"]
    pattern_order = ["insertion", "deletion", "substitution", "loop"]
    framings = [f for f in framing_order if f in framings]
    patterns = [p for p in pattern_order if p in patterns]
    framing_color = {"proposed": WARM_ORANGE, "legacy": COOL_GREY, "raw": ACCENT_TEAL, "all": WARM_ORANGE}

    metrics_to_plot = [
        ("activity_f1",        "activity F1 (auto)",       0,   1.0),
        ("judge_accuracy",     "judge accuracy (1-5)",     1.0, 5.0),
        ("judge_completeness", "judge completeness (1-5)", 1.0, 5.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), constrained_layout=True)
    x = np.arange(len(patterns))
    bar_w = 0.8 / max(1, len(framings))

    for ax, (key, ylabel, ymin, ymax) in zip(axes, metrics_to_plot):
        for i, framing in enumerate(framings):
            means = []
            stds = []
            for pat in patterns:
                label = f"{framing}-{pat}" if len(framings) > 1 or framing != "all" else pat
                a = aggregates.get(label, {}).get(key, {})
                means.append(a.get("mean") or 0)
                stds.append(a.get("std") or 0)
            offset = (i - (len(framings) - 1) / 2) * bar_w
            ax.bar(x + offset, means, bar_w, yerr=stds, capsize=3,
                   color=framing_color.get(framing, COOL_GREY),
                   edgecolor=DARK_TEXT, linewidth=0.5,
                   label=framing if ax is axes[0] else None)
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, fontsize=9)
        ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.08)
        ax.set_ylabel(ylabel, fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8, title="framing")

    return _savefig(fig, out_name)


# ---------------------------------------------------------------------------
# Figure 7 -- Component ablation
# ---------------------------------------------------------------------------


def fig_ablation(out_name: str = "fig7_ablation",
                 dataset_path: str | None = None) -> str:
    """Bar chart: drift score per ablation. Lower = component removed degrades signal."""
    set_paper_style()
    from drift.metrics import (
        activity_frequency_dist, align, dfg_dist, jsd, trace_variant_dist,
    )
    from drift.io import build_cases_dataframe
    from drift.ot_attribution import attribution_report

    df = _load_dataset(dataset_path)
    df_full, _, _ = _inject_on_back_half(
        df, "insertion",
        after_activity="Take in charge ticket", new_activity="AutoReview",
        fraction=0.6, seed=42,
    )
    case_end = df_full.groupby("Case ID")["Complete Timestamp"].max().sort_values()
    mid = len(case_end) // 2
    base_ids = set(case_end.iloc[:mid].index)
    df_b = df_full[df_full["Case ID"].isin(base_ids)]
    df_c = df_full[~df_full["Case ID"].isin(base_ids)]

    pa = activity_frequency_dist(df_b); pb = activity_frequency_dist(df_c)
    act_jsd = jsd(*align(pa, pb))
    da = dfg_dist(df_b); db = dfg_dist(df_c)
    dfg_j = jsd(*align(da, db))
    ta = trace_variant_dist(df_b); tb = trace_variant_dist(df_c)
    trace_j = jsd(*align(ta, tb))
    # Actual TV(traces) — used by "only legacy TV" variant. Defined as 0.5 * L1
    # on the JOINT support of variant distributions (matches §4.3 baseline def).
    ta_arr, tb_arr = align(ta, tb)
    legacy_tv = float(0.5 * np.sum(np.abs(ta_arr - tb_arr)))
    cb_b = build_cases_dataframe(df_b, *_COLS)
    cb_c = build_cases_dataframe(df_c, *_COLS)
    w1 = attribution_report(cb_b, cb_c, k_flows=1, k_changes=1)["w1"]

    variants = [
        ("full proposed\n(act + DFG + trace + $W_1$)", max(act_jsd, dfg_j, trace_j, w1), WARM_ORANGE),
        ("− OT ($W_1$)",     max(act_jsd, dfg_j, trace_j),               ACCENT_TEAL),
        ("− trace scale",    max(act_jsd, dfg_j, w1),                    ACCENT_TEAL),
        ("− DFG scale",      max(act_jsd, trace_j, w1),                  COOL_GREY),
        ("− activity scale", max(dfg_j, trace_j, w1),                    COOL_GREY),
        ("only legacy TV(traces)",   legacy_tv,                          COOL_GREY),
    ]
    labels = [v[0] for v in variants]
    values = [v[1] for v in variants]
    colors = [v[2] for v in variants]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor=DARK_TEXT, linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("max scale score")
    for yi, vi in zip(y, values):
        ax.text(vi + 0.005, yi, f"{vi:.3f}", va="center", fontsize=8)
    return _savefig(fig, out_name)


# ---------------------------------------------------------------------------
# Figure 8 -- H0 uniformity check for permutation test
# ---------------------------------------------------------------------------


def fig_h0_uniformity(out_name: str = "fig8_h0_uniformity",
                      n_splits: int = 40,
                      n_perm: int = 49,
                      dataset_path: str | None = None) -> str:
    """Histogram of permutation p-values across many random splits of a
    drift-free log. Under H0 the distribution should be approximately Uniform(0,1).

    To keep runtime manageable we use the activity-only JSD statistic (cheap)
    rather than the multi-scale aggregate; both are valid sample paths under H0,
    and activity-JSD has enough granularity at the Help Desk activity-set size
    (14) to avoid tie pathologies. Adjust n_splits / n_perm for tighter CIs.
    """
    set_paper_style()
    from drift.metrics import activity_frequency_dist, align, jsd
    from drift.significance import permutation_pvalue
    from scipy.stats import kstest

    df = _load_dataset(dataset_path)

    def stat_fn(a, b):
        pa = activity_frequency_dist(a, activity_col="Activity")
        pb = activity_frequency_dist(b, activity_col="Activity")
        return jsd(*align(pa, pb))

    rng = np.random.default_rng(2024)
    case_ids_all = df["Case ID"].drop_duplicates().to_numpy()

    pvals = []
    for i in range(n_splits):
        split_seed = int(rng.integers(0, 100_000))
        perm_seed  = int(rng.integers(0, 100_000))
        rng_split = np.random.default_rng(split_seed)
        shuffled = case_ids_all.copy()
        rng_split.shuffle(shuffled)
        mid = len(shuffled) // 2
        a_ids, b_ids = set(shuffled[:mid]), set(shuffled[mid:])
        df_a = df[df["Case ID"].isin(a_ids)]
        df_b = df[df["Case ID"].isin(b_ids)]
        p = permutation_pvalue(df_a, df_b, stat_fn,
                                B=n_perm, seed=perm_seed, case_id_col="Case ID")
        pvals.append(p)
        if (i + 1) % 5 == 0:
            print(f"    h0 split {i + 1}/{n_splits} — last p={p:.3f}", flush=True)
    pvals = np.asarray(pvals)
    ks_stat, ks_p = kstest(pvals, "uniform")

    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    n_bins = 10
    counts, edges, _ = ax.hist(pvals, bins=n_bins, range=(0, 1),
                                 color=WARM_ORANGE, edgecolor=DARK_TEXT, linewidth=0.8)
    expected = n_splits / n_bins
    ax.axhline(expected, color=DARK_TEXT, linestyle="--", linewidth=1.0,
               label=f"Uniform expectation ({expected:.1f} per bin)")
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"permutation p-value under $H_0$")
    ax.set_ylabel("count")
    # KS statistics annotated inside the axes (top-left of usable area)
    ax.text(
        0.02, 0.97,
        f"$n$ = {n_splits} splits × $B$ = {n_perm} permutations\n"
        f"KS stat = {ks_stat:.3f},  KS $p$ = {ks_p:.3f}\n"
        f"empirical mean = {pvals.mean():.3f}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8, color=DARK_TEXT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#D1D5DB", linewidth=0.6, alpha=0.92),
    )
    ax.legend(loc="upper right", fontsize=8)
    return _savefig(fig, out_name)


# ---------------------------------------------------------------------------
# Dispatch map (consumed by make_figures.py)
# ---------------------------------------------------------------------------


BUILDERS = {
    1: fig_teaser,
    2: fig_method_overview,
    3: fig_multi_scale,
    4: fig_detection_roc,
    5: fig_localization,
    6: fig_llm_rubric,
    7: fig_ablation,
    8: lambda **kw: fig_h0_uniformity(**kw),
}
