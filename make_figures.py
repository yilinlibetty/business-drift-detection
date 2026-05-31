"""Paper-figure dispatcher.

Usage:
    python make_figures.py --figure 1            # generate one
    python make_figures.py --figure all          # generate all
    python make_figures.py --list                # show available figures

Per-figure helpers live in drift/viz.py and are wired up in Phase 8.
"""

from __future__ import annotations

import argparse
import os
import sys

FIGURES = {
    1: ("teaser",            "Teaser: drift signal + CPD + transport plan + LLM excerpt"),
    2: ("method_overview",   "Method overview pipeline diagram"),
    3: ("multi_scale",       "Multi-scale signals across 4 injection patterns"),
    4: ("detection_roc",     "Detection ROC: proposed vs 3 baselines + bootstrap CI"),
    5: ("localization",      "Localization accuracy scatter: true vs detected CP"),
    6: ("llm_rubric",        "LLM rubric grouped bars: accuracy/completeness/actionability"),
    7: ("ablation",          "Ablation: drop each component, show metric drop"),
    8: ("h0_uniformity",     "H₀ uniformity histogram of permutation test p-values"),
}

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def _dispatch(figure_id: int):
    """Build a figure end-to-end via drift.viz."""
    from drift.viz import BUILDERS
    name, desc = FIGURES[figure_id]
    print(f"[fig {figure_id}] {desc}")
    builder = BUILDERS.get(figure_id)
    if builder is None:
        print(f"  no builder registered for figure {figure_id}")
        return
    out_path = builder()
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--figure", default=None,
                        help="Figure number (1..7) or 'all'")
    parser.add_argument("--list", action="store_true",
                        help="List available figures and exit")
    args = parser.parse_args()

    if args.list or args.figure is None:
        for fid, (name, desc) in FIGURES.items():
            print(f"  {fid}  {name:<20s}  {desc}")
        if args.figure is None:
            parser.print_help(sys.stderr)
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)

    if args.figure.lower() == "all":
        for fid in FIGURES:
            _dispatch(fid)
        return

    try:
        fid = int(args.figure)
    except ValueError:
        parser.error(f"--figure must be an int 1..7 or 'all', got {args.figure!r}")
    if fid not in FIGURES:
        parser.error(f"--figure {fid} unknown; valid ids: {list(FIGURES)}")
    _dispatch(fid)


if __name__ == "__main__":
    main()
