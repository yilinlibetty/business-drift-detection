# Multi-Scale Optimal-Transport Process-Drift Detection with LLM Root-Cause Evaluation

This repository implements an end-to-end framework for business-process drift detection that combines multi-scale distributional analysis with optimal-transport attribution and a systematic protocol for evaluating Large Language Model (LLM) root-cause explanations. The framework comprises five technical components and one evaluation protocol.

1. Multi-scale drift detection at the activity, directly-follows-graph (DFG), and trace levels using Jensen-Shannon divergence.
2. Variant-level optimal-transport (Wasserstein-1 with a normalised Levenshtein ground metric) for case-level attribution.
3. Pruned Exact Linear Time (PELT) change-point detection with bootstrap 95\% confidence intervals for temporal localisation.
4. A case-level permutation test that produces calibrated p-values, replacing hand-tuned thresholds.
5. An LLM analyst that consumes the structured report and emits a natural-language root-cause diagnosis.
6. A controlled-injection evaluation protocol that converts the analyst's natural-language explanation into eight quantitative metrics, including pattern-match rate, activity F1, case-ID Jaccard, and a three-dimensional oracle judge rubric.

---

## 1. Method Positioning

The earlier iteration of this codebase reduced drift to a single scalar `max(TV(trace_dist), W(duration)/median)` and triggered an alarm at the hand-picked threshold $0.05$. The present framework re-designs each stage from first principles to address five gaps that the legacy pipeline left open.

| Gap | Legacy | Present framework |
|---|---|---|
| Aggregation | `max(TV, W)` joins two heterogeneous distances | Three-scale JSD vector and variant-level $W_1$, each with its own p-value |
| Localisation | Mandatory 50/50 split | PELT change-point detection with a bootstrap 95\% confidence interval |
| Attribution | Top-$K$ trace frequencies only | Optimal-transport plan $\pi$ supplies case-level pairing of baseline and current cases |
| Significance | Hand-picked threshold 0.05 | Permutation test with the Phipson-Smyth unbiased estimator |
| Interpretation | LLM used as a downstream presentation tool | LLM treated as a measurable component with a grid of injected ground-truth scenarios |

---

## 2. Repository Structure

```text
business-drift-detection/
|-- drift/                       # Primary method package; one module per stage
|   |-- io.py                    # Event-log loaders (CSV and XES) and per-case dataframe builder
|   |-- injection.py             # Bose-style drift injection (insertion / deletion / substitution / loop)
|   |-- metrics.py               # Multi-scale JSD (activity / DFG / trace)
|   |-- significance.py          # Case-unit permutation test p-values
|   |-- localization.py          # Sliding-window signal, PELT detection, bootstrap CI
|   |-- ot_attribution.py        # Variant-level Wasserstein-1, transport plan, case-level attribution
|   |-- evaluation.py            # LLM extraction + precision/recall + judge rubric
|   `-- viz.py                   # Paper figure style and figure builders
|
|-- run_full_pipeline.py         # Pipeline orchestrator (~330 LOC)
|-- run_llm_evaluation.py        # LLM evaluation grid harness (patterns x seeds)
|-- make_figures.py              # Figure CLI dispatcher: --figure 1..8 | all
|-- convert_data.py              # Backward-compatible shim re-exporting drift.io.*
|
|-- tests/                       # 103 unit tests
|   |-- test_smoke.py
|   |-- test_injection.py
|   |-- test_metrics.py
|   |-- test_significance.py
|   |-- test_localization.py
|   |-- test_ot_attribution.py
|   `-- test_evaluation.py
|
|-- datasets/
|   |-- finale.csv               # Help Desk ticket log (4580 cases / 14 activities / 21k events)
|   |-- frequency-log.csv        # BPI Challenge-style loan process (24 activities / 71k events)
|   |-- frequency-log.xes        # Same log in XES format
|   |-- llm_analyst_official.py  # LLM analyst (schema v2 consumer; honours environment overrides)
|   `-- final_report_for_azure.json   # Pipeline output (schema v2)
|
|-- examples/
|   `-- Final_Drift_Analysis_Report.md   # Sample LLM analyst output
|
|-- outputs/                     # LLM evaluation grid JSON
|-- archive/                     # Historical experimental code (non-primary path)
|-- requirements.txt
|-- pytest.ini
`-- README.md
```

---

## 3. Environment and Dependencies

Python 3.10 or later is required; the framework has been verified on Python 3.11. The dependency stack is enumerated in `requirements.txt` and groups into three categories. Core scientific computing depends on `pandas`, `numpy`, `scipy`, and `matplotlib`. The LLM analyst path depends on `openai` and `httpx`. The new method stack additionally requires `pot >= 0.9`, `ruptures >= 1.1`, `editdistance >= 0.8`, `seaborn >= 0.13`, and `pytest >= 8.0`.

LLM credentials are read from environment variables when present. The supported variables are `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`. The scripts ship with non-secret fallback values for convenience, but environment-variable overrides are recommended in any deployment scenario.

Installation:

```bash
pip install -r requirements.txt
```

---

## 4. One-Pass Reproduction

```bash
# 0) Unit tests
pytest tests/

# 1) Main pipeline (Help Desk, CPD split, permutation test)
python run_full_pipeline.py
#    Writes datasets/final_report_for_azure.json (schema v2)
#    plus final_llm_input_prompt.txt

# 2) LLM analyst (consumes schema v2 attribution and drift_vector)
python datasets/llm_analyst_official.py
#    Writes examples/Final_Drift_Analysis_Report.md

# 3) All eight paper figures
python make_figures.py --figure all
#    Writes figures/fig1..8_*.pdf

# 4) LLM evaluation grid (4 patterns x 3 seeds x 3 framings = 36 cells; approximately 45 minutes of API time)
python run_llm_evaluation.py
#    Writes outputs/llm_evaluation_grid.json
```

---

## 5. Command-Line Interface and Environment Variables

### 5.1 `run_full_pipeline.py` flags

| Flag | Effect |
|---|---|
| `--legacy` | Reverts to the legacy `max(TV, W) > 0.05` path for compatibility |
| `--no-perm` | Skips the permutation test (faster, no p-values) |
| `--no-cpd` | Forces the 50/50 split and skips PELT detection |

### 5.2 Environment variables

| Variable | Default | Effect |
|---|---|---|
| `EVENT_LOG_PATH` | `datasets/finale.csv` | Input path. Supports CSV, XES, and XML |
| `COL_CASE_ID` / `COL_ACTIVITY` / `COL_TIMESTAMP` | Help Desk column names | Column mapping for arbitrary CSV schemas |
| `KEEP_ONLY_COMPLETE` | `true` | Retains only `lifecycle:transition=complete` events when loading XES |
| `INJECT_DRIFT` | `false` | Enables controlled drift injection |
| `DRIFT_PATTERN` | `insertion` | One of `insertion`, `deletion`, `substitution`, `loop` |
| `DRIFT_TARGET` | `Live Chat` | Anchor activity for the injection |
| `DRIFT_SECONDARY` | `AutoReview` | Secondary activity (insertion target or substitution destination) |
| `DRIFT_FRACTION` | `0.5` | Fraction of cases affected by injection |
| `DRIFT_SEED` | `42` | Seed for injection, permutation, and bootstrap |
| `SPLIT_METHOD` | `cpd` | One of `cpd` (PELT-based) or `midpoint` (50/50) |
| `PERMUTATION_B` | `100` | Number of permutation iterations |

### 5.3 Running on BPI Challenge 2017

```bash
EVENT_LOG_PATH=datasets/frequency-log.csv \
  COL_CASE_ID="case:concept:name" \
  COL_ACTIVITY="concept:name" \
  COL_TIMESTAMP="time:timestamp" \
  python run_full_pipeline.py
```

### 5.4 Running with injection enabled

```bash
INJECT_DRIFT=true DRIFT_PATTERN=insertion \
  DRIFT_TARGET="Take in charge ticket" DRIFT_SECONDARY=AutoReview \
  DRIFT_FRACTION=0.7 \
  python run_full_pipeline.py
```

---

## 6. Schema v2 Output Format

The pipeline writes a structured report to `datasets/final_report_for_azure.json` following schema v2. The schema is illustrated below.

```jsonc
{
  "schema_version": "2.0",
  "status": "DRIFT DETECTED" | "STABLE",
  "dataset": {
    "name": "finale", "n_events": 22909, "n_cases": 4580,
    "n_activities": 15, "n_variants_baseline": 158, "n_variants_current": 155,
    "n_variants_union": 269
  },
  "split": {
    "method": "cpd",
    "window": 200, "step": 50,
    "change_points": [2450],
    "ci_95": [[2200, 2550]]
  },
  "drift_vector": {
    "activity_jsd": 0.060, "activity_pvalue": 0.0099,
    "dfg_jsd":      0.231, "dfg_pvalue":      0.0099,
    "trace_jsd":    0.480,
    "trace_w1":     0.194,
    "aggregate_pvalue": 0.0099
  },
  "attribution": {
    "w1": 0.194,
    "top_transport_flows": [
      {
        "from_variant": ["Assign seriousness", "Take in charge ticket", "Resolve ticket", "Closed"],
        "to_variant":   ["Assign seriousness", "Take in charge ticket", "AutoReview", "Resolve ticket", "Closed"],
        "mass": 0.326, "edit_distance": 0.20,
        "from_case_ids_sample": ["Case 245", "Case 1300"],
        "to_case_ids_sample":   ["Case 2286", "Case 3900"]
      }
    ],
    "top_lost_variants":   [{"variant": [], "mass_baseline": 0.50, "mass_current": 0.13, "delta": -0.37}],
    "top_gained_variants": [{"variant": [], "mass_baseline": 0.00, "mass_current": 0.33, "delta": 0.33}]
  },
  "ground_truth": {
    "pattern": "insertion", "target_activity": "Take in charge ticket",
    "secondary_activity": "AutoReview", "fraction": 0.7, "seed": 42,
    "n_affected_cases": 1457, "affected_case_ids": [],
    "description": "Insert 'AutoReview' after every 'Take in charge ticket' in 1457 cases (1561 new events)."
  },
  "legacy": {
    "drift_score": 0.698, "trace_drift_score": 0.698,
    "duration_drift_score": 0.017, "drift_metric": "tv",
    "detection_threshold": 0.05, "status": "DRIFT DETECTED",
    "top_baseline_process_freq": {}, "top_current_process_freq": {},
    "baseline_count": 2450, "current_count": 2130
  }
}
```

The legacy LLM analyst path remains operable on the `legacy` block via the `--legacy-prompt` flag, which preserves compatibility with prior demonstrations.

---

## 7. Paper Figures

The figures are regenerated via `make_figures.py --figure N`, where `N` is an integer between 1 and 8 or the literal string `all`. The complete set is summarised below.

| # | File | Content | Approximate cost |
|---|---|---|---:|
| 1 | `fig1_teaser.pdf` | End-to-end case study: multi-scale signal, PELT change point with CI, top transport flows | $\sim$10 s |
| 2 | `fig2_method_overview.pdf` | Architecture overview (no data dependency) | $<$ 2 s |
| 3 | `fig3_multi_scale.pdf` | Three-scale plus $W_1$ bar chart across the four injection patterns, with NULL baseline | $\sim$10 s |
| 4 | `fig4_detection_roc.pdf` | Receiver-operating-characteristic comparison of proposed $W_1$ against TV and legacy baselines | $\sim$40 s |
| 5 | `fig5_localization.pdf` | True versus detected change-point scatter and delay box-plot | $\sim$2--3 min |
| 6 | `fig6_llm_rubric.pdf` | LLM judge rubric across three evidence framings (requires `run_llm_evaluation.py` output) | Depends on cache |
| 7 | `fig7_ablation.pdf` | Component ablation of the multi-scale plus OT score | $\sim$5 s |
| 8 | `fig8_h0_uniformity.pdf` | Empirical $H_0$ p-value distribution for the permutation test | $\sim$5 min |

The figure-style policy lives in `drift/viz.py:set_paper_style()`. It applies an Ocean Dusk colour palette, a serif type family, a 300 dpi vector backend, and a fixed colour convention: the proposed method renders in warm coral, baselines in cool grey.

---

## 8. Tests

```bash
pytest tests/                       # 103 tests; runtime approximately 23 s
pytest tests/test_significance.py   # H_0 uniformity and H_1 power
pytest tests/test_ot_attribution.py # Two-variant hand-computed transport
```

Test coverage spans injection correctness with seed determinism, JSD identity, bounds, and symmetry properties, $H_0$ uniformity of the permutation test, synthetic change-point signals (step, flat, double-step), the OT identity and a hand-computed transport, and the full LLM evaluation pipeline under a mocked client.

---

## 9. Archive Directory

The `archive/` directory preserves historical experimental code including model verification scripts, earlier visualisation prototypes, and KS-test demonstrations. The primary reproduction path is the trio of `run_full_pipeline.py`, `make_figures.py`, and `run_llm_evaluation.py`; the archive should not be consulted for first-time reproduction.

---

## 10. Single-Command Reproduction

```bash
pytest tests/ && \
  python run_full_pipeline.py && \
  python datasets/llm_analyst_official.py && \
  python make_figures.py --figure all
```

The command above produces the following artefacts.

- `datasets/final_report_for_azure.json` contains the schema v2 report.
- `examples/Final_Drift_Analysis_Report.md` contains the LLM analyst output.
- `figures/fig1..8_*.pdf` contains the eight paper figures.

To regenerate figure 6, which depends on the LLM evaluation grid, run the following two commands in addition.

```bash
python run_llm_evaluation.py --seeds 3 --patterns insertion deletion substitution loop
python make_figures.py --figure 6
```
