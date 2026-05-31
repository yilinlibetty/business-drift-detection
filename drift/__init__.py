"""Process-drift detection toolkit.

Modules:
    io              -- event-log loading (CSV/XES) and case-frame construction
    injection       -- M5: Bose-style controlled drift injection with ground truth
    metrics         -- M1: multi-scale Jensen-Shannon decomposition
    significance    -- M4: permutation-test p-values
    localization    -- M3: sliding-window signal + ruptures PELT + bootstrap CI
    ot_attribution  -- M2: variant-level W1 + transport plan + case-level attribution
    evaluation      -- M6: LLM root-cause extraction, precision/recall, judge rubric
    viz             -- M7: shared paper-figure style helpers
"""

__version__ = "2.0.0"
