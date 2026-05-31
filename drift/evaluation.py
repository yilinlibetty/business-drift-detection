"""M6 -- LLM root-cause evaluation protocol.

Three building blocks:

    extract_claimed_root_cause(report_md, client, model)
        Second-LLM call. Parses the analyst's Markdown report into a
        structured claim: {claimed_pattern, claimed_activities,
        claimed_n_affected_cases, claimed_case_ids}. Pure JSON output.

    compute_attribution_metrics(claimed, ground_truth)
        Pure-function comparison of the structured claim against the
        injected ground truth: pattern_match (bool), activity
        precision/recall/F1, case-ID Jaccard. No LLM call.

    llm_judge_rubric(report_md, ground_truth, client, model)
        Third-LLM call. Returns {accuracy, completeness, actionability,
        reasoning} on a 1-5 Likert. The judge sees BOTH the report and
        the ground truth -- this is an "oracle judge" and the scores
        upper-bound what a human grader would give.

Plus a harness:

    run_evaluation_grid(patterns, seeds, run_pipeline_fn, run_analyst_fn,
                        client, judge_model)
        For each (pattern, seed) cell, dispatch the pipeline, run the
        analyst, extract the claim, grade. Aggregate to mean / std.
        Saves the full grid + a summary table.

The harness is wired in run_full_pipeline.py / make_figures.py; the three
building blocks above are independently usable.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


EXTRACT_SYSTEM_PROMPT = (
    "You are an evaluator for BPM drift analyst reports.\n"
    "Given the analyst's Markdown report, extract the analyst's CLAIMED root cause\n"
    "as a strict JSON object with these keys (and no others):\n"
    "  claimed_pattern: one of [insertion, deletion, substitution, loop, unknown]\n"
    "  claimed_activities: list[str]  -- activities the analyst identifies as central to the drift\n"
    "  claimed_n_affected_cases: integer or null\n"
    "  claimed_case_ids: list[str]  -- explicit case IDs the analyst names\n"
    "Be conservative: only list activities/cases the analyst explicitly mentions.\n"
    "Output ONLY a JSON object. No markdown, no commentary.\n"
)


JUDGE_SYSTEM_PROMPT = (
    "You are an oracle grader for BPM drift analyst reports.\n"
    "You see (1) the ground-truth injection metadata and (2) the analyst's report.\n"
    "Score the report on three axes, each integer 1 (worst) to 5 (best):\n"
    "  accuracy       -- does the report's root cause match the ground-truth pattern\n"
    "                    and identify the correct activities?\n"
    "  completeness   -- does the report cover ALL claimed flows / activities affected,\n"
    "                    not just the dominant one?\n"
    "  actionability  -- are the improvement recommendations concrete enough that a BPM\n"
    "                    engineer could execute them tomorrow?\n"
    "Output strict JSON: {accuracy:int, completeness:int, actionability:int, reasoning:str}\n"
    "No markdown, no commentary outside the JSON object.\n"
)


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_json(text: str) -> dict:
    """Best-effort JSON object extraction from an LLM response string."""
    if not isinstance(text, str):
        raise ValueError(f"expected str, got {type(text).__name__}")
    text = text.strip()
    # Strip ```json fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            raise
        return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Building block 1 -- extract claim
# ---------------------------------------------------------------------------


_ALLOWED_PATTERNS = {"insertion", "deletion", "substitution", "loop", "unknown"}


def extract_claimed_root_cause(
    report_md: str,
    client: Any,
    model: str,
    temperature: float = 0.0,
) -> dict:
    """Second-LLM parse: turn analyst Markdown into a structured claim dict.

    ``client`` is anything with the ``.chat.completions.create(...)`` interface
    of the OpenAI SDK (or a stub that returns the same shape).
    """
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyst report (Markdown):\n\n{report_md}"},
        ],
    )
    content = response.choices[0].message.content
    raw = _coerce_json(content)

    pattern = str(raw.get("claimed_pattern", "unknown")).strip().lower()
    if pattern not in _ALLOWED_PATTERNS:
        pattern = "unknown"

    def _str_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        return [str(v).strip()]

    n_cases = raw.get("claimed_n_affected_cases")
    if isinstance(n_cases, str) and n_cases.isdigit():
        n_cases = int(n_cases)
    elif not isinstance(n_cases, (int, type(None))):
        n_cases = None

    return {
        "claimed_pattern": pattern,
        "claimed_activities": _str_list(raw.get("claimed_activities")),
        "claimed_n_affected_cases": n_cases,
        "claimed_case_ids": _str_list(raw.get("claimed_case_ids")),
    }


# ---------------------------------------------------------------------------
# Building block 2 -- precision/recall vs ground truth (pure function)
# ---------------------------------------------------------------------------


def _ground_truth_activities(gt: dict) -> set[str]:
    out = set()
    if gt.get("target_activity"):
        out.add(str(gt["target_activity"]))
    if gt.get("secondary_activity"):
        out.add(str(gt["secondary_activity"]))
    return out


def _jaccard(a: set, b: set) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_attribution_metrics(claimed: dict, ground_truth: dict) -> dict:
    """Score a structured claim against an injection ground truth.

    Returns:
        pattern_match            bool
        activity_precision       float in [0, 1]
        activity_recall          float in [0, 1]
        activity_f1              float in [0, 1]
        case_id_jaccard          float in [0, 1] or None if either side empty
    """
    pattern_match = (
        claimed.get("claimed_pattern") == ground_truth.get("pattern")
    )
    truth_acts = _ground_truth_activities(ground_truth)
    claimed_acts = set(claimed.get("claimed_activities", []))

    if not truth_acts:
        precision = recall = f1 = 0.0
    else:
        if not claimed_acts:
            precision = recall = f1 = 0.0
        else:
            tp = len(truth_acts & claimed_acts)
            precision = tp / len(claimed_acts)
            recall = tp / len(truth_acts)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    truth_cases = set(map(str, ground_truth.get("affected_case_ids", [])))
    claimed_cases = set(map(str, claimed.get("claimed_case_ids", [])))
    case_id_jaccard = (
        _jaccard(truth_cases, claimed_cases) if (truth_cases and claimed_cases) else None
    )

    return {
        "pattern_match": bool(pattern_match),
        "activity_precision": float(precision),
        "activity_recall": float(recall),
        "activity_f1": float(f1),
        "case_id_jaccard": case_id_jaccard,
    }


# ---------------------------------------------------------------------------
# Building block 3 -- LLM judge rubric
# ---------------------------------------------------------------------------


def _clip_int_score(v, lo: int = 1, hi: int = 5) -> int:
    try:
        x = int(round(float(v)))
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, x))


def llm_judge_rubric(
    report_md: str,
    ground_truth: dict,
    client: Any,
    model: str,
    temperature: float = 0.0,
) -> dict:
    """Oracle-judge rubric: rate accuracy / completeness / actionability 1-5."""
    user_payload = (
        "GROUND TRUTH (injection metadata):\n"
        f"{json.dumps(ground_truth, ensure_ascii=False, indent=2, default=str)}\n\n"
        "ANALYST REPORT (Markdown):\n"
        f"{report_md}"
    )
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
    )
    raw = _coerce_json(response.choices[0].message.content)
    return {
        "accuracy": _clip_int_score(raw.get("accuracy")),
        "completeness": _clip_int_score(raw.get("completeness")),
        "actionability": _clip_int_score(raw.get("actionability")),
        "reasoning": str(raw.get("reasoning", "")).strip(),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def aggregate_metric(values: list[float | int | None]) -> dict:
    """Aggregate a list of metric values to mean / std / n (ignoring Nones)."""
    valid = [v for v in values if v is not None]
    if not valid:
        return {"n": 0, "mean": None, "std": None}
    arr = np.asarray(valid, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def grade_one_scenario(
    report_md: str,
    ground_truth: dict,
    client: Any,
    judge_model: str,
    extract_model: str | None = None,
) -> dict:
    """Run extract + metrics + judge once. Returns a flat dict for the grid."""
    extract_model = extract_model or judge_model
    claimed = extract_claimed_root_cause(report_md, client, extract_model)
    metrics = compute_attribution_metrics(claimed, ground_truth)
    judge = llm_judge_rubric(report_md, ground_truth, client, judge_model)
    return {
        "claimed": claimed,
        "metrics": metrics,
        "judge": judge,
    }


def run_evaluation_grid(
    scenarios: list[dict],
    run_pipeline_fn: Callable[[dict], tuple[dict, dict]],
    run_analyst_fn: Callable[[dict], str],
    client: Any,
    judge_model: str,
    extract_model: str | None = None,
) -> dict:
    """Run the full evaluation grid.

    Parameters
    ----------
    scenarios : list[dict]
        Each scenario is a dict that ``run_pipeline_fn`` understands; it must
        induce the pipeline to inject a known drift so the pipeline returns a
        report containing ``ground_truth``.
    run_pipeline_fn : (scenario) -> (report_dict, ground_truth_dict)
    run_analyst_fn  : (report_dict) -> markdown string
    client, judge_model, extract_model : LLM dispatch.

    Returns dict with:
        rows         list of per-cell graded dicts (scenario + claim + metrics + judge)
        aggregates   per-metric mean/std across rows, grouped by scenario "label"
    """
    rows = []
    for scn in scenarios:
        report, gt = run_pipeline_fn(scn)
        analyst_md = run_analyst_fn(report)
        grade = grade_one_scenario(
            analyst_md, gt, client, judge_model, extract_model=extract_model,
        )
        rows.append({"scenario": scn, **grade})

    # group by scenario "label" if present, else single "all" bucket
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        label = row["scenario"].get("label", "all")
        grouped.setdefault(label, []).append(row)

    aggregates = {}
    for label, group in grouped.items():
        aggregates[label] = {
            "n": len(group),
            "pattern_match_rate": aggregate_metric(
                [1 if r["metrics"]["pattern_match"] else 0 for r in group]
            ),
            "activity_precision": aggregate_metric([r["metrics"]["activity_precision"] for r in group]),
            "activity_recall":    aggregate_metric([r["metrics"]["activity_recall"]    for r in group]),
            "activity_f1":        aggregate_metric([r["metrics"]["activity_f1"]        for r in group]),
            "case_id_jaccard":    aggregate_metric([r["metrics"]["case_id_jaccard"]    for r in group]),
            "judge_accuracy":      aggregate_metric([r["judge"]["accuracy"]      for r in group]),
            "judge_completeness":  aggregate_metric([r["judge"]["completeness"]  for r in group]),
            "judge_actionability": aggregate_metric([r["judge"]["actionability"] for r in group]),
        }
    return {"rows": rows, "aggregates": aggregates}
