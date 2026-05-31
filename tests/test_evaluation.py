"""Tests for drift.evaluation (Phase 7 / M6).

LLM calls are mocked via a tiny ``FakeClient`` that pretends to be the OpenAI
SDK and returns canned JSON strings.
"""

from __future__ import annotations

import json

import pytest

from drift.evaluation import (
    _coerce_json,
    _jaccard,
    aggregate_metric,
    compute_attribution_metrics,
    extract_claimed_root_cause,
    grade_one_scenario,
    llm_judge_rubric,
)


# ---------------------------------------------------------------------------
# Fake OpenAI-style client
# ---------------------------------------------------------------------------


class FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, responder):
        self._responder = responder

    def create(self, *, model, messages, temperature=0.0, **_):
        return FakeResponse(self._responder(messages, model))


class FakeChat:
    def __init__(self, responder):
        self.completions = FakeCompletions(responder)


class FakeClient:
    """Stand-in for an OpenAI client. ``responder`` is a function from
    (messages: list[dict], model: str) to the response string.
    """

    def __init__(self, responder):
        self.chat = FakeChat(responder)


# ---------------------------------------------------------------------------
# JSON coercion
# ---------------------------------------------------------------------------


def test_coerce_json_plain():
    assert _coerce_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_coerce_json_strips_code_fences():
    raw = "```json\n{\"a\": 1}\n```"
    assert _coerce_json(raw) == {"a": 1}


def test_coerce_json_falls_back_to_regex():
    raw = "Sure, here is the JSON: {\"x\": 5}\nLet me know if you need more."
    assert _coerce_json(raw) == {"x": 5}


def test_coerce_json_raises_on_garbage():
    with pytest.raises(json.JSONDecodeError):
        _coerce_json("no json here at all")


# ---------------------------------------------------------------------------
# extract_claimed_root_cause
# ---------------------------------------------------------------------------


def test_extract_claimed_root_cause_happy_path():
    canned = json.dumps({
        "claimed_pattern": "insertion",
        "claimed_activities": ["Take in charge ticket", "AutoReview"],
        "claimed_n_affected_cases": 1457,
        "claimed_case_ids": ["Case 1007", "Case 1010"],
    })
    client = FakeClient(lambda *_: canned)
    out = extract_claimed_root_cause("# Report ...", client, model="fake")
    assert out["claimed_pattern"] == "insertion"
    assert out["claimed_activities"] == ["Take in charge ticket", "AutoReview"]
    assert out["claimed_n_affected_cases"] == 1457
    assert out["claimed_case_ids"] == ["Case 1007", "Case 1010"]


def test_extract_claimed_root_cause_normalises_unknown_pattern():
    canned = json.dumps({
        "claimed_pattern": "MysteryDrift",
        "claimed_activities": ["A"],
    })
    client = FakeClient(lambda *_: canned)
    out = extract_claimed_root_cause("# X", client, model="fake")
    assert out["claimed_pattern"] == "unknown"


def test_extract_claimed_root_cause_handles_missing_fields():
    canned = "{}"
    client = FakeClient(lambda *_: canned)
    out = extract_claimed_root_cause("# X", client, model="fake")
    assert out == {
        "claimed_pattern": "unknown",
        "claimed_activities": [],
        "claimed_n_affected_cases": None,
        "claimed_case_ids": [],
    }


def test_extract_claimed_root_cause_handles_messy_response():
    canned = "Here's the JSON: ```json\n{\"claimed_pattern\":\"deletion\",\"claimed_activities\":[\"Wait\"]}\n```"
    client = FakeClient(lambda *_: canned)
    out = extract_claimed_root_cause("# X", client, model="fake")
    assert out["claimed_pattern"] == "deletion"
    assert out["claimed_activities"] == ["Wait"]


# ---------------------------------------------------------------------------
# compute_attribution_metrics (pure function)
# ---------------------------------------------------------------------------


def test_metrics_perfect_match():
    claimed = {
        "claimed_pattern": "insertion",
        "claimed_activities": ["Take in charge ticket", "AutoReview"],
        "claimed_case_ids": ["Case 1", "Case 2"],
    }
    gt = {
        "pattern": "insertion",
        "target_activity": "Take in charge ticket",
        "secondary_activity": "AutoReview",
        "affected_case_ids": ["Case 1", "Case 2"],
    }
    m = compute_attribution_metrics(claimed, gt)
    assert m["pattern_match"]
    assert m["activity_precision"] == 1.0
    assert m["activity_recall"] == 1.0
    assert m["activity_f1"] == 1.0
    assert m["case_id_jaccard"] == 1.0


def test_metrics_pattern_mismatch():
    claimed = {"claimed_pattern": "deletion", "claimed_activities": ["X"], "claimed_case_ids": []}
    gt = {"pattern": "insertion", "target_activity": "Y", "secondary_activity": None, "affected_case_ids": []}
    m = compute_attribution_metrics(claimed, gt)
    assert m["pattern_match"] is False


def test_metrics_partial_activity_overlap():
    claimed = {
        "claimed_pattern": "insertion",
        "claimed_activities": ["A", "B"],  # one correct, one wrong
        "claimed_case_ids": [],
    }
    gt = {"pattern": "insertion", "target_activity": "A", "secondary_activity": "C", "affected_case_ids": []}
    m = compute_attribution_metrics(claimed, gt)
    assert m["activity_precision"] == pytest.approx(0.5)
    assert m["activity_recall"] == pytest.approx(0.5)
    assert m["activity_f1"] == pytest.approx(0.5)


def test_metrics_empty_claim_gives_zero():
    claimed = {"claimed_pattern": "unknown", "claimed_activities": [], "claimed_case_ids": []}
    gt = {"pattern": "insertion", "target_activity": "A", "secondary_activity": "B", "affected_case_ids": ["c0"]}
    m = compute_attribution_metrics(claimed, gt)
    assert m["activity_precision"] == 0.0
    assert m["activity_recall"] == 0.0
    assert m["activity_f1"] == 0.0


def test_metrics_case_id_jaccard_skipped_when_either_empty():
    claimed = {"claimed_pattern": "insertion", "claimed_activities": ["A"], "claimed_case_ids": []}
    gt = {"pattern": "insertion", "target_activity": "A", "secondary_activity": None,
          "affected_case_ids": ["c0", "c1"]}
    m = compute_attribution_metrics(claimed, gt)
    assert m["case_id_jaccard"] is None


def test_jaccard_basics():
    assert _jaccard(set(), set()) == 1.0
    assert _jaccard({"a"}, set()) == 0.0
    assert _jaccard({"a"}, {"a"}) == 1.0
    assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# llm_judge_rubric
# ---------------------------------------------------------------------------


def test_judge_rubric_clips_to_1_5():
    canned = json.dumps({
        "accuracy": 7,         # out of range high -> clip to 5
        "completeness": 0,     # out of range low  -> clip to 1
        "actionability": 3.7,  # round to 4
        "reasoning": "x",
    })
    client = FakeClient(lambda *_: canned)
    out = llm_judge_rubric("# Report", {"pattern": "insertion"}, client, model="fake")
    assert out["accuracy"] == 5
    assert out["completeness"] == 1
    assert out["actionability"] == 4
    assert out["reasoning"] == "x"


def test_judge_rubric_defaults_to_one_on_garbage():
    canned = json.dumps({"reasoning": "no scores given"})
    client = FakeClient(lambda *_: canned)
    out = llm_judge_rubric("# Report", {"pattern": "insertion"}, client, model="fake")
    assert out["accuracy"] == 1
    assert out["completeness"] == 1
    assert out["actionability"] == 1


# ---------------------------------------------------------------------------
# grade_one_scenario integration
# ---------------------------------------------------------------------------


def test_grade_one_scenario_routes_correctly():
    """The fake client dispatches different responses based on the system prompt."""

    def responder(messages, model):
        sys = messages[0]["content"]
        if "evaluator" in sys.lower():
            return json.dumps({
                "claimed_pattern": "insertion",
                "claimed_activities": ["Take", "AutoReview"],
                "claimed_n_affected_cases": 100,
                "claimed_case_ids": ["Case 1"],
            })
        elif "oracle grader" in sys.lower():
            return json.dumps({"accuracy": 5, "completeness": 4, "actionability": 3, "reasoning": "ok"})
        raise AssertionError(f"unexpected system prompt: {sys[:80]}")

    client = FakeClient(responder)
    gt = {
        "pattern": "insertion",
        "target_activity": "Take",
        "secondary_activity": "AutoReview",
        "affected_case_ids": ["Case 1", "Case 2"],
    }
    out = grade_one_scenario("# Report", gt, client, judge_model="fake")
    assert out["claimed"]["claimed_pattern"] == "insertion"
    assert out["metrics"]["pattern_match"] is True
    assert out["metrics"]["activity_f1"] == 1.0
    assert out["judge"] == {"accuracy": 5, "completeness": 4, "actionability": 3, "reasoning": "ok"}


# ---------------------------------------------------------------------------
# aggregate_metric
# ---------------------------------------------------------------------------


def test_aggregate_metric_ignores_none():
    out = aggregate_metric([0.1, 0.3, None, 0.5])
    assert out["n"] == 3
    assert out["mean"] == pytest.approx(0.3)
    assert out["std"] is not None


def test_aggregate_metric_all_none():
    out = aggregate_metric([None, None])
    assert out == {"n": 0, "mean": None, "std": None}


def test_aggregate_metric_single_value_has_zero_std():
    out = aggregate_metric([0.7])
    assert out["n"] == 1
    assert out["mean"] == 0.7
    assert out["std"] == 0.0
