"""Run the M6 LLM evaluation grid end-to-end across multiple framings.

For each (framing, pattern, seed) cell:
  1. Run the pipeline with the corresponding injection enabled (shared per
     (pattern, seed); reused across framings).
  2. Build the analyst input under the chosen framing:
       - ``proposed`` : schema v2 (multi-scale + p-values + OT flows + cases)
       - ``legacy``   : v1-style block only (drift_score + top-K trace freq)
       - ``raw``      : first ~200 raw events on each side, no aggregation
  3. Call the analyst LLM to produce a Chinese-markdown drift report.
  4. Grade the report against the injected ground truth (extract claim,
     compute precision/recall, ask an oracle judge to score).

Aggregate per (framing, pattern), write the full grid + summary to
``outputs/llm_evaluation_grid.json``.

Provider auto-detection via ``drift.llm.get_client``: prefers Anthropic
(ANTHROPIC_API_KEY) over OpenAI. Default models: claude-sonnet-4-5 for
analyst + judge, claude-3-haiku for extract (when on Anthropic).

Cost estimate: 4 patterns × N seeds × 3 framings cells, each cell ≈ 3 LLM
calls. Default (N=3) → 36 cells → 108 calls. ~10-20 min wall-clock.

Usage:
    python run_llm_evaluation.py                                # all defaults
    python run_llm_evaluation.py --seeds 2 --framings proposed  # smoke
    python run_llm_evaluation.py --patterns insertion deletion  # subset
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

from drift.evaluation import aggregate_metric, grade_one_scenario
from drift.injection import inject as inject_drift
from drift.io import load_event_log
from drift.llm import default_model, get_client


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(HERE, "outputs", "llm_evaluation_grid.json")
PIPELINE_JSON = os.path.join(HERE, "datasets", "final_report_for_azure.json")


DEFAULT_TARGETS = {
    "insertion":    ("Take in charge ticket", "AutoReview"),
    "deletion":     ("Wait", None),
    "substitution": ("Wait", "QueuedWait"),
    "loop":         ("Resolve ticket", None),
}


# ---------------------------------------------------------------------------
# Analyst system prompts per framing
# ---------------------------------------------------------------------------


ANALYST_SYSTEM_PROMPT_PROPOSED = (
    "你是一位资深的 BPM 业务流程挖掘专家。用户提供漂移检测报告 (schema v2), 包含:\n"
    "  * drift_vector: activity_jsd / dfg_jsd / trace_jsd / trace_w1 + p-values\n"
    "  * attribution.top_transport_flows: 哪些 baseline 变体迁移到了哪些 current 变体, 含 case 样本\n"
    "  * attribution.top_lost_variants / top_gained_variants: 路径占比的最大变化\n"
    "  * split.change_points + ci_95\n\n"
    "撰写中文 Markdown 报告, 硬性要求:\n"
    "1) 固定结构: 总览 / 关键变化 (Baseline vs Current) / 根因推断 / 改进建议\n"
    "2) 必须引用至少 3 个 drift_vector 分量、3 个 transport flow 的 mass、1 个 p-value、1 个 change_point + 其 CI\n"
    "3) 根因推断基于 top_transport_flows, 不要仅靠 top_lost/top_gained\n"
    "4) 改进建议要具体可执行\n"
)


ANALYST_SYSTEM_PROMPT_LEGACY = (
    "你是一位资深的 BPM 业务流程挖掘专家。用户提供漂移检测数据 (v1 schema, 仅含 drift_score + "
    "top_baseline_process_freq + top_current_process_freq + 计数)。\n\n"
    "撰写中文 Markdown 报告, 硬性要求:\n"
    "1) 固定结构: 总览 / 关键变化 / 根因推断 / 改进建议\n"
    "2) 必须引用 drift_score 和 top_k 频率 / 计数\n"
    "3) 结论尽量量化\n"
)


ANALYST_SYSTEM_PROMPT_RAW = (
    "你是一位资深的 BPM 业务流程挖掘专家。用户提供两个时段的原始事件日志样本 (Case ID, Activity, Timestamp)。\n"
    "你需要自己判断: 这两个时段之间是否存在流程漂移; 若有, 是何种类型 "
    "(插入新步骤 / 删除现有步骤 / 替换步骤 / 出现重复循环 / 其他), 涉及哪些活动。\n\n"
    "撰写中文 Markdown 报告, 固定结构: 总览 / 关键变化 / 根因推断 / 改进建议。"
    "结论必须基于你在事件原文中观察到的证据, 不要假装看到了不存在的统计量。\n"
)


# ---------------------------------------------------------------------------
# Pipeline + analyst helpers
# ---------------------------------------------------------------------------


def _run_pipeline(dataset_path: str, pattern: str, target: str, secondary: str | None,
                  fraction: float, seed: int, python_bin: str,
                  cols: dict[str, str]) -> dict:
    """Run run_full_pipeline.py as subprocess, return schema v2 JSON dict."""
    env = os.environ.copy()
    env.update({
        "EVENT_LOG_PATH": dataset_path,
        "INJECT_DRIFT": "true",
        "DRIFT_PATTERN": pattern,
        "DRIFT_TARGET": target,
        "DRIFT_FRACTION": str(fraction),
        "DRIFT_SEED": str(seed),
        **cols,
    })
    if secondary:
        env["DRIFT_SECONDARY"] = secondary

    proc = subprocess.run(
        [python_bin, os.path.join(HERE, "run_full_pipeline.py"), "--no-perm"],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pipeline failed (rc={proc.returncode}):\n"
                           f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    with open(PIPELINE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _prune_v2_for_prompt(report: dict) -> dict:
    """Strip the legacy and ground_truth blocks so the analyst doesn't see GT."""
    out = {k: v for k, v in report.items() if k not in {"legacy", "ground_truth"}}
    return out


def _build_raw_events_sample(dataset_path: str, cols: dict[str, str],
                              pattern: str, target: str, secondary: str | None,
                              fraction: float, seed: int,
                              events_per_side: int = 200) -> str:
    """Re-inject the same drift in-process so we can hand raw events to the LLM."""
    case_id_col = cols["COL_CASE_ID"]
    activity_col = cols["COL_ACTIVITY"]
    timestamp_col = cols["COL_TIMESTAMP"]

    df = load_event_log(dataset_path, case_id_col, activity_col, timestamp_col, keep_only_complete=True)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format="ISO8601", errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values([case_id_col, timestamp_col]).reset_index(drop=True)

    case_end = df.groupby(case_id_col)[timestamp_col].max().sort_values()
    mid = len(case_end) // 2
    late_ids = set(case_end.iloc[mid:].index)
    df_late = df[df[case_id_col].isin(late_ids)]
    df_early = df[~df[case_id_col].isin(late_ids)]
    inj_kw = dict(case_id_col=case_id_col, activity_col=activity_col,
                  timestamp_col=timestamp_col,
                  fraction=fraction, seed=seed)
    if pattern == "insertion":
        df_late_inj, _ = inject_drift(df_late, "insertion",
                                      after_activity=target, new_activity=secondary,
                                      **inj_kw)
    elif pattern == "deletion":
        df_late_inj, _ = inject_drift(df_late, "deletion",
                                      target_activity=target, **inj_kw)
    elif pattern == "substitution":
        df_late_inj, _ = inject_drift(df_late, "substitution",
                                      src_activity=target, dst_activity=secondary,
                                      **inj_kw)
    elif pattern == "loop":
        df_late_inj, _ = inject_drift(df_late, "loop",
                                      target_activity=target, **inj_kw)
    else:
        raise ValueError(f"unknown pattern {pattern!r}")

    cols_kept = [case_id_col, activity_col, timestamp_col]
    sample_b = df_early.sort_values(timestamp_col).head(events_per_side)[cols_kept]
    sample_c = df_late_inj.sort_values(timestamp_col).head(events_per_side)[cols_kept]

    # Format timestamps compactly to keep the prompt small
    for s in (sample_b, sample_c):
        s[timestamp_col] = s[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S")

    return (
        f"## 基线时段事件样本 (前 {len(sample_b)} 条, 共 {len(df_early)} 条)\n\n"
        + sample_b.to_markdown(index=False)
        + f"\n\n## 当前时段事件样本 (前 {len(sample_c)} 条, 共 {len(df_late_inj)} 条)\n\n"
        + sample_c.to_markdown(index=False)
    )


def _build_analyst_input(framing: str, report: dict, dataset_path: str,
                          cols: dict[str, str], pattern: str, target: str,
                          secondary: str | None, fraction: float, seed: int,
                          ) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the chosen framing."""
    if framing == "proposed":
        data = _prune_v2_for_prompt(report)
        user = ("以下是系统检测到的漂移数据 (schema v2 JSON):\n```json\n"
                + json.dumps(data, indent=2, ensure_ascii=False, default=str)
                + "\n```")
        return ANALYST_SYSTEM_PROMPT_PROPOSED, user
    if framing == "legacy":
        data = report.get("legacy") or {}
        user = ("以下是系统检测到的漂移数据 (legacy v1 JSON):\n```json\n"
                + json.dumps(data, indent=2, ensure_ascii=False, default=str)
                + "\n```")
        return ANALYST_SYSTEM_PROMPT_LEGACY, user
    if framing == "raw":
        user = _build_raw_events_sample(
            dataset_path, cols, pattern, target, secondary, fraction, seed,
        )
        return ANALYST_SYSTEM_PROMPT_RAW, user
    raise ValueError(f"unknown framing {framing!r}")


def _call_analyst(client, model: str, system: str, user: str,
                   temperature: float = 0.5, max_tokens: int = 2500) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


_METRIC_KEYS = [
    ("pattern_match_rate", lambda r: 1 if r["metrics"]["pattern_match"] else 0),
    ("activity_precision", lambda r: r["metrics"]["activity_precision"]),
    ("activity_recall",    lambda r: r["metrics"]["activity_recall"]),
    ("activity_f1",        lambda r: r["metrics"]["activity_f1"]),
    ("case_id_jaccard",    lambda r: r["metrics"]["case_id_jaccard"]),
    ("judge_accuracy",     lambda r: r["judge"]["accuracy"]),
    ("judge_completeness", lambda r: r["judge"]["completeness"]),
    ("judge_actionability",lambda r: r["judge"]["actionability"]),
]


def _aggregate(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if "error" in row:
            continue
        grouped.setdefault(row["scenario"]["label"], []).append(row)
    agg = {}
    for label, group in grouped.items():
        agg[label] = {"n": len(group)}
        for name, getter in _METRIC_KEYS:
            agg[label][name] = aggregate_metric([getter(r) for r in group])
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--dataset", default=os.path.join(HERE, "datasets", "finale.csv"),
                        help="Event-log path forwarded as EVENT_LOG_PATH.")
    parser.add_argument("--col-case-id",  default="Case ID")
    parser.add_argument("--col-activity", default="Activity")
    parser.add_argument("--col-timestamp",default="Complete Timestamp")
    parser.add_argument("--patterns", nargs="+",
                        default=["insertion", "deletion", "substitution", "loop"])
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds per pattern.")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--framings", nargs="+",
                        default=["proposed", "legacy", "raw"],
                        choices=["proposed", "legacy", "raw"],
                        help="Which analyst framings to evaluate.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter for pipeline subprocess.")
    parser.add_argument("--analyst-model", default=None,
                        help="Override analyst model (default: drift.llm.default_model('high'))")
    parser.add_argument("--judge-model", default=None,
                        help="Override judge model (default: drift.llm.default_model('high'))")
    parser.add_argument("--extract-model", default=None,
                        help="Override extract model (default: drift.llm.default_model('low'))")
    args = parser.parse_args()

    cols = {
        "COL_CASE_ID":   args.col_case_id,
        "COL_ACTIVITY":  args.col_activity,
        "COL_TIMESTAMP": args.col_timestamp,
    }

    client = get_client()
    analyst_model = args.analyst_model or default_model("high")
    judge_model   = args.judge_model   or default_model("high")
    extract_model = args.extract_model or default_model("low")

    rows = []
    t_start = time.time()
    n_cells = len(args.patterns) * args.seeds * len(args.framings)
    print(f"Running {n_cells} cells "
          f"({len(args.patterns)} patterns × {args.seeds} seeds × {len(args.framings)} framings)")
    print(f"Models: analyst={analyst_model}  judge={judge_model}  extract={extract_model}")
    print(f"Dataset: {args.dataset}")
    print()

    cell_idx = 0
    for pattern in args.patterns:
        target, secondary = DEFAULT_TARGETS.get(pattern, (None, None))
        if target is None:
            print(f"  skip pattern={pattern}: no DEFAULT_TARGETS entry")
            continue
        for seed in range(args.seeds):
            # Run pipeline once per (pattern, seed); reused across framings.
            try:
                report = _run_pipeline(args.dataset, pattern, target, secondary,
                                       args.fraction, seed, args.python, cols)
            except Exception as e:
                print(f"  PIPELINE FAILED for {pattern}.seed{seed}: {e}")
                for framing in args.framings:
                    rows.append({
                        "scenario": {"pattern": pattern, "seed": seed,
                                      "framing": framing, "target": target,
                                      "secondary": secondary,
                                      "label": f"{framing}-{pattern}"},
                        "error": f"pipeline failed: {e}",
                    })
                cell_idx += len(args.framings)
                continue

            gt = report.get("ground_truth") or {}
            for framing in args.framings:
                cell_idx += 1
                cell_label = f"{framing}/{pattern}.seed{seed}"
                print(f"  [{cell_idx}/{n_cells}] {cell_label}", flush=True)
                t0 = time.time()
                try:
                    sys_p, usr_p = _build_analyst_input(
                        framing, report, args.dataset, cols,
                        pattern, target, secondary, args.fraction, seed,
                    )
                    analyst_md = _call_analyst(client, analyst_model, sys_p, usr_p)
                    grade = grade_one_scenario(
                        analyst_md, gt, client,
                        judge_model=judge_model, extract_model=extract_model,
                    )
                    rows.append({
                        "scenario": {"pattern": pattern, "seed": seed,
                                      "framing": framing, "target": target,
                                      "secondary": secondary,
                                      "label": f"{framing}-{pattern}"},
                        "duration_s": round(time.time() - t0, 1),
                        "ground_truth_summary": {
                            "pattern": gt.get("pattern"),
                            "target_activity": gt.get("target_activity"),
                            "secondary_activity": gt.get("secondary_activity"),
                            "n_affected_cases": gt.get("n_affected_cases"),
                        },
                        "analyst_md_len": len(analyst_md),
                        "claimed": grade["claimed"],
                        "metrics": grade["metrics"],
                        "judge": {k: grade["judge"][k] for k in
                                  ("accuracy", "completeness", "actionability")},
                        "judge_reasoning": grade["judge"].get("reasoning", ""),
                    })
                except Exception as e:
                    print(f"    FAILED: {e}")
                    rows.append({
                        "scenario": {"pattern": pattern, "seed": seed,
                                      "framing": framing, "target": target,
                                      "secondary": secondary,
                                      "label": f"{framing}-{pattern}"},
                        "error": str(e),
                    })

    aggregates = _aggregate(rows)

    out = {
        "args": {k: v for k, v in vars(args).items()},
        "models": {"analyst": analyst_model, "judge": judge_model, "extract": extract_model},
        "n_cells_total": n_cells,
        "n_cells_succeeded": sum(1 for r in rows if "error" not in r),
        "n_cells_failed":    sum(1 for r in rows if "error"     in r),
        "duration_total_s":  round(time.time() - t_start, 1),
        "rows": rows,
        "aggregates": aggregates,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nwrote {args.output}")
    print(f"  succeeded {out['n_cells_succeeded']} / {n_cells} cells in {out['duration_total_s']}s")
    print()
    print(f"  {'cell':28s}  n   F1            judge_acc")
    for label, agg in sorted(aggregates.items()):
        f1 = agg["activity_f1"]
        ja = agg["judge_accuracy"]
        print(f"  {label:28s}  {agg['n']:>2d}  "
              f"{f1['mean']:.2f}±{f1['std']:.2f}    "
              f"{ja['mean']:.2f}±{ja['std']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
