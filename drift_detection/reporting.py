from __future__ import annotations

from collections import Counter
from typing import Any


def render_markdown_report(result: dict[str, Any]) -> str:
    global_summary = result.get("global_summary", {})
    config = result.get("config", {})
    drift_points = result.get("drift_points", [])
    evaluation = result.get("evaluation") or {}
    llm_meta = result.get("llm", {})

    lines: list[str] = []
    lines.append("# Business Drift Analysis Report")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append(f"- 检测状态：**{global_summary.get('status', 'UNKNOWN')}**")
    lines.append(f"- 漂移区段数量：`{global_summary.get('drift_point_count', 0)}`")
    lines.append(f"- 时间线窗口数量：`{global_summary.get('window_count', 0)}`")
    lines.append(f"- 最高平滑分数：`{global_summary.get('peak_score', 0.0)}`")
    lines.append(f"- 判定阈值：`{global_summary.get('threshold', 0.0)}`")
    lines.append(f"- LLM 诊断：`{'enabled' if llm_meta.get('enabled') else 'disabled'}`，实际调用：`{llm_meta.get('used_llm', False)}`")
    if llm_meta.get("fallback_reason"):
        lines.append(f"- LLM fallback：{llm_meta['fallback_reason']}")
    lines.append("")
    lines.append("## 检测方法与阈值")
    lines.append("")
    lines.append(f"- 分析模式：`{config.get('analysis_mode', 'timeline')}`")
    lines.append(f"- 漂移度量：`trace={config.get('drift_metric', 'tv')}`，`mode={config.get('detection_mode', 'mixed')}`")
    lines.append(f"- 窗口参数：`window_size={config.get('window_size')}`，`step_size={config.get('step_size')}`")
    threshold_details = global_summary.get("threshold_details", {})
    if threshold_details:
        lines.append(f"- 阈值来源：`{threshold_details.get('source', 'configured')}`")
        if threshold_details.get("source") == "auto":
            lines.append(
                "- 自动阈值细节：median=`{}`，MAD=`{}`，auto_candidate=`{}`，最终阈值=`{}`".format(
                    threshold_details.get("median_score"),
                    threshold_details.get("mad_score"),
                    threshold_details.get("auto_candidate"),
                    global_summary.get("threshold"),
                )
            )
    lines.append("")
    lines.append("## 漂移时间线概览")
    lines.append("")
    if not drift_points:
        lines.append("- 当前时间线没有检测到超过阈值的漂移区段。")
    else:
        for point in drift_points:
            tag_names = ", ".join(tag["tag"] for tag in point.get("rule_based_tags", [])) or "无"
            lines.append(
                "- `{}`: {} 至 {}，峰值分数 `{}`，标签 `{}`".format(
                    point["id"],
                    point.get("interval_start_time"),
                    point.get("interval_end_time"),
                    point.get("peak_score"),
                    tag_names,
                )
            )
    lines.append("")
    lines.append("## 逐漂移点分析")
    lines.append("")
    if not drift_points:
        lines.append("- 无漂移点可分析。")
    else:
        for point in drift_points:
            lines.extend(_render_drift_point(point))
    lines.append("")
    lines.append("## 跨区段共性")
    lines.append("")
    lines.extend(_render_cross_interval_patterns(drift_points))
    lines.append("")
    lines.append("## 改进建议")
    lines.append("")
    lines.extend(_render_aggregated_recommendations(drift_points))
    if evaluation:
        lines.append("")
        lines.append("## Evaluation Snapshot")
        lines.append("")
        for key, value in evaluation.items():
            lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def render_human_review_rubric(result: dict[str, Any]) -> str:
    drift_points = result.get("drift_points", [])
    lines = [
        "# Human Review Rubric",
        "",
        "请对每个漂移点按 1-5 分评分，并补充证据备注。",
        "",
        "| Drift Point | 变化是否真实 | 原因是否有证据支撑 | 建议是否可执行 | 备注 |",
        "| --- | --- | --- | --- | --- |",
    ]
    if drift_points:
        for point in drift_points:
            lines.append(f"| {point['id']} |  |  |  |  |")
    else:
        lines.append("| N/A |  |  |  |  |")
    return "\n".join(lines) + "\n"


def _render_drift_point(point: dict[str, Any]) -> list[str]:
    evidence = point.get("evidence", {})
    diagnosis = point.get("llm_diagnosis") or {}
    lines = [f"### {point['id']}", ""]
    lines.append(
        "- 时间区段：`{}` 到 `{}`，峰值时间 `{}`".format(
            point.get("interval_start_time"),
            point.get("interval_end_time"),
            point.get("peak_time"),
        )
    )
    lines.append(
        "- 峰值分数：`peak={}`，`trace={}`，`duration={}`".format(
            point.get("peak_score"),
            point.get("trace_score"),
            point.get("duration_score"),
        )
    )
    delay = point.get("detection_delay_proxy", {})
    lines.append(
        "- Detection delay proxy：`cases_to_peak={}`，`hours_to_peak={}`".format(
            delay.get("cases_to_peak"),
            delay.get("hours_to_peak"),
        )
    )
    tag_summary = ", ".join(
        f"{tag['tag']}({tag['confidence']}, evidence={','.join(tag['evidence_ids'])})"
        for tag in point.get("rule_based_tags", [])
    ) or "无"
    lines.append(f"- 规则标签：{tag_summary}")
    lines.append("- 关键证据：")
    for evidence_line in _top_evidence_lines(point):
        lines.append(f"  - {evidence_line}")
    lines.append(f"- 诊断摘要：{diagnosis.get('summary', 'N/A')}")
    lines.append("- 候选根因：")
    for cause in diagnosis.get("candidate_causes", []):
        lines.append(
            "  - {} | confidence={} | evidence={}: {}".format(
                cause.get("title"),
                cause.get("confidence"),
                ",".join(cause.get("evidence_ids", [])) or "无",
                cause.get("details"),
            )
        )
    lines.append("- 建议：")
    for recommendation in diagnosis.get("recommendations", []):
        lines.append(
            "  - [{}] {} | evidence={}: {}".format(
                recommendation.get("priority", "medium"),
                recommendation.get("title"),
                ",".join(recommendation.get("evidence_ids", [])) or "无",
                recommendation.get("details"),
            )
        )
    missing_data = diagnosis.get("missing_data", [])
    if missing_data:
        lines.append("- 缺失数据：")
        for item in missing_data:
            lines.append(f"  - {item}")
    lines.append("")
    return lines


def _top_evidence_lines(point: dict[str, Any]) -> list[str]:
    evidence = point.get("evidence", {})
    evidence_index = evidence.get("evidence_index", {})
    preferred_ids = []
    for section_name in [
        "top_increased_traces",
        "top_decreased_traces",
        "top_changed_transitions",
        "activity_frequency_deltas",
    ]:
        for item in evidence.get(section_name, [])[:2]:
            preferred_ids.append(item.get("evidence_id"))
    for summary_name in ["rework_or_loop_rate_delta", "duration_stats_delta"]:
        item = evidence.get(summary_name)
        if item:
            preferred_ids.append(item.get("evidence_id"))
    unique_ids = []
    for evidence_id in preferred_ids:
        if evidence_id and evidence_id not in unique_ids:
            unique_ids.append(evidence_id)
    return [f"[{evidence_id}] {evidence_index.get(evidence_id, '')}" for evidence_id in unique_ids[:6]]


def _render_cross_interval_patterns(drift_points: list[dict[str, Any]]) -> list[str]:
    if not drift_points:
        return ["- 无可聚合的漂移区段。"]
    tag_counter = Counter()
    for point in drift_points:
        for tag in point.get("rule_based_tags", []):
            tag_counter[tag["tag"]] += 1
    if not tag_counter:
        return ["- 没有形成稳定的跨区段标签模式。"]
    lines = []
    for tag, count in tag_counter.most_common():
        lines.append(f"- `{tag}` 出现在 `{count}` 个漂移点中。")
    return lines


def _render_aggregated_recommendations(drift_points: list[dict[str, Any]]) -> list[str]:
    aggregated = Counter()
    recommendation_map: dict[str, tuple[str, str]] = {}
    for point in drift_points:
        for recommendation in (point.get("llm_diagnosis") or {}).get("recommendations", []):
            title = recommendation.get("title")
            if not title:
                continue
            aggregated[title] += 1
            recommendation_map[title] = (
                recommendation.get("priority", "medium"),
                recommendation.get("details", ""),
            )
    if not aggregated:
        return ["- 当前没有可聚合的建议。"]
    lines = []
    for title, count in aggregated.most_common():
        priority, details = recommendation_map[title]
        lines.append(f"- [{priority}] {title} | 出现 `{count}` 次 | {details}")
    return lines
