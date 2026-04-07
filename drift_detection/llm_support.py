from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL")


def load_llm_settings(enabled: bool = True) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        "model": os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
    }


def enrich_with_llm_diagnosis(
    drift_points: list[dict[str, Any]],
    global_summary: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics_meta = {
        "enabled": bool(settings.get("enabled")),
        "model": settings.get("model"),
        "base_url": settings.get("base_url"),
        "used_llm": False,
        "fallback_reason": None,
    }

    if not settings.get("enabled"):
        diagnostics_meta["fallback_reason"] = "LLM disabled by configuration."
        for point in drift_points:
            point["llm_diagnosis"] = fallback_diagnosis(point)
        return drift_points, diagnostics_meta

    if not settings.get("api_key"):
        diagnostics_meta["fallback_reason"] = "OPENAI_API_KEY is not configured."
        for point in drift_points:
            point["llm_diagnosis"] = fallback_diagnosis(point)
        return drift_points, diagnostics_meta

    client_kwargs = {"api_key": settings["api_key"]}
    if settings.get("base_url"):
        client_kwargs["base_url"] = settings["base_url"]
    client = OpenAI(**client_kwargs)

    for point in drift_points:
        try:
            point["llm_diagnosis"] = diagnose_drift_point(point, global_summary, client, settings["model"])
            diagnostics_meta["used_llm"] = True
        except Exception as exc:
            diagnostics_meta["fallback_reason"] = str(exc)
            point["llm_diagnosis"] = fallback_diagnosis(point, error_message=str(exc))
    return drift_points, diagnostics_meta


def diagnose_drift_point(
    drift_point: dict[str, Any],
    global_summary: dict[str, Any],
    client: OpenAI,
    model: str,
) -> dict[str, Any]:
    system_prompt = (
        "你是一位严谨的业务流程漂移分析专家。"
        "你只能基于提供的 evidence JSON 做推断，不能把推断写成已证实事实。"
        "输出必须是严格 JSON，且根因必须引用 evidence_ids。"
    )
    user_payload = {
        "global_summary": {
            "status": global_summary.get("status"),
            "threshold": global_summary.get("threshold"),
            "detection_mode": global_summary.get("detection_mode"),
            "drift_metric": global_summary.get("drift_metric"),
        },
        "drift_point": {
            "id": drift_point.get("id"),
            "interval_start_time": drift_point.get("interval_start_time"),
            "interval_end_time": drift_point.get("interval_end_time"),
            "peak_time": drift_point.get("peak_time"),
            "peak_score": drift_point.get("peak_score"),
            "trace_score": drift_point.get("trace_score"),
            "duration_score": drift_point.get("duration_score"),
            "rule_based_tags": drift_point.get("rule_based_tags", []),
            "evidence": drift_point.get("evidence", {}),
        },
        "required_schema": {
            "summary": "string",
            "candidate_causes": [
                {
                    "title": "string",
                    "details": "string",
                    "confidence": "number 0-1",
                    "evidence_ids": ["string"]
                }
            ],
            "recommendations": [
                {
                    "title": "string",
                    "details": "string",
                    "priority": "high|medium|low",
                    "evidence_ids": ["string"]
                }
            ],
            "confidence": "number 0-1",
            "missing_data": ["string"]
        },
        "requirements": [
            "全部内容用中文。",
            "candidate_causes 必须表述为候选根因或可能原因。",
            "candidate_causes 和 recommendations 都必须引用 evidence_ids。",
            "不要输出 Markdown，不要输出解释性前缀，只输出 JSON 对象。",
        ],
    }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or "{}"
    parsed = _parse_json_response(content)
    return normalize_diagnosis_payload(parsed, drift_point, source="llm", model=model)


def _parse_json_response(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        return json.loads(content[start:end + 1])
    raise ValueError("LLM response did not contain valid JSON.")


def normalize_diagnosis_payload(
    payload: dict[str, Any],
    drift_point: dict[str, Any],
    source: str,
    model: str | None = None,
) -> dict[str, Any]:
    fallback = fallback_diagnosis(drift_point)
    valid_ids = set(drift_point.get("evidence", {}).get("evidence_ids", []))

    candidate_causes = []
    for item in payload.get("candidate_causes", []) or []:
        evidence_ids = [evidence_id for evidence_id in item.get("evidence_ids", []) if evidence_id in valid_ids]
        candidate_causes.append(
            {
                "title": str(item.get("title") or "未命名候选根因"),
                "details": str(item.get("details") or ""),
                "confidence": _bounded_confidence(item.get("confidence", 0.5)),
                "evidence_ids": evidence_ids,
            }
        )

    recommendations = []
    for item in payload.get("recommendations", []) or []:
        evidence_ids = [evidence_id for evidence_id in item.get("evidence_ids", []) if evidence_id in valid_ids]
        recommendations.append(
            {
                "title": str(item.get("title") or "未命名建议"),
                "details": str(item.get("details") or ""),
                "priority": str(item.get("priority") or "medium").lower(),
                "evidence_ids": evidence_ids,
            }
        )

    normalized = {
        "summary": str(payload.get("summary") or fallback["summary"]),
        "candidate_causes": candidate_causes or fallback["candidate_causes"],
        "recommendations": recommendations or fallback["recommendations"],
        "confidence": _bounded_confidence(payload.get("confidence", fallback["confidence"])),
        "missing_data": [str(item) for item in payload.get("missing_data", [])],
        "source": source,
        "model": model,
    }
    return normalized


def fallback_diagnosis(drift_point: dict[str, Any], error_message: str | None = None) -> dict[str, Any]:
    tags = drift_point.get("rule_based_tags", [])
    candidate_causes = []
    recommendations = []
    for tag in tags:
        candidate_causes.append(
            {
                "title": _candidate_title(tag["tag"]),
                "details": tag["rationale"],
                "confidence": _bounded_confidence(tag.get("confidence", 0.5)),
                "evidence_ids": tag.get("evidence_ids", []),
            }
        )
        recommendations.append(
            {
                "title": _recommendation_title(tag["tag"]),
                "details": _recommendation_details(tag["tag"]),
                "priority": "high" if tag.get("confidence", 0.0) >= 0.75 else "medium",
                "evidence_ids": tag.get("evidence_ids", []),
            }
        )

    if not candidate_causes:
        evidence_ids = drift_point.get("evidence", {}).get("evidence_ids", [])[:2]
        candidate_causes.append(
            {
                "title": "行为分布发生变化",
                "details": "当前窗口与参考窗口之间存在统计差异，但不足以归入更强的候选根因分类。",
                "confidence": 0.35,
                "evidence_ids": evidence_ids,
            }
        )
        recommendations.append(
            {
                "title": "补充业务上下文字段",
                "details": "建议增加 resource/team/priority/channel 等字段，以便缩小候选根因范围。",
                "priority": "medium",
                "evidence_ids": evidence_ids,
            }
        )

    missing_data = []
    if not drift_point.get("evidence", {}).get("attribute_distribution_deltas"):
        missing_data.append("缺少 resource/team/priority/channel/region 等业务属性，无法进一步验证 case mix 或责任转移假设。")
    if error_message:
        missing_data.append(f"LLM fallback reason: {error_message}")

    return {
        "summary": "基于规则证据生成候选根因，当前结果应视为证据支持的诊断假设。",
        "candidate_causes": candidate_causes,
        "recommendations": recommendations,
        "confidence": round(max((item["confidence"] for item in candidate_causes), default=0.35), 2),
        "missing_data": missing_data,
        "source": "fallback",
        "model": None,
    }


def _bounded_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.5
    return round(min(1.0, max(0.0, numeric)), 2)


def _candidate_title(tag: str) -> str:
    mapping = {
        "path_added": "新增流程路径占比上升",
        "path_removed_or_skipped_step": "既有流程路径被删除或跳步",
        "delay_increase": "处理时长整体上升",
        "loop_increase": "返工或循环处理增加",
        "handoff_or_escalation_increase": "交接或升级行为增加",
        "case_mix_shift": "案件结构或业务类型组合变化",
    }
    return mapping.get(tag, tag)


def _recommendation_title(tag: str) -> str:
    mapping = {
        "path_added": "核查新路径的业务规则与日志口径",
        "path_removed_or_skipped_step": "审计被跳过的关键步骤",
        "delay_increase": "定位高时延活动与等待环节",
        "loop_increase": "排查返工触发条件",
        "handoff_or_escalation_increase": "检查升级与转派规则",
        "case_mix_shift": "补充分层维度做 case mix 验证",
    }
    return mapping.get(tag, "补充验证证据")


def _recommendation_details(tag: str) -> str:
    mapping = {
        "path_added": "对新增路径回看近期流程规则、系统发布和日志字段变化，确认是业务变化还是记录口径变化。",
        "path_removed_or_skipped_step": "抽样检查被削弱的路径，确认关键步骤是否被自动化、跳过或漏记。",
        "delay_increase": "按活动、等待段和高分位时长做切分，优先核查 p90 明显上升的环节。",
        "loop_increase": "检查返工触发原因，如资料缺失、审批驳回或多次指派。",
        "handoff_or_escalation_increase": "结合团队与资源字段，验证是否出现更频繁的升级、转派或跨团队处理。",
        "case_mix_shift": "补充优先级、渠道、地区或产品线维度，验证是否由案件组合变化引起。",
    }
    return mapping.get(tag, "结合现有证据做进一步人工核验。")
