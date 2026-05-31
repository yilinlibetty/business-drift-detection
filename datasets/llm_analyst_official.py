"""LLM analyst: read schema v2 (or legacy) JSON, generate Chinese Markdown report.

Provider auto-detection via ``drift.llm.get_client``: prefers Anthropic when
``ANTHROPIC_API_KEY`` is set, otherwise falls back to OpenAI (honors
``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` env vars; the hard-coded defaults
below remain only for legacy back-compat -- they may be expired).

Usage:
    python datasets/llm_analyst_official.py
    python datasets/llm_analyst_official.py --legacy-prompt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make project root importable so `drift.llm` resolves whether this script is
# invoked from project root or from datasets/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from drift.llm import default_model, get_client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# WARNING: leaked key kept ONLY as a legacy fallback for OpenAI gateway demos.
# Always prefer ANTHROPIC_API_KEY (auto-detected) or OPENAI_API_KEY env vars.
DEFAULT_API_KEY = "sk-40hwexwUh2GuW3jaUiMNoUARdEDd1CxZnQgr2I3VyD84soWg"
DEFAULT_BASE_URL = "http://35.220.164.252:3888/v1"

# Tier resolution: provider-dependent. Anthropic -> claude-sonnet-4-5@20250929;
# OpenAI -> OPENAI_MODEL env var if set else 'gpt-4o'.
MODEL_NAME = os.getenv("OPENAI_MODEL") or default_model(tier="high")

BASE_DIR = _HERE
INPUT_FILE = os.path.join(BASE_DIR, "final_report_for_azure.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "examples", "Final_Drift_Analysis_Report.md")


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


TOTAL_STEPS = 3


def print_progress(step: int, message: str):
    width = 24
    filled = int(width * step / TOTAL_STEPS)
    bar = "#" * filled + "-" * (width - filled)
    print(f"Progress: [{bar}] {step}/{TOTAL_STEPS} {message}")


def load_report_data(filepath: str) -> dict | None:
    print_progress(1, f"读取数据: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {filepath}")
        print("   -> 请先运行 run_full_pipeline.py 生成数据!")
        return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


SYSTEM_PROMPT_V2 = (
    "你是一位资深的 BPM 业务流程挖掘专家。\n"
    "用户会提供漂移检测报告 (schema v2), 其中包含:\n"
    "  * drift_vector: activity_jsd / dfg_jsd / trace_jsd / trace_w1 + p-values\n"
    "  * attribution.top_transport_flows: 哪些 baseline 变体迁移到了哪些 current 变体, 含案例样本\n"
    "  * attribution.top_lost_variants / top_gained_variants: 路径占比的最大变化\n"
    "  * split.change_points + ci_95: 漂移发生在哪些 case 位置 (含 95% 置信区间)\n"
    "  * ground_truth (可选): 若有, 不要引用为'已知的根因', 而是给出独立的诊断\n"
    "请撰写中文 Markdown 报告, 硬性要求:\n"
    "1) 使用固定结构标题: 总览 / 关键变化 (对比 Baseline vs Current) / 根因推断 / 改进建议\n"
    "2) 每部分使用短段落或项目符号\n"
    "3) 必须引用具体数值: 至少 3 个 drift_vector 分量, 至少 3 个 transport flow 的 mass, 至少 1 个 p-value, 1 个 change_point + 其 CI\n"
    "4) 根因推断必须基于 top_transport_flows 而不是仅靠 top_lost/top_gained\n"
    "5) 改进建议要可执行 (具体到流程步骤, 不要空话)\n"
)


SYSTEM_PROMPT_LEGACY = (
    "你是一位资深的 BPM 业务流程挖掘专家。\n"
    "请根据用户提供的 JSON 漂移检测数据 (legacy schema v1), 撰写一份清晰、可读性高的 Markdown 分析报告。\n\n"
    "硬性要求:\n"
    "1) 使用固定结构标题: 总览 / 关键变化 (对比 Baseline vs Current) / 根因推断 / 改进建议\n"
    "2) 每个部分使用短段落或项目符号, 避免长句堆叠\n"
    "3) 必须引用 JSON 里的具体数据 (例如 drift_score、top_k 频率/计数)\n"
    "4) 结论尽量量化, 避免空泛描述\n"
    "5) 用中文输出, 格式为 Markdown\n"
)


def _shrink_for_prompt(json_data: dict, use_legacy: bool) -> dict:
    """Strip oversized fields so the prompt stays small."""
    if use_legacy or json_data.get("schema_version") != "2.0":
        # legacy mode: just use the legacy block (or the v1 root if it's already legacy)
        return json_data.get("legacy", json_data)

    shrunk = {k: v for k, v in json_data.items() if k not in {"legacy"}}
    # drop bulky ground_truth.affected_case_ids (kept summary stats)
    if isinstance(shrunk.get("ground_truth"), dict):
        gt = {**shrunk["ground_truth"]}
        if "affected_case_ids" in gt and len(gt["affected_case_ids"]) > 10:
            gt["affected_case_ids_sample"] = gt["affected_case_ids"][:10]
            gt["affected_case_ids_total"] = len(gt["affected_case_ids"])
            del gt["affected_case_ids"]
        shrunk["ground_truth"] = gt
    return shrunk


def run_analysis_official(json_data: dict, use_legacy: bool) -> str:
    print_progress(2, f"正在呼叫 LLM ({MODEL_NAME}) [{'legacy' if use_legacy else 'v2'} prompt]")

    client = get_client(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)

    system_prompt = SYSTEM_PROMPT_LEGACY if use_legacy else SYSTEM_PROMPT_V2
    data_to_send = _shrink_for_prompt(json_data, use_legacy)
    data_str = json.dumps(data_to_send, indent=2, ensure_ascii=False, default=str)

    user_prompt = (
        "以下是系统检测到的漂移数据 (JSON), 请严格按要求输出报告:\n"
        "```json\n"
        f"{data_str}\n"
        "```"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--legacy-prompt", action="store_true",
                        help="Use the v1 prompt structure (consumes the legacy block).")
    parser.add_argument("--input", default=INPUT_FILE,
                        help=f"Input JSON path (default: {INPUT_FILE}).")
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help=f"Output Markdown path (default: {OUTPUT_FILE}).")
    args = parser.parse_args()

    data = load_report_data(args.input)
    if data is None:
        return 1

    report = run_analysis_official(data, use_legacy=args.legacy_prompt)
    if not report:
        return 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print_progress(3, f"报告生成成功: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
