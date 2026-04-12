from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from drift_detection.pipeline import PipelineConfig, serialize_value
from drift_detection.reporting import render_markdown_report
from drift_detection.visualization import (
    figures_to_zip_bytes,
    plot_activity_delta,
    plot_attribute_delta,
    plot_dominant_signal_distribution,
    plot_drift_point_score_breakdown,
    plot_duration_comparison,
    plot_multiview_radar,
    plot_score_timeline,
    plot_score_component_heatmap,
    plot_threshold_sensitivity,
    plot_transition_delta,
    plot_trace_distribution,
    save_analysis_figures,
)
from run_full_pipeline import run_pipeline


st.set_page_config(
    page_title="Business Drift Detection",
    page_icon=None,
    layout="wide",
)


def main() -> None:
    st.title("Business Drift Detection")
    st.caption("Single-run research prototype for evidence-driven process drift diagnosis.")

    defaults = PipelineConfig.from_env()
    uploaded_file, config = _render_sidebar(defaults)

    status_area = st.empty()
    if st.sidebar.button("Run drift detection", type="primary", use_container_width=True):
        _run_detection(uploaded_file, config, status_area)

    result = st.session_state.get("drift_result")
    if not result:
        st.info(
            "Default demo is ready. If you do not upload a file, the app uses `datasets/finale.csv`; "
            "you can click `Run drift detection` directly."
        )
        return

    _render_results(result)


def _render_sidebar(defaults: PipelineConfig) -> tuple[Any, PipelineConfig]:
    st.sidebar.header("Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload event log",
        type=["csv", "xes", "xml"],
        help="CSV uses selectable columns; XES/XML uses the configured names for parser output.",
    )

    csv_columns = _read_uploaded_csv_columns(uploaded_file) or _read_default_csv_columns(defaults.file_path)
    if uploaded_file is None:
        st.sidebar.success("One-click demo: using the default dataset.")
        st.sidebar.caption(f"Dataset: `{defaults.file_path}`")
    elif csv_columns:
        st.sidebar.caption(f"Uploaded CSV with {len(csv_columns)} columns.")
    else:
        st.sidebar.caption("Uploaded XES/XML or CSV header could not be previewed; use manual column names.")
    st.sidebar.caption(
        "Recommended defaults: columns auto-detected, window/step auto, mode=mixed, "
        "score_profile=trace-duration, metric=tv, auto threshold, LLM off."
    )

    with st.sidebar.expander("Columns (auto-filled)", expanded=uploaded_file is not None):
        col_case_id = _column_input("Case ID column", defaults.col_case_id, csv_columns, _CASE_ID_CANDIDATES)
        col_activity = _column_input("Activity column", defaults.col_activity, csv_columns, _ACTIVITY_CANDIDATES)
        col_timestamp = _column_input("Timestamp column", defaults.col_timestamp, csv_columns, _TIMESTAMP_CANDIDATES)
        keep_only_complete = st.checkbox("Keep only complete lifecycle events", value=defaults.keep_only_complete)

    with st.sidebar.expander("Detection settings", expanded=False):
        window_size_raw = st.number_input(
            "Window size (0 = auto)",
            min_value=0,
            value=int(defaults.window_size or 0),
            step=1,
        )
        step_size_raw = st.number_input(
            "Step size (0 = auto)",
            min_value=0,
            value=int(defaults.step_size or 0),
            step=1,
        )
        detection_mode = st.selectbox(
            "Detection mode",
            options=["mixed", "structure", "delay"],
            index=["mixed", "structure", "delay"].index(defaults.detection_mode)
            if defaults.detection_mode in {"mixed", "structure", "delay"}
            else 0,
        )
        score_profile = st.selectbox(
            "Score profile",
            options=["trace-duration", "multi-view"],
            index=0 if defaults.score_profile != "multi-view" else 1,
        )
        drift_metric = st.selectbox(
            "Trace drift metric",
            options=["tv", "l1"],
            index=0 if defaults.drift_metric != "l1" else 1,
        )
        top_k = st.number_input("Top-K evidence items", min_value=1, max_value=50, value=int(defaults.top_k), step=1)

    with st.sidebar.expander("Threshold settings", expanded=False):
        auto_threshold = st.checkbox("Auto threshold", value=defaults.auto_threshold)
        threshold = st.number_input(
            "Base threshold",
            min_value=0.0,
            value=float(defaults.threshold),
            step=0.01,
            format="%.4f",
        )
        mad_multiplier = st.number_input(
            "MAD multiplier",
            min_value=0.0,
            value=float(defaults.mad_multiplier),
            step=0.5,
            format="%.2f",
        )

    with st.sidebar.expander("LLM diagnosis", expanded=False):
        has_api_key = bool(os.getenv("OPENAI_API_KEY"))
        llm_enabled = st.checkbox(
            "Enable LLM diagnosis",
            value=False,
            help="Disabled by default for reproducible runs. Requires OPENAI_API_KEY when enabled.",
        )
        if llm_enabled and not has_api_key:
            st.warning("OPENAI_API_KEY is not configured; the pipeline will use fallback diagnosis.")

    config = PipelineConfig.from_env()
    config.file_path = defaults.file_path
    config.col_case_id = col_case_id
    config.col_activity = col_activity
    config.col_timestamp = col_timestamp
    config.keep_only_complete = keep_only_complete
    config.window_size = int(window_size_raw) or None
    config.step_size = int(step_size_raw) or None
    config.detection_mode = detection_mode
    config.score_profile = score_profile
    config.drift_metric = drift_metric
    config.top_k = int(top_k)
    config.auto_threshold = auto_threshold
    config.threshold = float(threshold)
    config.mad_multiplier = float(mad_multiplier)
    config.llm_enabled = llm_enabled
    config.inject_drift = False
    config.analysis_mode = "timeline"
    return uploaded_file, config


def _run_detection(uploaded_file: Any, config: PipelineConfig, status_area: Any) -> None:
    temp_path: str | None = None
    logs: list[str] = []

    def log_callback(message: str) -> None:
        logs.append(message)
        status_area.info(message)

    try:
        if uploaded_file is not None:
            temp_path = _save_uploaded_file(uploaded_file)
            config.file_path = temp_path

        with st.spinner("Running drift detection..."):
            result = run_pipeline(config, evaluate_requested=False, verbose=False, log_callback=log_callback)

        st.session_state["drift_result"] = result
        st.session_state["drift_run_logs"] = logs
        status_area.success("Detection finished.")
    except Exception as exc:
        st.session_state["drift_run_logs"] = logs
        status_area.error(f"Detection failed: {exc}")
        st.exception(exc)
    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _render_results(result: dict[str, Any]) -> None:
    global_summary = result.get("global_summary", {}) or {}
    drift_points = result.get("drift_points", []) or []
    config = result.get("config", {}) or {}
    llm_meta = result.get("llm", {}) or {}

    st.subheader("Summary")
    columns = st.columns(6)
    columns[0].metric("Status", str(global_summary.get("status", "UNKNOWN")))
    columns[1].metric("Drift points", int(global_summary.get("drift_point_count", len(drift_points)) or 0))
    columns[2].metric("Threshold", _format_number(global_summary.get("threshold")))
    columns[3].metric("Peak score", _format_number(global_summary.get("peak_score")))
    columns[4].metric("Score profile", str(global_summary.get("score_profile", config.get("score_profile", "N/A"))))
    columns[5].metric("LLM used", "yes" if llm_meta.get("used_llm") else "no")

    download_left, download_right = st.columns(2)
    download_left.download_button(
        "Download JSON",
        data=json.dumps(serialize_value(result), ensure_ascii=False, indent=2),
        file_name="drift_analysis.json",
        mime="application/json",
        use_container_width=True,
    )
    download_right.download_button(
        "Download Markdown report",
        data=render_markdown_report(result),
        file_name="final_drift_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    figure_left, figure_right = st.columns(2)
    figure_left.download_button(
        "Download figures ZIP",
        data=figures_to_zip_bytes(result),
        file_name="drift_analysis_figures.zip",
        mime="application/zip",
        use_container_width=True,
    )
    if figure_right.button("Save figures to outputs/figures", use_container_width=True):
        saved_paths = save_analysis_figures(result, output_dir="outputs/figures")
        st.success(f"Saved {len(saved_paths)} figure file(s) to outputs/figures.")

    st.subheader("Figure 1. Drift Score Timeline")
    st.pyplot(plot_score_timeline(result), clear_figure=True)

    with st.expander("Figure 5. Threshold sensitivity", expanded=False):
        st.pyplot(plot_threshold_sensitivity(result), clear_figure=True)
    with st.expander("More global analysis figures", expanded=False):
        global_tabs = st.tabs(["Score heatmap", "Dominant signal distribution"])
        with global_tabs[0]:
            st.pyplot(plot_score_component_heatmap(result), clear_figure=True)
        with global_tabs[1]:
            st.pyplot(plot_dominant_signal_distribution(result), clear_figure=True)

    st.subheader("Drift Points")
    if not drift_points:
        st.info("No drift points exceeded the configured threshold.")
        return

    for point in drift_points:
        _render_drift_point(point)

    with st.expander("Pipeline logs", expanded=False):
        for line in st.session_state.get("drift_run_logs", []):
            st.write(line)


def _render_drift_point(point: dict[str, Any]) -> None:
    point_id = point.get("id", "Drift point")
    title = (
        f"{point_id} | {point.get('interval_start_time', 'N/A')} to "
        f"{point.get('interval_end_time', 'N/A')} | peak={point.get('peak_score', 'N/A')}"
    )
    with st.expander(title, expanded=True):
        metric_cols = st.columns(5)
        metric_cols[0].metric("Peak score", _format_number(point.get("peak_score")))
        metric_cols[1].metric("Trace", _format_number(point.get("trace_score")))
        metric_cols[2].metric("Duration", _format_number(point.get("duration_score")))
        metric_cols[3].metric("Dominant", str(point.get("dominant_signal") or "N/A"))
        metric_cols[4].metric("Threshold excess", _format_number(point.get("threshold_excess")))

        tags = point.get("rule_based_tags", []) or []
        tag_text = ", ".join(f"{tag.get('tag')} ({tag.get('confidence')})" for tag in tags) or "None"
        diagnosis = point.get("llm_diagnosis") or {}
        st.markdown(f"**Rule tags:** {tag_text}")
        st.markdown(f"**Diagnosis summary:** {diagnosis.get('summary', 'N/A')}")

        with st.expander("Evidence details", expanded=False):
            tab_trace, tab_activity, tab_transition, tab_attribute, tab_duration, tab_radar, tab_breakdown = st.tabs(
                [
                    "Trace comparison",
                    "Activity delta",
                    "Transition delta",
                    "Attribute delta",
                    "Duration",
                    "Multi-view radar",
                    "Score breakdown",
                ]
            )
            with tab_trace:
                st.pyplot(plot_trace_distribution(point), clear_figure=True)
            with tab_activity:
                st.pyplot(plot_activity_delta(point), clear_figure=True)
            with tab_transition:
                st.pyplot(plot_transition_delta(point), clear_figure=True)
            with tab_attribute:
                st.pyplot(plot_attribute_delta(point), clear_figure=True)
            with tab_duration:
                st.pyplot(plot_duration_comparison(point), clear_figure=True)
            with tab_radar:
                st.pyplot(plot_multiview_radar(point), clear_figure=True)
            with tab_breakdown:
                st.pyplot(plot_drift_point_score_breakdown(point), clear_figure=True)


_CASE_ID_CANDIDATES = ("Case ID", "case_id", "case:concept:name", "case", "id")
_ACTIVITY_CANDIDATES = ("Activity", "activity", "concept:name", "event", "task")
_TIMESTAMP_CANDIDATES = ("Complete Timestamp", "timestamp", "time:timestamp", "Timestamp", "datetime")


def _column_input(label: str, default: str, columns: list[str], candidates: tuple[str, ...]) -> str:
    if columns:
        selected_default = _pick_column(columns, default, candidates)
        index = columns.index(selected_default)
        return str(st.selectbox(label, options=columns, index=index))
    return st.text_input(label, value=default)


def _pick_column(columns: list[str], default: str, candidates: tuple[str, ...]) -> str:
    normalized = {column.strip().lower(): column for column in columns}
    for candidate in (default, *candidates):
        match = normalized.get(candidate.strip().lower())
        if match is not None:
            return match
    for column in columns:
        lowered = column.strip().lower()
        if any(candidate.strip().lower() in lowered for candidate in candidates):
            return column
    return columns[0]


def _read_uploaded_csv_columns(uploaded_file: Any) -> list[str]:
    if uploaded_file is None:
        return []
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix != ".csv":
        return []
    try:
        columns = list(pd.read_csv(uploaded_file, nrows=0).columns)
    except Exception:
        columns = []
    finally:
        uploaded_file.seek(0)
    return [str(column) for column in columns]


def _read_default_csv_columns(file_path: str) -> list[str]:
    if Path(file_path).suffix.lower() != ".csv":
        return []
    try:
        return [str(column) for column in pd.read_csv(file_path, nrows=0).columns]
    except Exception:
        return []


def _save_uploaded_file(uploaded_file: Any) -> str:
    suffix = Path(uploaded_file.name).suffix.lower() or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="drift_upload_") as handle:
        handle.write(uploaded_file.getbuffer())
        return handle.name


def _format_number(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


if __name__ == "__main__":
    main()
