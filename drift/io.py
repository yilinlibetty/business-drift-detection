"""Event-log loading + canonical case-frame construction.

Relocated from convert_data.py (kept there as a thin re-export shim for back-compat).

Public API:
    load_event_log(path, case_id_col, activity_col, timestamp_col, keep_only_complete) -> DataFrame
    build_cases_dataframe(df, case_id_col, activity_col, timestamp_col) -> DataFrame
        columns: CaseID, Trace (tuple of activities), TraceStr (joined), StartTime, EndTime,
                 EventCount, Duration (minutes)
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import pandas as pd

try:
    import pm4py
except ImportError:
    pm4py = None


def load_event_log(file_path, case_id_col, activity_col, timestamp_col, keep_only_complete=True):
    """Read CSV or XES into a flat DataFrame with at least case-id, activity, timestamp columns."""
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == ".csv":
        return pd.read_csv(file_path)
    if file_ext in {".xes", ".xml"}:
        return _read_xes(
            file_path,
            case_id_col,
            activity_col,
            timestamp_col,
            keep_only_complete=keep_only_complete,
        )
    raise ValueError(f"Unsupported file type: {file_ext}")


def build_cases_dataframe(df, case_id_col, activity_col, timestamp_col):
    """Aggregate flat events into per-case rows.

    Returns DataFrame with columns: CaseID, Trace (tuple[str]), TraceStr (str),
    StartTime, EndTime, EventCount, Duration (minutes), sorted by EndTime.

    Trace is stored as a tuple so it is hashable for variant-level computation;
    TraceStr is kept for human-readable JSON output and legacy compatibility.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values([case_id_col, timestamp_col])

    grouped = df.groupby(case_id_col, sort=False)
    rows = []
    for case_id, sub in grouped:
        activities = tuple(map(str, sub[activity_col].tolist()))
        ts = sub[timestamp_col]
        rows.append({
            "CaseID": case_id,
            "Trace": activities,
            "TraceStr": " -> ".join(activities),
            "StartTime": ts.min(),
            "EndTime": ts.max(),
            "EventCount": len(activities),
            "Duration": (ts.max() - ts.min()).total_seconds() / 60.0,
        })
    cases = pd.DataFrame(rows)
    cases = cases.sort_values("EndTime").reset_index(drop=True)
    return cases


def _read_xes(file_path, case_id_col, activity_col, timestamp_col, keep_only_complete=True):
    if pm4py is not None:
        log = pm4py.read_xes(file_path)
        df = pm4py.convert_to_dataframe(log)
        df = df.rename(
            columns={
                "case:concept:name": case_id_col,
                "concept:name": activity_col,
                "time:timestamp": timestamp_col,
            }
        )
        return df
    return _parse_xes_to_dataframe(
        file_path,
        case_id_col,
        activity_col,
        timestamp_col,
        keep_only_complete=keep_only_complete,
    )


def _parse_xes_to_dataframe(
    file_path,
    case_id_col,
    activity_col,
    timestamp_col,
    keep_only_complete=True,
):
    rows = []
    current_case_id = None
    in_event = False
    event_activity = None
    event_timestamp = None
    event_transition = None
    trace_index = 0

    for parse_event, elem in ET.iterparse(file_path, events=("start", "end")):
        tag = _strip_tag(elem.tag)
        if parse_event == "start":
            if tag == "trace":
                trace_index += 1
                current_case_id = None
            elif tag == "event":
                in_event = True
                event_activity = None
                event_timestamp = None
                event_transition = None
            continue

        if tag in {"string", "date"}:
            key = elem.get("key")
            value = elem.get("value")
            if in_event:
                if key == "concept:name" and event_activity is None:
                    event_activity = value
                elif key == "time:timestamp" and event_timestamp is None:
                    event_timestamp = value
                elif key == "lifecycle:transition" and event_transition is None:
                    event_transition = value
            else:
                if key == "concept:name" and current_case_id is None:
                    current_case_id = value
            elem.clear()
            continue

        if tag == "event":
            in_event = False
            if _should_keep_event(event_transition, keep_only_complete):
                if event_activity and event_timestamp:
                    case_id_value = current_case_id or f"trace_{trace_index}"
                    rows.append((case_id_value, event_activity, event_timestamp))
            event_activity = None
            event_timestamp = None
            event_transition = None
            elem.clear()
            continue

        if tag == "trace":
            current_case_id = None
            elem.clear()
            continue

        elem.clear()

    return pd.DataFrame(rows, columns=[case_id_col, activity_col, timestamp_col])


def _should_keep_event(event_transition, keep_only_complete):
    if not keep_only_complete:
        return True
    if event_transition is None:
        return True
    return event_transition.lower() == "complete"


def _strip_tag(tag):
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag
