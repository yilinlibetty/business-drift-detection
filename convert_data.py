"""Back-compat shim — re-exports load_event_log from drift.io.

The real implementation lives in drift/io.py. This file is preserved so that:
  * existing imports `from convert_data import load_event_log` keep working;
  * the CLI behavior `python convert_data.py` (XES -> CSV conversion) is unchanged.
"""

from __future__ import annotations

import os

from drift.io import (
    load_event_log,
    _parse_xes_to_dataframe,  # re-export for downstream callers
    _read_xes,
    _should_keep_event,
    _strip_tag,
)

__all__ = [
    "load_event_log",
    "_parse_xes_to_dataframe",
    "_read_xes",
    "_should_keep_event",
    "_strip_tag",
]


if __name__ == "__main__":
    input_file = os.getenv("EVENT_LOG_PATH", "datasets/frequency-log.xes")
    output_file = os.getenv("EVENT_LOG_OUTPUT", "datasets/frequency-log.csv")
    keep_only_complete = os.getenv("KEEP_ONLY_COMPLETE", "true").lower() in {"1", "true", "yes"}

    print(f"Loading {input_file}...")
    df = load_event_log(
        input_file,
        case_id_col="case:concept:name",
        activity_col="concept:name",
        timestamp_col="time:timestamp",
        keep_only_complete=keep_only_complete,
    )
    df.to_csv(output_file, index=False)
    print(f"Saved CSV to {output_file}")
