"""Normalized CSV export: one row per TextGrad engine call."""

import csv
from pathlib import Path
from typing import Any, Dict, List


GRADIENT_FIELDS = [
    "run_id",
    "recorded_at_utc",
    "iteration",
    "stage",
    "call_index",
    "kind",
    "system_prompt",
    "gradient_prompt",
    "response",
]


def gradient_rows(
    result: Dict[str, Any], *, run_id: str, recorded_at: str
) -> List[Dict[str, Any]]:
    rows = []
    for step in result.get("steps", []):
        iteration = step["iteration"]
        traces = {
            "V": step.get("v_training_interaction", {}).get("gradient_trace", []),
            "G": step.get("g_training_interaction", {}).get("gradient_trace", []),
        }
        for stage, trace in traces.items():
            for call_index, call in enumerate(trace, start=1):
                rows.append(
                    {
                        "run_id": run_id,
                        "recorded_at_utc": recorded_at,
                        "iteration": iteration,
                        "stage": stage,
                        "call_index": call_index,
                        "kind": call.get("kind", ""),
                        "system_prompt": call.get("system_prompt", ""),
                        "gradient_prompt": call.get("prompt", ""),
                        "response": call.get("response", ""),
                    }
                )
    return rows


def append_gradient_csv(
    result: Dict[str, Any],
    csv_path: Path,
    *,
    run_id: str,
    recorded_at: str,
) -> None:
    rows = gradient_rows(result, run_id=run_id, recorded_at=recorded_at)
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GRADIENT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
