"""Human-friendly CSV trajectory reporting."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


CSV_FIELDS = [
    "run_id",
    "recorded_at_utc",
    "phase",
    "iteration",
    "dataset",
    "split",
    "case_index",
    "generator_model",
    "verifier_model",
    "backward_model",
    "correct",
    "verifier_label",
    "verifier_confidence",
    "answer",
    "verifier_critique",
    "generator_prompt",
    "verifier_prompt",
    "question",
    "ground_truth",
    "generator_prompt_before",
    "generator_prompt_after",
    "generator_training_output",
    "verifier_prompt_before",
    "verifier_prompt_after",
    "verifier_training_outputs",
    "generator_loss_output",
    "v_textgrad_trace",
    "g_textgrad_trace",
    "post_update_generator_output",
    "post_update_verifier_output",
]


def _json(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False)


def _row(
    result: Dict[str, Any],
    evaluation: Dict[str, Any],
    *,
    run_id: str,
    recorded_at: str,
    phase: str,
    iteration: int,
    generator_prompt: str,
    verifier_prompt: str,
    details: Dict[str, Any] = None,
) -> Dict[str, Any]:
    case = evaluation["case"]
    verdict = evaluation["verdict"]
    models = result.get("models", {})
    details = details or {}
    row = {
        "run_id": run_id,
        "recorded_at_utc": recorded_at,
        "phase": phase,
        "iteration": iteration,
        "dataset": result.get("dataset", ""),
        "split": case.get("split", ""),
        "case_index": case.get("index", ""),
        "generator_model": models.get("generator", ""),
        "verifier_model": models.get("verifier", ""),
        "backward_model": models.get("backward", ""),
        "correct": evaluation["correct"],
        "verifier_label": verdict.get("label", ""),
        "verifier_confidence": verdict.get("confidence", ""),
        "answer": evaluation.get("answer", ""),
        "verifier_critique": verdict.get("critique", ""),
        "generator_prompt": generator_prompt,
        "verifier_prompt": verifier_prompt,
        "question": case.get("question", ""),
        "ground_truth": case.get("answer", ""),
        "generator_prompt_before": details.get(
            "generator_prompt_before", generator_prompt
        ),
        "generator_prompt_after": details.get(
            "generator_prompt_after", generator_prompt
        ),
        "generator_training_output": details.get("generator_training_output", ""),
        "verifier_prompt_before": details.get(
            "verifier_prompt_before", verifier_prompt
        ),
        "verifier_prompt_after": details.get(
            "verifier_prompt_after", verifier_prompt
        ),
        "verifier_training_outputs": _json(
            details.get("verifier_training_outputs")
        ),
        "generator_loss_output": details.get("generator_loss_output", ""),
        "v_textgrad_trace": _json(details.get("v_textgrad_trace")),
        "g_textgrad_trace": _json(details.get("g_textgrad_trace")),
        "post_update_generator_output": evaluation.get("answer", ""),
        "post_update_verifier_output": evaluation.get("verdict_raw", ""),
    }
    return row


def result_rows(
    result: Dict[str, Any], *, run_id: str, recorded_at: str
) -> List[Dict[str, Any]]:
    steps = result["steps"]
    first = steps[0]
    rows = [
        _row(
            result,
            result["initial"],
            run_id=run_id,
            recorded_at=recorded_at,
            phase="initial",
            iteration=0,
            generator_prompt=first["g_prompt_before"],
            verifier_prompt=first["v_prompt_before"],
        )
    ]
    if result.get("check_initial") is not None:
        rows.append(
            _row(
                result,
                result["check_initial"],
                run_id=run_id,
                recorded_at=recorded_at,
                phase="check_initial",
                iteration=0,
                generator_prompt=first["g_prompt_before"],
                verifier_prompt=first["v_prompt_before"],
            )
        )

    for step in steps:
        iteration = step["iteration"]
        rows.append(
            _row(
                result,
                step["evaluation"],
                run_id=run_id,
                recorded_at=recorded_at,
                phase="iteration",
                iteration=iteration,
                generator_prompt=step["g_prompt_after"],
                verifier_prompt=step["v_prompt_after"],
                details={
                    "generator_prompt_before": step["g_prompt_before"],
                    "generator_prompt_after": step["g_prompt_after"],
                    "generator_training_output": step.get(
                        "g_training_interaction", {}
                    ).get("answer", ""),
                    "verifier_prompt_before": step["v_prompt_before"],
                    "verifier_prompt_after": step["v_prompt_after"],
                    "verifier_training_outputs": step.get(
                        "v_training_interaction", {}
                    ).get("examples", []),
                    "generator_loss_output": step.get(
                        "g_training_interaction", {}
                    ).get("loss_output", ""),
                    "v_textgrad_trace": step.get(
                        "v_training_interaction", {}
                    ).get("gradient_trace", []),
                    "g_textgrad_trace": step.get(
                        "g_training_interaction", {}
                    ).get("gradient_trace", []),
                },
            )
        )
        if step.get("check") is not None:
            rows.append(
                _row(
                    result,
                    step["check"],
                    run_id=run_id,
                    recorded_at=recorded_at,
                    phase="check_iteration",
                    iteration=iteration,
                    generator_prompt=step["g_prompt_after"],
                    verifier_prompt=step["v_prompt_after"],
                )
            )

    last = steps[-1]
    rows.append(
        _row(
            result,
            result["final"],
            run_id=run_id,
            recorded_at=recorded_at,
            phase="final",
            iteration=last["iteration"],
            generator_prompt=last["g_prompt_after"],
            verifier_prompt=last["v_prompt_after"],
        )
    )
    if result.get("check_final") is not None:
        rows.append(
            _row(
                result,
                result["check_final"],
                run_id=run_id,
                recorded_at=recorded_at,
                phase="check_final",
                iteration=last["iteration"],
                generator_prompt=last["g_prompt_after"],
                verifier_prompt=last["v_prompt_after"],
            )
        )
    return rows


def append_result_csv(
    result: Dict[str, Any],
    csv_path: Path,
    *,
    run_id: str,
    recorded_at: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            old_fields = reader.fieldnames or []
            old_rows = list(reader)
        if old_fields != CSV_FIELDS:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(
                    {field: row.get(field, "") for field in CSV_FIELDS}
                    for row in old_rows
                )
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(
            result_rows(result, run_id=run_id, recorded_at=recorded_at)
        )
