"""One-pass Generator baselines over the balanced GSM8K CSV."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import textgrad as tg
from dotenv import load_dotenv
from textgrad.tasks.big_bench_hard import parse_integer_answer

from .agents import GeneratorAgent
from .engines import TokenUsageTracker, build_engine, merge_token_usage
from .evaluation import is_correct
from .prompts import (
    GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
    GSM8K_GENERATOR_PROMPT,
    GSM8K_GENERATOR_STRATEGY_PROMPT,
)


PREDICTION_FIELDS = (
    "position",
    "wrong_id",
    "collection_split",
    "difficulty",
    "source_split",
    "source_index",
    "question",
    "ground_truth",
    "parsed_prediction",
    "correct",
    "model",
    "generator_output",
)


@dataclass(frozen=True)
class BaselineCase:
    position: int
    wrong_id: int
    collection_split: str
    difficulty: str
    source_split: str
    source_index: int
    question: str
    ground_truth: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser(
    *,
    default_model: str,
    default_base_url: str,
    default_output_dir: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Generator inference per balanced GSM8K example with the "
            "unchanged initial prompt. No Verifier, backward pass, or update is used."
        )
    )
    parser.add_argument(
        "--data",
        default="runs/gsm8k_qwen4b_balanced_1to1/mixed_samples.csv",
    )
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--model", default=default_model)
    parser.add_argument(
        "--vllm-base-url",
        default=os.getenv("VLLM_BASE_URL", default_base_url),
    )
    parser.add_argument("--vllm-api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument(
        "--vllm-enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking; disabled by default for both baselines.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit newly processed samples for a smoke test; omit for the full set.",
    )
    return parser


def load_cases(path: Path) -> list[BaselineCase]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "wrong_id",
        "collection_split",
        "split",
        "index",
        "question",
        "ground_truth",
    }
    if not rows:
        raise ValueError(f"baseline dataset is empty: {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"baseline dataset is missing columns: {sorted(missing)}")
    cases = [
        BaselineCase(
            position=position,
            wrong_id=int(row["wrong_id"]),
            collection_split=row["collection_split"],
            difficulty=row.get("difficulty", ""),
            source_split=row["split"],
            source_index=int(row["index"]),
            question=row["question"],
            ground_truth=row["ground_truth"],
        )
        for position, row in enumerate(rows, start=1)
    ]
    split_counts = {
        split: sum(case.collection_split == split for case in cases)
        for split in ("train", "val", "test")
    }
    if any(count == 0 for count in split_counts.values()):
        raise ValueError(
            "expected non-empty train, val, and test collection splits, got "
            f"{split_counts}"
        )
    return cases


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
    positions = [int(record["position"]) for record in records]
    if len(positions) != len(set(positions)):
        raise ValueError(f"duplicate prediction positions in {path}")
    return records


def _write_predictions_csv(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for record in sorted(records, key=lambda item: int(item["position"])):
            writer.writerow({field: record.get(field) for field in PREDICTION_FIELDS})
    temporary.replace(path)


def _metric(records: Sequence[Dict[str, Any]], target_count: int) -> Dict[str, Any]:
    completed = len(records)
    correct = sum(bool(record["correct"]) for record in records)
    return {
        "target_count": target_count,
        "completed_count": completed,
        "correct_count": correct,
        "incorrect_count": completed - correct,
        "accuracy": correct / completed if completed else None,
        "coverage": completed / target_count if target_count else 1.0,
    }


def summarize(
    cases: Sequence[BaselineCase],
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_cases = [case for case in cases if case.collection_split == split]
        split_records = [
            record for record in records if record["collection_split"] == split
        ]
        result[split] = _metric(split_records, len(split_cases))
    result["overall"] = _metric(records, len(cases))
    return result


def _configuration(args, data_path: Path) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "balanced-gsm8k-one-pass-initial-generator-baseline",
        "data": str(data_path.resolve()),
        "model": args.model,
        "vllm_base_url": args.vllm_base_url.rstrip("/"),
        "vllm_enable_thinking": bool(args.vllm_enable_thinking),
        "temperature": 0,
        "max_tokens": 2000,
        "generator_fixed_system_prompt": GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
        "generator_strategy_prompt": GSM8K_GENERATOR_STRATEGY_PROMPT,
        "generator_prompt": GSM8K_GENERATOR_PROMPT,
        "generator_prompt_sha256": hashlib.sha256(
            GSM8K_GENERATOR_PROMPT.encode("utf-8")
        ).hexdigest(),
    }


def _validate_or_create_config(path: Path, expected: Dict[str, Any]) -> None:
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            differing = {
                key: {"existing": actual.get(key), "requested": value}
                for key, value in expected.items()
                if actual.get(key) != value
            }
            raise ValueError(
                f"output directory belongs to a different baseline: {differing}"
            )
        return
    _write_json(path, expected)


def run(
    argv: Sequence[str] | None = None,
    *,
    default_model: str,
    default_base_url: str,
    default_output_dir: str,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    parser = build_parser(
        default_model=default_model,
        default_base_url=default_base_url,
        default_output_dir=default_output_dir,
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    if not args.vllm_base_url:
        parser.error("--vllm-base-url or VLLM_BASE_URL is required")

    data_path = Path(args.data)
    cases = load_cases(data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    predictions_jsonl = output_dir / "predictions.jsonl"
    predictions_csv = output_dir / "predictions.csv"
    errors_jsonl = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"
    config = _configuration(args, data_path)
    _validate_or_create_config(config_path, config)

    records = _load_jsonl(predictions_jsonl)
    completed_positions = {int(record["position"]) for record in records}
    known_positions = {case.position for case in cases}
    unknown_positions = completed_positions.difference(known_positions)
    if unknown_positions:
        raise ValueError(
            f"predictions contain positions absent from this dataset: {unknown_positions}"
        )
    pending = [case for case in cases if case.position not in completed_positions]
    if not pending:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            f"baseline already complete: model={args.model} "
            f"accuracy={summary['metrics']['overall']['accuracy']:.6f} "
            f"output={summary_path}",
            flush=True,
        )
        return

    prior_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    token_usage_base = prior_summary.get("token_usage", {})
    created_at = prior_summary.get("created_at_utc", utc_now())
    tracker = TokenUsageTracker()
    engine = build_engine(
        args.model,
        vllm_base_url=args.vllm_base_url,
        vllm_api_key=args.vllm_api_key,
        vllm_enable_thinking=args.vllm_enable_thinking,
        token_usage_tracker=tracker,
        usage_role="generator",
    )
    fixed_prompt = tg.Variable(
        GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
        requires_grad=False,
        role_description="fixed initial GSM8K Generator rules",
    )
    strategy_prompt = tg.Variable(
        GSM8K_GENERATOR_STRATEGY_PROMPT,
        requires_grad=False,
        role_description="fixed initial GSM8K Generator strategy",
    )
    generator = GeneratorAgent(engine, strategy_prompt, fixed_prompt)

    invocation_limit = len(pending)
    if args.max_samples is not None:
        invocation_limit = min(invocation_limit, args.max_samples)
    invocation_cases = pending[:invocation_limit]

    def answer_one(case: BaselineCase) -> Dict[str, Any]:
        question = tg.Variable(
            case.question,
            requires_grad=False,
            role_description="balanced GSM8K baseline question",
        )
        output = generator.run(question).value
        return {
            **asdict(case),
            "parsed_prediction": parse_integer_answer(output),
            "correct": is_correct(output, case.ground_truth),
            "model": args.model,
            "generator_output": output,
        }

    print(
        f"baseline start model={args.model} endpoint={args.vllm_base_url} "
        f"completed={len(records)}/{len(cases)} processing={len(invocation_cases)} "
        f"workers={args.workers} prompt_sha256={config['generator_prompt_sha256']}",
        flush=True,
    )
    invocation_errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for batch_start in range(0, len(invocation_cases), args.workers):
            batch = invocation_cases[batch_start : batch_start + args.workers]
            futures = [executor.submit(answer_one, case) for case in batch]
            for case, future in zip(batch, futures):
                try:
                    record = future.result()
                except Exception as exc:
                    invocation_errors += 1
                    error_record = {
                        "timestamp_utc": utc_now(),
                        **asdict(case),
                        "model": args.model,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    _append_jsonl(errors_jsonl, error_record)
                    print(
                        f"error position={case.position} wrong_id={case.wrong_id} "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                _append_jsonl(predictions_jsonl, record)
                records.append(record)
                print(
                    f"done={len(records)}/{len(cases)} position={case.position} "
                    f"split={case.collection_split} wrong_id={case.wrong_id} "
                    f"correct={record['correct']}",
                    flush=True,
                )

            _write_predictions_csv(predictions_csv, records)
            metrics = summarize(cases, records)
            complete = len(records) == len(cases)
            summary = {
                **config,
                "created_at_utc": created_at,
                "updated_at_utc": utc_now(),
                "complete": complete,
                "workers": args.workers,
                "metrics": metrics,
                "token_usage": merge_token_usage(
                    token_usage_base,
                    tracker.snapshot(),
                ),
                "predictions_jsonl": str(predictions_jsonl.resolve()),
                "predictions_csv": str(predictions_csv.resolve()),
                "errors_jsonl": str(errors_jsonl.resolve()),
            }
            _write_json(summary_path, summary)

    remaining = len(cases) - len(records)
    if remaining and args.max_samples is None:
        raise RuntimeError(
            f"baseline incomplete after {invocation_errors} errors; "
            f"{remaining} samples remain. Rerun the same command to retry them."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    print(
        "baseline result "
        f"model={args.model} complete={summary['complete']} "
        f"train={metrics['train']['accuracy']} "
        f"val={metrics['val']['accuracy']} "
        f"test={metrics['test']['accuracy']} "
        f"overall={metrics['overall']['accuracy']} "
        f"output={summary_path}",
        flush=True,
    )

