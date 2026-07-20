"""Vanilla TextGrad prompt optimization on the balanced GSM8K CSV.

This deliberately removes the Verifier and the GAN-like alternating objective,
while aligning the G-side training protocol with the completed GVGAN run:
fixed system text plus a trainable strategy, the same batches, TGD constraints,
backward length limit, and evaluation checkpoints. The only training objective
is TextGrad's final-answer string equality.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import textgrad as tg
from dotenv import load_dotenv
from textgrad.autograd import StringBasedFunction
from textgrad.engine import EngineLM
from textgrad.tasks.big_bench_hard import string_based_equality_fn

from .agents import GeneratorAgent
from .batch_trainer import GENERATOR_OPTIMIZER_CONSTRAINTS
from .engines import TokenUsageTracker, build_engine, merge_token_usage
from .evaluation import is_correct
from .initial_baseline import BaselineCase, load_cases, summarize
from .prompts import (
    GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
    GSM8K_GENERATOR_PROMPT,
    GSM8K_GENERATOR_STRATEGY_PROMPT,
)
from .recording import BACKWARD_LENGTH_INSTRUCTION, BACKWARD_MAX_TOKENS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


class AuditedEngine(EngineLM):
    """Persist backward/optimizer calls without changing vanilla prompts or limits."""

    def __init__(
        self,
        engine: EngineLM,
        trace_path: Path,
        backward_max_tokens: int | None = BACKWARD_MAX_TOKENS,
    ):
        self.engine = engine
        self.model_string = getattr(engine, "model_string", type(engine).__name__)
        self.trace_path = trace_path
        self.backward_max_tokens = backward_max_tokens
        self._lock = threading.Lock()
        self._call_count = len(_load_jsonl(trace_path))

    @staticmethod
    def _kind(system_prompt: Any, prompt: Any) -> str:
        system_text = str(system_prompt or "").lower()
        prompt_text = str(prompt).lower()
        if "optimization system that improves text" in system_text:
            return "optimizer_update"
        if "gradient (feedback) engine" in system_text or "feedback to a variable" in prompt_text:
            return "backward_gradient"
        if "reduce" in system_text and "gradient" in system_text:
            return "gradient_reduction"
        return "other"

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        kind = self._kind(system_prompt, prompt)
        if kind == "backward_gradient" and self.backward_max_tokens is not None:
            kwargs.setdefault("max_tokens", self.backward_max_tokens)
            system_prompt = (
                f"{str(system_prompt).rstrip()}\n\n{BACKWARD_LENGTH_INSTRUCTION}"
                if system_prompt
                else BACKWARD_LENGTH_INSTRUCTION
            )
        with self._lock:
            self._call_count += 1
            call_index = self._call_count
        print(f"    textgrad call={call_index} start kind={kind}", flush=True)
        response = self.engine(prompt, system_prompt=system_prompt, **kwargs)
        record = {
            "call_index": call_index,
            "timestamp_utc": utc_now(),
            "kind": kind,
            "model": self.model_string,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
        }
        with self._lock:
            _append_jsonl(self.trace_path, record)
        print(f"    textgrad call={call_index} done kind={kind}", flush=True)
        return response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the shared initial Qwen4B GSM8K prompt using vanilla "
            "TextGrad with Qwen32B as backward/optimizer engine."
        )
    )
    parser.add_argument(
        "--data",
        default="runs/gsm8k_qwen4b_balanced_1to1/mixed_samples.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/qwen4b_vanilla_textgrad_qwen32b_gvgan_aligned",
    )
    parser.add_argument("--forward-model", default="qwen4b-api")
    parser.add_argument("--backward-model", default="qwen32b-api")
    parser.add_argument(
        "--forward-vllm-base-url",
        default=os.getenv("GENERATOR_VLLM_BASE_URL", "http://127.0.0.1:8004/v1"),
    )
    parser.add_argument(
        "--backward-vllm-base-url",
        default=os.getenv("BACKWARD_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--vllm-api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument("--forward-enable-thinking", action="store_true")
    parser.add_argument("--backward-enable-thinking", action="store_true")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=None,
        help="Invocation-only early stop for smoke/resume testing.",
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch_schedule(
    train_cases: Sequence[BaselineCase],
    *,
    batch_size: int,
) -> list[list[int]]:
    return [
        list(range(start, min(start + batch_size, len(train_cases))))
        for start in range(0, len(train_cases), batch_size)
    ]


def _configuration(args, data_path: Path, schedule: list[list[int]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "balanced-gsm8k-vanilla-textgrad-gvgan-aligned-ablation",
        "data": str(data_path.resolve()),
        "data_sha256": _sha256_file(data_path),
        "forward_model": args.forward_model,
        "backward_model": args.backward_model,
        "forward_vllm_base_url": args.forward_vllm_base_url.rstrip("/"),
        "backward_vllm_base_url": args.backward_vllm_base_url.rstrip("/"),
        "forward_enable_thinking": bool(args.forward_enable_thinking),
        "backward_enable_thinking": bool(args.backward_enable_thinking),
        "temperature": 0,
        "engine_default_max_tokens": 2000,
        "initial_prompt": GSM8K_GENERATOR_PROMPT,
        "initial_prompt_sha256": hashlib.sha256(
            GSM8K_GENERATOR_PROMPT.encode("utf-8")
        ).hexdigest(),
        "fixed_system_prompt": GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
        "initial_trainable_strategy": GSM8K_GENERATOR_STRATEGY_PROMPT,
        "trainable_variable": "strategy only; fixed Generator system prompt is immutable",
        "objective": "TextGrad GSM8K final-integer string equality (0/1)",
        "optimizer": "TextualGradientDescent",
        "optimizer_constraints": GENERATOR_OPTIMIZER_CONSTRAINTS,
        "gradient_memory": 0,
        "batch_aggregation": "tg.sum",
        "backward_max_tokens": BACKWARD_MAX_TOKENS,
        "backward_length_instruction": BACKWARD_LENGTH_INSTRUCTION,
        "update_acceptance": "unconditional, matching GVGAN",
        "evaluation_policy": "initial and after every batch on train/val/test, matching GVGAN",
        "batch_size": args.batch_size,
        "epochs": 1,
        "total_steps": len(schedule),
        "batch_schedule_train_offsets": schedule,
    }


def _validate_or_create_config(path: Path, expected: dict[str, Any]) -> None:
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            differing = {
                key: {"existing": actual.get(key), "requested": value}
                for key, value in expected.items()
                if actual.get(key) != value
            }
            raise ValueError(f"output directory has a different configuration: {differing}")
        return
    _atomic_json(path, expected)


def _prediction(case: BaselineCase, output: str, prompt_sha256: str) -> dict[str, Any]:
    return {
        **asdict(case),
        "generator_output": output,
        "correct": is_correct(output, case.ground_truth),
        "prompt_sha256": prompt_sha256,
    }


def _combined_generator_prompt(strategy: str) -> str:
    return (
        GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT
        + "\n\n<TRAINABLE_STRATEGY>\n"
        + strategy
        + "\n</TRAINABLE_STRATEGY>"
    )


def evaluate_cases(
    cases: Sequence[BaselineCase],
    *,
    prompt: str,
    engine: EngineLM,
    workers: int,
    label: str,
) -> list[dict[str, Any]]:
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def answer(case: BaselineCase) -> dict[str, Any]:
        output = engine(case.question, system_prompt=prompt)
        return _prediction(case, output, prompt_sha256)

    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(answer, case): case for case in cases}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"    {label} {len(records)}/{len(cases)} "
                f"id={record['wrong_id']} correct={record['correct']}",
                flush=True,
            )
    return sorted(records, key=lambda item: int(item["position"]))


def _accuracy(records: Sequence[dict[str, Any]]) -> float:
    return sum(bool(record["correct"]) for record in records) / len(records)


def _write_predictions_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fields = (
        "position",
        "wrong_id",
        "collection_split",
        "difficulty",
        "source_split",
        "source_index",
        "question",
        "ground_truth",
        "correct",
        "prompt_sha256",
        "generator_output",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda item: int(item["position"])):
            writer.writerow({field: record.get(field) for field in fields})
    temporary.replace(path)


def run(argv: Sequence[str] | None = None) -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.stop_after_step is not None and args.stop_after_step < 1:
        parser.error("stop-after-step must be positive")

    data_path = Path(args.data)
    cases = load_cases(data_path)
    train_cases = [case for case in cases if case.collection_split == "train"]
    schedule = _batch_schedule(
        train_cases,
        batch_size=args.batch_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    state_path = output_dir / "state.json"
    trace_path = output_dir / "textgrad_calls.jsonl"
    final_jsonl = output_dir / "final_predictions.jsonl"
    final_csv = output_dir / "final_predictions.csv"
    summary_path = output_dir / "summary.json"
    config = _configuration(args, data_path, schedule)
    _validate_or_create_config(config_path, config)

    prior_state = (
        json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    )
    if prior_state and prior_state.get("status") == "complete":
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            f"baseline already complete: final overall="
            f"{summary['metrics']['overall']['accuracy']:.6f} output={summary_path}",
            flush=True,
        )
        return

    forward_tracker = TokenUsageTracker()
    backward_tracker = TokenUsageTracker()
    forward_engine = build_engine(
        args.forward_model,
        vllm_base_url=args.forward_vllm_base_url,
        vllm_api_key=args.vllm_api_key,
        vllm_enable_thinking=args.forward_enable_thinking,
        token_usage_tracker=forward_tracker,
        usage_role="forward",
    )
    raw_backward_engine = build_engine(
        args.backward_model,
        vllm_base_url=args.backward_vllm_base_url,
        vllm_api_key=args.vllm_api_key,
        vllm_enable_thinking=args.backward_enable_thinking,
        token_usage_tracker=backward_tracker,
        usage_role="backward_optimizer",
    )
    backward_engine = AuditedEngine(raw_backward_engine, trace_path)
    tg.set_backward_engine(backward_engine, override=True)

    persisted_usage = (prior_state or {}).get("token_usage", {})
    state = prior_state or {
        "schema_version": 1,
        "status": "initializing",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "next_step": 0,
        "current_strategy": GSM8K_GENERATOR_STRATEGY_PROMPT,
        "history": [],
        "evaluations": [],
        "pending_evaluation_batch": None,
        "token_usage": persisted_usage,
    }

    fixed_prompt = tg.Variable(
        GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
        requires_grad=False,
        role_description="immutable Generator role, reasoning rules, and output format",
    )
    strategy_variable = tg.Variable(
        state["current_strategy"],
        requires_grad=True,
        role_description="trainable GSM8K Generator problem-solving strategy",
    )
    generator = GeneratorAgent(forward_engine, strategy_variable, fixed_prompt)
    objective = StringBasedFunction(
        string_based_equality_fn,
        function_purpose=(
            "The runtime of string-based function that checks if the prediction "
            "is correct."
        ),
    )
    optimizer = tg.TextualGradientDescent(
        engine=backward_engine,
        parameters=generator.parameters(),
        constraints=GENERATOR_OPTIMIZER_CONSTRAINTS,
    )

    def current_usage():
        return merge_token_usage(
            persisted_usage, forward_tracker.snapshot(), backward_tracker.snapshot()
        )

    def run_evaluation(batch_index: int) -> None:
        stage = "initial" if batch_index == 0 else "after_batch"
        prompt = _combined_generator_prompt(strategy_variable.value)
        print(
            f"{stage} evaluation batch={batch_index} start ({len(cases)} examples)",
            flush=True,
        )
        rows = evaluate_cases(
            cases,
            prompt=prompt,
            engine=forward_engine,
            workers=args.workers,
            label=f"{stage}-{batch_index}",
        )
        metrics = summarize(cases, rows)
        state["evaluations"].append(
            {
                "stage": stage,
                "batch_index": batch_index,
                "train": metrics["train"],
                "val": metrics["val"],
                "test": metrics["test"],
                "overall": metrics["overall"],
                "rows": rows,
            }
        )
        state["pending_evaluation_batch"] = None
        if batch_index > 0:
            state["next_step"] = batch_index
        state["status"] = "training"
        state["updated_at_utc"] = utc_now()
        state["token_usage"] = current_usage()
        _atomic_json(state_path, state)
        print(
            f"{stage} batch={batch_index} "
            f"train={metrics['train']['accuracy']:.6f} "
            f"val={metrics['val']['accuracy']:.6f} "
            f"test={metrics['test']['accuracy']:.6f}",
            flush=True,
        )

    if not state["evaluations"]:
        run_evaluation(0)
    elif state.get("pending_evaluation_batch") is not None:
        run_evaluation(int(state["pending_evaluation_batch"]))

    print(
        f"vanilla TextGrad start step={state['next_step']}/{len(schedule)} "
        f"batch_size={args.batch_size} forward={args.forward_model} "
        f"backward={args.backward_model} backward_max_tokens={BACKWARD_MAX_TOKENS}",
        flush=True,
    )
    invocation_steps = 0
    for step_index in range(int(state["next_step"]), len(schedule)):
        if args.stop_after_step is not None and invocation_steps >= args.stop_after_step:
            print(f"intentional stop after {invocation_steps} new step(s)", flush=True)
            return
        batch_offsets = schedule[step_index]
        batch = [train_cases[offset] for offset in batch_offsets]
        old_strategy = strategy_variable.value
        old_prompt = _combined_generator_prompt(old_strategy)
        print(
            f"step {step_index + 1}/{len(schedule)} train ids="
            f"{[case.wrong_id for case in batch]}",
            flush=True,
        )
        optimizer.zero_grad()
        losses = []
        train_predictions = []
        for case in batch:
            question = tg.Variable(
                case.question,
                requires_grad=False,
                role_description="multi-step GSM8K question",
            )
            answer = tg.Variable(
                case.ground_truth,
                requires_grad=False,
                role_description="correct answer for the query",
            )
            response = generator.run(question)
            train_predictions.append(
                _prediction(
                    case,
                    response.value,
                    hashlib.sha256(old_prompt.encode("utf-8")).hexdigest(),
                )
            )
            losses.append(
                objective(
                    inputs={"prediction": response, "ground_truth_answer": answer}
                )
            )
        tg.sum(losses).backward()
        gradient_text = strategy_variable.get_gradient_text()
        optimizer.step()
        new_strategy = strategy_variable.value
        new_prompt = _combined_generator_prompt(new_strategy)
        state["current_strategy"] = new_strategy
        state["history"].append(
            {
                "step": step_index + 1,
                "epoch": 1,
                "step_in_epoch": step_index + 1,
                "batch_train_offsets": batch_offsets,
                "batch_wrong_ids": [case.wrong_id for case in batch],
                "train_predictions": train_predictions,
                "train_batch_accuracy": _accuracy(train_predictions),
                "strategy_before": old_strategy,
                "strategy_after": new_strategy,
                "generator_prompt_before": old_prompt,
                "generator_prompt_after": new_prompt,
                "update_status": "updated",
                "gradient_text": gradient_text,
            }
        )
        # Persist the unconditional update before the expensive all-set
        # evaluation. A restart completes this pending evaluation without
        # applying the optimizer update a second time.
        state["pending_evaluation_batch"] = step_index + 1
        state["status"] = "evaluating"
        state["updated_at_utc"] = utc_now()
        state["token_usage"] = current_usage()
        _atomic_json(state_path, state)
        print(
            f"step {step_index + 1} updated unconditionally: "
            f"strategy chars {len(old_strategy)} -> {len(new_strategy)}",
            flush=True,
        )
        run_evaluation(step_index + 1)
        invocation_steps += 1

    final_evaluation = state["evaluations"][-1]
    final_records = final_evaluation["rows"]
    _write_predictions_csv(final_csv, final_records)
    final_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final_records),
        encoding="utf-8",
    )
    metrics = {
        key: final_evaluation[key] for key in ("train", "val", "test", "overall")
    }
    final_usage = current_usage()
    summary = {
        **config,
        "created_at_utc": state["created_at_utc"],
        "completed_at_utc": utc_now(),
        "complete": True,
        "initial_metrics": {
            key: state["evaluations"][0][key]
            for key in ("train", "val", "test", "overall")
        },
        "updates": len(state["history"]),
        "final_strategy": state["current_strategy"],
        "final_prompt": _combined_generator_prompt(state["current_strategy"]),
        "final_prompt_sha256": hashlib.sha256(
            _combined_generator_prompt(state["current_strategy"]).encode("utf-8")
        ).hexdigest(),
        "metrics": metrics,
        "token_usage": final_usage,
        "state": str(state_path.resolve()),
        "textgrad_calls": str(trace_path.resolve()),
        "final_predictions_jsonl": str(final_jsonl.resolve()),
        "final_predictions_csv": str(final_csv.resolve()),
    }
    _atomic_json(summary_path, summary)
    state["status"] = "complete"
    state["updated_at_utc"] = utc_now()
    state["token_usage"] = final_usage
    _atomic_json(state_path, state)
    print(
        "vanilla TextGrad complete "
        f"train={metrics['train']['accuracy']:.6f} "
        f"val={metrics['val']['accuracy']:.6f} "
        f"test={metrics['test']['accuracy']:.6f} "
        f"overall={metrics['overall']['accuracy']:.6f} output={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    run()
