"""Original TextGrad GSM8K prompt-optimization protocol on our balanced CSV.

Only the dataset and engines are substituted: Qwen3-4B is the forward/test
engine and Qwen3-32B is the evaluation/backward/optimizer engine. Batch size,
shuffle, update count, prompt architecture, objective, validation gate,
evaluation schedule, and optimizer options follow
``textgrad/evaluation/prompt_optimization.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import textgrad as tg
from dotenv import load_dotenv
from textgrad.autograd import StringBasedFunction
from textgrad.tasks.big_bench_hard import string_based_equality_fn

from .engines import TokenUsageTracker, build_engine, merge_token_usage
from .evaluation import is_correct
from .initial_baseline import BaselineCase, load_cases
from .vanilla_textgrad_baseline import (
    AuditedEngine,
    _accuracy,
    _atomic_json,
    evaluate_cases,
    utc_now,
)


ORIGINAL_GSM8K_SYSTEM_PROMPT = (
    "You will answer a mathemetical reasoning question. Think step by step. "
    "The last line of your response should be of the following format: "
    "'Answer: $VALUE' where VALUE is a numerical value."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the original TextGrad GSM8K prompt-optimization protocol on "
            "our balanced CSV with Qwen4B forward and Qwen32B backward."
        )
    )
    parser.add_argument(
        "--data",
        default="runs/gsm8k_qwen4b_balanced_1to1/mixed_samples.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/qwen4b_original_textgrad_qwen32b",
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
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=32)
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


def original_batch_schedule(
    train_size: int,
    *,
    batch_size: int = 3,
    max_epochs: int = 3,
    steps_per_epoch: int = 4,
    seed: int = 42,
) -> list[list[int]]:
    """Reproduce DataLoader's in-place NumPy shuffle and four-batch break."""
    if train_size < batch_size * steps_per_epoch:
        raise ValueError("training split is too small for the original schedule")
    rng = np.random.RandomState(seed)
    indices = np.arange(train_size)
    schedule: list[list[int]] = []
    for _ in range(max_epochs):
        rng.shuffle(indices)
        for step in range(steps_per_epoch):
            start = step * batch_size
            schedule.append([int(value) for value in indices[start : start + batch_size]])
    return schedule


def _configuration(args, data_path: Path, schedule: list[list[int]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "balanced-gsm8k-original-textgrad-protocol",
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
        "starting_system_prompt": ORIGINAL_GSM8K_SYSTEM_PROMPT,
        "trainable_variable": "entire system prompt",
        "objective": "TextGrad GSM8K final-integer string equality (0/1)",
        "optimizer": "TextualGradientDescent",
        "optimizer_constraints": [],
        "gradient_memory": 0,
        "momentum": 0,
        "backward_max_tokens_override": None,
        "backward_length_instruction": None,
        "batch_aggregation": "tg.sum",
        "validation_gate": "reject only when candidate accuracy is lower; accept ties",
        "evaluation_policy": "initial val/test and val/test after every update",
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "total_steps": len(schedule),
        "seed": args.seed,
        "evaluation_workers": args.workers,
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


def run(argv: Sequence[str] | None = None) -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.batch_size, args.max_epochs, args.steps_per_epoch, args.workers) < 1:
        parser.error("batch-size, max-epochs, steps-per-epoch, and workers must be positive")
    if args.stop_after_step is not None and args.stop_after_step < 1:
        parser.error("stop-after-step must be positive")

    data_path = Path(args.data)
    cases = load_cases(data_path)
    train = [case for case in cases if case.collection_split == "train"]
    val = [case for case in cases if case.collection_split == "val"]
    test = [case for case in cases if case.collection_split == "test"]
    schedule = original_batch_schedule(
        len(train),
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        steps_per_epoch=args.steps_per_epoch,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    state_path = output_dir / "state.json"
    trace_path = output_dir / "textgrad_calls.jsonl"
    summary_path = output_dir / "summary.json"
    config = _configuration(args, data_path, schedule)
    _validate_or_create_config(config_path, config)

    prior_state = (
        json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    )
    if prior_state and prior_state.get("status") == "complete":
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            f"original TextGrad baseline already complete: "
            f"test={summary['final_test_accuracy']:.6f} output={summary_path}",
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
    raw_backward = build_engine(
        args.backward_model,
        vllm_base_url=args.backward_vllm_base_url,
        vllm_api_key=args.vllm_api_key,
        vllm_enable_thinking=args.backward_enable_thinking,
        token_usage_tracker=backward_tracker,
        usage_role="backward_optimizer",
    )
    # None is deliberate: original TextGrad does not apply our 512-token cap or
    # append our length instruction to the backward system prompt.
    backward_engine = AuditedEngine(
        raw_backward,
        trace_path,
        backward_max_tokens=None,
    )
    tg.set_backward_engine(backward_engine, override=True)

    persisted_usage = (prior_state or {}).get("token_usage", {})
    state = prior_state or {
        "schema_version": 1,
        "status": "initializing",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "next_step": 0,
        "current_prompt": ORIGINAL_GSM8K_SYSTEM_PROMPT,
        "accepted_val_accuracy": None,
        "accepted_val_records": [],
        "initial_test": None,
        "history": [],
        "pending": None,
        "token_usage": persisted_usage,
    }

    prompt_variable = tg.Variable(
        state["current_prompt"],
        requires_grad=True,
        role_description=(
            "structured system prompt to a somewhat capable language model that "
            "specifies the behavior and strategies for the QA task"
        ),
    )
    model = tg.BlackboxLLM(forward_engine, prompt_variable)
    objective = StringBasedFunction(
        string_based_equality_fn,
        function_purpose=(
            "The runtime of string-based function that checks if the prediction "
            "is correct."
        ),
    )
    optimizer = tg.TextualGradientDescent(
        engine=backward_engine,
        parameters=[prompt_variable],
    )

    def current_usage():
        return merge_token_usage(
            persisted_usage, forward_tracker.snapshot(), backward_tracker.snapshot()
        )

    def save_state() -> None:
        state["updated_at_utc"] = utc_now()
        state["token_usage"] = current_usage()
        _atomic_json(state_path, state)

    if state["initial_test"] is None:
        print(f"initial test start ({len(test)} examples)", flush=True)
        initial_test = evaluate_cases(
            test,
            prompt=prompt_variable.value,
            engine=forward_engine,
            workers=args.workers,
            label="initial-test",
        )
        state["initial_test"] = {
            "accuracy": _accuracy(initial_test),
            "records": initial_test,
        }
        save_state()
    if state["accepted_val_accuracy"] is None:
        print(f"initial validation start ({len(val)} examples)", flush=True)
        initial_val = evaluate_cases(
            val,
            prompt=prompt_variable.value,
            engine=forward_engine,
            workers=args.workers,
            label="initial-val",
        )
        state["accepted_val_accuracy"] = _accuracy(initial_val)
        state["accepted_val_records"] = initial_val
        state["status"] = "training"
        save_state()

    def finish_pending() -> None:
        pending = state["pending"]
        if pending is None:
            return
        step = int(pending["step"])
        if pending["phase"] == "validation":
            prompt_variable.set_value(pending["candidate_prompt"])
            candidate_val = evaluate_cases(
                val,
                prompt=pending["candidate_prompt"],
                engine=forward_engine,
                workers=args.workers,
                label=f"step-{step}-val",
            )
            candidate_accuracy = _accuracy(candidate_val)
            previous_accuracy = float(pending["previous_val_accuracy"])
            accepted = candidate_accuracy >= previous_accuracy
            if accepted:
                state["current_prompt"] = pending["candidate_prompt"]
                state["accepted_val_accuracy"] = candidate_accuracy
                state["accepted_val_records"] = candidate_val
                decision = "improved" if candidate_accuracy > previous_accuracy else "tied"
            else:
                state["current_prompt"] = pending["previous_prompt"]
                prompt_variable.set_value(pending["previous_prompt"])
                decision = "rejected"
            pending.update(
                {
                    "candidate_val_accuracy": candidate_accuracy,
                    "accepted_val_accuracy": state["accepted_val_accuracy"],
                    "accepted": accepted,
                    "decision": decision,
                    "phase": "test",
                }
            )
            state["pending"] = pending
            save_state()
            print(
                f"step {step} {decision}: val {previous_accuracy:.6f} -> "
                f"{candidate_accuracy:.6f}; kept={state['accepted_val_accuracy']:.6f}",
                flush=True,
            )

        pending = state["pending"]
        if pending["phase"] == "test":
            prompt_variable.set_value(state["current_prompt"])
            test_records = evaluate_cases(
                test,
                prompt=state["current_prompt"],
                engine=forward_engine,
                workers=args.workers,
                label=f"step-{step}-test",
            )
            pending["test_accuracy"] = _accuracy(test_records)
            pending["test_records"] = test_records
            pending.pop("phase", None)
            state["history"].append(pending)
            state["pending"] = None
            state["next_step"] = step
            state["status"] = "training"
            save_state()
            print(
                f"step {step}/{len(schedule)} test={_accuracy(test_records):.6f} "
                f"prompt_chars={len(state['current_prompt'])}",
                flush=True,
            )

    if state["pending"] is not None:
        finish_pending()

    print(
        f"original TextGrad start step={state['next_step']}/{len(schedule)} "
        f"batch_size={args.batch_size} epochs={args.max_epochs} "
        f"constraints=0 backward_cap=None",
        flush=True,
    )
    invocation_steps = 0
    for step_index in range(int(state["next_step"]), len(schedule)):
        if args.stop_after_step is not None and invocation_steps >= args.stop_after_step:
            print(f"intentional stop after {invocation_steps} new step(s)", flush=True)
            return
        offsets = schedule[step_index]
        batch = [train[offset] for offset in offsets]
        previous_prompt = prompt_variable.value
        optimizer.zero_grad()
        losses = []
        train_records = []
        print(
            f"step {step_index + 1}/{len(schedule)} epoch="
            f"{step_index // args.steps_per_epoch + 1} "
            f"batch={step_index % args.steps_per_epoch + 1} "
            f"ids={[case.wrong_id for case in batch]}",
            flush=True,
        )
        for case in batch:
            question = tg.Variable(
                case.question,
                requires_grad=False,
                role_description="query to the language model",
            )
            answer = tg.Variable(
                case.ground_truth,
                requires_grad=False,
                role_description="correct answer for the query",
            )
            response = model(question)
            train_records.append(
                {
                    "wrong_id": case.wrong_id,
                    "question": case.question,
                    "ground_truth": case.ground_truth,
                    "generator_output": response.value,
                    "correct": is_correct(response.value, case.ground_truth),
                }
            )
            losses.append(
                objective(
                    inputs={"prediction": response, "ground_truth_answer": answer}
                )
            )
        tg.sum(losses).backward()
        gradient_text = prompt_variable.get_gradient_text()
        optimizer.step()
        candidate_prompt = prompt_variable.value
        state["current_prompt"] = candidate_prompt
        state["pending"] = {
            "step": step_index + 1,
            "epoch": step_index // args.steps_per_epoch + 1,
            "step_in_epoch": step_index % args.steps_per_epoch + 1,
            "batch_train_offsets": offsets,
            "batch_wrong_ids": [case.wrong_id for case in batch],
            "train_records": train_records,
            "previous_prompt": previous_prompt,
            "candidate_prompt": candidate_prompt,
            "previous_val_accuracy": state["accepted_val_accuracy"],
            "gradient_text": gradient_text,
            "phase": "validation",
        }
        state["status"] = "validating"
        save_state()
        finish_pending()
        invocation_steps += 1

    final_step = state["history"][-1]
    summary = {
        **config,
        "created_at_utc": state["created_at_utc"],
        "completed_at_utc": utc_now(),
        "complete": True,
        "initial_test_accuracy": state["initial_test"]["accuracy"],
        "initial_val_accuracy": state["history"][0]["previous_val_accuracy"],
        "final_test_accuracy": final_step["test_accuracy"],
        "final_val_accuracy": state["accepted_val_accuracy"],
        "accepted_updates": sum(item["accepted"] for item in state["history"]),
        "rejected_updates": sum(not item["accepted"] for item in state["history"]),
        "tie_updates": sum(item["decision"] == "tied" for item in state["history"]),
        "final_prompt": state["current_prompt"],
        "final_prompt_sha256": hashlib.sha256(
            state["current_prompt"].encode("utf-8")
        ).hexdigest(),
        "token_usage": current_usage(),
        "state": str(state_path.resolve()),
        "textgrad_calls": str(trace_path.resolve()),
    }
    _atomic_json(summary_path, summary)
    state["status"] = "complete"
    state["token_usage"] = summary["token_usage"]
    save_state()
    print(
        f"original TextGrad complete val={summary['final_val_accuracy']:.6f} "
        f"test={summary['final_test_accuracy']:.6f} output={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    run()
