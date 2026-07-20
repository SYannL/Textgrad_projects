#!/usr/bin/env python
"""Run strict alternating G/V prompt optimization on a GSM8K CSV split."""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import textgrad as tg
from dotenv import load_dotenv

from adversarial_gv.agents import GeneratorAgent, VerifierAgent
from adversarial_gv.batch_trainer import BatchAdversarialGVTrainer, HardCase
from adversarial_gv.engines import (
    TokenUsageTracker,
    build_engine,
    merge_token_usage,
    resolve_role_base_urls,
)
from adversarial_gv.prompts import (
    GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
    GSM8K_GENERATOR_STRATEGY_PROMPT,
    VERIFIER_FIXED_SYSTEM_PROMPT,
    VERIFIER_STRATEGY_PROMPT,
)
from adversarial_gv.progress_logging import (
    configure_textgrad_progress_logging,
    set_log_progress,
)
from adversarial_gv.progress_planning import measure_progress_line_totals
from adversarial_gv.recording import RecordingEngine
from adversarial_gv.wandb_monitoring import init_wandb_monitor


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--data", default="runs/gsm8k_initial_wrong/wrong_samples.csv")
    result.add_argument("--output-dir", default="runs/hardset_main")
    result.add_argument("--batch-size", type=int, default=5)
    result.add_argument("--generator-model", default="gpt-4o-mini")
    result.add_argument("--verifier-model", default="gpt-4o-mini")
    result.add_argument("--backward-model", default="experimental:gpt-5-mini")
    result.add_argument(
        "--vllm-base-url",
        default=os.getenv("VLLM_BASE_URL"),
        help="Legacy shared vLLM endpoint used as a fallback for every role.",
    )
    result.add_argument(
        "--generator-vllm-base-url",
        default=os.getenv("GENERATOR_VLLM_BASE_URL"),
        help="Generator vLLM endpoint; overrides --vllm-base-url.",
    )
    result.add_argument(
        "--verifier-vllm-base-url",
        default=os.getenv("VERIFIER_VLLM_BASE_URL"),
        help="Verifier vLLM endpoint; overrides --vllm-base-url.",
    )
    result.add_argument(
        "--backward-vllm-base-url",
        default=os.getenv("BACKWARD_VLLM_BASE_URL"),
        help=(
            "TextGrad backward, CoT labeler, and optimizer endpoint; "
            "overrides --vllm-base-url."
        ),
    )
    result.add_argument("--vllm-api-key", default=os.getenv("VLLM_API_KEY"))
    result.add_argument(
        "--vllm-enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode; disabled by default for concise TextGrad outputs.",
    )
    result.add_argument("--evaluation-workers", type=int, default=8)
    result.add_argument("--training-workers", type=int, default=1)
    result.add_argument(
        "--wandb-entity",
        default=os.getenv("WANDB_ENTITY", "siyann"),
    )
    result.add_argument(
        "--wandb-project",
        default=os.getenv("WANDB_PROJECT", "GANAgent"),
    )
    result.add_argument("--wandb-name", default=None)
    result.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.getenv("WANDB_MODE", "online"),
    )
    return result


def load_cases(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cases = [
        HardCase(
            wrong_id=int(row["wrong_id"]),
            collection_split=row["collection_split"],
            source_split=row["split"],
            source_index=int(row["index"]),
            question=row["question"],
            ground_truth=row["ground_truth"],
        )
        for row in rows
    ]
    train = [case for case in cases if case.collection_split == "train"]
    val = [case for case in cases if case.collection_split == "val"]
    test = [case for case in cases if case.collection_split == "test"]
    if not train or not val or not test:
        raise ValueError(
            "expected non-empty train, val, and test splits, got "
            f"{len(train)}, {len(val)}, and {len(test)}"
        )
    return train, val, test


def save_json(path: Path, value: Dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def export_metrics(path: Path, experiment: Dict) -> None:
    rows = []
    for item in experiment["evaluations"]:
        rows.append(
            {
                "stage": item["stage"],
                "batch_index": item["batch_index"],
                "train_accuracy": item["train"]["accuracy"],
                "train_accept_rate": item["train"]["accept_rate"],
                "train_challenge_rate": item["train"]["challenge_rate"],
                "train_reject_rate": item["train"]["reject_rate"],
                "train_invalid_rate": item["train"]["invalid_rate"],
                "train_evaluated_count": item["train"].get("evaluated_count", 0),
                "train_skipped_count": item["train"].get("skipped_count", 0),
                "val_accuracy": item["val"]["accuracy"],
                "val_accept_rate": item["val"]["accept_rate"],
                "val_challenge_rate": item["val"]["challenge_rate"],
                "val_reject_rate": item["val"]["reject_rate"],
                "val_invalid_rate": item["val"]["invalid_rate"],
                "val_evaluated_count": item["val"].get("evaluated_count", 0),
                "val_skipped_count": item["val"].get("skipped_count", 0),
                "test_accuracy": item["test"]["accuracy"],
                "test_accept_rate": item["test"]["accept_rate"],
                "test_challenge_rate": item["test"]["challenge_rate"],
                "test_reject_rate": item["test"]["reject_rate"],
                "test_invalid_rate": item["test"]["invalid_rate"],
                "test_evaluated_count": item["test"].get("evaluated_count", 0),
                "test_skipped_count": item["test"].get("skipped_count", 0),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_accuracy_plot(path: Path, experiment: Dict) -> None:
    """Export an English-labelled train/validation accuracy plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip accuracy plot: {exc}", flush=True)
        return

    evaluations = experiment["evaluations"]
    x_values = [item["batch_index"] for item in evaluations]
    x_labels = ["Initial" if item["batch_index"] == 0 else f"B{item['batch_index']}" for item in evaluations]
    train = [item["train"]["accuracy"] for item in evaluations]
    val = [item["val"]["accuracy"] for item in evaluations]

    plt.figure(figsize=(8, 5), dpi=180)
    plt.plot(x_values, train, marker="o", linewidth=2.5, label="Train Accuracy", color="#2563eb")
    plt.plot(x_values, val, marker="o", linewidth=2.5, label="Validation Accuracy", color="#dc2626")
    plt.xticks(x_values, x_labels)
    upper = max(0.8, min(1.0, max(train + val + [0.0]) + 0.1))
    plt.ylim(0, upper)
    plt.grid(True, alpha=0.28)
    plt.title("GAN-like G/V Prompt Optimization Accuracy", fontsize=13)
    plt.xlabel("Training Progress", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.legend()
    for x, y in zip(x_values, train):
        plt.text(x, y + 0.025, f"{y:.3f}", ha="center", fontsize=8, color="#1e40af")
    for x, y in zip(x_values, val):
        offset = -0.045 if y > 0.08 else 0.025
        plt.text(x, y + offset, f"{y:.3f}", ha="center", fontsize=8, color="#991b1b")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def export_gradients(path: Path, experiment: Dict) -> None:
    fields = [
        "batch_index",
        "stage",
        "call_index",
        "kind",
        "system_prompt",
        "gradient_prompt",
        "response",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for batch in experiment["batches"]:
            for stage, trace in (
                ("V", batch["v_step"]["gradient_trace"]),
                ("G", batch["g_step"]["gradient_trace"]),
            ):
                for index, call in enumerate(trace, start=1):
                    writer.writerow(
                        {
                            "batch_index": batch["batch_index"],
                            "stage": stage,
                            "call_index": index,
                            "kind": call["kind"],
                            "system_prompt": call["system_prompt"],
                            "gradient_prompt": call["prompt"],
                            "response": call["response"],
                        }
                    )


def main(argv: Sequence[str] = None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    args = parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    role_base_urls = resolve_role_base_urls(
        shared=args.vllm_base_url,
        generator=args.generator_vllm_base_url,
        verifier=args.verifier_vllm_base_url,
        backward=args.backward_vllm_base_url,
    )
    train, val, test = load_cases(Path(args.data))
    batches = [
        train[i : i + args.batch_size]
        for i in range(0, len(train), args.batch_size)
    ]
    total_batches = len(batches)
    initial_line_total, batch_line_totals = measure_progress_line_totals(
        batches,
        (train, val, test),
    )
    configure_textgrad_progress_logging()
    set_log_progress(
        0,
        total_batches,
        initial_line_total,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "experiment.json"
    prompt_architecture = "fixed-system-plus-trainable-strategy-three-verdict-v2"

    if checkpoint.exists():
        experiment = json.loads(checkpoint.read_text(encoding="utf-8"))
        if experiment.get("prompt_architecture") != prompt_architecture:
            raise ValueError(
                "Existing checkpoint uses the old monolithic prompt architecture. "
                "Use a new --output-dir for fixed system + trainable strategy prompts."
            )
        if (
            experiment.get("generator_fixed_system_prompt")
            != GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT
            or experiment.get("verifier_fixed_system_prompt")
            != VERIFIER_FIXED_SYSTEM_PROMPT
        ):
            raise ValueError(
                "Checkpoint fixed system prompts differ from the current immutable "
                "prompts. Use a new --output-dir."
            )
        expected_models = {
            "generator": args.generator_model,
            "verifier": args.verifier_model,
            "backward": args.backward_model,
        }
        if experiment.get("models") != expected_models:
            raise ValueError(
                "Checkpoint model roles differ from the requested Generator, "
                "Verifier, or backward models. Use a new --output-dir."
            )
        stored_config = experiment.get("config", {})
        stored_role_base_urls = stored_config.get("vllm_base_urls")
        if stored_role_base_urls is None:
            stored_role_base_urls = resolve_role_base_urls(
                shared=stored_config.get("vllm_base_url")
            )
        if stored_role_base_urls != role_base_urls:
            raise ValueError(
                "Checkpoint vLLM role endpoints differ from the requested "
                "Generator, Verifier, or backward endpoints. Use a new --output-dir."
            )
        g_strategy_value = experiment["current_generator_strategy_prompt"]
        v_strategy_value = experiment["current_verifier_strategy_prompt"]
    else:
        experiment = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "prompt_architecture": prompt_architecture,
            "models": {
                "generator": args.generator_model,
                "verifier": args.verifier_model,
                "backward": args.backward_model,
            },
            "config": {
                "epochs": 1,
                "total_batches": total_batches,
                "batch_size": args.batch_size,
                "evaluation_workers": args.evaluation_workers,
                "training_workers": args.training_workers,
                "backend": (
                    "role-specific-vllm"
                    if any(role_base_urls.values())
                    else "textgrad-default"
                ),
                "vllm_base_urls": role_base_urls,
                "vllm_enable_thinking": args.vllm_enable_thinking,
            },
            "generator_fixed_system_prompt": GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
            "verifier_fixed_system_prompt": VERIFIER_FIXED_SYSTEM_PROMPT,
            "initial_generator_strategy_prompt": GSM8K_GENERATOR_STRATEGY_PROMPT,
            "initial_verifier_strategy_prompt": VERIFIER_STRATEGY_PROMPT,
            "current_generator_strategy_prompt": GSM8K_GENERATOR_STRATEGY_PROMPT,
            "current_verifier_strategy_prompt": VERIFIER_STRATEGY_PROMPT,
            "evaluations": [],
            "batches": [],
            "complete": False,
        }
        g_strategy_value = GSM8K_GENERATOR_STRATEGY_PROMPT
        v_strategy_value = VERIFIER_STRATEGY_PROMPT

    existing_wandb = experiment.get("wandb", {})
    wandb_project = existing_wandb.get("project", args.wandb_project)
    wandb_entity = existing_wandb.get("entity", args.wandb_entity)
    wandb_name = existing_wandb.get(
        "name",
        args.wandb_name or output_dir.name,
    )
    wandb_run_id = existing_wandb.get("run_id")
    wandb_monitor = init_wandb_monitor(
        entity=wandb_entity,
        project=wandb_project,
        name=wandb_name,
        mode=args.wandb_mode,
        output_dir=output_dir,
        run_id=wandb_run_id,
        config={
            "architecture": "GAN-like Generator-Verifier TextGrad",
            "dataset": "GSM8K balanced 1:1",
            "data_path": str(Path(args.data).resolve()),
            "epochs": 1,
            "batch_size": args.batch_size,
            "total_batches": total_batches,
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "generator_model": args.generator_model,
            "verifier_model": args.verifier_model,
            "backward_model": args.backward_model,
            "prompt_architecture": prompt_architecture,
            "vllm_base_urls": role_base_urls,
            "vllm_enable_thinking": args.vllm_enable_thinking,
        },
    )
    if wandb_monitor is not None:
        if not wandb_run_id:
            experiment["wandb"] = {
                "run_id": wandb_monitor.id,
                "entity": wandb_entity,
                "project": wandb_project,
                "name": wandb_name,
                "url": wandb_monitor.url,
            }
            save_json(checkpoint, experiment)
            for historical_evaluation in experiment["evaluations"]:
                wandb_monitor.log_evaluation(historical_evaluation)
            if experiment.get("token_usage") and experiment["evaluations"]:
                wandb_monitor.log_tokens(
                    experiment["evaluations"][-1]["batch_index"],
                    "resume_state",
                    experiment["token_usage"],
                )
        print(
            f"wandb run={wandb_monitor.id} url={wandb_monitor.url}",
            flush=True,
        )

    g_fixed = tg.Variable(
        GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
        requires_grad=False,
        role_description="immutable Generator role, rules, and output format",
    )
    v_fixed = tg.Variable(
        VERIFIER_FIXED_SYSTEM_PROMPT,
        requires_grad=False,
        role_description="immutable Verifier role, audit rules, and output format",
    )
    g_strategy = tg.Variable(
        g_strategy_value,
        requires_grad=True,
        role_description="trainable GSM8K Generator problem-solving strategy",
    )
    v_strategy = tg.Variable(
        v_strategy_value,
        requires_grad=True,
        role_description="trainable Verifier trajectory-audit strategy",
    )
    shared_engine_options = {
        "vllm_api_key": args.vllm_api_key,
        "vllm_enable_thinking": args.vllm_enable_thinking,
    }
    token_usage_base = experiment.get("token_usage", {})
    token_tracker = TokenUsageTracker()

    def current_token_usage():
        return merge_token_usage(token_usage_base, token_tracker.snapshot())

    backward = RecordingEngine(
        build_engine(
            args.backward_model,
            vllm_base_url=role_base_urls["backward"],
            **shared_engine_options,
            token_usage_tracker=token_tracker,
            usage_role="backward",
        )
    )
    trainer = BatchAdversarialGVTrainer(
        GeneratorAgent(
            build_engine(
                args.generator_model,
                vllm_base_url=role_base_urls["generator"],
                **shared_engine_options,
                token_usage_tracker=token_tracker,
                usage_role="generator",
            ),
            g_strategy,
            g_fixed,
        ),
        VerifierAgent(
            build_engine(
                args.verifier_model,
                vllm_base_url=role_base_urls["verifier"],
                **shared_engine_options,
                token_usage_tracker=token_tracker,
                usage_role="verifier",
            ),
            v_strategy,
            v_fixed,
        ),
        backward,
        evaluation_workers=args.evaluation_workers,
        training_workers=args.training_workers,
    )

    if not experiment["evaluations"]:
        initial = {
            "stage": "initial",
            "batch_index": 0,
            "train": trainer.evaluate(train),
            "val": trainer.evaluate(val),
            "test": trainer.evaluate(test),
        }
        experiment["evaluations"].append(initial)
        experiment["token_usage"] = current_token_usage()
        save_json(checkpoint, experiment)
        if wandb_monitor is not None:
            wandb_monitor.log_evaluation(
                initial,
                experiment["token_usage"],
                len(g_strategy.value),
                len(v_strategy.value),
            )
        print(
            f"initial train={initial['train']['accuracy']:.3f} "
            f"val={initial['val']['accuracy']:.3f} "
            f"test={initial['test']['accuracy']:.3f} "
            f"skipped={initial['train'].get('skipped_count', 0)}/"
            f"{initial['val'].get('skipped_count', 0)}/"
            f"{initial['test'].get('skipped_count', 0)}",
            flush=True,
        )

    completed = len(experiment["batches"])
    for batch_index, batch in enumerate(batches, start=1):
        if batch_index <= completed:
            continue
        set_log_progress(
            batch_index,
            total_batches,
            batch_line_totals[batch_index - 1],
        )
        print(f"starting batch={batch_index}/{len(batches)} cases={[case.wrong_id for case in batch]}", flush=True)
        record = trainer.train_batch(batch, batch_index)
        experiment["token_usage"] = current_token_usage()
        save_json(checkpoint, experiment)
        if wandb_monitor is not None:
            wandb_monitor.log_tokens(
                batch_index,
                "training_complete",
                experiment["token_usage"],
            )
        evaluation = {
            "stage": "after_batch",
            "batch_index": batch_index,
            "train": trainer.evaluate(train),
            "val": trainer.evaluate(val),
            "test": trainer.evaluate(test),
        }
        experiment["batches"].append(record)
        experiment["evaluations"].append(evaluation)
        experiment["current_generator_strategy_prompt"] = g_strategy.value
        experiment["current_verifier_strategy_prompt"] = v_strategy.value
        experiment["token_usage"] = current_token_usage()
        save_json(checkpoint, experiment)
        export_metrics(output_dir / "metrics.csv", experiment)
        export_accuracy_plot(output_dir / "accuracy.png", experiment)
        export_gradients(output_dir / "gradient_traces.csv", experiment)
        if wandb_monitor is not None:
            wandb_monitor.log_evaluation(
                evaluation,
                experiment["token_usage"],
                len(g_strategy.value),
                len(v_strategy.value),
            )
        print(
            f"batch={batch_index}/{len(batches)} "
            f"train={evaluation['train']['accuracy']:.3f} "
            f"val={evaluation['val']['accuracy']:.3f} "
            f"test={evaluation['test']['accuracy']:.3f} "
            f"skipped={evaluation['train'].get('skipped_count', 0)}/"
            f"{evaluation['val'].get('skipped_count', 0)}/"
            f"{evaluation['test'].get('skipped_count', 0)}",
            flush=True,
        )

    experiment["complete"] = True
    experiment["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(checkpoint, experiment)
    export_metrics(output_dir / "metrics.csv", experiment)
    export_accuracy_plot(output_dir / "accuracy.png", experiment)
    export_gradients(output_dir / "gradient_traces.csv", experiment)
    if wandb_monitor is not None:
        wandb_monitor.finish()
    print(f"complete: {checkpoint}")


if __name__ == "__main__":
    main()
