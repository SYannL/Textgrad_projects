#!/usr/bin/env python
"""Run strict alternating G/V prompt optimization on a GSM8K CSV split."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import textgrad as tg
from dotenv import load_dotenv

from adversarial_gv.agents import GeneratorAgent, VerifierAgent
from adversarial_gv.batch_trainer import BatchAdversarialGVTrainer, HardCase
from adversarial_gv.prompts import GSM8K_GENERATOR_PROMPT, VERIFIER_PROMPT
from adversarial_gv.recording import RecordingEngine


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--data", default="runs/gsm8k_initial_wrong/wrong_samples.csv")
    result.add_argument("--output-dir", default="runs/hardset_main")
    result.add_argument("--batch-size", type=int, default=5)
    result.add_argument("--generator-model", default="gpt-4o-mini")
    result.add_argument("--verifier-model", default="gpt-4o-mini")
    result.add_argument("--backward-model", default="experimental:gpt-5-mini")
    result.add_argument("--evaluation-workers", type=int, default=8)
    result.add_argument("--training-workers", type=int, default=1)
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
    if not train or not val:
        raise ValueError(f"expected non-empty train and val splits, got {len(train)} and {len(val)}")
    return train, val


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
                "val_accuracy": item["val"]["accuracy"],
                "val_accept_rate": item["val"]["accept_rate"],
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
    train, val = load_cases(Path(args.data))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "experiment.json"

    if checkpoint.exists():
        experiment = json.loads(checkpoint.read_text(encoding="utf-8"))
        g_prompt_value = experiment["current_generator_prompt"]
        v_prompt_value = experiment["current_verifier_prompt"]
    else:
        experiment = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "models": {
                "generator": args.generator_model,
                "verifier": args.verifier_model,
                "backward": args.backward_model,
            },
            "config": {
                "epochs": 1,
                "batch_size": args.batch_size,
                "evaluation_workers": args.evaluation_workers,
                "training_workers": args.training_workers,
            },
            "initial_generator_prompt": GSM8K_GENERATOR_PROMPT,
            "initial_verifier_prompt": VERIFIER_PROMPT,
            "current_generator_prompt": GSM8K_GENERATOR_PROMPT,
            "current_verifier_prompt": VERIFIER_PROMPT,
            "evaluations": [],
            "batches": [],
            "complete": False,
        }
        g_prompt_value = GSM8K_GENERATOR_PROMPT
        v_prompt_value = VERIFIER_PROMPT

    g_prompt = tg.Variable(
        g_prompt_value,
        requires_grad=True,
        role_description="general GSM8K Generator system prompt",
    )
    v_prompt = tg.Variable(
        v_prompt_value,
        requires_grad=True,
        role_description="general reasoning Verifier system prompt",
    )
    backward = RecordingEngine(tg.get_engine(args.backward_model))
    trainer = BatchAdversarialGVTrainer(
        GeneratorAgent(tg.get_engine(args.generator_model), g_prompt),
        VerifierAgent(tg.get_engine(args.verifier_model), v_prompt),
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
        }
        experiment["evaluations"].append(initial)
        save_json(checkpoint, experiment)
        print(
            f"initial train={initial['train']['accuracy']:.3f} "
            f"val={initial['val']['accuracy']:.3f}",
            flush=True,
        )

    completed = len(experiment["batches"])
    batches = [train[i : i + args.batch_size] for i in range(0, len(train), args.batch_size)]
    for batch_index, batch in enumerate(batches, start=1):
        if batch_index <= completed:
            continue
        print(f"starting batch={batch_index}/{len(batches)} cases={[case.wrong_id for case in batch]}", flush=True)
        record = trainer.train_batch(batch, batch_index)
        experiment["batches"].append(record)
        evaluation = {
            "stage": "after_batch",
            "batch_index": batch_index,
            "train": trainer.evaluate(train),
            "val": trainer.evaluate(val),
        }
        experiment["evaluations"].append(evaluation)
        experiment["current_generator_prompt"] = g_prompt.value
        experiment["current_verifier_prompt"] = v_prompt.value
        save_json(checkpoint, experiment)
        export_metrics(output_dir / "metrics.csv", experiment)
        export_accuracy_plot(output_dir / "accuracy.png", experiment)
        export_gradients(output_dir / "gradient_traces.csv", experiment)
        print(
            f"batch={batch_index}/{len(batches)} "
            f"train={evaluation['train']['accuracy']:.3f} "
            f"val={evaluation['val']['accuracy']:.3f}",
            flush=True,
        )

    experiment["complete"] = True
    experiment["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(checkpoint, experiment)
    export_metrics(output_dir / "metrics.csv", experiment)
    export_accuracy_plot(output_dir / "accuracy.png", experiment)
    export_gradients(output_dir / "gradient_traces.csv", experiment)
    print(f"complete: {checkpoint}")


if __name__ == "__main__":
    main()
