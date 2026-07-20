"""Optional Weights & Biases monitoring for the hard-set training run."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .engines import TOKEN_FIELDS


def token_metrics(token_usage: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    return {
        f"tokens/{role}/{field.removesuffix('_tokens')}": int(value)
        for role, values in (token_usage or {}).items()
        for field, value in values.items()
        if field in TOKEN_FIELDS
    }


@dataclass
class WandbMonitor:
    run: Any

    @property
    def id(self) -> str:
        return self.run.id

    @property
    def url(self) -> Optional[str]:
        return getattr(self.run, "url", None)

    def log_tokens(
        self,
        batch_index: int,
        phase: str,
        token_usage: Dict[str, Dict[str, int]],
    ) -> None:
        self.run.log(
            {
                "batch": batch_index,
                "phase": phase,
                **token_metrics(token_usage),
            }
        )

    def log_evaluation(
        self,
        evaluation: Dict[str, Any],
        token_usage: Optional[Dict[str, Dict[str, int]]] = None,
        generator_prompt_length: Optional[int] = None,
        verifier_prompt_length: Optional[int] = None,
    ) -> None:
        metrics: Dict[str, Any] = {
            "batch": evaluation["batch_index"],
            "phase": evaluation["stage"],
            "accuracy/train": evaluation["train"]["accuracy"],
            "accuracy/val": evaluation["val"]["accuracy"],
            "accept_rate/train": evaluation["train"]["accept_rate"],
            "accept_rate/val": evaluation["val"]["accept_rate"],
            "challenge_rate/train": evaluation["train"]["challenge_rate"],
            "challenge_rate/val": evaluation["val"]["challenge_rate"],
            "reject_rate/train": evaluation["train"]["reject_rate"],
            "reject_rate/val": evaluation["val"]["reject_rate"],
            "invalid_rate/train": evaluation["train"]["invalid_rate"],
            "invalid_rate/val": evaluation["val"]["invalid_rate"],
            "skipped_cases/train": evaluation["train"].get("skipped_count", 0),
            "skipped_cases/val": evaluation["val"].get("skipped_count", 0),
        }
        if token_usage is not None:
            metrics.update(token_metrics(token_usage))
        if generator_prompt_length is not None:
            metrics["prompt_chars/generator_strategy"] = generator_prompt_length
        if verifier_prompt_length is not None:
            metrics["prompt_chars/verifier_strategy"] = verifier_prompt_length
        self.run.log(metrics)

    def finish(self) -> None:
        self.run.finish()


def init_wandb_monitor(
    *,
    entity: str,
    project: str,
    name: str,
    mode: str,
    output_dir: Path,
    config: Dict[str, Any],
    run_id: Optional[str] = None,
) -> Optional[WandbMonitor]:
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B monitoring is enabled but wandb is not installed"
        ) from exc

    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        id=run_id,
        resume="allow" if run_id else None,
        mode=mode,
        dir=str(output_dir),
        config=config,
        save_code=True,
    )
    run.define_metric("batch")
    run.define_metric("accuracy/*", step_metric="batch")
    run.define_metric("accept_rate/*", step_metric="batch")
    run.define_metric("challenge_rate/*", step_metric="batch")
    run.define_metric("reject_rate/*", step_metric="batch")
    run.define_metric("invalid_rate/*", step_metric="batch")
    run.define_metric("skipped_cases/*", step_metric="batch")
    run.define_metric("tokens/*", step_metric="batch")
    run.define_metric("prompt_chars/*", step_metric="batch")
    return WandbMonitor(run)
