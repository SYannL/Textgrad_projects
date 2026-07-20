"""Add experiment progress as the leading fields of TextGrad JSONL logs."""

import json
import logging
import threading


class _Progress:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current = 0
        self.total = 0
        self.line_current = 0
        self.line_total = 1
        self.epoch = "1/1"

    def set(
        self,
        current: int,
        total: int,
        line_total: int,
        epoch: str = "1/1",
    ) -> None:
        with self.lock:
            self.current = current
            self.total = total
            self.line_current = 0
            self.line_total = line_total
            self.epoch = epoch

    def advance(self):
        with self.lock:
            self.line_current += 1
            progress = (
                f"{self.current}/{self.total} "
                f"{self.line_current}/{self.line_total}"
            )
            return progress, self.epoch


_PROGRESS = _Progress()


class ProgressJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        super().format(record)
        snapshot_attribute = "_adversarial_gv_progress_snapshot"
        snapshot = getattr(record, snapshot_attribute, None)
        if snapshot is None:
            snapshot = _PROGRESS.advance()
            setattr(record, snapshot_attribute, snapshot)
        progress, epoch = snapshot
        output = {
            "progress": progress,
            "epoch": epoch,
            **{
                key: str(value)
                for key, value in record.__dict__.items()
                if key != snapshot_attribute
            },
        }
        return json.dumps(output)


def configure_textgrad_progress_logging() -> None:
    from textgrad import logger

    formatter = ProgressJsonFormatter()
    for handler in logger.handlers:
        handler.setFormatter(formatter)


def set_log_progress(
    current: int,
    total: int,
    line_total: int,
    epoch: str = "1/1",
) -> None:
    if total < 1 or not 0 <= current <= total:
        raise ValueError(f"invalid progress {current}/{total}")
    if line_total < 1:
        raise ValueError("line total must be positive")
    _PROGRESS.set(current, total, line_total, epoch)
