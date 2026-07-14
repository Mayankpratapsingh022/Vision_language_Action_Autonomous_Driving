from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PAIR = re.compile(r"\b(step|loss|grdn|lr|gpu_mem_gb|mem_gb|samples_per_s|smp/s):\s*([-+0-9.eE]+)")
EVAL = re.compile(r"\bstep\s+(\d+):\s+eval_loss=([-+0-9.eE]+)")


def parse_metric_line(line: str) -> dict[str, float | int] | None:
    clean = ANSI_ESCAPE.sub("", line)
    eval_match = EVAL.search(clean)
    if eval_match:
        return {"step": int(eval_match.group(1)), "eval_loss": float(eval_match.group(2))}
    values: dict[str, float | int] = {}
    for name, raw in PAIR.findall(clean):
        normalized = {"grdn": "gradient_norm", "mem_gb": "gpu_mem_gb", "smp/s": "samples_per_s"}.get(
            name, name
        )
        values[normalized] = int(float(raw)) if normalized == "step" else float(raw)
    return values if "step" in values and ("loss" in values or "eval_loss" in values) else None


def append_metric(path: str | Path, event: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]


def write_training_plot(events: Iterable[dict[str, Any]], output_path: str | Path) -> bool:
    events = list(events)
    train = [event for event in events if "loss" in event]
    evaluation = [event for event in events if "eval_loss" in event]
    if not train and not evaluation:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    if train:
        axis.plot([event["step"] for event in train], [event["loss"] for event in train], label="train loss")
    if evaluation:
        axis.plot(
            [event["step"] for event in evaluation],
            [event["eval_loss"] for event in evaluation],
            marker="o",
            label="held-out loss",
        )
    axis.set(title="SmolVLA left-turn fine-tuning", xlabel="optimizer step", ylabel="flow-matching loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return True


__all__ = ["append_metric", "load_metrics", "parse_metric_line", "write_training_plot"]
