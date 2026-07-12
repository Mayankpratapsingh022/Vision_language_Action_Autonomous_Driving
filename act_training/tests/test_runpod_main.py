from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from runpod_main import (
    _validate_args,
    build_runtime_environment,
    build_training_command,
    load_env_file,
)


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "run_name": "act-driving-v1",
        "max_steps": 10_000,
        "batch_size": 64,
        "num_workers": 4,
        "resume": "auto",
        "timeout_hours": 12.0,
        "no_push_to_hub": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_manual_runpod_command_uses_persistent_workspace() -> None:
    workspace = Path("/workspace/act-driving")
    command = build_training_command(_args(), workspace)

    assert command[0]
    assert "runpod_train.py" in command[1]
    assert command[command.index("--batch-size") + 1] == "64"
    assert command[command.index("--max-steps") + 1] == "10000"
    assert command[command.index("--artifact-root") + 1] == "/workspace/act-driving/artifacts/runs"
    assert command[command.index("--cache-dir") + 1] == "/workspace/act-driving/cache/huggingface"


def test_runtime_environment_keeps_all_caches_under_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    environment = build_runtime_environment(Path("/workspace/act-driving"))

    assert environment["HF_TOKEN"] == "test-token"
    assert environment["HF_HOME"].startswith("/workspace/act-driving/")
    assert environment["TORCH_HOME"].startswith("/workspace/act-driving/")
    assert environment["MPLCONFIGDIR"].startswith("/workspace/act-driving/")
    assert environment["TOKENIZERS_PARALLELISM"] == "false"


def test_manual_dotenv_loader_does_not_override_pod_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=file-token\nACT_WORKSPACE_ROOT='/workspace/custom'\n")
    monkeypatch.setenv("HF_TOKEN", "pod-secret")
    monkeypatch.delenv("ACT_WORKSPACE_ROOT", raising=False)

    load_env_file(env_file)

    assert os.environ["HF_TOKEN"] == "pod-secret"
    assert os.environ["ACT_WORKSPACE_ROOT"] == "/workspace/custom"


def test_manual_entrypoint_rejects_invalid_resource_values() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        _validate_args(_args(batch_size=0))


def test_manual_entrypoint_can_disable_hub_upload() -> None:
    command = build_training_command(_args(no_push_to_hub=True), Path("/workspace/act-driving"))

    assert command[-1] == "--no-push-to-hub"
