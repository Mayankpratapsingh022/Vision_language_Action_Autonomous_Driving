from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from left_turn_vla.commands import build_train_command, find_resume_config, latest_pretrained_model  # noqa: E402
from left_turn_vla.config import TrainConfig  # noqa: E402
from left_turn_vla.env import load_env_file  # noqa: E402
from left_turn_vla.training_metrics import (  # noqa: E402
    append_metric,
    load_metrics,
    parse_metric_line,
    write_training_plot,
)

DEFAULT_CONFIG = PROJECT_DIR / "configs" / "left_turn.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA for the simulator protected left turn.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-name")
    parser.add_argument("--dataset-repo")
    parser.add_argument("--model-repo")
    parser.add_argument("--steps", "--max-steps", dest="steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--resume", choices=("auto", "never", "require"), default="auto")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_env_file(PROJECT_DIR / ".env")
    args = parse_args()
    base = TrainConfig.from_json(args.config)
    config = base.with_overrides(
        run_name=args.run_name,
        dataset_repo=args.dataset_repo or os.environ.get("HF_DATASET_REPO"),
        model_repo=args.model_repo or os.environ.get("HF_MODEL_REPO"),
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        push_to_hub=args.push_to_hub,
    )

    default_workspace = Path(os.environ.get("VLA_WORKSPACE_ROOT", PROJECT_DIR / "outputs"))
    output_root = Path(args.output_root).expanduser() if args.output_root else default_workspace / "runs"
    output_dir = output_root.resolve() / config.run_name
    artifacts_dir = output_dir / "run_artifacts"
    log_dir = default_workspace.resolve() / "logs"
    log_path = log_dir / f"{config.run_name}.log"
    metrics_path = artifacts_dir / "training_metrics.jsonl"
    resume_config = find_resume_config(output_dir)

    if args.resume == "never":
        resume_config = None
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Run directory already contains data: {output_dir}")
    elif args.resume == "require" and resume_config is None:
        raise FileNotFoundError(f"No resumable checkpoint found in {output_dir}")

    executable = shutil.which("lerobot-train") or str(PROJECT_DIR / ".venv" / "bin" / "lerobot-train")
    wandb_enabled = args.wandb if args.wandb is not None else bool(os.environ.get("WANDB_API_KEY"))
    command = build_train_command(
        config,
        executable=executable,
        output_dir=output_dir,
        resume_config=resume_config,
        wandb_enabled=wandb_enabled,
        wandb_project=os.environ.get("WANDB_PROJECT", "urban-left-turn-vla"),
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": config.to_dict(),
        "output_dir": str(output_dir),
        "resume_config": str(resume_config) if resume_config else None,
        "wandb_enabled": wandb_enabled,
        "command": command,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        print("\n" + shlex.join(command), flush=True)
        return

    if not Path(executable).exists() and shutil.which(executable) is None:
        raise FileNotFoundError("lerobot-train is unavailable. Run `python -m pip install -e .` first.")
    if config.push_to_hub and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required because push_to_hub is enabled")
    _verify_cuda()

    log_dir.mkdir(parents=True, exist_ok=True)
    runtime_env = _runtime_environment(default_workspace.resolve())

    with log_path.open("a", buffering=1) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            env=runtime_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            event = parse_metric_line(line)
            if event is not None:
                append_metric(metrics_path, event)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"SmolVLA training exited with code {return_code}. See {log_path}")

    # LeRobot rejects a fresh run when its output directory already exists. Keep
    # wrapper artifacts out of that directory until LeRobot has created it.
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_training_plot(load_metrics(metrics_path), artifacts_dir / "training_curves.png")
    model_path = latest_pretrained_model(output_dir)
    if not args.skip_eval:
        _run_evaluation(config, model_path, artifacts_dir, runtime_env)
    if config.push_to_hub:
        _upload_run_artifacts(config, artifacts_dir, log_path)
    print(f"Training completed. Local checkpoint: {model_path}", flush=True)
    if config.push_to_hub:
        print(f"Published model: https://huggingface.co/{config.model_repo}", flush=True)


def _verify_cuda() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not installed") from error
    if not torch.cuda.is_available():
        raise RuntimeError("SmolVLA fine-tuning requires a CUDA GPU")
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "bf16": torch.cuda.is_bf16_supported(),
            }
        ),
        flush=True,
    )


def _runtime_environment(workspace: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("HF_HOME", str(workspace / "cache" / "huggingface"))
    environment.setdefault("HF_LEROBOT_HOME", str(workspace / "cache" / "lerobot"))
    environment.setdefault("TORCH_HOME", str(workspace / "cache" / "torch"))
    environment.setdefault("MPLCONFIGDIR", str(workspace / "cache" / "matplotlib"))
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    for key in ("HF_HOME", "HF_LEROBOT_HOME", "TORCH_HOME", "MPLCONFIGDIR"):
        Path(environment[key]).mkdir(parents=True, exist_ok=True)
    return environment


def _run_evaluation(
    config: TrainConfig,
    model_path: Path,
    artifacts_dir: Path,
    environment: dict[str, str],
) -> None:
    command = [
        sys.executable,
        str(PROJECT_DIR / "evaluate.py"),
        f"--model-path={model_path}",
        f"--dataset-repo={config.dataset_repo}",
        f"--output-dir={artifacts_dir / 'evaluation'}",
        f"--eval-split={config.eval_split}",
        "--device=cuda",
    ]
    subprocess.run(command, cwd=PROJECT_DIR, env=environment, check=True)


def _upload_run_artifacts(config: TrainConfig, artifacts_dir: Path, log_path: Path) -> None:
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.upload_folder(
            repo_id=config.model_repo,
            repo_type="model",
            folder_path=artifacts_dir,
            path_in_repo=f"runs/{config.run_name}",
            commit_message=f"Add {config.run_name} metrics and evaluation",
        )
        api.upload_file(
            repo_id=config.model_repo,
            repo_type="model",
            path_or_fileobj=log_path,
            path_in_repo=f"runs/{config.run_name}/training.log",
            commit_message=f"Add {config.run_name} training log",
        )
    except Exception as error:  # The trained checkpoint remains valid even if the auxiliary upload fails.
        print(f"Warning: model trained, but run-artifact upload failed: {error}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
