from __future__ import annotations

import sys
from pathlib import Path

import train


def test_fresh_run_does_not_precreate_lerobot_output(monkeypatch, tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    output_dir = output_root / "fresh-run"
    fake_train = tmp_path / "fake_train.py"
    fake_train.write_text(
        "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                "output = Path(sys.argv[1])",
                "assert not output.exists(), 'wrapper created output before LeRobot'",
                "checkpoint = output / 'checkpoints' / 'last' / 'pretrained_model'",
                "checkpoint.mkdir(parents=True)",
                "(checkpoint / 'train_config.json').write_text('{}')",
                "print('step:1 loss:0.1')",
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(
        train,
        "build_train_command",
        lambda *_args, **_kwargs: [sys.executable, str(fake_train), str(output_dir)],
    )
    monkeypatch.setattr(train, "_verify_cuda", lambda: None)
    monkeypatch.setattr(train, "_run_evaluation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train.shutil, "which", lambda _name: sys.executable)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--run-name",
            "fresh-run",
            "--max-steps",
            "2000",
            "--output-root",
            str(output_root),
            "--no-push-to-hub",
        ],
    )

    train.main()

    assert (output_dir / "run_artifacts" / "run_manifest.json").is_file()
    assert (output_dir / "run_artifacts" / "training_metrics.jsonl").is_file()
