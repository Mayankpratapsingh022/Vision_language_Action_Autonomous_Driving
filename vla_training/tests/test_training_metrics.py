from left_turn_vla.training_metrics import parse_metric_line


def test_parses_training_progress() -> None:
    event = parse_metric_line("step:200 loss:0.123 grdn:1.50 lr:1.0e-04 mem_gb:12.5")
    assert event == {
        "step": 200,
        "loss": 0.123,
        "gradient_norm": 1.5,
        "lr": 1.0e-4,
        "gpu_mem_gb": 12.5,
    }


def test_parses_held_out_loss() -> None:
    assert parse_metric_line("step 1000: eval_loss=0.4567") == {"step": 1000, "eval_loss": 0.4567}

