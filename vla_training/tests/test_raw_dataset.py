import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from left_turn_vla.constants import LEFT_TURN_INSTRUCTION
from left_turn_vla.raw_dataset import analyze_directory, iter_lerobot_frames, load_episode


def _image_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128), (20, 40, 60)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _sample(index: int, *, collision: bool = False) -> dict:
    return {
        "seed": 42,
        "capture_resolution": 128,
        "image": _image_url(),
        "language_id": 0,
        "language_text": LEFT_TURN_INSTRUCTION,
        "control": {"throttle": 0.4 + index * 0.1, "brake": 0.0, "steer": -0.2},
        "ego": {"speed": 0.5, "steering": -0.1},
        "events": {"collision": collision, "offRoute": False, "redLightViolation": False},
        "task": {"routeProgress": 0.96},
    }


def _write_episode(path: Path, *, collision: bool = False) -> None:
    samples = [_sample(index % 2, collision=collision and index == 10) for index in range(60)]
    path.write_text(json.dumps({"metadata": {"schema_version": "vla-urban-3"}, "samples": samples}))


def test_analysis_ignores_cumulative_exports_and_rejects_collision(tmp_path: Path) -> None:
    _write_episode(tmp_path / "human-clean.json")
    _write_episode(tmp_path / "human-collision.json", collision=True)
    (tmp_path / "vla_urban_dataset_123.json").write_text("{}")
    inspections, report = analyze_directory(tmp_path)
    assert report["independent_files"] == 2
    assert report["old_cumulative_exports_ignored"] == 1
    assert report["accepted_episodes"] == 1
    assert sum(item.collision for item in inspections) == 1


def test_held_out_tail_is_seed_disjoint(tmp_path: Path) -> None:
    for seed in range(10):
        path = tmp_path / f"human-{seed}.json"
        _write_episode(path)
        payload = json.loads(path.read_text())
        for sample in payload["samples"]:
            sample["seed"] = seed
        path.write_text(json.dumps(payload))
    inspections, report = analyze_directory(tmp_path, eval_split=0.2)
    accepted = [item for item in inspections if item.accepted]
    held_out = accepted[-report["held_out_episodes"] :]
    training = accepted[: -report["held_out_episodes"]]
    assert {item.seed for item in training}.isdisjoint(item.seed for item in held_out)
    assert report["seed_disjoint_holdout"]


def test_frame_conversion_aligns_previous_controls_and_speed_units(tmp_path: Path) -> None:
    path = tmp_path / "human-clean.json"
    _write_episode(path)
    frames = list(iter_lerobot_frames(load_episode(path)))
    np.testing.assert_allclose(frames[0]["observation.state"], [12.0, -0.1, 0.0, 0.0])
    np.testing.assert_allclose(frames[1]["observation.state"], [12.0, -0.1, 0.4, 0.0])
    np.testing.assert_allclose(frames[1]["action"], [0.5, 0.0, -0.2])
    assert frames[0]["observation.images.front"].shape == (128, 128, 3)
