import base64
import io

import numpy as np
import pytest
from PIL import Image

from inference_server import decode_image_data_url, parse_prediction_request, sanitize_action
from left_turn_vla.constants import LEFT_TURN_INSTRUCTION


def _data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128), (0, 0, 0)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_prediction_request_contract() -> None:
    image, state, instruction = parse_prediction_request(
        {"type": "predict", "image": _data_url(), "state": [0.0, 0.0, 0.0, 0.0], "instruction": LEFT_TURN_INSTRUCTION}
    )
    assert image.shape == (128, 128, 3)
    np.testing.assert_array_equal(state, np.zeros(4, dtype=np.float32))
    assert instruction == LEFT_TURN_INSTRUCTION


def test_invalid_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="base64"):
        decode_image_data_url("data:image/png;base64,not-valid")


def test_action_sanitizer_bounds_and_excludes_overlap() -> None:
    action = sanitize_action(np.asarray([1.4, 0.4, -2.0], dtype=np.float32))
    assert action == {"throttle": 1.0, "brake": 0.0, "steer": -1.0}
