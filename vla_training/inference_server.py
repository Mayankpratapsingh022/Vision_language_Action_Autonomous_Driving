from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image, UnidentifiedImageError

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from left_turn_vla.constants import LEFT_TURN_INSTRUCTION  # noqa: E402
from left_turn_vla.env import load_env_file  # noqa: E402
from left_turn_vla.inference import SmolVLADriver  # noqa: E402

DEFAULT_MODEL = "Mayank022/urban-vla-left-turn-smolvla"
MAX_IMAGE_BYTES = 2_000_000
MAX_IMAGE_SIDE = 1_024


class Driver(Protocol):
    device: str
    action_steps: int
    chunk_size: int

    def predict(self, image: np.ndarray, state: np.ndarray, instruction: str) -> np.ndarray: ...

    def reset(self) -> None: ...

    def supports_instruction(self, instruction: str) -> bool: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the fine-tuned left-turn SmolVLA policy.")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("VLA_MODEL_PATH") or os.environ.get("HF_MODEL_REPO") or DEFAULT_MODEL,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    return parser.parse_args()


def create_app(driver: Driver) -> FastAPI:
    app = FastAPI(title="SmolVLA Left-Turn Driving Inference", version="1.0.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ready",
            "policy": "smolvla",
            "task": "protected-left-turn",
            "instruction": LEFT_TURN_INSTRUCTION,
            "device": str(driver.device),
            "image_size": 128,
            "state_dim": 4,
            "action_dim": 3,
            "chunk_size": int(driver.chunk_size),
            "action_steps": int(driver.action_steps),
        }

    @app.websocket("/ws")
    async def inference_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        driver.reset()
        try:
            while True:
                payload = await websocket.receive_json()
                message_type = payload.get("type", "predict") if isinstance(payload, dict) else None
                if message_type == "reset":
                    driver.reset()
                    await websocket.send_json({"type": "reset_ack"})
                    continue
                request_id = payload.get("request_id") if isinstance(payload, dict) else None
                try:
                    image, state, instruction = parse_prediction_request(payload)
                    if not driver.supports_instruction(instruction):
                        raise ValueError("This model only supports the protected left-turn instruction")
                    started = time.perf_counter()
                    raw_action = await asyncio.to_thread(driver.predict, image, state, instruction)
                    latency_ms = (time.perf_counter() - started) * 1_000
                    action = sanitize_action(raw_action)
                    await websocket.send_json(
                        {
                            "type": "prediction",
                            "request_id": request_id,
                            "action": action,
                            "raw_action": {
                                "throttle": float(raw_action[0]),
                                "brake": float(raw_action[1]),
                                "steer": float(raw_action[2]),
                            },
                            "latency_ms": round(latency_ms, 2),
                        }
                    )
                except (TypeError, ValueError) as error:
                    await websocket.send_json({"type": "error", "request_id": request_id, "error": str(error)})
        except WebSocketDisconnect:
            return

    return app


def parse_prediction_request(payload: Any) -> tuple[np.ndarray, np.ndarray, str]:
    if not isinstance(payload, dict) or payload.get("type", "predict") != "predict":
        raise ValueError("Expected a predict message")
    image = decode_image_data_url(payload.get("image"))
    raw_state = payload.get("state")
    if not isinstance(raw_state, list) or len(raw_state) != 4:
        raise ValueError("state must contain [speed, steering, previous throttle, previous brake]")
    try:
        state = np.asarray(raw_state, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("state values must be numeric") from error
    if not np.isfinite(state).all():
        raise ValueError("state values must be finite")
    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    if len(instruction) > 512:
        raise ValueError("instruction must be at most 512 characters")
    return image, state, instruction.strip()


def decode_image_data_url(value: Any) -> np.ndarray:
    if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value:
        raise ValueError("image must be a base64 image data URL")
    encoded = value.split(",", maxsplit=1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("image contains invalid base64 data") from error
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"image must be between 1 and {MAX_IMAGE_BYTES} bytes")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width > MAX_IMAGE_SIDE or image.height > MAX_IMAGE_SIDE:
                raise ValueError(f"image dimensions must not exceed {MAX_IMAGE_SIDE} x {MAX_IMAGE_SIDE}")
            rgb = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
            return np.asarray(rgb, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("image data is not a supported image") from error


def sanitize_action(action: np.ndarray) -> dict[str, float]:
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("model returned an invalid action")
    throttle = float(np.clip(values[0], 0.0, 1.0))
    brake = float(np.clip(values[1], 0.0, 1.0))
    steer = float(np.clip(values[2], -1.0, 1.0))
    if throttle < 0.03:
        throttle = 0.0
    if brake < 0.05:
        brake = 0.0
    if abs(steer) < 0.015:
        steer = 0.0
    if throttle > 0 and brake > 0:
        if throttle >= brake:
            brake = 0.0
        else:
            throttle = 0.0
    return {"throttle": throttle, "brake": brake, "steer": steer}


def main() -> None:
    load_env_file(PROJECT_DIR / ".env")
    args = parse_args()
    driver = SmolVLADriver(args.model_path, device=args.device, cache_dir=PROJECT_DIR / ".cache")
    print(
        f"SmolVLA ready: device={driver.device}, chunk={driver.chunk_size}, execute={driver.action_steps}",
        flush=True,
    )
    uvicorn.run(create_app(driver), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
