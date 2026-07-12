from __future__ import annotations

from pathlib import Path

from huggingface_hub.errors import HfHubHTTPError
from requests import Response

from urban_act.config import TrainConfig
from urban_act.train import _download_dataset_snapshot


def _hub_error(status_code: int) -> HfHubHTTPError:
    response = Response()
    response.status_code = status_code
    response.url = "https://huggingface.co/test"
    return HfHubHTTPError(f"HTTP {status_code}", response=response)


def test_dataset_download_retries_transient_hub_errors(monkeypatch) -> None:
    attempts = iter((_hub_error(502), _hub_error(503), "/workspace/cache/snapshot"))
    delays: list[float] = []

    def fake_download(**_kwargs):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("urban_act.train.snapshot_download", fake_download)

    result = _download_dataset_snapshot(
        TrainConfig(),
        token="test-token",
        attempts=3,
        sleep=delays.append,
    )

    assert result == Path("/workspace/cache/snapshot")
    assert delays == [1, 2]


def test_dataset_download_does_not_retry_authentication_errors(monkeypatch) -> None:
    calls = 0

    def fake_download(**_kwargs):
        nonlocal calls
        calls += 1
        raise _hub_error(401)

    monkeypatch.setattr("urban_act.train.snapshot_download", fake_download)

    try:
        _download_dataset_snapshot(TrainConfig(), token="bad-token", sleep=lambda _delay: None)
    except HfHubHTTPError:
        pass
    else:
        raise AssertionError("Expected authentication failure")

    assert calls == 1
