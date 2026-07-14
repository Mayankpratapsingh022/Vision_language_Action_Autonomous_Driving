from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path) -> None:
    source = Path(path)
    if not source.exists():
        return
    for raw_line in source.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


__all__ = ["load_env_file"]
