from __future__ import annotations

import os

os.environ.setdefault("VLA_WORKSPACE_ROOT", "/workspace/vla-driving")

from train import main  # noqa: E402, I001


if __name__ == "__main__":
    main()
