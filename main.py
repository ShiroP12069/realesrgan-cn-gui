from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap_path()

from realesrgan_gui.app import run_gui  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_gui())
