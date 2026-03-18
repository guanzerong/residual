from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[4]
_PI05_VENDOR_DIR = _REPO_ROOT / ".vendor" / "pi05_transformers_only"


def ensure_pi05_dependencies() -> None:
    """Add isolated pi05-only Python deps to sys.path without mutating the base environment."""

    if _PI05_VENDOR_DIR.exists() and str(_PI05_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(_PI05_VENDOR_DIR))
