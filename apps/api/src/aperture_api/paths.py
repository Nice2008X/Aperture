"""Shared filesystem locations."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
MODELS_DIR = REPO_ROOT / "data" / "models"
