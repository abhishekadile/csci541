"""API settings, read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ml.config import PROJECT_ROOT, resolve_path


def _bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    checkpoint_path: Path = field(default_factory=lambda: resolve_path(
        os.environ.get("MODEL_CHECKPOINT_PATH", "weight/model_b/best.pt")))
    results_dir: Path = field(default_factory=lambda: resolve_path(os.environ.get("RESULTS_DIR", "artifacts/results")))
    frontend_dist: Path = field(default_factory=lambda: resolve_path(os.environ.get("FRONTEND_DIST", "frontend/dist")))
    max_upload_bytes: int = field(default_factory=lambda: int(os.environ.get("MAX_UPLOAD_BYTES", 5 * 1024 * 1024)))
    device: str = field(default_factory=lambda: os.environ.get("INFERENCE_DEVICE", "cpu"))
    allow_non_trained: bool = field(default_factory=lambda: _bool("ALLOW_NON_TRAINED_CHECKPOINTS", False))
    project_root: Path = PROJECT_ROOT
