from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "static"
GARMENTS_DIR = ROOT_DIR / "garments"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "models"
POSE_MODEL_PATH = Path(
    os.getenv("POSE_MODEL_PATH", str(MODELS_DIR / "pose_landmarker_lite.task"))
)

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_SIDE = 1920
ANALYZE_MAX_SIDE = 640


def ensure_directories() -> None:
    for directory in (STATIC_DIR, GARMENTS_DIR, OUTPUTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

