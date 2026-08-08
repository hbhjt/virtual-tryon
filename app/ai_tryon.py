from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .ai_masks import target_garment_mask
from .config import MODELS_DIR, OUTPUTS_DIR, ROOT_DIR
from .garments import Garment
from .image_utils import read_image_path, write_png_path
from .pose import PoseResult, screen_sides
from .tryon import warp_garment


AI_PYTHON = ROOT_DIR / ".venv-ai" / "Scripts" / "python.exe"
AI_WORKER = ROOT_DIR / "scripts" / "catvton_worker.py"
CATVTON_SOURCE = ROOT_DIR / "vendor" / "CatVTON"
CATVTON_CACHE = MODELS_DIR / "catvton-cache"


@dataclass(frozen=True)
class AIBackendStatus:
    installed: bool
    model_cached: bool
    backend: str = "CatVTON / PyTorch CPU"
    resolution: str = "512×768"
    default_steps: int = 12

    def as_dict(self) -> dict[str, object]:
        return {
            "installed": self.installed,
            "model_cached": self.model_cached,
            "backend": self.backend,
            "resolution": self.resolution,
            "default_steps": self.default_steps,
        }


def backend_status() -> AIBackendStatus:
    installed = AI_PYTHON.exists() and AI_WORKER.exists() and CATVTON_SOURCE.exists()
    cached_repositories = list((CATVTON_CACHE / "hub").glob("models--*")) if CATVTON_CACHE.exists() else []
    names = {path.name.lower() for path in cached_repositories}
    required = (
        any("zhengchong--catvton" in name for name in names)
        and any("stable-diffusion-inpainting" in name for name in names)
        and any("sd-vae-ft-mse" in name for name in names)
    )
    return AIBackendStatus(installed=installed, model_cached=required)


def _garment_condition(garment: Garment) -> np.ndarray:
    image = read_image_path(garment.image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("服装素材无法读取")
    if image.shape[2] == 3:
        return image
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    return np.clip(image[:, :, :3] * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)


def _target_mask(frame: np.ndarray, garment: Garment, pose: PoseResult) -> np.ndarray:
    garment_rgba = read_image_path(garment.image_path, cv2.IMREAD_UNCHANGED)
    if garment_rgba is None:
        raise RuntimeError("服装素材无法读取")
    warped = warp_garment(garment_rgba, garment, pose, frame.shape[:2])
    return target_garment_mask(warped[:, :, 3])


def _hands_mask(frame: np.ndarray, pose: PoseResult, garment: Garment) -> np.ndarray:
    height, width = frame.shape[:2]
    sides = screen_sides(pose)

    def point(name: str) -> np.ndarray:
        landmark = sides[name]
        return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)

    shoulder_span = max(float(np.linalg.norm(point("shoulder_r") - point("shoulder_l"))), 24.0)
    radius = max(8, round(shoulder_span * 0.09))
    mask = np.zeros((height, width), dtype=np.uint8)
    for side in ("l", "r"):
        shoulder = point(f"shoulder_{side}")
        elbow = point(f"elbow_{side}")
        wrist = point(f"wrist_{side}")
        if garment.template != "long_sleeve":
            # Start skin protection below the intended cuff. Starting exactly on
            # the cuff lets line thickness erase the last few pixels of the sleeve.
            protect_start = float(np.clip(garment.sleeve_length + 0.10, 0.52, 0.92))
            cuff = shoulder + (elbow - shoulder) * protect_start
            cv2.line(
                mask,
                tuple(np.int32(cuff)),
                tuple(np.int32(elbow)),
                255,
                max(9, round(shoulder_span * 0.125)),
                cv2.LINE_AA,
            )
            cv2.line(
                mask,
                tuple(np.int32(elbow)),
                tuple(np.int32(wrist)),
                255,
                max(8, round(shoulder_span * 0.12)),
                cv2.LINE_AA,
            )
        palm_center = wrist + (wrist - elbow) * 0.08
        cv2.circle(mask, tuple(np.int32(palm_center)), radius, 255, -1, cv2.LINE_AA)
    return mask


def _required_sleeve_mask(frame: np.ndarray, pose: PoseResult, garment: Garment) -> np.ndarray:
    """Pose corridor that must remain cloth after generated-image parsing."""
    height, width = frame.shape[:2]
    sides = screen_sides(pose)

    def point(name: str) -> np.ndarray:
        landmark = sides[name]
        return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)

    shoulder_span = max(float(np.linalg.norm(point("shoulder_r") - point("shoulder_l"))), 24.0)
    mask = np.zeros((height, width), dtype=np.uint8)
    for side in ("l", "r"):
        shoulder = point(f"shoulder_{side}")
        elbow = point(f"elbow_{side}")
        wrist = point(f"wrist_{side}")
        if garment.template == "long_sleeve":
            sleeve_end = elbow + (wrist - elbow) * 0.92
            cv2.line(
                mask,
                tuple(np.int32(shoulder)),
                tuple(np.int32(elbow)),
                255,
                max(12, round(shoulder_span * 0.32)),
                cv2.LINE_AA,
            )
            cv2.line(
                mask,
                tuple(np.int32(elbow)),
                tuple(np.int32(sleeve_end)),
                255,
                max(10, round(shoulder_span * 0.24)),
                cv2.LINE_AA,
            )
        else:
            cuff_t = float(np.clip(garment.sleeve_length, 0.42, 0.82))
            band_start = shoulder + (elbow - shoulder) * max(0.34, cuff_t - 0.13)
            band_end = shoulder + (elbow - shoulder) * min(0.88, cuff_t + 0.04)
            cv2.line(
                mask,
                tuple(np.int32(band_start)),
                tuple(np.int32(band_end)),
                255,
                max(11, round(shoulder_span * 0.30)),
                cv2.LINE_AA,
            )
    return mask


def run_ai_tryon(
    frame: np.ndarray,
    garment: Garment,
    pose: PoseResult,
    output_path: Path,
    *,
    steps: int = 12,
    width: int = 512,
    height: int = 768,
    timeout_seconds: int = 1800,
) -> dict[str, object]:
    status = backend_status()
    if not status.installed:
        raise RuntimeError("AI 高质量环境尚未安装")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="catvton-job-", dir=OUTPUTS_DIR) as job_dir_raw:
        job_dir = Path(job_dir_raw)
        person_path = job_dir / "person.png"
        garment_path = job_dir / "garment.png"
        target_mask_path = job_dir / "target-mask.png"
        hands_mask_path = job_dir / "hands-mask.png"
        sleeve_mask_path = job_dir / "sleeve-mask.png"
        write_png_path(person_path, frame)
        write_png_path(garment_path, _garment_condition(garment))
        write_png_path(target_mask_path, _target_mask(frame, garment, pose))
        write_png_path(hands_mask_path, _hands_mask(frame, pose, garment))
        write_png_path(sleeve_mask_path, _required_sleeve_mask(frame, pose, garment))

        command = [
            str(AI_PYTHON),
            str(AI_WORKER),
            "--person",
            str(person_path),
            "--garment",
            str(garment_path),
            "--target-mask",
            str(target_mask_path),
            "--hands-mask",
            str(hands_mask_path),
            "--sleeve-mask",
            str(sleeve_mask_path),
            "--output",
            str(output_path),
            "--width",
            str(width),
            "--height",
            str(height),
            "--steps",
            str(max(4, min(30, steps))),
        ]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT_DIR,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"AI processing exceeded {timeout_seconds} seconds") from exc
        if completed.returncode != 0 or not output_path.exists():
            detail = (completed.stderr or completed.stdout or "AI worker failed").strip()
            raise RuntimeError(detail[-1800:])
        worker_metrics: dict[str, object] = {}
        for line in reversed(completed.stdout.splitlines()):
            try:
                worker_metrics = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    return {
        "backend": status.backend,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        **worker_metrics,
    }
