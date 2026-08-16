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
from .ai_progress import write_progress
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


class AIInputError(RuntimeError):
    """The camera frame cannot be safely prepared for portrait virtual try-on."""


@dataclass(frozen=True)
class AICropRegion:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def ai_portrait_crop_region(
    frame: np.ndarray,
    pose: PoseResult,
    target_mask: np.ndarray,
    garment: Garment,
    *,
    target_aspect: float = 2.0 / 3.0,
) -> AICropRegion:
    """Return a 2:3 upper-body crop or reject frames that would cut the person."""
    if not pose.detected or pose.fallback:
        raise AIInputError("未检测到完整人体；请露出头部、双肩、手肘和腰部后重新拍摄")
    height, width = frame.shape[:2]
    critical = {
        "头部": pose.landmarks["nose"],
        "左肩": pose.landmarks["left_shoulder"],
        "右肩": pose.landmarks["right_shoulder"],
        "左手肘": pose.landmarks["left_elbow"],
        "右手肘": pose.landmarks["right_elbow"],
        "左腰": pose.landmarks["left_hip"],
        "右腰": pose.landmarks["right_hip"],
    }
    missing = [name for name, point in critical.items() if point.visibility < 0.34]
    nose = pose.landmarks["nose"]
    hips = (pose.landmarks["left_hip"], pose.landmarks["right_hip"])
    if nose.y <= 0.035:
        missing.append("完整头部")
    if max(point.y for point in hips) >= 0.975:
        missing.append("腰部")
    if missing:
        names = "、".join(dict.fromkeys(missing))
        raise AIInputError(f"画面缺少{names}；请后退一些，让完整上半身进入画面")

    sides = screen_sides(pose)
    shoulder_l = np.array([sides["shoulder_l"].x * width, sides["shoulder_l"].y * height])
    shoulder_r = np.array([sides["shoulder_r"].x * width, sides["shoulder_r"].y * height])
    hip_l = np.array([sides["hip_l"].x * width, sides["hip_l"].y * height])
    hip_r = np.array([sides["hip_r"].x * width, sides["hip_r"].y * height])
    shoulder_span = max(float(np.linalg.norm(shoulder_r - shoulder_l)), 24.0)
    torso_height = max(float(np.linalg.norm((hip_l + hip_r - shoulder_l - shoulder_r) * 0.5)), 40.0)

    names = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_hip", "right_hip"]
    if garment.template == "long_sleeve":
        names.extend(["left_wrist", "right_wrist"])
    points = np.array(
        [[pose.landmarks[name].x * width, pose.landmarks[name].y * height] for name in names],
        dtype=np.float32,
    )
    target_points = cv2.findNonZero(np.where(target_mask > 0, 255, 0).astype(np.uint8))
    if target_points is None:
        raise AIInputError("无法确定目标衣服区域，请重新选择衣服或拍摄")
    tx, ty, tw, th = cv2.boundingRect(target_points)
    left = min(float(points[:, 0].min()), float(tx)) - shoulder_span * 0.14
    right = max(float(points[:, 0].max()), float(tx + tw)) + shoulder_span * 0.14
    head_top = nose.y * height - torso_height * 0.34
    top = min(float(points[:, 1].min()), float(ty), head_top) - torso_height * 0.06
    bottom = max(float(points[:, 1].max()), float(ty + th)) + torso_height * 0.12
    left = max(0.0, left)
    right = min(float(width), right)
    top = max(0.0, top)
    bottom = min(float(height), bottom)
    required_width = right - left
    required_height = bottom - top

    maximum_width = min(float(width), float(height) * target_aspect)
    maximum_height = maximum_width / target_aspect
    if required_width > maximum_width * 0.98 or required_height > maximum_height * 0.98:
        raise AIInputError(
            "人物距离摄像头太近，2:3 AI 画面会裁掉肩部或手臂；请后退约半步后重新拍摄"
        )

    crop_width = max(required_width * 1.08, required_height * target_aspect * 1.08)
    crop_width = min(maximum_width, crop_width)
    crop_height = crop_width / target_aspect
    crop_width_i = max(2, min(width, int(round(crop_width))))
    crop_height_i = max(3, min(height, int(round(crop_height))))
    # Keep the integer crop as close to exactly 2:3 as possible.
    crop_width_i = min(crop_width_i, int(crop_height_i * target_aspect))
    crop_height_i = min(height, int(round(crop_width_i / target_aspect)))
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    x0 = int(np.clip(round(center_x - crop_width_i * 0.5), 0, width - crop_width_i))
    y0 = int(np.clip(round(center_y - crop_height_i * 0.5), 0, height - crop_height_i))
    region = AICropRegion(x0, y0, x0 + crop_width_i, y0 + crop_height_i)
    if left < region.x0 or right > region.x1 or top < region.y0 or bottom > region.y1:
        raise AIInputError("人物靠近画面边缘；请站到中央并露出完整上半身后重新拍摄")
    return region


def restore_ai_crop(
    original: np.ndarray,
    generated_crop: np.ndarray,
    region: AICropRegion,
) -> np.ndarray:
    """Paste a generated crop back while preserving every pixel outside it."""
    if generated_crop.shape[:2] != (region.height, region.width):
        generated_crop = cv2.resize(
            generated_crop,
            (region.width, region.height),
            interpolation=cv2.INTER_LANCZOS4,
        )
    result = original.copy()
    result[region.y0 : region.y1, region.x0 : region.x1] = generated_crop
    return result


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
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    skin = np.where(
        (y_channel >= 35)
        & (cr_channel >= 132)
        & (cr_channel <= 178)
        & (cb_channel >= 72)
        & (cb_channel <= 135),
        255,
        0,
    ).astype(np.uint8)
    skin = cv2.morphologyEx(
        skin,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    for side in ("l", "r"):
        shoulder = point(f"shoulder_{side}")
        elbow = point(f"elbow_{side}")
        wrist = point(f"wrist_{side}")
        if garment.template != "long_sleeve":
            # Protect only skin-colored pixels in the forearm corridor. A broad
            # keypoint line alone can preserve old colored sleeves as elbow patches.
            protect_start = float(np.clip(garment.sleeve_length + 0.10, 0.52, 0.92))
            cuff = shoulder + (elbow - shoulder) * protect_start
            corridor = np.zeros_like(mask)
            cv2.line(
                corridor,
                tuple(np.int32(cuff)),
                tuple(np.int32(elbow)),
                255,
                max(9, round(shoulder_span * 0.125)),
                cv2.LINE_AA,
            )
            cv2.line(
                corridor,
                tuple(np.int32(elbow)),
                tuple(np.int32(wrist)),
                255,
                max(8, round(shoulder_span * 0.12)),
                cv2.LINE_AA,
            )
            protected_skin = cv2.bitwise_and(corridor, skin)
            protected_skin = cv2.dilate(
                protected_skin,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            mask = np.maximum(mask, protected_skin)
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
    progress_file: Path | None = None,
) -> dict[str, object]:
    status = backend_status()
    if not status.installed:
        raise RuntimeError("AI 高质量环境尚未安装")
    if progress_file is not None:
        write_progress(
            progress_file,
            progress=3,
            stage="preflight",
            message="正在检查人物姿态和画面构图",
        )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    target_mask_full = _target_mask(frame, garment, pose)
    hands_mask_full = _hands_mask(frame, pose, garment)
    sleeve_mask_full = _required_sleeve_mask(frame, pose, garment)
    crop_region = ai_portrait_crop_region(
        frame,
        pose,
        target_mask_full,
        garment,
        target_aspect=width / height,
    )
    crop_slice = np.s_[crop_region.y0 : crop_region.y1, crop_region.x0 : crop_region.x1]
    person_crop = frame[crop_slice]
    target_mask_crop = target_mask_full[crop_slice]
    hands_mask_crop = hands_mask_full[crop_slice]
    sleeve_mask_crop = sleeve_mask_full[crop_slice]
    if progress_file is not None:
        write_progress(
            progress_file,
            progress=8,
            stage="preparing",
            message="2:3 人像区域与生成蒙版已准备完成",
        )
    with tempfile.TemporaryDirectory(prefix="catvton-job-", dir=OUTPUTS_DIR) as job_dir_raw:
        job_dir = Path(job_dir_raw)
        person_path = job_dir / "person.png"
        garment_path = job_dir / "garment.png"
        target_mask_path = job_dir / "target-mask.png"
        hands_mask_path = job_dir / "hands-mask.png"
        sleeve_mask_path = job_dir / "sleeve-mask.png"
        debug_enabled = os.environ.get("CATVTON_SAVE_DEBUG") == "1"
        worker_output_path = (
            output_path.with_name(f"{output_path.stem}-crop-output.png")
            if debug_enabled
            else job_dir / "generated-crop.png"
        )
        write_png_path(person_path, person_crop)
        write_png_path(garment_path, _garment_condition(garment))
        write_png_path(target_mask_path, target_mask_crop)
        write_png_path(hands_mask_path, hands_mask_crop)
        write_png_path(sleeve_mask_path, sleeve_mask_crop)

        if debug_enabled:
            debug_prefix = output_path.with_suffix("")
            write_png_path(debug_prefix.with_name(f"{debug_prefix.name}-input-crop.png"), person_crop)
            write_png_path(debug_prefix.with_name(f"{debug_prefix.name}-target-mask.png"), target_mask_crop)
            write_png_path(debug_prefix.with_name(f"{debug_prefix.name}-hands-mask.png"), hands_mask_crop)

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
            str(worker_output_path),
            "--width",
            str(width),
            "--height",
            str(height),
            "--steps",
            str(max(4, min(30, steps))),
        ]
        if progress_file is not None:
            command.extend(["--progress-file", str(progress_file)])
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
        if completed.returncode != 0 or not worker_output_path.exists():
            detail = (completed.stderr or completed.stdout or "AI worker failed").strip()
            raise RuntimeError(detail[-1800:])
        generated_crop = read_image_path(worker_output_path, cv2.IMREAD_COLOR)
        if generated_crop is None:
            raise RuntimeError("AI 生成结果无法读取")
        restored = restore_ai_crop(frame, generated_crop, crop_region)
        write_png_path(output_path, restored)
        if progress_file is not None:
            write_progress(
                progress_file,
                progress=99,
                stage="restoring",
                message="正在保存最终试穿照片",
            )
        worker_metrics: dict[str, object] = {}
        for line in reversed(completed.stdout.splitlines()):
            try:
                worker_metrics = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        worker_metrics["output"] = str(output_path)
        worker_metrics["crop"] = {
            "x": crop_region.x0,
            "y": crop_region.y0,
            "width": crop_region.width,
            "height": crop_region.height,
            "source_width": frame.shape[1],
            "source_height": frame.shape[0],
        }
    return {
        "backend": status.backend,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        **worker_metrics,
    }
