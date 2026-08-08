from __future__ import annotations

import threading
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .image_utils import clamp01, distance


LANDMARK_NAMES = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    visibility: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "x": round(self.x, 5),
            "y": round(self.y, 5),
            "visibility": round(self.visibility, 4),
        }


@dataclass(frozen=True)
class PoseResult:
    landmarks: dict[str, Landmark]
    detected: bool
    fallback: bool
    confidence: float
    segmentation_mask: np.ndarray | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "detected": self.detected,
            "fallback": self.fallback,
            "confidence": round(self.confidence, 4),
            "landmarks": {name: point.as_dict() for name, point in self.landmarks.items()},
        }


def fallback_pose() -> PoseResult:
    points = {
        "nose": Landmark(0.50, 0.13, 0.25),
        "left_eye": Landmark(0.52, 0.115, 0.25),
        "right_eye": Landmark(0.48, 0.115, 0.25),
        "left_ear": Landmark(0.56, 0.14, 0.25),
        "right_ear": Landmark(0.44, 0.14, 0.25),
        "left_shoulder": Landmark(0.66, 0.28, 0.25),
        "right_shoulder": Landmark(0.34, 0.28, 0.25),
        "left_elbow": Landmark(0.73, 0.47, 0.25),
        "right_elbow": Landmark(0.27, 0.47, 0.25),
        "left_wrist": Landmark(0.76, 0.66, 0.25),
        "right_wrist": Landmark(0.24, 0.66, 0.25),
        "left_hip": Landmark(0.59, 0.68, 0.25),
        "right_hip": Landmark(0.41, 0.68, 0.25),
    }
    return PoseResult(points, detected=False, fallback=True, confidence=0.25)


class PoseEstimator:
    """Thread-safe MediaPipe pose wrapper with a deterministic fallback."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._landmarker = None
        self._lock = threading.Lock()
        self.load_error: str | None = None
        self._load()

    @property
    def available(self) -> bool:
        return self._landmarker is not None

    def _load(self) -> None:
        if not self.model_path.exists():
            self.load_error = f"未找到姿态模型：{self.model_path.name}"
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # MediaPipe's Windows native file loader cannot open model paths that
            # contain Chinese characters, so keep an ASCII runtime copy.
            runtime_model_path = self.model_path
            try:
                str(self.model_path).encode("ascii")
            except UnicodeEncodeError:
                cache_dir = Path(tempfile.gettempdir()) / "virtual_tryon_models"
                cache_dir.mkdir(parents=True, exist_ok=True)
                runtime_model_path = cache_dir / self.model_path.name
                if (
                    not runtime_model_path.exists()
                    or runtime_model_path.stat().st_size != self.model_path.stat().st_size
                ):
                    runtime_model_path.write_bytes(self.model_path.read_bytes())

            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(runtime_model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.45,
                min_pose_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
            self._mp = mp
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self.load_error = None
        except Exception as exc:  # pragma: no cover - depends on native runtime
            self.load_error = f"姿态模型加载失败：{exc}"
            self._landmarker = None

    def detect(self, bgr_image: np.ndarray, *, allow_fallback: bool = True) -> PoseResult:
        if self._landmarker is None:
            return fallback_pose() if allow_fallback else PoseResult({}, False, False, 0.0)

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        try:
            with self._lock:
                result = self._landmarker.detect(mp_image)
        except Exception:
            return fallback_pose() if allow_fallback else PoseResult({}, False, False, 0.0)

        if not result.pose_landmarks:
            return fallback_pose() if allow_fallback else PoseResult({}, False, False, 0.0)

        raw = result.pose_landmarks[0]
        landmarks: dict[str, Landmark] = {}
        for name, index in LANDMARK_NAMES.items():
            point = raw[index]
            landmarks[name] = Landmark(
                clamp01(point.x),
                clamp01(point.y),
                clamp01(point.visibility if point.visibility is not None else 1.0),
            )
        confidence = float(np.mean([p.visibility for p in landmarks.values()]))
        return PoseResult(
            landmarks,
            detected=True,
            fallback=False,
            confidence=confidence,
            segmentation_mask=None,
        )


def screen_sides(pose: PoseResult) -> dict[str, Landmark]:
    """Return joints ordered by their screen x-coordinate, independent of anatomy."""
    lm = pose.landmarks
    anatomical_left_is_screen_left = lm["left_shoulder"].x < lm["right_shoulder"].x
    if anatomical_left_is_screen_left:
        left_prefix, right_prefix = "left", "right"
    else:
        left_prefix, right_prefix = "right", "left"
    return {
        "shoulder_l": lm[f"{left_prefix}_shoulder"],
        "elbow_l": lm[f"{left_prefix}_elbow"],
        "wrist_l": lm[f"{left_prefix}_wrist"],
        "hip_l": lm[f"{left_prefix}_hip"],
        "shoulder_r": lm[f"{right_prefix}_shoulder"],
        "elbow_r": lm[f"{right_prefix}_elbow"],
        "wrist_r": lm[f"{right_prefix}_wrist"],
        "hip_r": lm[f"{right_prefix}_hip"],
    }


def evaluate_frame(image: np.ndarray, pose: PoseResult) -> dict[str, object]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = clamp01(1.0 - np.exp(-blur_variance / 150.0))

    brightness_mean = float(np.mean(gray))
    brightness = clamp01(1.0 - abs(brightness_mean - 135.0) / 120.0)
    contrast = clamp01(float(np.std(gray)) / 55.0)
    lighting = 0.72 * brightness + 0.28 * contrast

    sides = screen_sides(pose)
    sl, sr = sides["shoulder_l"], sides["shoulder_r"]
    hl, hr = sides["hip_l"], sides["hip_r"]
    shoulder_span = max(distance((sl.x, sl.y), (sr.x, sr.y)), 1e-4)
    hip_span = max(distance((hl.x, hl.y), (hr.x, hr.y)), 1e-4)
    level_score = clamp01(1.0 - abs(sl.y - sr.y) / max(shoulder_span * 0.35, 0.03))
    hip_level_score = clamp01(1.0 - abs(hl.y - hr.y) / max(hip_span * 0.45, 0.03))
    frontality = 0.68 * level_score + 0.32 * hip_level_score

    center_x = (sl.x + sr.x + hl.x + hr.x) / 4.0
    centering = clamp01(1.0 - abs(center_x - 0.5) / 0.32)
    scale_score = clamp01(1.0 - abs(shoulder_span - 0.32) / 0.28)
    visibility = pose.confidence if pose.detected else 0.35

    score = (
        visibility * 0.28
        + frontality * 0.24
        + sharpness * 0.19
        + lighting * 0.11
        + centering * 0.10
        + scale_score * 0.08
    )
    if not pose.detected:
        score = min(score, 0.54)

    guidance = "姿态很好，请保持不动"
    ready = bool(pose.detected and score >= 0.66)
    if not pose.detected:
        guidance = "请站到画面中央，露出肩膀和腰部"
    elif visibility < 0.58:
        guidance = "请露出双肩和手臂"
    elif frontality < 0.62:
        guidance = "请正对摄像头站立"
    elif centering < 0.62:
        guidance = "请移动到画面中央"
    elif scale_score < 0.55:
        guidance = "请调整距离，让上半身占据画面"
    elif sharpness < 0.42:
        guidance = "画面有些模糊，请保持不动"
    elif lighting < 0.42:
        guidance = "光线不足，请面向光源"

    return {
        "score": round(float(score), 4),
        "ready": ready,
        "guidance": guidance,
        "sharpness": round(float(sharpness), 4),
        "lighting": round(float(lighting), 4),
        "frontality": round(float(frontality), 4),
        "centering": round(float(centering), 4),
        "visibility": round(float(visibility), 4),
        "blur_variance": round(blur_variance, 2),
    }
