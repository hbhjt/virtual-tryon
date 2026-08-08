from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


class InvalidImageError(ValueError):
    pass


def decode_image(data: bytes, *, unchanged: bool = False) -> np.ndarray:
    if not data:
        raise InvalidImageError("图片内容为空")
    flags = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    if image is None or image.size == 0:
        raise InvalidImageError("无法识别该图片，请使用 JPG 或 PNG")
    return image


def resize_long_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def encode_jpeg(image: np.ndarray, quality: int = 92) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("结果图片编码失败")
    return encoded.tobytes()


def read_image_path(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """Read through bytes because OpenCV's Windows file API can fail on Unicode paths."""
    try:
        return cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), flags)
    except OSError:
        return None


def write_png_path(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"图片编码失败：{path.name}")
    path.write_bytes(encoded.tobytes())


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
