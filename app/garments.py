from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import GARMENTS_DIR
from .image_utils import write_png_path


DEFAULT_ANCHORS = {
    "sleeve_l": [0.08, 0.29],
    "shoulder_l": [0.30, 0.16],
    "neck_l": [0.43, 0.145],
    "neck_r": [0.57, 0.145],
    "shoulder_r": [0.70, 0.16],
    "sleeve_r": [0.92, 0.29],
    "armpit_r": [0.75, 0.39],
    "hem_r": [0.73, 0.92],
    "hem_l": [0.27, 0.92],
    "armpit_l": [0.25, 0.39],
    "sleeve_inner_l": [0.22, 0.36],
    "sleeve_inner_r": [0.78, 0.36],
}


@dataclass(frozen=True)
class Garment:
    garment_id: str
    name: str
    category: str
    image_path: Path
    metadata_path: Path
    anchors: dict[str, list[float]]
    template: str = "short_sleeve"
    fit: float = 1.0
    hem_ratio: float = 1.04
    shoulder_extension: float = 0.14
    shoulder_lift: float = 0.045
    sleeve_length: float = 0.62
    hem_width_ratio: float = 0.54
    hem_overlap: float = 0.055
    collar_type: str = "round"
    collar_width: float = 0.30
    collar_depth: float = 0.035

    def public_dict(self) -> dict[str, object]:
        return {
            "id": self.garment_id,
            "name": self.name,
            "category": self.category,
            "template": self.template,
            "image_url": f"/garments/{self.garment_id}/image.png",
        }


def _draw_default_garment(
    path: Path,
    base_color: tuple[int, int, int],
    style: str,
) -> None:
    height, width = 700, 600
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    points = np.array(
        [
            [48, 203], [180, 112], [258, 102], [342, 102], [420, 112],
            [552, 203], [450, 273], [438, 644], [162, 644], [150, 273],
        ],
        dtype=np.int32,
    )
    color = (*base_color, 255)
    cv2.fillPoly(canvas, [points], color)

    # Neck opening and simple fabric details make the built-in samples useful for demos.
    neck = np.array([[258, 101], [300, 145], [342, 101]], dtype=np.int32)
    cv2.fillPoly(canvas, [neck], (0, 0, 0, 0))
    seam = tuple(max(0, channel - 28) for channel in base_color) + (255,)
    cv2.line(canvas, (153, 272), (438, 272), seam, 3, cv2.LINE_AA)
    cv2.line(canvas, (181, 115), (151, 272), seam, 3, cv2.LINE_AA)
    cv2.line(canvas, (419, 115), (449, 272), seam, 3, cv2.LINE_AA)
    cv2.line(canvas, (164, 632), (436, 632), seam, 4, cv2.LINE_AA)

    if style == "polo":
        cv2.line(canvas, (300, 142), (300, 244), (235, 235, 235, 255), 18)
        cv2.line(canvas, (300, 148), (300, 244), seam, 3)
        for y in (174, 207, 238):
            cv2.circle(canvas, (308, y), 5, (235, 235, 235, 255), -1, cv2.LINE_AA)
    elif style == "stripe":
        for y in range(300, 600, 72):
            cv2.rectangle(canvas, (158, y), (442, y + 18), (235, 235, 235, 210), -1)
    elif style == "jacket":
        cv2.line(canvas, (300, 145), (300, 638), (210, 210, 210, 255), 6)
        for y in range(210, 590, 54):
            cv2.circle(canvas, (315, y), 5, (185, 185, 185, 255), -1, cv2.LINE_AA)

    write_png_path(path, canvas)


class GarmentStore:
    def __init__(self, root: Path = GARMENTS_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.ensure_defaults()

    def ensure_defaults(self) -> None:
        defaults = [
            ("ocean-polo", "海蓝色 Polo", (198, 112, 32), "polo"),
            ("coral-tee", "珊瑚红 T 恤", (78, 91, 220), "stripe"),
            ("midnight-jacket", "深色休闲外套", (56, 47, 42), "jacket"),
        ]
        for garment_id, name, color, style in defaults:
            directory = self.root / garment_id
            image_path = directory / "image.png"
            metadata_path = directory / "metadata.json"
            directory.mkdir(parents=True, exist_ok=True)
            if not image_path.exists():
                _draw_default_garment(image_path, color, style)
            if not metadata_path.exists():
                metadata = {
                    "id": garment_id,
                    "name": name,
                    "category": "upper_body",
                    "anchors": DEFAULT_ANCHORS,
                }
                metadata_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )

    def list(self) -> list[Garment]:
        garments: list[Garment] = []
        for metadata_path in sorted(self.root.glob("*/metadata.json")):
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
                image_path = metadata_path.parent / "image.png"
                if image_path.exists():
                    garments.append(
                        Garment(
                            garment_id=str(data["id"]),
                            name=str(data["name"]),
                            category=str(data.get("category", "upper_body")),
                            image_path=image_path,
                            metadata_path=metadata_path,
                            anchors=data.get("anchors", DEFAULT_ANCHORS),
                            template=str(data.get("template", "short_sleeve")),
                            fit=float(data.get("fit", 1.0)),
                            hem_ratio=float(data.get("hem_ratio", 1.04)),
                            shoulder_extension=float(data.get("shoulder_extension", 0.14)),
                            shoulder_lift=float(data.get("shoulder_lift", 0.045)),
                            sleeve_length=float(data.get("sleeve_length", 0.62)),
                            hem_width_ratio=float(data.get("hem_width_ratio", 0.54)),
                            hem_overlap=float(data.get("hem_overlap", 0.055)),
                            collar_type=str(data.get("collar_type", "round")),
                            collar_width=float(data.get("collar_width", 0.30)),
                            collar_depth=float(data.get("collar_depth", 0.035)),
                        )
                    )
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return garments

    def get(self, garment_id: str) -> Garment:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", garment_id):
            raise KeyError(garment_id)
        for garment in self.list():
            if garment.garment_id == garment_id:
                return garment
        raise KeyError(garment_id)

    def add(self, name: str, image: np.ndarray) -> Garment:
        safe_name = name.strip()[:40] or "自定义上衣"
        garment_id = f"custom-{uuid.uuid4().hex[:10]}"
        directory = self.root / garment_id
        directory.mkdir(parents=True, exist_ok=False)
        rgba = self._prepare_alpha(image)

        # Normalize transparent padding so all garments share the same anchor convention.
        normalized = np.zeros((700, 600, 4), dtype=np.uint8)
        alpha = rgba[:, :, 3]
        coords = cv2.findNonZero((alpha > 12).astype(np.uint8))
        if coords is None:
            raise ValueError("没有识别到服装主体，请上传透明背景或纯色背景图片")
        x, y, width, height = cv2.boundingRect(coords)
        crop = rgba[y : y + height, x : x + width]
        scale = min(510 / max(width, 1), 580 / max(height, 1))
        resized = cv2.resize(
            crop,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        rh, rw = resized.shape[:2]
        x0, y0 = (600 - rw) // 2, 85
        normalized[y0 : y0 + rh, x0 : x0 + rw] = resized
        write_png_path(directory / "image.png", normalized)
        metadata = {
            "id": garment_id,
            "name": safe_name,
            "category": "upper_body",
            "anchors": DEFAULT_ANCHORS,
            "template": "short_sleeve",
            "fit": 1.0,
            "hem_ratio": 1.04,
            "shoulder_extension": 0.14,
            "shoulder_lift": 0.045,
            "sleeve_length": 0.62,
            "hem_width_ratio": 0.54,
            "hem_overlap": 0.055,
            "collar_type": "round",
            "collar_width": 0.30,
            "collar_depth": 0.035,
        }
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.get(garment_id)

    @staticmethod
    def _prepare_alpha(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] == 4:
            return image
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        bgr = image[:, :, :3]
        height, width = bgr.shape[:2]
        if min(height, width) < 20:
            raise ValueError("服装图片尺寸过小")

        mask = np.zeros((height, width), dtype=np.uint8)
        bg_model = np.zeros((1, 65), dtype=np.float64)
        fg_model = np.zeros((1, 65), dtype=np.float64)
        margin_x = max(1, int(width * 0.035))
        margin_y = max(1, int(height * 0.035))
        rect = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)
        try:
            cv2.grabCut(bgr, mask, rect, bg_model, fg_model, 4, cv2.GC_INIT_WITH_RECT)
            alpha = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
        except cv2.error:
            gray_distance = np.linalg.norm(bgr.astype(np.float32) - 245.0, axis=2)
            alpha = np.clip((gray_distance - 8.0) * 12.0, 0, 255).astype(np.uint8)
        alpha = cv2.medianBlur(alpha, 5)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
        return np.dstack([bgr, alpha])
