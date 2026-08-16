from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.ai_tryon import AIInputError, AICropRegion, ai_portrait_crop_region, restore_ai_crop
from app.garments import GarmentStore
from app.pose import Landmark, PoseResult, fallback_pose


def detected_pose(**updates: tuple[float, float]) -> PoseResult:
    base = fallback_pose().landmarks
    landmarks = {
        name: Landmark(
            updates.get(name, (point.x, point.y))[0],
            updates.get(name, (point.x, point.y))[1],
            0.95,
        )
        for name, point in base.items()
    }
    return PoseResult(landmarks, detected=True, fallback=False, confidence=0.95)


def test_portrait_crop_is_two_by_three_and_contains_target() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    pose = detected_pose(
        nose=(0.50, 0.16),
        left_shoulder=(0.56, 0.31),
        right_shoulder=(0.44, 0.31),
        left_elbow=(0.59, 0.48),
        right_elbow=(0.41, 0.48),
        left_hip=(0.55, 0.70),
        right_hip=(0.45, 0.70),
    )
    target = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(target, (500, 190), (780, 570), 255, -1)
    garment = GarmentStore().get("coral-tee")

    region = ai_portrait_crop_region(frame, pose, target, garment)
    assert abs(region.width / region.height - 2 / 3) < 0.003
    assert region.x0 <= 500 and region.x1 >= 781
    assert region.y0 <= 190 and region.y1 >= 571


def test_close_landscape_person_is_rejected_instead_of_cutting_shoulders() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    pose = detected_pose(
        nose=(0.50, 0.10),
        left_shoulder=(0.72, 0.28),
        right_shoulder=(0.28, 0.28),
        left_elbow=(0.78, 0.52),
        right_elbow=(0.22, 0.52),
        left_hip=(0.62, 0.88),
        right_hip=(0.38, 0.88),
    )
    target = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(target, (300, 170), (980, 710), 255, -1)

    with pytest.raises(AIInputError, match="太近"):
        ai_portrait_crop_region(frame, pose, target, GarmentStore().get("coral-tee"))


def test_restore_crop_preserves_every_pixel_outside_region() -> None:
    original = np.full((120, 200, 3), 70, dtype=np.uint8)
    region = AICropRegion(60, 15, 140, 105)
    crop = np.full((90, 80, 3), 210, dtype=np.uint8)
    result = restore_ai_crop(original, crop, region)
    outside = np.ones(original.shape[:2], dtype=bool)
    outside[15:105, 60:140] = False
    assert result.shape == original.shape
    assert np.array_equal(result[outside], original[outside])
    assert np.all(result[20:100, 65:135] == 210)
