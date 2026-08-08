from __future__ import annotations

import cv2
import numpy as np

from app.pose import evaluate_frame, fallback_pose


def test_fallback_pose_never_claims_auto_capture_ready() -> None:
    image = np.full((480, 640, 3), 135, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (639, 479), (100, 160, 190), 4)
    result = evaluate_frame(image, fallback_pose())

    assert result["ready"] is False
    assert result["score"] <= 0.54
    assert "画面中央" in result["guidance"]


def test_sharp_pattern_scores_higher_than_blurred_pattern() -> None:
    tile = np.indices((480, 640)).sum(axis=0) // 12 % 2
    sharp = np.repeat((tile * 170 + 50).astype(np.uint8)[:, :, None], 3, axis=2)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 12)

    sharp_score = evaluate_frame(sharp, fallback_pose())["sharpness"]
    blurred_score = evaluate_frame(blurred, fallback_pose())["sharpness"]

    assert sharp_score > blurred_score
    assert sharp_score > 0.8

