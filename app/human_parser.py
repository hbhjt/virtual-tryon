from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .pose import PoseResult, screen_sides


@dataclass(frozen=True)
class BodyPartMasks:
    person: np.ndarray
    skin: np.ndarray
    arms_hands: np.ndarray
    neck: np.ndarray
    lower_body: np.ndarray
    occlusion: np.ndarray


def _pixel(point, width: int, height: int) -> np.ndarray:
    return np.array([point.x * width, point.y * height], dtype=np.float32)


def _soft(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    return np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)


def _skin_mask(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    y, cr, cb = cv2.split(ycrcb)
    h, s, v = cv2.split(hsv)
    ycrcb_skin = (cr >= 128) & (cr <= 184) & (cb >= 72) & (cb <= 142) & (y > 35)
    hsv_skin = ((h <= 25) | (h >= 170)) & (s >= 18) & (s <= 190) & (v >= 45)
    mask = np.where(ycrcb_skin & hsv_skin, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def parse_body_parts(
    frame: np.ndarray,
    pose: PoseResult,
    garment_template: str,
    sleeve_length: float = 0.62,
) -> BodyPartMasks:
    """CPU-friendly hybrid parsing from MediaPipe silhouette, joints and skin cues."""
    height, width = frame.shape[:2]
    sides = screen_sides(pose)
    sl = _pixel(sides["shoulder_l"], width, height)
    sr = _pixel(sides["shoulder_r"], width, height)
    hl = _pixel(sides["hip_l"], width, height)
    hr = _pixel(sides["hip_r"], width, height)
    shoulder_span = max(float(np.linalg.norm(sr - sl)), 24.0)

    if pose.segmentation_mask is not None:
        person_u8 = np.clip(
            cv2.resize(pose.segmentation_mask, (width, height), interpolation=cv2.INTER_LINEAR)
            * 255.0,
            0,
            255,
        ).astype(np.uint8)
        person_u8 = cv2.GaussianBlur(person_u8, (0, 0), 1.2)
    else:
        person_u8 = np.full((height, width), 255, dtype=np.uint8)

    skin_u8 = _skin_mask(frame)
    arm_region = np.zeros((height, width), dtype=np.uint8)
    arm_width = max(10, round(shoulder_span * 0.15))
    hand_radius = max(8, round(shoulder_span * 0.085))
    is_long_sleeve = garment_template == "long_sleeve"
    for side in ("l", "r"):
        shoulder = _pixel(sides[f"shoulder_{side}"], width, height)
        elbow = _pixel(sides[f"elbow_{side}"], width, height)
        wrist = _pixel(sides[f"wrist_{side}"], width, height)
        if not is_long_sleeve:
            cuff = shoulder + (elbow - shoulder) * float(np.clip(sleeve_length, 0.42, 0.82))
            cv2.line(
                arm_region,
                tuple(np.int32(cuff)),
                tuple(np.int32(elbow)),
                255,
                arm_width,
                cv2.LINE_AA,
            )
            cv2.line(
                arm_region,
                tuple(np.int32(elbow)),
                tuple(np.int32(wrist)),
                255,
                max(8, round(arm_width * 0.82)),
                cv2.LINE_AA,
            )
        cv2.circle(arm_region, tuple(np.int32(wrist)), hand_radius, 255, -1, cv2.LINE_AA)

    arms_hands_u8 = cv2.bitwise_and(arm_region, skin_u8)
    arms_hands_u8 = cv2.bitwise_and(arms_hands_u8, person_u8)

    torso_down = ((hl - sl) + (hr - sr)) * 0.5
    neck_top = (sl + sr) * 0.5 - torso_down * 0.18
    neck_left = sl + (sr - sl) * 0.35
    neck_right = sl + (sr - sl) * 0.65
    neck_bottom = (sl + sr) * 0.5 + torso_down * 0.08
    neck_poly = np.array(
        [neck_top, neck_right, neck_bottom + (sr - sl) * 0.08, neck_bottom - (sr - sl) * 0.08, neck_left],
        dtype=np.int32,
    )
    neck_region = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(neck_region, neck_poly, 255, cv2.LINE_AA)
    neck_u8 = cv2.bitwise_and(cv2.bitwise_and(neck_region, skin_u8), person_u8)

    lower_u8 = np.zeros((height, width), dtype=np.uint8)
    hip_y = int(np.clip(((hl[1] + hr[1]) * 0.5) - shoulder_span * 0.02, 0, height - 1))
    lower_u8[hip_y:, :] = person_u8[hip_y:, :]

    # Upper-body try-on must never restore trousers or the old garment hem.
    parts = np.maximum(arms_hands_u8, neck_u8)
    return BodyPartMasks(
        person=_soft(person_u8, 0.8),
        skin=_soft(skin_u8, 0.8),
        arms_hands=_soft(arms_hands_u8, 0.65),
        neck=_soft(neck_u8, 0.65),
        lower_body=_soft(lower_u8, 1.4),
        occlusion=_soft(parts, 0.65),
    )
