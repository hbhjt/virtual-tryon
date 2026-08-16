from __future__ import annotations

import cv2
import numpy as np

from .garments import DEFAULT_ANCHORS, Garment
from .human_parser import parse_body_parts
from .image_utils import read_image_path
from .pose import PoseResult, screen_sides


ANCHOR_ORDER = [
    "sleeve_l",
    "shoulder_l",
    "neck_l",
    "neck_r",
    "shoulder_r",
    "sleeve_r",
    "armpit_r",
    "hem_r",
    "hem_l",
    "armpit_l",
]


def _point(point, width: int, height: int) -> np.ndarray:
    return np.array([point.x * width, point.y * height], dtype=np.float32)


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector / max(length, 1e-6)


def _target_geometry(
    pose: PoseResult,
    width: int,
    height: int,
    fit: float = 1.0,
    hem_ratio: float = 1.04,
    shoulder_extension: float = 0.14,
    shoulder_lift: float = 0.045,
    sleeve_length: float = 0.62,
    hem_width_ratio: float = 0.54,
    hem_overlap: float = 0.055,
    collar_width: float = 0.30,
    collar_depth: float = 0.035,
) -> dict[str, np.ndarray]:
    sides = screen_sides(pose)
    sl_raw = _point(sides["shoulder_l"], width, height)
    sr_raw = _point(sides["shoulder_r"], width, height)
    el = _point(sides["elbow_l"], width, height)
    er = _point(sides["elbow_r"], width, height)
    wl = _point(sides["wrist_l"], width, height)
    wr = _point(sides["wrist_r"], width, height)
    hl_raw = _point(sides["hip_l"], width, height)
    hr_raw = _point(sides["hip_r"], width, height)

    shoulder_center = (sl_raw + sr_raw) * 0.5
    across = sr_raw - sl_raw
    shoulder_span = max(float(np.linalg.norm(across)), 20.0)
    across_unit = _unit(across)
    torso_down_l = hl_raw - sl_raw
    torso_down_r = hr_raw - sr_raw
    torso_down = (torso_down_l + torso_down_r) * 0.5

    sl = sl_raw - across_unit * shoulder_span * shoulder_extension * fit - torso_down * shoulder_lift
    sr = sr_raw + across_unit * shoulder_span * shoulder_extension * fit - torso_down * shoulder_lift
    shoulder_peak_l = sl_raw - across_unit * shoulder_span * 0.025 - torso_down * (shoulder_lift + 0.035)
    shoulder_peak_r = sr_raw + across_unit * shoulder_span * 0.025 - torso_down * (shoulder_lift + 0.035)
    neck_center = shoulder_center - torso_down * (0.105 - collar_depth * 0.25)
    neck_l = neck_center - across * (collar_width * 0.5)
    neck_r = neck_center + across * (collar_width * 0.5)

    armpit_l_raw = sl_raw + torso_down_l * 0.31 + across * 0.02
    armpit_r_raw = sr_raw + torso_down_r * 0.31 - across * 0.02
    armpit_center = (armpit_l_raw + armpit_r_raw) * 0.5
    armpit_half_width = shoulder_span * 0.66 * fit
    armpit_l = armpit_center - across_unit * armpit_half_width
    armpit_r = armpit_center + across_unit * armpit_half_width

    torso_height = float(np.mean([np.linalg.norm(torso_down_l), np.linalg.norm(torso_down_r)]))
    drop = max(torso_height * hem_overlap, shoulder_span * 0.045)
    down_unit = _unit(torso_down)
    hem_l_raw = hl_raw - across_unit * shoulder_span * 0.05 + down_unit * drop
    hem_r_raw = hr_raw + across_unit * shoulder_span * 0.05 + down_unit * drop
    hem_l_raw = sl + (hem_l_raw - sl) * hem_ratio
    hem_r_raw = sr + (hem_r_raw - sr) * hem_ratio
    hem_center = (hem_l_raw + hem_r_raw) * 0.5
    hem_half_width = max(
        float(np.linalg.norm(hr_raw - hl_raw)) * 0.72,
        shoulder_span * hem_width_ratio,
    ) * fit
    hem_l = hem_center - across_unit * hem_half_width
    hem_r = hem_center + across_unit * hem_half_width

    sleeve_l = sl_raw + (el - sl_raw) * sleeve_length - across_unit * shoulder_span * 0.15
    sleeve_r = sr_raw + (er - sr_raw) * sleeve_length + across_unit * shoulder_span * 0.15
    sleeve_inner_l = sl_raw + (el - sl_raw) * max(0.38, sleeve_length - 0.06) + across_unit * shoulder_span * 0.08
    sleeve_inner_r = sr_raw + (er - sr_raw) * max(0.38, sleeve_length - 0.06) - across_unit * shoulder_span * 0.08

    geometry = {
        "sleeve_l": sleeve_l,
        "shoulder_l": sl,
        "shoulder_peak_l": shoulder_peak_l,
        "neck_l": neck_l,
        "neck_center": neck_center,
        "neck_r": neck_r,
        "shoulder_r": sr,
        "shoulder_peak_r": shoulder_peak_r,
        "sleeve_r": sleeve_r,
        "sleeve_inner_l": sleeve_inner_l,
        "sleeve_inner_r": sleeve_inner_r,
        "armpit_l": armpit_l,
        "armpit_r": armpit_r,
        "hem_l": hem_l,
        "hem_r": hem_r,
    }

    def arm_sections(
        shoulder: np.ndarray,
        elbow: np.ndarray,
        wrist: np.ndarray,
        armpit: np.ndarray,
        outward: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        def section(center: np.ndarray, tangent: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
            normal = _unit(np.array([-tangent[1], tangent[0]], dtype=np.float32))
            if float(np.dot(normal, outward)) < 0:
                normal = -normal
            return center + normal * radius, center - normal * radius

        top_center = shoulder + (elbow - shoulder) * 0.08
        top_outer, _ = section(top_center, elbow - shoulder, shoulder_span * 0.18)
        elbow_outer, elbow_inner = section(elbow, wrist - shoulder, shoulder_span * 0.14)
        cuff_center = wrist - _unit(wrist - elbow) * shoulder_span * 0.015
        cuff_outer, cuff_inner = section(cuff_center, wrist - elbow, shoulder_span * 0.09)
        return top_outer, armpit, elbow_outer, elbow_inner, cuff_outer, cuff_inner

    left_sections = arm_sections(sl_raw, el, wl, armpit_l, -across_unit)
    right_sections = arm_sections(sr_raw, er, wr, armpit_r, across_unit)
    for side, values in (("l", left_sections), ("r", right_sections)):
        for key, value in zip(
            ("sleeve_top_outer", "sleeve_top_inner", "elbow_outer", "elbow_inner", "cuff_outer", "cuff_inner"),
            values,
        ):
            geometry[f"{key}_{side}"] = value
    return geometry


def target_anchors(pose: PoseResult, width: int, height: int) -> np.ndarray:
    geometry = _target_geometry(pose, width, height)
    return np.array([geometry[name] for name in ANCHOR_ORDER], dtype=np.float32)


def source_anchors(garment: Garment, width: int, height: int) -> np.ndarray:
    return np.array(
        [[garment.anchors[name][0] * width, garment.anchors[name][1] * height] for name in ANCHOR_ORDER],
        dtype=np.float32,
    )


def _source_point(
    garment: Garment,
    name: str,
    width: int,
    height: int,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    raw = garment.anchors.get(name) or DEFAULT_ANCHORS.get(name)
    if raw is None:
        if fallback is None:
            raise KeyError(f"Missing garment anchor: {name}")
        return fallback.astype(np.float32)
    return np.array([raw[0] * width, raw[1] * height], dtype=np.float32)


def _add_grid(
    source_points: list[np.ndarray],
    target_points: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    source_rows: list[list[np.ndarray]],
    target_rows: list[list[np.ndarray]],
) -> None:
    base = len(source_points)
    rows, columns = len(source_rows), len(source_rows[0])
    for source_row, target_row in zip(source_rows, target_rows):
        source_points.extend(source_row)
        target_points.extend(target_row)
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = base + row * columns + column
            b = a + 1
            c = a + columns
            d = c + 1
            triangles.extend(((a, b, c), (b, d, c)))


def _interpolate_row(left: np.ndarray, right: np.ndarray, columns: int = 5) -> list[np.ndarray]:
    return [left * (1.0 - t) + right * t for t in np.linspace(0.0, 1.0, columns)]


def build_category_mesh(
    garment: Garment,
    pose: PoseResult,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    geo = _target_geometry(
        pose,
        target_width,
        target_height,
        garment.fit,
        garment.hem_ratio,
        garment.shoulder_extension,
        garment.shoulder_lift,
        garment.sleeve_length,
        garment.hem_width_ratio,
        garment.hem_overlap,
        garment.collar_width,
        garment.collar_depth,
    )

    ssl = _source_point(garment, "shoulder_l", source_width, source_height)
    ssr = _source_point(garment, "shoulder_r", source_width, source_height)
    snl = _source_point(garment, "neck_l", source_width, source_height)
    snr = _source_point(garment, "neck_r", source_width, source_height)
    sal = _source_point(garment, "armpit_l", source_width, source_height)
    sar = _source_point(garment, "armpit_r", source_width, source_height)
    shl = _source_point(garment, "hem_l", source_width, source_height)
    shr = _source_point(garment, "hem_r", source_width, source_height)

    columns = 7
    source_rows = [
        [ssl, ssl * 0.55 + snl * 0.45, snl, (snl + snr) * 0.5, snr, snr * 0.45 + ssr * 0.55, ssr],
        _interpolate_row(sal, sar, columns),
        _interpolate_row(sal * 0.66 + shl * 0.34, sar * 0.66 + shr * 0.34, columns),
        _interpolate_row(sal * 0.32 + shl * 0.68, sar * 0.32 + shr * 0.68, columns),
        _interpolate_row(shl, shr, columns),
    ]
    target_rows = [
        [geo["shoulder_l"], geo["shoulder_peak_l"], geo["neck_l"], geo["neck_center"], geo["neck_r"], geo["shoulder_peak_r"], geo["shoulder_r"]],
        _interpolate_row(geo["armpit_l"], geo["armpit_r"], columns),
        _interpolate_row(geo["armpit_l"] * 0.66 + geo["hem_l"] * 0.34, geo["armpit_r"] * 0.66 + geo["hem_r"] * 0.34, columns),
        _interpolate_row(geo["armpit_l"] * 0.32 + geo["hem_l"] * 0.68, geo["armpit_r"] * 0.32 + geo["hem_r"] * 0.68, columns),
        _interpolate_row(geo["hem_l"], geo["hem_r"], columns),
    ]
    _add_grid(src, dst, triangles, source_rows, target_rows)

    if garment.template == "long_sleeve":
        for side in ("l", "r"):
            source_sleeve_rows = [
                [
                    _source_point(garment, f"shoulder_{side}", source_width, source_height),
                    _source_point(garment, f"armpit_{side}", source_width, source_height),
                ],
                [
                    _source_point(garment, f"elbow_outer_{side}", source_width, source_height),
                    _source_point(garment, f"elbow_inner_{side}", source_width, source_height),
                ],
                [
                    _source_point(garment, f"cuff_outer_{side}", source_width, source_height),
                    _source_point(garment, f"cuff_inner_{side}", source_width, source_height),
                ],
            ]
            target_sleeve_rows = [
                [geo[f"sleeve_top_outer_{side}"], geo[f"sleeve_top_inner_{side}"]],
                [geo[f"elbow_outer_{side}"], geo[f"elbow_inner_{side}"]],
                [geo[f"cuff_outer_{side}"], geo[f"cuff_inner_{side}"]],
            ]
            _add_grid(src, dst, triangles, source_sleeve_rows, target_sleeve_rows)
    else:
        for side in ("l", "r"):
            shoulder = _source_point(garment, f"shoulder_{side}", source_width, source_height)
            outer = _source_point(garment, f"sleeve_{side}", source_width, source_height)
            inner = _source_point(
                garment,
                f"sleeve_inner_{side}",
                source_width,
                source_height,
                fallback=(outer + _source_point(garment, f"armpit_{side}", source_width, source_height)) * 0.5,
            )
            armpit = _source_point(garment, f"armpit_{side}", source_width, source_height)
            dpoints = [geo[f"shoulder_{side}"], geo[f"sleeve_{side}"], geo[f"sleeve_inner_{side}"], geo[f"armpit_{side}"]]
            spoints = [shoulder, outer, inner, armpit]
            center_s = sum(spoints) * 0.25
            center_d = sum(dpoints) * 0.25
            base = len(src)
            src.extend(spoints + [center_s])
            dst.extend(dpoints + [center_d])
            triangles.extend((base + a, base + b, base + 4) for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)))

    return np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32), triangles


def _warp_triangle(
    source: np.ndarray,
    destination: np.ndarray,
    source_triangle: np.ndarray,
    destination_triangle: np.ndarray,
) -> None:
    raw_sx, raw_sy, raw_sw, raw_sh = cv2.boundingRect(source_triangle)
    sx = max(0, raw_sx - 3)
    sy = max(0, raw_sy - 3)
    source_x1 = min(source.shape[1], raw_sx + raw_sw + 3)
    source_y1 = min(source.shape[0], raw_sy + raw_sh + 3)
    sw, sh = source_x1 - sx, source_y1 - sy
    dx, dy, dw, dh = cv2.boundingRect(destination_triangle)
    if min(sw, sh, dw, dh) <= 0:
        return
    source_local = source_triangle - np.array([sx, sy], dtype=np.float32)
    destination_local = destination_triangle - np.array([dx, dy], dtype=np.float32)
    source_crop = source[sy : sy + sh, sx : sx + sw]
    transform = cv2.getAffineTransform(source_local.astype(np.float32), destination_local.astype(np.float32))
    warped = cv2.warpAffine(
        source_crop,
        transform,
        (dw, dh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    x0, y0 = max(dx, 0), max(dy, 0)
    x1, y1 = min(dx + dw, destination.shape[1]), min(dy + dh, destination.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    crop_x0, crop_y0 = x0 - dx, y0 - dy
    crop_x1, crop_y1 = crop_x0 + (x1 - x0), crop_y0 + (y1 - y0)
    mask = np.zeros((dh, dw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(destination_local), 255, cv2.LINE_8)
    local_mask = mask[crop_y0:crop_y1, crop_x0:crop_x1, None] > 0
    local_warp = warped[crop_y0:crop_y1, crop_x0:crop_x1].astype(np.float32)
    roi = destination[y0:y1, x0:x1]
    destination[y0:y1, x0:x1] = np.where(local_mask, local_warp, roi)


def warp_garment(
    garment_rgba: np.ndarray,
    garment: Garment,
    pose: PoseResult,
    shape: tuple[int, int],
) -> np.ndarray:
    frame_height, frame_width = shape
    source, target, triangles = build_category_mesh(
        garment,
        pose,
        garment_rgba.shape[1],
        garment_rgba.shape[0],
        frame_width,
        frame_height,
    )
    canvas = np.zeros((frame_height, frame_width, 4), dtype=np.float32)
    source_float = garment_rgba.astype(np.float32)
    for triangle in triangles:
        indices = np.array(triangle)
        _warp_triangle(source_float, canvas, source[indices], target[indices])
    rendered = np.clip(canvas, 0, 255).astype(np.uint8)
    # Rasterized adjacent triangles can leave one-pixel cracks after rounding.
    # Close only those internal cracks and propagate neighboring cloth color into them.
    alpha = rendered[:, :, 3]
    closed = cv2.morphologyEx(
        alpha,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    missing = np.where((closed > 20) & (alpha < 12), 255, 0).astype(np.uint8)
    if np.any(missing):
        rendered[:, :, :3] = cv2.inpaint(rendered[:, :, :3], missing, 3, cv2.INPAINT_TELEA)
        rendered[:, :, 3] = np.maximum(alpha, closed)
    rendered = _limit_collar_opening(rendered, garment, pose)
    return rendered


def _rigid_placement_transform(
    garment_rgba: np.ndarray,
    garment: Garment,
    pose: PoseResult,
    shape: tuple[int, int],
    *,
    scale_multiplier: float = 1.0,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    frame_height, frame_width = shape
    alpha = garment_rgba[:, :, 3]
    points = cv2.findNonZero(np.where(alpha >= 12, 255, 0).astype(np.uint8))
    if points is None:
        return np.eye(2, 3, dtype=np.float32), 20.0, 30.0, np.zeros(2, dtype=np.float32)
    x, y, box_width, box_height = cv2.boundingRect(points)

    sides = screen_sides(pose)
    sl = _point(sides["shoulder_l"], frame_width, frame_height)
    sr = _point(sides["shoulder_r"], frame_width, frame_height)
    hl = _point(sides["hip_l"], frame_width, frame_height)
    hr = _point(sides["hip_r"], frame_width, frame_height)
    shoulder_vector = sr - sl
    shoulder_span = max(float(np.linalg.norm(shoulder_vector)), 20.0)
    torso_down = ((hl - sl) + (hr - sr)) * 0.5
    torso_height = max(float(np.linalg.norm(torso_down)), 30.0)

    desired_width = shoulder_span * 1.72 * garment.fit
    desired_height = torso_height * 1.13 * garment.hem_ratio
    base_scale = min(desired_width / max(box_width, 1), desired_height / max(box_height, 1))
    base_scale = float(np.clip(base_scale, 0.08, 2.5))
    scale_factor = float(np.clip(scale_multiplier, 0.8, 1.6))

    source_l = _source_point(garment, "shoulder_l", garment_rgba.shape[1], garment_rgba.shape[0])
    source_r = _source_point(garment, "shoulder_r", garment_rgba.shape[1], garment_rgba.shape[0])
    source_vector = source_r - source_l
    source_angle = float(np.arctan2(source_vector[1], source_vector[0]))
    target_angle = float(np.arctan2(shoulder_vector[1], shoulder_vector[0]))
    angle = target_angle - source_angle
    cosine, sine = np.cos(angle) * base_scale, np.sin(angle) * base_scale
    base_linear = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    source_mid = (source_l + source_r) * 0.5
    target_mid = (sl + sr) * 0.5 - torso_down * 0.035
    base_translation = target_mid - base_linear @ source_mid

    # Once the initial keypoint placement is known, user scaling is performed
    # around the visible garment's fixed geometric center. Therefore neither
    # pose detection nor the garment's center position changes between clicks.
    source_center = np.array(
        [x + box_width * 0.5, y + box_height * 0.5],
        dtype=np.float32,
    )
    fixed_center = base_linear @ source_center + base_translation
    linear = base_linear * scale_factor
    translation = fixed_center - linear @ source_center
    transform = np.column_stack([linear, translation]).astype(np.float32)
    return transform, shoulder_span, torso_height, fixed_center


def place_garment_rigid(
    garment_rgba: np.ndarray,
    garment: Garment,
    pose: PoseResult,
    shape: tuple[int, int],
    *,
    scale_multiplier: float = 1.0,
) -> np.ndarray:
    """Place the unmodified garment with one uniform scale, rotation and translation."""
    frame_height, frame_width = shape
    transform, _, _, _ = _rigid_placement_transform(
        garment_rgba,
        garment,
        pose,
        shape,
        scale_multiplier=scale_multiplier,
    )
    return cv2.warpAffine(
        garment_rgba,
        transform,
        (frame_width, frame_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _limit_collar_opening(rendered: np.ndarray, garment: Garment, pose: PoseResult) -> np.ndarray:
    """Fill only the part of a warped neckline that falls below the configured collar depth."""
    height, width = rendered.shape[:2]
    sides = screen_sides(pose)
    sl = _point(sides["shoulder_l"], width, height)
    sr = _point(sides["shoulder_r"], width, height)
    hl = _point(sides["hip_l"], width, height)
    hr = _point(sides["hip_r"], width, height)
    center = (sl + sr) * 0.5
    across = sr - sl
    down = ((hl - sl) + (hr - sr)) * 0.5
    collar_bottom = center + down * float(np.clip(garment.collar_depth, 0.0, 0.10))
    half_width = across * float(np.clip(garment.collar_width * 0.56, 0.10, 0.24))
    region = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(
        [
            collar_bottom - half_width,
            collar_bottom + half_width,
            collar_bottom + down * 0.20 + half_width * 0.78,
            collar_bottom + down * 0.20 - half_width * 0.78,
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(region, polygon, 255, cv2.LINE_8)
    alpha = rendered[:, :, 3]
    missing = np.where((region > 0) & (alpha < 12), 255, 0).astype(np.uint8)
    if np.any(missing):
        rendered = rendered.copy()
        rendered[:, :, :3] = cv2.inpaint(rendered[:, :, :3], missing, 3, cv2.INPAINT_TELEA)
        rendered[:, :, 3][missing > 0] = 255
    return rendered


def arm_occlusion_mask(pose: PoseResult, width: int, height: int) -> np.ndarray:
    sides = screen_sides(pose)
    sl = _point(sides["shoulder_l"], width, height)
    sr = _point(sides["shoulder_r"], width, height)
    shoulder_span = max(float(np.linalg.norm(sr - sl)), 20.0)
    mask = np.zeros((height, width), dtype=np.uint8)
    thickness = max(8, round(shoulder_span * 0.115))
    hand_radius = max(6, round(shoulder_span * 0.07))
    for side in ("l", "r"):
        elbow = _point(sides[f"elbow_{side}"], width, height)
        wrist = _point(sides[f"wrist_{side}"], width, height)
        cv2.line(mask, tuple(np.int32(elbow)), tuple(np.int32(wrist)), 255, thickness, cv2.LINE_AA)
        cv2.circle(mask, tuple(np.int32(wrist)), hand_radius, 255, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (0, 0), max(1.2, thickness * 0.08))


def _relight_cloth(frame: np.ndarray, cloth: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    region = alpha > 0.18
    scene_level = float(np.median(gray[region])) if np.any(region) else float(np.median(gray))
    cloth_gray = cv2.cvtColor(np.clip(cloth, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    cloth_level = float(np.median(cloth_gray[region])) if np.any(region) else scene_level
    global_gain = float(np.clip((scene_level + 28.0) / (cloth_level + 28.0), 0.76, 1.22))

    sigma = max(9.0, min(frame.shape[:2]) / 32.0)
    illumination = cv2.GaussianBlur(gray, (0, 0), sigma)
    reference = float(np.median(illumination[region])) if np.any(region) else float(np.median(illumination))
    light_field = np.clip((illumination + 25.0) / (reference + 25.0), 0.74, 1.24)
    relit = cloth.astype(np.float32) * global_gain * light_field[:, :, None]

    # Preserve material detail after geometric resampling without producing halos.
    soft = cv2.GaussianBlur(relit, (0, 0), 0.75)
    relit = relit + (relit - soft) * 0.20
    return np.clip(relit, 0, 255)


def crisp_alpha(alpha_u8: np.ndarray, edge_width: float = 1.5) -> np.ndarray:
    """Return an opaque garment with only a one-to-two-pixel anti-aliased rim."""
    binary = np.where(alpha_u8 >= 12, 1, 0).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
    signed = inside - outside
    alpha = np.clip(0.5 + signed / max(2.0 * edge_width, 1.0), 0.0, 1.0)
    alpha[inside >= edge_width] = 1.0
    alpha[outside >= edge_width] = 0.0
    return alpha.astype(np.float32)


def _contact_shadow(alpha: np.ndarray, radius: int = 4) -> np.ndarray:
    binary = np.where(alpha >= 0.5, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    outer = cv2.subtract(cv2.dilate(binary, kernel), binary).astype(np.float32) / 255.0
    return cv2.GaussianBlur(outer, (0, 0), 0.65) * 0.085


def _repair_collar(
    frame: np.ndarray,
    pose: PoseResult,
    garment_alpha: np.ndarray,
    skin: np.ndarray,
) -> np.ndarray:
    """Extend visible neck skin into a small non-skin collar remnant."""
    height, width = frame.shape[:2]
    sides = screen_sides(pose)
    sl = _point(sides["shoulder_l"], width, height)
    sr = _point(sides["shoulder_r"], width, height)
    hl = _point(sides["hip_l"], width, height)
    hr = _point(sides["hip_r"], width, height)
    center = (sl + sr) * 0.5
    down = ((hl - sl) + (hr - sr)) * 0.5
    across = sr - sl

    source_u8 = np.zeros((height, width), dtype=np.uint8)
    source_polygon = np.array(
        [
            center - down * 0.22 - across * 0.07,
            center - down * 0.22 + across * 0.07,
            center - down * 0.07 + across * 0.09,
            center - down * 0.07 - across * 0.09,
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(source_u8, source_polygon, 255, cv2.LINE_8)
    region_u8 = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(
        [
            center - down * 0.105 - across * 0.075,
            center - down * 0.105 + across * 0.075,
            center + down * 0.035 + across * 0.115,
            center + down * 0.035 - across * 0.115,
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(region_u8, polygon, 255, cv2.LINE_8)
    source_mask = (source_u8 > 0) & (skin > 0.72)
    if int(source_mask.sum()) < 20:
        return frame

    skin_color = np.median(frame[source_mask].astype(np.float32), axis=0)
    color_distance = np.linalg.norm(frame.astype(np.float32) - skin_color[None, None, :], axis=2)
    repair = (
        (region_u8 > 0)
        & (garment_alpha < 0.08)
        & ((skin < 0.28) | (color_distance > 24.0))
    ).astype(np.uint8)
    repair = cv2.morphologyEx(
        repair,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    if not np.any(repair):
        return frame

    # Keep the neck from looking flat by retaining the local luminance variation.
    luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    source_level = float(np.median(luminance[source_mask]))
    local_light = cv2.GaussianBlur(luminance, (0, 0), 2.2) - source_level
    patch = np.clip(skin_color[None, None, :] + local_light[:, :, None] * 0.35, 0, 255)
    blend = cv2.GaussianBlur(repair.astype(np.float32), (0, 0), 0.55)
    repaired = frame.astype(np.float32) * (1.0 - blend[:, :, None]) + patch * blend[:, :, None]
    return np.clip(repaired, 0, 255).astype(np.uint8)


def compose_tryon(
    frame: np.ndarray,
    garment: Garment,
    pose: PoseResult,
    garment_scale: float = 1.0,
) -> np.ndarray:
    garment_rgba = read_image_path(garment.image_path, cv2.IMREAD_UNCHANGED)
    if garment_rgba is None:
        raise RuntimeError("服装素材无法读取")
    if garment_rgba.ndim == 2:
        garment_rgba = cv2.cvtColor(garment_rgba, cv2.COLOR_GRAY2BGRA)
    elif garment_rgba.shape[2] == 3:
        garment_rgba = np.dstack([garment_rgba, np.full(garment_rgba.shape[:2], 255, dtype=np.uint8)])

    height, width = frame.shape[:2]
    # Fast preview uses only pose keypoints and one similarity transform. The
    # optional UI adjustment scales the whole source garment uniformly.
    warped = place_garment_rigid(
        garment_rgba,
        garment,
        pose,
        (height, width),
        scale_multiplier=garment_scale,
    )
    alpha = warped[:, :, 3].astype(np.float32) / 255.0
    # Keep the PNG's own outline. Only normalize near-binary interpolation noise;
    # do not close, dilate or erode the source alpha.
    alpha[alpha >= 0.98] = 1.0
    alpha[alpha <= 0.02] = 0.0
    parts = parse_body_parts(frame, pose, garment.template, garment.sleeve_length)

    # Strict silhouette preservation: do not intersect the garment alpha with a
    # person mask and do not punch arm/skin masks back through it. Those operations
    # make a rigidly placed catalog garment look locally deformed.
    cloth = _relight_cloth(frame, warped[:, :, :3].astype(np.float32), alpha)

    # Only the outside rim receives a contact shadow; the cloth RGB is never blurred.
    base_frame = _repair_collar(frame, pose, alpha, parts.skin)
    contact = _contact_shadow(alpha)
    base = base_frame.astype(np.float32) * (1.0 - contact[:, :, None])
    result = base * (1.0 - alpha[:, :, None]) + cloth * alpha[:, :, None]
    return np.clip(result, 0, 255).astype(np.uint8)
