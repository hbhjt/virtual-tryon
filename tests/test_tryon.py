from __future__ import annotations

import json

import cv2
import numpy as np

from app.garments import DEFAULT_ANCHORS, Garment, GarmentStore
from app.human_parser import parse_body_parts
from app.pose import fallback_pose
from app.tryon import build_category_mesh, compose_tryon, crisp_alpha, place_garment_rigid, target_anchors, warp_garment


def make_test_garment(tmp_path) -> tuple[Garment, np.ndarray]:
    directory = tmp_path / "test-top"
    directory.mkdir()
    rgba = np.zeros((700, 600, 4), dtype=np.uint8)
    polygon = np.array(
        [[48, 203], [180, 112], [258, 102], [342, 102], [420, 112], [552, 203],
         [450, 273], [438, 644], [162, 644], [150, 273]],
        dtype=np.int32,
    )
    cv2.fillPoly(rgba, [polygon], (30, 80, 220, 255))
    image_path = directory / "image.png"
    metadata_path = directory / "metadata.json"
    cv2.imwrite(str(image_path), rgba)
    metadata_path.write_text(json.dumps({"id": "test-top"}), encoding="utf-8")
    garment = Garment("test-top", "测试上衣", "upper_body", image_path, metadata_path, DEFAULT_ANCHORS)
    return garment, rgba


def test_target_anchors_are_ordered_and_inside_typical_frame() -> None:
    anchors = target_anchors(fallback_pose(), 640, 960)
    assert anchors.shape == (10, 2)
    assert np.all(anchors[:, 0] > 0)
    assert np.all(anchors[:, 0] < 640)
    assert np.all(anchors[:, 1] > 0)
    assert np.all(anchors[:, 1] < 960)
    assert anchors[1, 0] < anchors[4, 0]


def test_warp_and_composite_change_torso_pixels(tmp_path) -> None:
    garment, rgba = make_test_garment(tmp_path)
    pose = fallback_pose()
    frame = np.full((960, 640, 3), 180, dtype=np.uint8)

    warped = warp_garment(rgba, garment, pose, frame.shape[:2])
    result = compose_tryon(frame, garment, pose)

    assert np.count_nonzero(warped[:, :, 3]) > 20_000
    changed = np.any(result != frame, axis=2)
    assert np.count_nonzero(changed) > 20_000
    assert result.shape == frame.shape


def test_category_mesh_uses_dense_torso_and_long_sleeve_anchors() -> None:
    store = GarmentStore()
    pose = fallback_pose()
    tee = store.get("coral-tee")
    jacket = store.get("midnight-jacket")

    tee_source, tee_target, tee_triangles = build_category_mesh(tee, pose, 1254, 1254, 640, 960)
    jacket_source, jacket_target, jacket_triangles = build_category_mesh(jacket, pose, 1254, 1254, 640, 960)

    assert tee_source.shape[0] >= 35
    assert jacket_source.shape[0] > tee_source.shape[0]
    assert len(tee_triangles) >= 40
    assert len(jacket_triangles) >= len(tee_triangles)
    assert np.isfinite(tee_target).all()
    assert np.isfinite(jacket_target).all()


def test_fast_alpha_is_opaque_inside_with_at_most_two_pixel_soft_edge() -> None:
    source = np.zeros((80, 80), dtype=np.uint8)
    cv2.rectangle(source, (20, 20), (59, 59), 210, -1)
    alpha = crisp_alpha(source)
    partial = (alpha > 0) & (alpha < 1)
    assert np.all(alpha[23:57, 23:57] == 1.0)
    assert np.count_nonzero(partial[18:62, 18:62]) <= 4 * 44 * 2
    assert np.count_nonzero(partial[16:18]) == 0
    assert np.count_nonzero(partial[62:64]) == 0


def test_upper_body_occlusion_does_not_restore_lower_body() -> None:
    frame = np.full((960, 640, 3), (90, 135, 190), dtype=np.uint8)
    parts = parse_body_parts(frame, fallback_pose(), "short_sleeve", 0.62)
    assert parts.lower_body[820, 320] > 0.9
    assert parts.occlusion[820, 320] == 0.0


def test_rigid_preview_preserves_garment_bounding_box_aspect_ratio(tmp_path) -> None:
    garment, rgba = make_test_garment(tmp_path)
    placed = place_garment_rigid(rgba, garment, fallback_pose(), (960, 640))
    source_box = cv2.boundingRect(cv2.findNonZero((rgba[:, :, 3] > 12).astype(np.uint8)))
    placed_box = cv2.boundingRect(cv2.findNonZero((placed[:, :, 3] > 12).astype(np.uint8)))
    source_ratio = source_box[2] / source_box[3]
    placed_ratio = placed_box[2] / placed_box[3]
    assert abs(source_ratio - placed_ratio) < 0.03
