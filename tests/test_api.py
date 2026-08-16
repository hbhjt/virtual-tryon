from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.ai_progress import write_progress
from app.image_utils import write_png_path


client = TestClient(app)


def image_bytes() -> bytes:
    image = np.full((720, 480, 3), 145, dtype=np.uint8)
    # A simple centered person-like silhouette exercises the fallback path deterministically.
    cv2.circle(image, (240, 130), 60, (105, 115, 135), -1)
    cv2.rectangle(image, (145, 190), (335, 620), (90, 105, 120), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_health_and_catalog() -> None:
    health = client.get("/api/health")
    catalog = client.get("/api/garments")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["ai"]["backend"].startswith("CatVTON")
    assert isinstance(health.json()["ai"]["installed"], bool)
    assert health.json()["ai"]["resolution"] == "512×768"
    assert health.json()["ai"]["default_steps"] == 12
    assert catalog.status_code == 200
    assert len(catalog.json()["items"]) >= 3


def test_analyze_returns_pose_and_quality_metrics() -> None:
    response = client.post(
        "/api/analyze", files={"image": ("person.jpg", image_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200
    payload = response.json()
    assert "pose" in payload
    assert 0 <= payload["metrics"]["score"] <= 1
    assert payload["elapsed_ms"] >= 0


def test_tryon_creates_downloadable_result() -> None:
    catalog = client.get("/api/garments").json()["items"]
    response = client.post(
        "/api/tryon",
        data={"garment_id": catalog[0]["id"], "garment_scale": "1.15"},
        files={"image": ("person.jpg", image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_url"].startswith("/outputs/tryon-")
    assert payload["mode"] == "fast"
    assert payload["garment_scale"] == 1.15
    result = client.get(payload["image_url"])
    assert result.status_code == 200
    assert result.headers["content-type"].startswith("image/jpeg")
    assert len(result.content) > 10_000


def test_burst_tryon_selects_one_of_three_frames() -> None:
    catalog = client.get("/api/garments").json()["items"]
    photos = [
        ("images", (f"frame-{index}.jpg", image_bytes(), "image/jpeg"))
        for index in range(3)
    ]
    response = client.post(
        "/api/tryon/burst", data={"garment_id": catalog[0]["id"]}, files=photos
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["burst_frames"] == 3
    assert payload["selected_frame"] in (0, 1, 2)
    assert client.get(payload["image_url"]).status_code == 200


def test_scale_existing_result_keeps_session_and_does_not_need_image_upload() -> None:
    garment_id = client.get("/api/garments").json()["items"][0]["id"]
    initial = client.post(
        "/api/tryon",
        data={"garment_id": garment_id, "garment_scale": "1.0"},
        files={"image": ("person.jpg", image_bytes(), "image/jpeg")},
    )
    assert initial.status_code == 200
    initial_payload = initial.json()
    initial_bytes = client.get(initial_payload["image_url"]).content

    resized = client.post(
        "/api/tryon/scale",
        data={"result_id": initial_payload["id"], "garment_scale": "1.2"},
    )
    assert resized.status_code == 200
    payload = resized.json()
    assert payload["id"] == initial_payload["id"]
    assert payload["image_url"] == initial_payload["image_url"]
    assert payload["garment_scale"] == 1.2
    assert payload["placement_locked"] is True
    assert client.get(payload["image_url"]).content != initial_bytes


def test_invalid_image_is_rejected() -> None:
    response = client.post(
        "/api/analyze", files={"image": ("bad.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400


def test_ai_request_exposes_completed_progress(monkeypatch) -> None:
    job_id = "1234567890abcdef1234567890abcdef"

    def fake_ai_tryon(frame, garment, pose, output_path, *, progress_file, **_kwargs):
        write_progress(
            progress_file,
            progress=55,
            stage="generating",
            message="正在生成衣服细节：第 6/12 步",
            step=6,
            total_steps=12,
        )
        write_png_path(output_path, frame)
        return {"generation_seconds": 0.01}

    monkeypatch.setattr("app.main.run_ai_tryon", fake_ai_tryon)
    garment_id = client.get("/api/garments").json()["items"][0]["id"]
    response = client.post(
        "/api/tryon",
        data={"garment_id": garment_id, "mode": "ai", "job_id": job_id},
        files={"image": ("person.jpg", image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["job_id"] == job_id
    progress = client.get(f"/api/tryon/progress/{job_id}")
    assert progress.status_code == 200
    assert progress.json()["status"] == "completed"
    assert progress.json()["progress"] == 100
