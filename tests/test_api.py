from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


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
        data={"garment_id": catalog[0]["id"]},
        files={"image": ("person.jpg", image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_url"].startswith("/outputs/tryon-")
    assert payload["mode"] == "fast"
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


def test_invalid_image_is_rejected() -> None:
    response = client.post(
        "/api/analyze", files={"image": ("bad.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
