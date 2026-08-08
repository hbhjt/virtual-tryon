from __future__ import annotations

import numpy as np

from app.garments import GarmentStore


def test_default_catalog_works_under_unicode_path(tmp_path) -> None:
    store = GarmentStore(tmp_path / "中文衣橱")

    items = store.list()

    assert len(items) == 3
    assert all(item.image_path.exists() for item in items)
    assert all(item.image_path.stat().st_size > 1_000 for item in items)


def test_transparent_garment_can_be_imported(tmp_path) -> None:
    store = GarmentStore(tmp_path / "衣橱")
    image = np.zeros((800, 700, 4), dtype=np.uint8)
    image[80:740, 120:580, :3] = (40, 180, 90)
    image[80:740, 120:580, 3] = 255

    imported = store.add("绿色测试上衣", image)

    assert imported.name == "绿色测试上衣"
    assert imported.image_path.exists()
    assert imported.image_path.stat().st_size > 1_000
    assert len(store.list()) == 4

