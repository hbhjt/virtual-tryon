from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import GARMENTS_DIR, MAX_IMAGE_SIDE, MODELS_DIR, ROOT_DIR
from app.garments import GarmentStore
from app.image_utils import encode_jpeg, read_image_path, resize_long_side, write_png_path
from app.pose import PoseEstimator, evaluate_frame
from app.tryon import compose_tryon


INPUT_DIR = ROOT_DIR / "samples" / "inputs"
OUTPUT_DIR = ROOT_DIR / "samples" / "outputs"


def panel(image: np.ndarray, label: str, width: int = 360, height: int = 540) -> np.ndarray:
    canvas = np.full((height + 48, width, 3), 242, dtype=np.uint8)
    resized, scale = resize_long_side(image, max(width, height))
    if resized.shape[1] > width or resized.shape[0] > height:
        fit = min(width / resized.shape[1], height / resized.shape[0])
        resized = cv2.resize(
            resized,
            (round(resized.shape[1] * fit), round(resized.shape[0] * fit)),
            interpolation=cv2.INTER_AREA,
        )
    y0 = 48 + (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    cv2.putText(
        canvas,
        label,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (42, 42, 42),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    store = GarmentStore(GARMENTS_DIR)
    catalog = {item.garment_id: item for item in store.list()}
    estimator = PoseEstimator(MODELS_DIR / "pose_landmarker_lite.task")
    if not estimator.available:
        raise RuntimeError(estimator.load_error or "姿态模型不可用")

    people = {
        "woman": INPUT_DIR / "person-woman-front.png",
        "man": INPUT_DIR / "person-man-front.png",
    }
    garment_ids = ("coral-tee", "ocean-polo", "cream-sweater", "midnight-jacket")
    manifest: dict[str, object] = {"pose_model": True, "samples": []}

    for person_id, input_path in people.items():
        original = read_image_path(input_path, cv2.IMREAD_COLOR)
        if original is None:
            raise RuntimeError(f"无法读取 {input_path}")
        original, _ = resize_long_side(original, MAX_IMAGE_SIDE)
        pose_input, _ = resize_long_side(original, 640)
        pose = estimator.detect(pose_input, allow_fallback=False)
        if not pose.detected:
            raise RuntimeError(f"没有在 {input_path.name} 中检测到人体姿态")
        quality = evaluate_frame(pose_input, pose)
        panels = [panel(original, "ORIGINAL")]

        for garment_id in garment_ids:
            started = time.perf_counter()
            result = compose_tryon(original, catalog[garment_id], pose)
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
            filename = f"{person_id}-{garment_id}.jpg"
            (OUTPUT_DIR / filename).write_bytes(encode_jpeg(result, quality=94))
            panels.append(panel(result, garment_id.upper()))
            manifest["samples"].append(
                {
                    "person": person_id,
                    "input": f"../inputs/{input_path.name}",
                    "garment": garment_id,
                    "output": filename,
                    "elapsed_ms": elapsed_ms,
                    "quality_score": quality["score"],
                    "pose_detected": True,
                }
            )

        comparison = np.hstack(panels)
        write_png_path(OUTPUT_DIR / f"comparison-{person_id}.png", comparison)

        ai_path = OUTPUT_DIR / f"{person_id}-coral-tee-ai.png"
        fast_path = OUTPUT_DIR / f"{person_id}-coral-tee.jpg"
        if ai_path.exists() and fast_path.exists():
            ai_result = read_image_path(ai_path, cv2.IMREAD_COLOR)
            fast_result = read_image_path(fast_path, cv2.IMREAD_COLOR)
            if ai_result is not None and fast_result is not None:
                fast_ai_comparison = np.hstack(
                    [
                        panel(original, "ORIGINAL"),
                        panel(fast_result, "FAST / HARD EDGE"),
                        panel(ai_result, "AI / 512x768 / 12 STEPS"),
                    ]
                )
                write_png_path(
                    OUTPUT_DIR / f"comparison-{person_id}-fast-ai.png",
                    fast_ai_comparison,
                )

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
