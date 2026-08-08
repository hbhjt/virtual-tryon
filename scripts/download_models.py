from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12]
        print(f"姿态模型已存在：{MODEL_PATH.name} (sha256 {digest})")
        return 0

    temporary = MODEL_PATH.with_suffix(".download")
    print("正在下载 MediaPipe 轻量姿态模型……")
    try:
        urllib.request.urlretrieve(MODEL_URL, temporary)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("下载到的模型文件不完整")
        temporary.replace(MODEL_PATH)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        print(f"模型下载失败：{exc}", file=sys.stderr)
        print("应用仍可启动，但只能使用手动抓拍和标准站姿合成。", file=sys.stderr)
        return 1
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12]
    print(f"姿态模型下载完成：{MODEL_PATH.name} (sha256 {digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

