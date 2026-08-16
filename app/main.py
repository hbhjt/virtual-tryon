from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .ai_tryon import AIInputError, backend_status, run_ai_tryon
from .ai_progress import normalize_job_id, progress_path, read_progress, write_progress
from .config import (
    ANALYZE_MAX_SIDE,
    GARMENTS_DIR,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_SIDE,
    OUTPUTS_DIR,
    POSE_MODEL_PATH,
    STATIC_DIR,
    ensure_directories,
)
from .garments import Garment, GarmentStore
from .image_utils import InvalidImageError, decode_image, encode_jpeg, resize_long_side
from .pose import PoseEstimator, PoseResult, evaluate_frame
from .tryon import compose_tryon


ensure_directories()
garments = GarmentStore()
pose_estimator = PoseEstimator(POSE_MODEL_PATH)


@dataclass
class FastTryonSession:
    frame: np.ndarray
    garment: Garment
    pose: PoseResult
    output_path: Path


FAST_SESSION_LIMIT = 4
fast_tryon_sessions: OrderedDict[str, FastTryonSession] = OrderedDict()

app = FastAPI(
    title="智能柜 CPU 虚拟换衣",
    version=__version__,
    description="从摄像头视频流选择最佳帧，并在 CPU 上完成上衣几何换装。",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/garments", StaticFiles(directory=GARMENTS_DIR), name="garments")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


async def _read_upload(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 12MB")
    return data


def _clamp_garment_scale(value: float) -> float:
    value = float(value)
    return max(0.8, min(1.6, value)) if math.isfinite(value) else 1.0


def _remember_fast_session(
    result_id: str,
    frame: np.ndarray,
    garment: Garment,
    pose: PoseResult,
    output_path: Path,
) -> None:
    fast_tryon_sessions[result_id] = FastTryonSession(
        frame=frame.copy(),
        garment=garment,
        pose=pose,
        output_path=output_path,
    )
    fast_tryon_sessions.move_to_end(result_id)
    while len(fast_tryon_sessions) > FAST_SESSION_LIMIT:
        fast_tryon_sessions.popitem(last=False)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": __version__,
        "pose_model_available": pose_estimator.available,
        "pose_model_error": pose_estimator.load_error,
        "garment_count": len(garments.list()),
        "ai": backend_status().as_dict(),
    }


@app.get("/api/tryon/progress/{job_id}")
def ai_tryon_progress(job_id: str) -> dict[str, object]:
    try:
        normalized = normalize_job_id(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    payload = read_progress(progress_path(normalized))
    if payload is None:
        payload = {
            "status": "queued",
            "progress": 1.0,
            "stage": "queued",
            "message": "任务已提交，正在等待本地 CPU",
        }
    return {"job_id": normalized, **payload}


@app.get("/api/garments")
def list_garments() -> dict[str, object]:
    return {"items": [garment.public_dict() for garment in garments.list()]}


@app.post("/api/garments", status_code=201)
async def add_garment(
    name: str = Form("自定义上衣"), image: UploadFile = File(...)
) -> dict[str, object]:
    try:
        decoded = decode_image(await _read_upload(image), unchanged=True)
        garment = garments.add(name, decoded)
        return garment.public_dict()
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)) -> dict[str, object]:
    started = time.perf_counter()
    try:
        frame = decode_image(await _read_upload(image))
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    frame, _ = resize_long_side(frame, ANALYZE_MAX_SIDE)
    pose = pose_estimator.detect(frame, allow_fallback=True)
    metrics = evaluate_frame(frame, pose)
    return {
        "pose": pose.as_dict(),
        "metrics": metrics,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }


@app.post("/api/tryon")
async def tryon(
    garment_id: str = Form(...),
    mode: str = Form("fast"),
    garment_scale: float = Form(1.0),
    job_id: str = Form(""),
    image: UploadFile = File(...),
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        garment = garments.get(garment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="服装不存在") from exc
    try:
        frame = decode_image(await _read_upload(image))
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    frame, scale = resize_long_side(frame, MAX_IMAGE_SIDE)
    pose_input, _ = resize_long_side(frame, ANALYZE_MAX_SIDE)
    pose_small = pose_estimator.detect(pose_input, allow_fallback=True)
    # Pose coordinates are normalized, so the same result can be applied to the full image.
    output_id = uuid.uuid4().hex
    mode = "ai" if mode == "ai" else "fast"
    garment_scale = _clamp_garment_scale(garment_scale)
    extension = "png" if mode == "ai" else "jpg"
    filename = f"tryon-{mode}-{output_id}.{extension}"
    output_path = OUTPUTS_DIR / filename
    ai_metrics = None
    ai_job_id = None
    if mode == "ai":
        try:
            ai_job_id = normalize_job_id(job_id or uuid.uuid4().hex)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        ai_progress_file = progress_path(ai_job_id)
        write_progress(
            ai_progress_file,
            progress=2,
            stage="queued",
            message="照片已接收，正在启动 AI 任务",
        )
        try:
            ai_metrics = await asyncio.to_thread(
                run_ai_tryon,
                frame,
                garment,
                pose_small,
                output_path,
                progress_file=ai_progress_file,
            )
        except AIInputError as exc:
            write_progress(
                ai_progress_file,
                progress=0,
                stage="input_error",
                message=str(exc),
                status="error",
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (RuntimeError, TimeoutError) as exc:
            write_progress(
                ai_progress_file,
                progress=0,
                stage="failed",
                message=f"AI 生成失败：{exc}",
                status="error",
            )
            raise HTTPException(status_code=503, detail=f"AI 高质量换装失败：{exc}") from exc
        write_progress(
            ai_progress_file,
            progress=100,
            stage="completed",
            message="AI 高质量试穿完成",
            status="completed",
        )
    else:
        result = compose_tryon(frame, garment, pose_small, garment_scale=garment_scale)
        output_path.write_bytes(encode_jpeg(result, quality=93))
        _remember_fast_session(output_id, frame, garment, pose_small, output_path)
    metrics = evaluate_frame(pose_input, pose_small)
    return {
        "id": output_id,
        "image_url": f"/outputs/{filename}",
        "garment": garment.public_dict(),
        "pose_detected": pose_small.detected,
        "used_fallback": pose_small.fallback,
        "warning": (
            None
            if pose_small.detected
            else "未检测到完整姿态，已按画面中央的标准站姿合成；正面全身照效果更好。"
        ),
        "quality": metrics,
        "input_scale": round(scale, 4),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
        "mode": mode,
        "garment_scale": round(garment_scale, 2),
        "ai_metrics": ai_metrics,
        "job_id": ai_job_id,
    }


@app.post("/api/tryon/burst")
async def tryon_burst(
    garment_id: str = Form(...),
    mode: str = Form("fast"),
    garment_scale: float = Form(1.0),
    job_id: str = Form(""),
    images: list[UploadFile] = File(...),
) -> dict[str, object]:
    """Choose the best frame from a short camera burst before compositing."""
    started = time.perf_counter()
    if not 1 <= len(images) <= 5:
        raise HTTPException(status_code=422, detail="连拍图片数量必须为 1～5 张")
    try:
        garment = garments.get(garment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="服装不存在") from exc

    candidates: list[tuple[float, object, object, dict[str, object]]] = []
    for upload in images:
        try:
            frame = decode_image(await _read_upload(upload))
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        frame, _ = resize_long_side(frame, MAX_IMAGE_SIDE)
        pose_input, _ = resize_long_side(frame, ANALYZE_MAX_SIDE)
        pose = pose_estimator.detect(pose_input, allow_fallback=True)
        metrics = evaluate_frame(pose_input, pose)
        rank = float(metrics["score"]) * 0.68 + float(metrics["sharpness"]) * 0.32
        candidates.append((rank, frame, pose, metrics))

    best_index = max(range(len(candidates)), key=lambda index: candidates[index][0])
    _, best_frame, best_pose, best_metrics = candidates[best_index]
    output_id = uuid.uuid4().hex
    mode = "ai" if mode == "ai" else "fast"
    garment_scale = _clamp_garment_scale(garment_scale)
    extension = "png" if mode == "ai" else "jpg"
    filename = f"tryon-{mode}-{output_id}.{extension}"
    output_path = OUTPUTS_DIR / filename
    ai_metrics = None
    ai_job_id = None
    if mode == "ai":
        try:
            ai_job_id = normalize_job_id(job_id or uuid.uuid4().hex)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        ai_progress_file = progress_path(ai_job_id)
        write_progress(
            ai_progress_file,
            progress=2,
            stage="queued",
            message="已选择最佳画面，正在启动 AI 任务",
        )
        try:
            ai_metrics = await asyncio.to_thread(
                run_ai_tryon,
                best_frame,
                garment,
                best_pose,
                output_path,
                progress_file=ai_progress_file,
            )
        except AIInputError as exc:
            write_progress(
                ai_progress_file,
                progress=0,
                stage="input_error",
                message=str(exc),
                status="error",
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (RuntimeError, TimeoutError) as exc:
            write_progress(
                ai_progress_file,
                progress=0,
                stage="failed",
                message=f"AI 生成失败：{exc}",
                status="error",
            )
            raise HTTPException(status_code=503, detail=f"AI 高质量换装失败：{exc}") from exc
        write_progress(
            ai_progress_file,
            progress=100,
            stage="completed",
            message="AI 高质量试穿完成",
            status="completed",
        )
    else:
        result = compose_tryon(
            best_frame,
            garment,
            best_pose,
            garment_scale=garment_scale,
        )
        output_path.write_bytes(encode_jpeg(result, quality=93))
        _remember_fast_session(output_id, best_frame, garment, best_pose, output_path)
    return {
        "id": output_id,
        "image_url": f"/outputs/{filename}",
        "garment": garment.public_dict(),
        "pose_detected": best_pose.detected,
        "used_fallback": best_pose.fallback,
        "warning": (
            None
            if best_pose.detected
            else "未检测到完整姿态，已按画面中央的标准站姿合成；正面全身照效果更好。"
        ),
        "quality": best_metrics,
        "burst_frames": len(candidates),
        "selected_frame": best_index,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
        "mode": mode,
        "garment_scale": round(garment_scale, 2),
        "ai_metrics": ai_metrics,
        "job_id": ai_job_id,
    }


@app.post("/api/tryon/scale")
async def scale_generated_tryon(
    result_id: str = Form(...),
    garment_scale: float = Form(...),
) -> dict[str, object]:
    """Resize an existing fast garment around its locked center without redetecting pose."""
    started = time.perf_counter()
    session = fast_tryon_sessions.get(result_id)
    if session is None:
        raise HTTPException(status_code=404, detail="该试穿结果已失效，请重新生成一次快速预览")
    fast_tryon_sessions.move_to_end(result_id)
    garment_scale = _clamp_garment_scale(garment_scale)
    result = compose_tryon(
        session.frame,
        session.garment,
        session.pose,
        garment_scale=garment_scale,
    )
    session.output_path.write_bytes(encode_jpeg(result, quality=93))
    return {
        "id": result_id,
        "image_url": f"/outputs/{session.output_path.name}",
        "mode": "fast",
        "garment_scale": round(garment_scale, 2),
        "placement_locked": True,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }
