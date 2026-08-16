from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from .config import OUTPUTS_DIR


PROGRESS_DIR = OUTPUTS_DIR / ".ai-progress"


def normalize_job_id(value: str) -> str:
    job_id = value.replace("-", "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("AI 任务编号无效")
    return job_id


def progress_path(job_id: str) -> Path:
    return PROGRESS_DIR / f"{normalize_job_id(job_id)}.json"


def write_progress(
    path: Path,
    *,
    progress: float,
    stage: str,
    message: str,
    status: str = "running",
    step: int | None = None,
    total_steps: int | None = None,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "status": status,
        "progress": round(max(0.0, min(100.0, float(progress))), 1),
        "stage": stage,
        "message": message,
        "updated_at": time.time(),
    }
    if step is not None:
        payload["step"] = int(step)
    if total_steps is not None:
        payload["total_steps"] = int(total_steps)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return payload


def read_progress(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
