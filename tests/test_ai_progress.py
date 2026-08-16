from __future__ import annotations

from app.ai_progress import normalize_job_id, read_progress, write_progress


def test_progress_file_round_trip_is_atomic(tmp_path) -> None:
    path = tmp_path / "progress.json"
    write_progress(
        path,
        progress=47.5,
        stage="generating",
        message="正在生成衣服细节：第 3/12 步",
        step=3,
        total_steps=12,
    )
    payload = read_progress(path)
    assert payload is not None
    assert payload["progress"] == 47.5
    assert payload["stage"] == "generating"
    assert payload["step"] == 3
    assert payload["total_steps"] == 12


def test_job_id_normalization_accepts_uuid_and_rejects_paths() -> None:
    assert normalize_job_id("01234567-89ab-cdef-0123-456789abcdef") == (
        "0123456789abcdef0123456789abcdef"
    )
    try:
        normalize_job_id("../../progress")
    except ValueError:
        pass
    else:
        raise AssertionError("path-like job id must be rejected")
