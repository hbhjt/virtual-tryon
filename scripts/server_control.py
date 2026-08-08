from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
STATE_PATH = Path(tempfile.gettempdir()) / "virtual-tryon-dev-server.json"
LOG_PATH = Path(tempfile.gettempdir()) / "virtual-tryon-dev-server.log"


def is_running(pid: int) -> bool:
    if sys.platform == "win32":
        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def terminate(pid: int) -> None:
    if sys.platform == "win32":
        process_terminate = 0x0001
        handle = ctypes.windll.kernel32.OpenProcess(process_terminate, False, pid)
        if not handle:
            raise OSError(f"无法打开进程 {pid}")
        try:
            if not ctypes.windll.kernel32.TerminateProcess(handle, 0):
                raise OSError(f"无法终止进程 {pid}")
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
        return
    os.kill(pid, signal.SIGTERM)


def stop() -> int:
    if not STATE_PATH.exists():
        print("没有正在记录的开发服务")
        return 0
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    pid = int(state["pid"])
    if is_running(pid):
        terminate(pid)
        for _ in range(20):
            if not is_running(pid):
                break
            time.sleep(0.1)
        print(f"已停止开发服务（PID {pid}）")
    else:
        print("记录的开发服务已经退出")
    STATE_PATH.unlink(missing_ok=True)
    return 0


def start(port: int) -> int:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if is_running(int(state["pid"])):
            print(f"开发服务已在运行：http://127.0.0.1:{state['port']}")
            return 0
        STATE_PATH.unlink(missing_ok=True)

    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    log = LOG_PATH.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT_DIR,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )
    log.close()
    STATE_PATH.write_text(
        json.dumps({"pid": process.pid, "port": port}), encoding="utf-8"
    )

    health_url = f"http://127.0.0.1:{port}/api/health"
    for _ in range(50):
        if process.poll() is not None:
            print(LOG_PATH.read_text(encoding="utf-8", errors="replace"), file=sys.stderr)
            STATE_PATH.unlink(missing_ok=True)
            return 1
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                health = json.loads(response.read())
                print(
                    f"开发服务已启动：http://127.0.0.1:{port} "
                    f"(pose={health['pose_model_available']}, garments={health['garment_count']})"
                )
                return 0
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.2)
    print("服务启动超时", file=sys.stderr)
    stop()
    return 1


def status() -> int:
    if not STATE_PATH.exists():
        print("开发服务未运行")
        return 1
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    running = is_running(int(state["pid"]))
    print(json.dumps({**state, "running": running}, ensure_ascii=False))
    return 0 if running else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.action == "start":
        return start(args.port)
    if args.action == "stop":
        return stop()
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
