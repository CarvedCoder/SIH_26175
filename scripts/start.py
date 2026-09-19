#!/usr/bin/env python3
"""DepthWizard one-command development startup.

    python scripts/start.py

Idempotent orchestration of the full local stack:
    1. load .env (from the repo root) into the environment
    2. create/update the Python virtualenv (.venv) and install deps
    3. install frontend deps (npm ci when package-lock.json exists and is
       in sync, else npm install)
    4. validate required configuration
    5. start infrastructure (PostgreSQL + MinIO) via docker compose
    6. start the FastAPI backend (uvicorn)
    7. start the Vite frontend dev server
    8. print URLs; on Ctrl+C, stop the child processes cleanly

All subprocess calls use argument lists — never shell string concatenation.
Running it twice is safe: containers are reused, servers are not duplicated
(a stale port owner is reported instead).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
FRONTEND = ROOT / "frontend"

BACKEND_PORT = int(os.environ.get("DW_PORT", "8000"))
FRONTEND_PORT = int(os.environ.get("VITE_PORT", "5173"))

processes: list[subprocess.Popen] = []


# ── helpers ────────────────────────────────────────────────────────────────

def log(message: str) -> None:
    print(f"[start] {message}", flush=True)


def fail(message: str) -> None:
    print(f"[start] ERROR: {message}", flush=True)
    shutdown(1)


def load_env_file() -> None:
    """Load repo-root .env (KEY=VALUE lines) without overwriting the
    caller's environment."""
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    log("loaded .env")


def run(cmd: list[str], cwd: Path | None = None, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=False, **kwargs)


def port_open(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


# ── steps ──────────────────────────────────────────────────────────────────

def ensure_python_env() -> None:
    if (VENV / "bin" / "python").exists():
        log("virtualenv found (.venv)")
    else:
        log("creating virtualenv (.venv)…")
        if run([sys.executable, "-m", "venv", str(VENV)]).returncode != 0:
            fail("could not create .venv — is python-venv installed?")
    pip = str(VENV / "bin" / "pip")
    log("installing Python dependencies…")
    if run([pip, "install", "-e", ".", "--quiet"]).returncode != 0:
        fail("pip install failed")


def ensure_frontend_deps() -> None:
    if not (FRONTEND / "package.json").is_file():
        fail("frontend/package.json not found")
    node_modules = FRONTEND / "node_modules"
    lock = FRONTEND / "package-lock.json"
    if node_modules.is_dir():
        # Only reinstall when the lockfile changed since the last install.
        stamp = node_modules / ".dw-install-stamp"
        if stamp.is_file() and lock.is_file() and stamp.read_text() == lock.read_text():
            log("frontend dependencies up to date")
            return
    npm = shutil.which("npm")
    if npm is None:
        fail("npm not found — install Node.js first")
    cmd = [npm, "ci"] if lock.is_file() else [npm, "install"]
    log(f"running {' '.join(cmd)} in frontend/…")
    if run(cmd, cwd=FRONTEND).returncode != 0:
        fail("frontend dependency installation failed")
    if lock.is_file():
        (node_modules / ".dw-install-stamp").write_text(lock.read_text())


def validate_config() -> None:
    missing = []
    # MinIO access requires credentials when the object store is enabled.
    if os.environ.get("STORAGE_BACKEND", "minio") == "minio" and not (
        os.environ.get("MINIO_ACCESS_KEY") and os.environ.get("MINIO_SECRET_KEY")
    ):
        missing.append(
            "MINIO_ACCESS_KEY / MINIO_SECRET_KEY (or set STORAGE_BACKEND=local)"
        )
    if missing:
        fail(
            "missing configuration:\n  - " + "\n  - ".join(missing) +
            "\nCopy .env.example to .env and fill it in."
        )
    log("configuration OK")


def start_infra() -> None:
    compose = shutil.which("docker")
    if compose is None:
        fail("Docker not found — install Docker to run PostgreSQL + MinIO, "
             "or point DATABASE_URL/MINIO_ENDPOINT at external services")
    log("starting PostgreSQL + MinIO via docker compose…")
    result = run(
        [compose, "compose", "up", "-d", "db", "minio"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail(f"docker compose failed:\n{result.stderr.strip()}")
    # Wait for Postgres readiness (backend fails fast without it).
    for _ in range(30):
        ready = run(
            [compose, "compose", "exec", "-T", "db", "pg_isready", "-U", "depthwizard"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if ready.returncode == 0:
            log("PostgreSQL is ready")
            return
        time.sleep(1)
    fail("PostgreSQL did not become ready within 30s")


def start_backend() -> None:
    if port_open(BACKEND_PORT):
        fail(f"port {BACKEND_PORT} is already in use — stop the old backend first")
    uvicorn = str(VENV / "bin" / "uvicorn")
    if not Path(uvicorn).exists():
        uvicorn = str(VENV / "bin" / "python")
        cmd = [uvicorn, "-m", "uvicorn"]
    else:
        cmd = [uvicorn]
    cmd += ["backend.app.main:app", "--host", "0.0.0.0", "--port", str(BACKEND_PORT)]
    log(f"starting backend on :{BACKEND_PORT}…")
    processes.append(subprocess.Popen(cmd, cwd=ROOT))


def start_frontend() -> None:
    if port_open(FRONTEND_PORT):
        log(f"port {FRONTEND_PORT} already in use — skipping frontend dev server")
        return
    npm = shutil.which("npm")
    if npm is None:
        log("npm not found — frontend not started")
        return
    log(f"starting frontend dev server on :{FRONTEND_PORT}…")
    processes.append(
        subprocess.Popen([npm, "run", "dev"], cwd=FRONTEND)
    )


def shutdown(exit_code: int = 0) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 5
    for proc in processes:
        try:
            proc.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.kill()
    print("\n[start] stopped. Infrastructure (db/minio) is left running — "
          "stop it with: docker compose stop", flush=True)
    sys.exit(exit_code)


def main() -> None:
    load_env_file()
    signal.signal(signal.SIGINT, lambda *_: shutdown(0))
    ensure_python_env()
    ensure_frontend_deps()
    validate_config()
    start_infra()
    start_backend()
    start_frontend()

    print(
        "\n[start] DepthWizard is up:\n"
        f"          frontend  http://localhost:{FRONTEND_PORT}\n"
        f"          backend   http://localhost:{BACKEND_PORT}  (docs: /docs)\n"
        "          minio     http://localhost:9001 (console)\n"
        "          postgres  localhost:5432\n"
        "Press Ctrl+C to stop.\n",
        flush=True,
    )
    try:
        while True:
            for proc in processes:
                if proc.poll() is not None:
                    fail("a child process exited unexpectedly — see its output above")
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(0)


if __name__ == "__main__":
    main()
