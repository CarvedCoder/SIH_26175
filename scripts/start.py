#!/usr/bin/env python3
"""DepthWizard one-command development startup.

Usage:
    uv run scripts/start.py

Local stack:
    - Python dependencies via uv
    - Frontend dependencies via npm
    - PostgreSQL via DATABASE_URL
        * Supabase/external PostgreSQL: no Docker PostgreSQL started
        * localhost PostgreSQL: optionally managed by Docker Compose
    - MinIO:
        * local Docker MinIO when MINIO_ENDPOINT points to localhost
        * external MinIO when MINIO_ENDPOINT points elsewhere
    - FastAPI backend
    - Vite frontend

The script is intentionally idempotent.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"

BACKEND_PORT = int(os.environ.get("DW_PORT", "8000"))
FRONTEND_PORT = int(os.environ.get("VITE_PORT", "5173"))

processes: list[subprocess.Popen] = []
infrastructure_started = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[start] {message}", flush=True)


def fail(message: str) -> None:
    print(f"[start] ERROR: {message}", flush=True)
    shutdown(1, startup_failed=True)


def load_env_file() -> None:
    """Load repo-root .env without overwriting existing environment variables."""
    env_file = ROOT / ".env"

    if not env_file.is_file():
        log(".env not found — using existing environment")
        return

    for line in env_file.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value

    log("loaded .env")


def run(
    cmd: list[str],
    cwd: Path | None = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        **kwargs,
    )


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Return True when TCP host:port accepts a connection."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def wait_for_port(
    host: str,
    port: int,
    timeout: float = 30.0,
) -> bool:
    """Wait until host:port becomes reachable."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if port_open(port, host):
            return True

        time.sleep(0.5)

    return False


def parse_host_port(value: str) -> tuple[str, int]:
    """Extract host/port from a URL or host:port value."""
    value = value.strip()

    # Allow both:
    #   localhost:9000
    #   http://localhost:9000
    #   https://example.com:9000
    if "://" not in value:
        value = f"http://{value}"

    parsed = urlparse(value)

    if not parsed.hostname:
        raise ValueError(f"Could not determine hostname from endpoint: {value}")

    if parsed.port:
        port = parsed.port
    elif parsed.scheme in {"postgres", "postgresql"}:
        port = 5432
    elif parsed.scheme in {"http", "https"}:
        port = 443 if parsed.scheme == "https" else 80
    else:
        raise ValueError(f"Could not determine port from endpoint: {value}")

    return parsed.hostname, port


def is_local_host(host: str) -> bool:
    return host in {
        "localhost",
        "127.0.0.1",
        "::1",
    }


def docker_command() -> str:
    docker = shutil.which("docker")

    if docker is None:
        fail(
            "Docker executable not found in PATH. "
            "Install Docker or configure external services."
        )

    return docker


# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------


def ensure_python_env() -> None:
    uv = shutil.which("uv")

    if uv is None:
        fail("uv not found — install uv first")

    log("syncing Python dependencies…")

    result = run(
        [uv, "sync"],
        cwd=ROOT,
    )

    if result.returncode != 0:
        fail("uv sync failed")

    log("Python dependencies ready")


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


def ensure_frontend_deps() -> None:
    if not (FRONTEND / "package.json").is_file():
        fail("frontend/package.json not found")

    node_modules = FRONTEND / "node_modules"
    lock = FRONTEND / "package-lock.json"
    stamp = node_modules / ".dw-install-stamp"

    if (
        node_modules.is_dir()
        and stamp.is_file()
        and lock.is_file()
        and stamp.read_text() == lock.read_text()
    ):
        log("frontend dependencies up to date")
        return

    npm = shutil.which("npm")

    if npm is None:
        fail("npm not found — install Node.js first")

    if lock.is_file():
        cmd = [npm, "ci"]
    else:
        cmd = [npm, "install"]

    log(f"running {' '.join(cmd)} in frontend/…")

    result = run(
        cmd,
        cwd=FRONTEND,
    )

    if result.returncode != 0:
        fail("frontend dependency installation failed")

    if lock.is_file():
        stamp.write_text(lock.read_text())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def validate_config() -> None:
    missing: list[str] = []

    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        missing.append("DATABASE_URL")

    storage_backend = os.environ.get("STORAGE_BACKEND", "minio").lower()

    if storage_backend == "minio":
        minio_endpoint = os.environ.get("MINIO_ENDPOINT")

        if not minio_endpoint:
            missing.append("MINIO_ENDPOINT")

        if not os.environ.get("MINIO_ACCESS_KEY"):
            missing.append("MINIO_ACCESS_KEY")

        if not os.environ.get("MINIO_SECRET_KEY"):
            missing.append("MINIO_SECRET_KEY")

    if missing:
        fail(
            "missing configuration:\n  - "
            + "\n  - ".join(missing)
            + "\nCheck your .env file."
        )

    # Validate DATABASE_URL without printing credentials.
    try:
        db_host, db_port = parse_host_port(database_url)
    except ValueError as exc:
        fail(f"invalid DATABASE_URL: {exc}")
        return

    log(
        f"database configured: "
        f"{'local' if is_local_host(db_host) else 'external'} "
        f"PostgreSQL ({db_host}:{db_port})"
    )

    log(f"storage backend: {storage_backend}")

    if storage_backend == "minio":
        endpoint = os.environ["MINIO_ENDPOINT"]
        try:
            minio_host, minio_port = parse_host_port(endpoint)
        except ValueError as exc:
            fail(f"invalid MINIO_ENDPOINT: {exc}")
            return

        log(
            f"MinIO configured: "
            f"{'local' if is_local_host(minio_host) else 'external'} "
            f"({minio_host}:{minio_port})"
        )

    log("configuration OK")


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


def check_external_database() -> None:
    """Verify that DATABASE_URL is reachable using the actual DB driver."""
    database_url = os.environ["DATABASE_URL"]

    log("checking PostgreSQL connection…")

    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(
            database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10},
        )

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        engine.dispose()

    except Exception as exc:
        fail(
            "could not connect to PostgreSQL using DATABASE_URL:\n"
            f"  {type(exc).__name__}: {exc}"
        )

    log("PostgreSQL connection OK")


def start_local_postgres() -> None:
    """Start the Docker PostgreSQL service only when explicitly needed."""
    docker = docker_command()

    log("starting local PostgreSQL via docker compose…")

    result = run(
        [docker, "compose", "up", "-d", "db"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        fail("Docker PostgreSQL startup failed:\n" + result.stderr.strip())

    log("local PostgreSQL container started")


def start_minio() -> None:
    """Start local MinIO only when MINIO_ENDPOINT points to localhost."""
    storage_backend = os.environ.get("STORAGE_BACKEND", "minio").lower()

    if storage_backend != "minio":
        log("MinIO disabled — STORAGE_BACKEND is not 'minio'")
        return

    endpoint = os.environ["MINIO_ENDPOINT"]

    host, port = parse_host_port(endpoint)

    if not is_local_host(host):
        log(f"using external MinIO at {host}:{port} — Docker MinIO will not be started")

        if not wait_for_port(host, port, timeout=15):
            fail(
                f"external MinIO is not reachable at {host}:{port}. "
                "Check MINIO_ENDPOINT and network connectivity."
            )

        log("external MinIO is reachable")
        return

    docker = docker_command()

    log("starting local MinIO via docker compose…")

    result = run(
        [docker, "compose", "up", "-d", "minio"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        fail("Docker MinIO startup failed:\n" + result.stderr.strip())

    if not wait_for_port(host, port, timeout=30):
        fail(f"MinIO did not become reachable at {host}:{port} within 30 seconds")

    log("MinIO is ready")

    global infrastructure_started
    infrastructure_started = True


def start_infra() -> None:
    """Start/check only infrastructure actually required by .env."""
    database_url = os.environ["DATABASE_URL"]

    db_host, _ = parse_host_port(database_url)

    # PostgreSQL
    if is_local_host(db_host):
        start_local_postgres()
    else:
        check_external_database()

    # MinIO
    start_minio()


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


def start_backend() -> None:
    if port_open(BACKEND_PORT):
        fail(f"port {BACKEND_PORT} is already in use — stop the old backend first")

    uv = shutil.which("uv")

    if uv is None:
        fail("uv not found")

    cmd = [
        uv,
        "run",
        "uvicorn",
        "backend.app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(BACKEND_PORT),
    ]

    log(f"starting backend on :{BACKEND_PORT}…")

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
    )

    processes.append(proc)

    # Give Uvicorn a moment to start.
    if not wait_for_port("127.0.0.1", BACKEND_PORT, timeout=15):
        if proc.poll() is not None:
            fail("backend exited during startup. Check the backend output above.")

        fail(f"backend did not become reachable on 127.0.0.1:{BACKEND_PORT}")

    log(f"backend ready on http://localhost:{BACKEND_PORT}")


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


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
        subprocess.Popen(
            [npm, "run", "dev", "--", "--host", "0.0.0.0"],
            cwd=FRONTEND,
        )
    )


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def shutdown(
    exit_code: int = 0,
    startup_failed: bool = False,
) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()

    deadline = time.monotonic() + 5

    for proc in processes:
        try:
            remaining = max(0.1, deadline - time.monotonic())
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()

    if startup_failed:
        print(
            "\n[start] stopped.",
            flush=True,
        )
    else:
        print(
            "\n[start] stopped. Docker infrastructure was left running.",
            flush=True,
        )

    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    load_env_file()

    signal.signal(
        signal.SIGINT,
        lambda *_: shutdown(0),
    )

    ensure_python_env()
    ensure_frontend_deps()
    validate_config()
    start_infra()
    start_backend()
    start_frontend()

    database_url = os.environ["DATABASE_URL"]
    db_host, db_port = parse_host_port(database_url)

    storage_backend = os.environ.get(
        "STORAGE_BACKEND",
        "minio",
    ).lower()

    print(
        "\n"
        "[start] DepthWizard is up:\n"
        f"          frontend  http://localhost:{FRONTEND_PORT}\n"
        f"          backend   http://localhost:{BACKEND_PORT} "
        f"(docs: /docs)\n"
        f"          postgres  {db_host}:{db_port}\n"
        f"          storage   {storage_backend}\n"
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
