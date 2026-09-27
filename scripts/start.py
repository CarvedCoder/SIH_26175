#!/usr/bin/env python3
"""DepthWizard one-command development startup.

Usage:
    uv run scripts/start.py

Local stack:
    - Python dependencies via uv
    - Frontend dependencies via npm
    - Native C++ acceleration built (native/ — non-fatal, pure-Python
      fallback when the toolchain is unavailable)
    - Model assets pre-fetched (released RDAH checkpoint unless
      DW_CKPT_RDAH overrides it; Depth Anything V2 weights unless
      DW_NO_LIVE=1) so the first processing job never stalls
    - PostgreSQL via DATABASE_URL
        * Supabase/external PostgreSQL: no Docker PostgreSQL started
        * localhost PostgreSQL: optionally managed by Docker Compose
    - Object storage (RustFS, S3-compatible):
        * local Docker RustFS when S3_ENDPOINT points to localhost
        * external S3 endpoint when S3_ENDPOINT points elsewhere
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
VENV = ROOT / ".venv"
PID_FILE = ROOT / "data" / ".dw-dev.pids"

# Ports are resolved lazily via backend_port()/frontend_port() AFTER
# load_env_file() runs, so DW_PORT/VITE_PORT set in .env are honored.
DEFAULT_BACKEND_PORT = int(os.environ.get("DW_PORT", "8010"))
DEFAULT_FRONTEND_PORT = int(os.environ.get("VITE_PORT", "5173"))


def backend_port() -> int:
    return int(os.environ.get("DW_PORT", str(DEFAULT_BACKEND_PORT)))


def frontend_port() -> int:
    return int(os.environ.get("VITE_PORT", str(DEFAULT_FRONTEND_PORT)))

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


def backend_healthy(port: int) -> bool:
    """True when something answering on ``port`` is OUR backend (its
    /health reports status ok + a version)."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=2
        ) as resp:
            data = json.loads(resp.read().decode())
        return data.get("status") == "ok" and "version" in data
    except Exception:
        return False


def frontend_healthy(port: int) -> bool:
    """True when something answering on ``port`` serves the Vite app."""
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/", timeout=2
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


def reap_orphans() -> None:
    """Terminate dev servers left over from a previous run that died
    without cleanup (SIGKILL, closed terminal). Children are recorded in
    ``data/.dw-dev.pids``."""
    if not PID_FILE.is_file():
        return

    try:
        pids = [int(p) for p in PID_FILE.read_text().split() if p.strip()]
    except ValueError:
        pids = []

    stopped = 0
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            stopped += 1
        except ProcessLookupError:
            continue
        except PermissionError:
            log(f"cannot stop leftover pid {pid} (not ours)")

    PID_FILE.unlink(missing_ok=True)
    if stopped:
        log(f"stopped {stopped} leftover dev server(s)")
        time.sleep(1.5)


def record_children() -> None:
    """Persist child PIDs so a later run can reap orphans."""
    try:
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(
            " ".join(str(p.pid) for p in processes if p.poll() is None)
        )
    except OSError:
        pass


def ensure_venv_interpreter() -> None:
    """Re-exec under the managed interpreter when launched with a foreign
    Python. The in-process PostgreSQL driver check imports the project's
    dependencies, which only exist in ``.venv`` (a bare ``python
    scripts/start.py`` with system Python would otherwise crash with
    ModuleNotFoundError)."""
    venv_python = VENV / "bin" / "python"
    if not venv_python.is_file():
        return
    # No .resolve() here: it would follow the venv symlink to the BASE
    # interpreter and wrongly conclude we are already inside .venv.
    if Path(sys.executable) == venv_python:
        return
    log(f"re-launching under {VENV.name} interpreter…")
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())])


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


def docker_command_optional() -> str | None:
    """Like :func:`docker_command` but returns None instead of failing —
    used where a local-services fallback exists."""
    return shutil.which("docker")


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
# Native acceleration + model assets (build once, then serve)
# ---------------------------------------------------------------------------


def build_native_extension() -> None:
    """Compile the C++ SIMD raster kernels (native/ -> dw_native) so the
    serving stack uses the fast path for tile serving / layer PNGs.

    NEVER fatal: the backend has a documented pure-Python fallback with
    identical outputs (backend/app/services/native_accel.py). A failed
    build only means slower serving, so it logs a warning and continues.
    """
    # pybind11 is a build-time-only dependency; `uv sync` (no extras) can
    # strip it from the venv, so reinstall on demand before compiling.
    try:
        import pybind11  # noqa: F401
    except ImportError:
        uv = shutil.which("uv")
        if uv is None:
            log("WARNING: pybind11 missing and uv not found — skipping native build")
            return
        log("installing build-time dependency pybind11…")
        run([uv, "pip", "install", "--python", sys.executable, "pybind11"])

    log("building native acceleration (native/ -> dw_native)…")

    result = run(
        [sys.executable, str(ROOT / "native" / "build.py")],
        cwd=ROOT,
    )

    if result.returncode != 0:
        log(
            "native extension build failed — continuing with the "
            "pure-Python fallback (identical outputs, slower serving)"
        )
        return

    log("native acceleration ready")


def prefetch_model_assets() -> None:
    """Pre-download model weights so the FIRST processing job never stalls
    inside an HTTP request. All steps are idempotent (cached artifacts are
    detected and skipped) and non-fatal (an offline machine still gets a
    working stack — the downloads just happen lazily on first use).

      * RDAH-Net checkpoint: the released Track1 model is fetched +
        MD5-verified (depthwizard/rdah.py) unless DW_CKPT_RDAH points at a
        local fine-tune, in which case only its presence is checked;
      * Depth Anything V2 backbone: warmed when live inference is enabled.
    """
    # The project is run from the source tree (package = false): the backend
    # gets ROOT on sys.path via `python -m uvicorn`, this script does not.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    rdah_override = os.environ.get("DW_CKPT_RDAH")
    if rdah_override:
        if (ROOT / rdah_override).is_file() or Path(rdah_override).is_file():
            log(f"RDAH checkpoint override present: {rdah_override}")
        else:
            log(
                f"WARNING: DW_CKPT_RDAH={rdah_override} does not exist — "
                "RDAH jobs will fail until it does"
            )
    else:
        try:
            from depthwizard.rdah import ensure_rdah_checkpoint

            path = ensure_rdah_checkpoint(
                str(ROOT / "checkpoints" / "rdah" / "rdah_track1_best_model.pth")
            )
            log(f"RDAH checkpoint ready: {path}")
        except Exception as exc:
            log(
                "WARNING: could not fetch the released RDAH checkpoint "
                f"({exc}) — it will be downloaded on the first RDAH job"
            )

    if os.environ.get("DW_NO_LIVE") == "1":
        log("live DAv2 disabled (DW_NO_LIVE=1) — backbone not warmed")
        return

    backbone_id = os.environ.get(
        "DW_BACKBONE", "depth-anything/Depth-Anything-V2-Base-hf"
    )
    try:
        from depthwizard.backbone import get_backbone

        backbone = get_backbone(model_id=backbone_id, device="cpu")
        if not backbone.loaded:
            log(f"fetching {backbone_id} weights (first run only)…")
        backbone.load()
        log(f"Depth Anything V2 backbone ready: {backbone_id}")
    except Exception as exc:
        log(
            f"WARNING: could not pre-fetch {backbone_id} ({exc}) — "
            "weights will download on the first live inference"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def validate_config() -> None:
    missing: list[str] = []

    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        missing.append("DATABASE_URL")

    storage_backend = os.environ.get("STORAGE_BACKEND", "s3").lower()

    if storage_backend in ("s3", "minio"):  # "minio": legacy alias
        s3_endpoint = os.environ.get("S3_ENDPOINT")

        if not s3_endpoint:
            missing.append("S3_ENDPOINT")

        if not os.environ.get("S3_ACCESS_KEY"):
            missing.append("S3_ACCESS_KEY")

        if not os.environ.get("S3_SECRET_KEY"):
            missing.append("S3_SECRET_KEY")

    if missing:
        fail(
            "missing configuration:\n  - "
            + "\n  - ".join(missing)
            + "\nCheck your .env file."
        )

    # Validate DATABASE_URL without printing credentials.
    if database_url.strip().startswith("sqlite"):
        # Zero-config local dev: the SQLite file is created by the backend
        # on startup (backend/app/db/database.py); no host/port to parse
        # and no database infrastructure to start.
        log(f"database configured: SQLite (local file: "
            f"{database_url.split('sqlite:///')[-1]})")
    else:
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

    if storage_backend in ("s3", "minio"):
        endpoint = os.environ["S3_ENDPOINT"]
        try:
            s3_host, s3_port = parse_host_port(endpoint)
        except ValueError as exc:
            fail(f"invalid S3_ENDPOINT: {exc}")
            return

        log(
            f"S3 object storage configured: "
            f"{'local' if is_local_host(s3_host) else 'external'} "
            f"({s3_host}:{s3_port})"
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


def start_storage() -> None:
    """Start local RustFS only when S3_ENDPOINT points to localhost."""
    storage_backend = os.environ.get("STORAGE_BACKEND", "s3").lower()

    if storage_backend not in ("s3", "minio"):  # "minio": legacy alias
        log("object storage disabled — STORAGE_BACKEND is not 's3'")
        return

    endpoint = os.environ["S3_ENDPOINT"]

    host, port = parse_host_port(endpoint)

    if not is_local_host(host):
        log(f"using external S3 storage at {host}:{port} — Docker RustFS will not be started")

        if not wait_for_port(host, port, timeout=15):
            fail(
                f"external S3 storage is not reachable at {host}:{port}. "
                "Check S3_ENDPOINT and network connectivity."
            )

        log("external S3 storage is reachable")
        return

    docker = docker_command()

    log("starting local RustFS via docker compose…")

    result = run(
        [docker, "compose", "up", "-d", "rustfs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        fail("Docker RustFS startup failed:\n" + result.stderr.strip())

    # Readiness, not just ordering: the backend's startup bucket creation
    # must not race a still-initializing object store.
    if not wait_for_port(host, port, timeout=30):
        fail(f"RustFS did not become reachable at {host}:{port} within 30 seconds")

    log("RustFS is ready")

    global infrastructure_started
    infrastructure_started = True


def start_infra() -> None:
    """Start/check only infrastructure actually required by .env."""
    database_url = os.environ["DATABASE_URL"]

    if database_url.strip().startswith("sqlite"):
        # local dev default: the file is created by the backend on startup;
        # nothing to start or check here.
        log("database: SQLite (local file) — no database infrastructure needed")
    else:
        db_host, _ = parse_host_port(database_url)

        # PostgreSQL
        if is_local_host(db_host):
            start_local_postgres()
        else:
            check_external_database()

    # Object storage (RustFS)
    start_storage()


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


def start_backend() -> None:
    if port_open(backend_port()):
        fail(f"port {backend_port()} is already in use — stop the old backend first")

    uv = shutil.which("uv")

    if uv is None:
        fail("uv not found")

    cmd = [
    sys.executable,
    "-m",
    "uvicorn",
    "backend.app.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    str(backend_port()),
    ]   

    log(f"starting backend on :{backend_port()}…")

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
    )

    processes.append(proc)

    # Give Uvicorn a moment to start.
    if not wait_for_port("127.0.0.1", backend_port(), timeout=30):
        if proc.poll() is not None:
            fail("backend exited during startup. Check the backend output above.")

        fail(f"backend did not become reachable on 127.0.0.1:{backend_port()}")

    log(f"backend ready on http://localhost:{backend_port()}")


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------


def start_frontend() -> None:
    if port_open(frontend_port()):
        if frontend_healthy(frontend_port()):
            log(f"frontend already running on :{frontend_port()} — reusing it")
            return
        log(f"port {frontend_port()} already in use by another program — frontend not started")
        return

    npm = shutil.which("npm")

    if npm is None:
        log("npm not found — frontend not started")
        return

    log(f"starting frontend dev server on :{frontend_port()}…")

    # Pin the API base to the port THIS script started the backend on, so a
    # stale VITE_API_BASE_URL in the environment can never point the browser
    # at a dead port.
    env = dict(os.environ)
    env["VITE_API_BASE_URL"] = f"http://localhost:{backend_port()}/api/v1"
    env["VITE_PORT"] = str(frontend_port())

    processes.append(
        subprocess.Popen(
            [npm, "run", "dev", "--", "--host", "0.0.0.0"],
            cwd=FRONTEND,
            env=env,
        )
    )
    record_children()


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
    ensure_venv_interpreter()
    ensure_frontend_deps()
    build_native_extension()
    prefetch_model_assets()
    validate_config()
    start_infra()
    start_backend()
    start_frontend()

    database_url = os.environ["DATABASE_URL"]
    if database_url.strip().startswith("sqlite"):
        db_summary = f"sqlite     {database_url.split('sqlite:///')[-1]}"
    else:
        db_host, db_port = parse_host_port(database_url)
        db_summary = f"postgres   {db_host}:{db_port}"

    storage_backend = os.environ.get(
        "STORAGE_BACKEND",
        "s3",
    ).lower()

    print(
        "\n"
        "[start] DepthWizard is up:\n"
        f"          frontend  http://localhost:{frontend_port()}\n"
        f"          backend   http://localhost:{backend_port()} "
        f"(docs: /docs)\n"
        f"          {db_summary}\n"
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