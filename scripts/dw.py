#!/usr/bin/env python3
"""DepthWizard master CLI — clone-and-run bootstrap + day-2 operations.

    ./dw doctor|setup|start|test|status|logs|stop|restart|clean|help

Design contract:
    * STANDARD LIBRARY ONLY — this file must run on a bare Python >= 3.8 so
      it can bootstrap a machine that has none of the project's deps.
    * ORCHESTRATION, not duplication: the application itself is started by
      scripts/start.py (deps sync, native build, model prefetch, RustFS,
      backend, frontend, cancellation). This CLI prepares prerequisites,
      configures the project, verifies readiness, and then hands over.
    * NEVER silent about large downloads or destructive actions.
    * NEVER installs system packages (no sudo/apt/brew/choco ever runs here).

Exit codes:
    0 success          1 general failure        2 invalid configuration
    3 missing prerequisite                      4 test failure
    5 user cancelled a required action
"""

from __future__ import annotations

import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
FRONTEND = ROOT / "frontend"
PID_FILE = ROOT / "data" / ".dw-dev.pids"
RUSTFS_IMAGE = "rustfs/rustfs:latest"
S3_HEALTH_TIMEOUT = 60

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CONFIG = 2
EXIT_PREREQ = 3
EXIT_TESTS = 4
EXIT_CANCELLED = 5

# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def ok(msg: str) -> None:
    print(f"  [ok] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"  [!!] {msg}", flush=True)


def bad(msg: str) -> None:
    print(f"  [x ] {msg}", flush=True)


def section(title: str) -> None:
    print(f"\n{title}", flush=True)


def fail(msg: str, code: int = EXIT_FAIL, hint: str | None = None) -> "NoReturn":  # type: ignore[name-defined]
    bad(msg)
    if hint:
        print(f"\n{hint}", flush=True)
    sys.exit(code)


def confirm(question: str, assume_yes: bool = False, default: bool = True) -> bool:
    """Ask before installing/downloading/deleting. --yes accepts NORMAL
    confirmations; destructive paths call confirm_destructive instead."""
    if assume_yes:
        print(f"  (auto-yes) {question} -> yes", flush=True)
        return True
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{question} {suffix} ").strip().lower()
    except EOFError:
        answer = ""
    if not answer:
        return default
    return answer in ("y", "yes")


def confirm_destructive(question: str) -> bool:
    """Destructive confirmations are NEVER bypassed by --yes."""
    try:
        answer = input(f"{question}\nType 'delete' to confirm: ").strip().lower()
    except EOFError:
        answer = ""
    return answer == "delete"


def sh(cmd: list, cwd: Path | None = None, check: bool = False,
       capture: bool = False, project_env: bool = False) -> subprocess.CompletedProcess:
    """Run a command. With project_env=True, os.environ is merged with the
    repo .env (file never overrides the real environment) — needed because
    backend Settings read os.environ, not the .env file."""
    env = None
    if project_env:
        env = dict(os.environ)
        for key, value in read_env_file().items():
            env.setdefault(key, value)
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, check=check,
        capture_output=capture, text=True, env=env,
    )


def popen(cmd: list, cwd: Path | None = None) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=str(cwd) if cwd else None)


def http_ok(url: str, timeout: float = 3.0) -> bool:
    """True when a URL answers with any HTTP status (connection is what
    matters; 403/404 still prove the service is up)."""
    try:
        req = urllib.request.Request(url, method="GET")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True  # reached the server; status is the server's business
    except Exception:
        return False


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def read_env_file(path: Path | None = None) -> dict:
    """Parse a .env file into a dict (values only, never exported)."""
    env: dict = {}
    path = path or (ROOT / ".env")
    if not path.is_file():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def which_one(*names: str) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def backend_port(env: dict) -> int:
    """Host-side dev backend port (compose publishes 8000; scripts/start.py
    defaults to DW_PORT=8010)."""
    try:
        return int(env.get("DW_PORT") or os.environ.get("DW_PORT") or 8010)
    except ValueError:
        return 8010


def frontend_port(env: dict) -> int:
    try:
        return int(env.get("VITE_PORT") or os.environ.get("VITE_PORT") or 5173)
    except ValueError:
        return 5173


def backend_healthy(port: int) -> bool:
    try:
        req = urllib.request.Request(f"http://localhost:{port}/api/v1/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def docker_cmd() -> str | None:
    return which_one("docker", "podman")


def compose_running(service: str) -> bool:
    docker = docker_cmd()
    if not docker:
        return False
    try:
        result = sh([docker, "compose", "ps", "--format", "{{.Service}} {{.State}}"],
                    cwd=ROOT, capture=True)
        return any(
            line.split()[0] == service and line.split()[1] == "running"
            for line in result.stdout.splitlines() if len(line.split()) >= 2
        )
    except Exception:
        return False


def venv_ready() -> bool:
    """The venv exists AND the serving stack's core imports work."""
    if not VENV_PYTHON.is_file():
        return False
    probe = (
        "import importlib.util as u; "
        "mods = ['fastapi','uvicorn','sqlalchemy','boto3','yaml']; "
        "raise SystemExit(0 if all(u.find_spec(m) is not None for m in mods) else 1)"
    )
    result = sh([str(VENV_PYTHON), "-c", probe], capture=True, project_env=True)
    return result.returncode == 0


def node_modules_ready(env: dict | None = None) -> bool:
    """Same stamp contract as scripts/start.py: node_modules present and its
    install stamp matches package-lock.json."""
    node_modules = FRONTEND / "node_modules"
    lock = FRONTEND / "package-lock.json"
    stamp = node_modules / ".dw-install-stamp"
    return (
        node_modules.is_dir()
        and stamp.is_file()
        and lock.is_file()
        and stamp.read_text() == lock.read_text()
    )


def s3_endpoint_url(env: dict) -> str:
    endpoint = env.get("S3_ENDPOINT") or os.environ.get("S3_ENDPOINT") or "localhost:9000"
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    secure = (env.get("S3_SECURE") or "false").lower() == "true"
    return f"{'https' if secure else 'http'}://{endpoint}"


def s3_public_url(env: dict) -> str:
    return env.get("S3_PUBLIC_ENDPOINT") or s3_endpoint_url(env)


def checkpoint_paths(env: dict) -> dict:
    """Resolve the serving/RDAH checkpoint paths exactly like
    ProcessingService._resolve_checkpoint (defaults included)."""
    serving = env.get("DW_CKPT") or os.environ.get("DW_CKPT")
    if serving:
        serving_path = Path(serving)
        if not serving_path.is_absolute():
            serving_path = ROOT / serving_path
    else:
        serving_path = first_existing(
            ROOT / "outputs" / "calib_net" / "postproc_flagship" / "best.pt",
            ROOT / "best.pt",
        )
    rdah = env.get("DW_CKPT_RDAH") or os.environ.get("DW_CKPT_RDAH")
    if rdah:
        rdah_path = Path(rdah)
        if not rdah_path.is_absolute():
            rdah_path = ROOT / rdah_path
    else:
        rdah_path = ROOT / "checkpoints" / "rdah" / "rdah_track1_best_model.pth"
    return {
        "serving": serving_path,
        "rdah": rdah_path,
        "serving_explicit": bool(serving),
        "rdah_explicit": bool(rdah),
    }


def backbone_id(env: dict) -> str:
    return (
        env.get("DW_BACKBONE")
        or os.environ.get("DW_BACKBONE")
        or "depth-anything/Depth-Anything-V2-Base-hf"
    )


def backbone_cache_status(env: dict) -> tuple:
    """(present, path) of the DAv2 weights in the local HF cache. The cache
    layout is ~/.cache/huggingface/hub/models--<org>--<model> (with
    HF_HOME honored when set). Presence of the directory means the weights
    were downloaded at least once; a partial download still fails later —
    start.py's prefetch completes it."""
    model_id = backbone_id(env)
    org, _, name = model_id.partition("/")
    hf_home = Path(
        env.get("HF_HOME") or os.environ.get("HF_HOME")
        or (Path.home() / ".cache" / "huggingface")
    )
    cache_dir = hf_home / "hub" / f"models--{org}--{name}"
    return cache_dir.is_dir(), cache_dir


def native_extension_path() -> Path | None:
    build = ROOT / "native" / "build"
    if not build.is_dir():
        return None
    for pattern in ("dw_native*.pyd", "dw_native*.so"):
        matches = sorted(build.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_child_pids() -> list:
    """Host-side dev-server PIDs recorded by scripts/start.py."""
    if not PID_FILE.is_file():
        return []
    pids = []
    for token in PID_FILE.read_text().split():
        try:
            pids.append(int(token))
        except ValueError:
            continue
    return pids


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        result = sh(["tasklist", "/FI", f"PID eq {pid}"], capture=True)
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# dw doctor — read-only diagnostics
# ---------------------------------------------------------------------------

def tool_version(cmd: list) -> str | None:
    try:
        result = sh(cmd, capture=True)
        first = (result.stdout or result.stderr).strip().splitlines()
        return first[0].strip() if first else None
    except Exception:
        return None


def doctor_checks(verbose: bool = False) -> dict:
    """Every check returns (state, detail): state in ok|warn|missing.
    READ-ONLY: doctor must never modify the machine."""
    env = read_env_file()
    results: dict = {}

    # -- system prerequisites
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    py_ok = sys.version_info >= (3, 8)  # the CLI itself; project needs 3.12+
    results["python"] = (py_ok, f"Python {py} ({platform.python_implementation()})")

    uv = which_one("uv")
    results["uv"] = (uv is not None, tool_version(["uv", "--version"]) if uv else "not found")

    node = which_one("node", "node.exe")
    results["node"] = (node is not None, tool_version(["node", "--version"]) if node else "not found")

    npm = which_one("npm", "npm.cmd")
    results["npm"] = (npm is not None, tool_version(["npm", "--version"]) if npm else "not found")

    docker = docker_cmd()
    results["docker"] = (docker is not None, "found" if docker else "not found")

    daemon = False
    compose = False
    compose_config = False
    if docker:
        daemon = sh([docker, "info"], capture=True).returncode == 0
        compose_result = sh([docker, "compose", "version"], capture=True)
        compose = compose_result.returncode == 0
        if compose:
            compose_config = sh([docker, "compose", "config", "--quiet"],
                                cwd=ROOT, capture=True).returncode == 0
    results["docker_daemon"] = (daemon, "running" if daemon else "not running")
    results["docker_compose"] = (compose, "available" if compose else "unavailable")
    results["compose_config"] = (compose_config, "docker compose config parses"
                                 if compose else "skipped")

    cc = which_one("cc", "gcc", "clang", "cl")
    results["cxx"] = (cc is not None, cc or "not found (native build optional)")
    cmake = which_one("cmake")
    results["cmake"] = (cmake is not None, cmake or "not found (optional)")
    git = which_one("git")
    results["git"] = (git is not None, "found" if git else "not found")

    # -- project files
    for label, rel in (
        ("pyproject.toml", "pyproject.toml"),
        ("uv.lock", "uv.lock"),
        ("frontend/package.json", "frontend/package.json"),
        ("frontend/package-lock.json", "frontend/package-lock.json"),
        (".env", ".env"),
        (".env.example", ".env.example"),
        ("docker-compose.yml", "docker-compose.yml"),
        ("Dockerfile", "Dockerfile"),
        ("frontend/Dockerfile", "frontend/Dockerfile"),
        ("scripts/start.py", "scripts/start.py"),
        ("scripts/dw.py", "scripts/dw.py"),
    ):
        results[label] = ((ROOT / rel).is_file(), rel)

    # -- environments
    results["python_env"] = (venv_ready(), ".venv with serving dependencies"
                             if venv_ready() else
                             (".venv exists but dependencies incomplete" if VENV_PYTHON.is_file()
                              else ".venv missing"))
    results["node_modules"] = (node_modules_ready(), "matches package-lock.json"
                               if node_modules_ready() else "missing or stale")

    # -- configuration
    database_url = env.get("DATABASE_URL")
    if not database_url:
        results["database_url"] = (
            False,
            "DATABASE_URL not set — a fresh clone fails scripts/start.py "
            "validation; `dw setup` writes the local SQLite default",
        )
    elif database_url.startswith("sqlite"):
        db_file = database_url.split("sqlite:///")[-1]
        results["database_url"] = (True, f"SQLite: {db_file} (created on first start)")
    else:
        results["database_url"] = (True, "external database configured "
                                   f"({database_url.split('@')[-1]})")
    storage_backend = (env.get("STORAGE_BACKEND") or "s3").lower()
    results["storage_config"] = (
        storage_backend in ("s3", "minio", "local"),
        f"STORAGE_BACKEND={storage_backend}, S3_ENDPOINT={env.get('S3_ENDPOINT', 'localhost:9000')}",
    )

    # -- RustFS / storage
    s3_url = s3_endpoint_url(env)
    s3_up = http_ok(f"{s3_url}/health")
    results["rustfs_endpoint"] = (s3_up, s3_url if s3_up else f"{s3_url} unreachable")
    console_port = "9001"
    results["rustfs_console"] = (
        http_ok(f"http://localhost:{console_port}", timeout=2),
        f"http://localhost:{console_port}",
    )
    image_present = False
    if docker and daemon:
        image_present = sh([docker, "image", "inspect", RUSTFS_IMAGE],
                           capture=True).returncode == 0
    results["rustfs_image"] = (image_present, RUSTFS_IMAGE if image_present
                               else f"{RUSTFS_IMAGE} not pulled (auto-pulled on setup/start)")

    # -- checkpoints / models
    ckpts = checkpoint_paths(env)
    results["serving_checkpoint"] = (
        ckpts["serving"].is_file() if ckpts["serving"] else False,
        str(ckpts["serving"].relative_to(ROOT)) if ckpts["serving"] and ckpts["serving"].is_file()
        else f"{ckpts['serving']} MISSING" + (" (WILL DOWNLOAD: released Track1 via auto-fetch)"
                                              if not ckpts["rdah_explicit"] else ""),
    )
    rdah_present = ckpts["rdah"].is_file()
    results["rdah_checkpoint"] = (
        rdah_present,
        str(ckpts["rdah"].relative_to(ROOT)) if rdah_present
        else f"{ckpts['rdah']} MISSING (WILL DOWNLOAD: figshare, ~65 MB, auto-verified)",
    )
    bb_present, bb_path = backbone_cache_status(env)
    results["model_cache"] = (
        bb_present,
        f"{backbone_id(env)} cached at {bb_path}" if bb_present
        else f"{backbone_id(env)} not cached (WILL DOWNLOAD: 0.3-1.3 GB on first use)",
    )

    # -- native acceleration
    native = native_extension_path()
    results["native"] = (native is not None,
                         str(native.relative_to(ROOT)) if native
                         else "MISSING — can be built automatically (pure-Python fallback works)")

    if verbose:
        # extra detail that would be noise by default
        results["docker_services"] = (
            compose_running("rustfs"),
            "rustfs container running" if compose_running("rustfs") else "rustfs not running",
        )

    return results


STATE_LABEL = {True: "OK", False: "MISSING"}


def cmd_doctor(args: list) -> int:
    verbose = "--verbose" in args or "-v" in args
    section("DepthWizard Doctor")
    results = doctor_checks(verbose=verbose)

    for name, (state, detail) in results.items():
        label = STATE_LABEL.get(state, "MISSING")
        print(f"[{label:7s}] {name}: {detail}", flush=True)

    required = [
        "python", "uv", "node", "npm", "docker", "docker_daemon",
        "docker_compose", "compose_config", ".env", "database_url",
        "python_env", "node_modules", "rustfs_endpoint",
    ]
    warnings_only = {"cxx", "cmake", "native", "rustfs_image", "rustfs_console",
                     "serving_checkpoint", "rdah_checkpoint", "model_cache", "git"}
    blocking = [name for name in required if not results[name][0]]
    warned = [name for name, (state, _) in results.items()
              if not state and name in warnings_only]

    print("", flush=True)
    if blocking:
        bad(f"Result: READY = false  ({len(blocking)} blocking, {len(warned)} optional items missing)")
        print("\nRun:\n    ./dw setup", flush=True)
        return EXIT_PREREQ
    ok("Result: READY = true"
       + (f"  ({len(warned)} optional items missing: {', '.join(warned)})" if warned else ""))
    print("\nRun:\n    ./dw start", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------
# dw setup — idempotent first-run bootstrap
# ---------------------------------------------------------------------------

def ensure_python() -> None:
    if sys.version_info >= (3, 8):
        required_minor = 12
        if sys.version_info >= (3, required_minor):
            ok(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")
            return
        fail(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is too old — "
            f"the project requires >= 3.{required_minor} (pyproject.toml).",
            code=EXIT_PREREQ,
            hint=(
                "Install Python 3.12+ and re-run ./dw setup:\n"
                "  Linux:   use your package manager or https://www.python.org/downloads/\n"
                "  macOS:   brew install python@3.12   (or python.org installer)\n"
                "  Windows: https://www.python.org/downloads/  (check 'Add to PATH')"
            ),
        )
    else:
        fail("No usable Python interpreter found.", code=EXIT_PREREQ,
             hint="Install Python 3.12+ from https://www.python.org/downloads/")


def ensure_uv(assume_yes: bool) -> None:
    if which_one("uv"):
        ok(f"uv detected ({tool_version(['uv', '--version'])})")
        return
    print("  [!] uv is not installed (the project's Python dependency manager).", flush=True)
    if not confirm("Install uv (user-level, no admin rights) now?", assume_yes=assume_yes):
        print(
            "Install it manually, then re-run ./dw setup:\n"
            "  Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh\n"
            "  Windows:     powershell -ExecutionPolicy ByPass -c "
            "\"irm https://astral.sh/uv/install.ps1 | iex\"",
            flush=True,
        )
        sys.exit(EXIT_CANCELLED)
    system = platform.system()
    if system == "Windows":
        cmd = ["powershell", "-ExecutionPolicy", "ByPass", "-c",
               "irm https://astral.sh/uv/install.ps1 | iex"]
    else:
        cmd = ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"]
    result = sh(cmd)
    if result.returncode != 0:
        fail("uv installation failed.", hint="Install manually: https://docs.astral.sh/uv/getting-started/installation/")
    # the installer drops uv into ~/.local/bin or similar — look it up again
    uv = which_one("uv") or str(Path.home() / ".local" / "bin" / "uv")
    verify = tool_version([uv, "--version"])
    if not verify:
        fail("uv was installed but is not callable from PATH.",
             hint="Open a new shell (so PATH reloads) and re-run ./dw setup.")
    os.environ["PATH"] = f"{Path(uv).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    ok(f"uv installed ({verify})")


def ensure_node_npm(assume_yes: bool) -> None:
    node = which_one("node", "node.exe")
    npm = which_one("npm", "npm.cmd")
    if node and npm:
        ok(f"Node.js detected ({tool_version(['node', '--version'])})")
        ok(f"npm detected ({tool_version(['npm', '--version'])})")
        return
    print("  [!] Node.js/npm are not installed (required for the frontend).", flush=True)
    print(
        "Installing Node.js needs ADMIN rights, which this script never uses.\n"
        "Install it yourself, then re-run ./dw setup:\n"
        "  Linux:   use your package manager, e.g. sudo apt install nodejs npm\n"
        "           (or https://nodejs.org/en/download)\n"
        "  macOS:   brew install node   (or https://nodejs.org/en/download)\n"
        "  Windows: https://nodejs.org/en/download  (LTS installer)",
        flush=True,
    )
    sys.exit(EXIT_PREREQ)


def ensure_docker(assume_yes: bool) -> None:
    docker = docker_cmd()
    if not docker:
        print("  [!] Docker is not installed (required for RustFS object storage).", flush=True)
        print(
            "Installing Docker needs ADMIN rights, which this script never uses.\n"
            "Install it yourself, then re-run ./dw setup:\n"
            "  Linux:   https://docs.docker.com/engine/install/  (or Docker Desktop)\n"
            "  macOS:   Docker Desktop — https://docs.docker.com/desktop/install/mac/\n"
            "  Windows: Docker Desktop — https://docs.docker.com/desktop/install/windows/",
            flush=True,
        )
        sys.exit(EXIT_PREREQ)
    ok("Docker detected")
    if sh([docker, "info"], capture=True).returncode != 0:
        print(
            "  [!] Docker is installed but the daemon is not running.\n"
            "  Start it yourself (this script never uses sudo/systemctl):\n"
            "  Linux:   sudo systemctl start docker   (or start the docker service)\n"
            "  macOS/Windows: launch Docker Desktop and wait for it to be ready",
            flush=True,
        )
        sys.exit(EXIT_PREREQ)
    ok("Docker daemon running")
    if sh([docker, "compose", "version"], capture=True).returncode != 0:
        fail("Docker Compose is unavailable.", code=EXIT_PREREQ,
             hint="Install the compose plugin: https://docs.docker.com/compose/install/")
    ok("Docker Compose available")
    result = sh([docker, "compose", "config", "--quiet"], cwd=ROOT, capture=True)
    if result.returncode != 0:
        fail("docker compose config failed — docker-compose.yml is invalid.",
             code=EXIT_CONFIG, hint=result.stderr.strip() or None)
    ok("docker compose config valid")


def ensure_env_file(assume_yes: bool) -> None:
    env_path = ROOT / ".env"
    example = ROOT / ".env.example"
    if not env_path.is_file():
        if not example.is_file():
            fail(".env and .env.example are both missing.", code=EXIT_CONFIG)
        shutil.copyfile(example, env_path)
        ok(".env created from .env.example")

    env = read_env_file(env_path)
    changed = []
    if not env.get("DATABASE_URL"):
        # scripts/start.py requires DATABASE_URL; give fresh clones the
        # zero-config local default (no Supabase / external Postgres needed).
        lines = env_path.read_text().splitlines()
        out = []
        inserted = False
        sqlite_line = "DATABASE_URL=sqlite:///data/depthwizard.db"
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") and "DATABASE_URL=sqlite" in stripped and not inserted:
                out.append(sqlite_line)  # activate the documented local default
                inserted = True
                changed.append("DATABASE_URL=sqlite:///data/depthwizard.db (local default)")
                continue
            out.append(line)
        if not inserted:
            out.insert(0, sqlite_line)
            changed.append(sqlite_line)
        env_path.write_text("\n".join(out) + "\n")
        env = read_env_file(env_path)
    if changed:
        for item in changed:
            ok(f".env configured: {item}")
    else:
        ok(".env configuration present")
    if not read_env_file(env_path).get("DATABASE_URL"):
        fail(".env still has no DATABASE_URL after bootstrap.", code=EXIT_CONFIG)


def ensure_python_deps(assume_yes: bool) -> None:
    uv = which_one("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if venv_ready():
        ok("Python environment ready (.venv, serving dependencies present)")
        # dev extras (pytest) — only needed for `dw test`, install lazily there
        return
    log("  .. syncing Python dependencies (uv sync — uses uv.lock; large "
        "packages like torch come from the local uv cache when present)…")
    result = sh([uv, "sync", "--extra", "dev"], cwd=ROOT)
    if result.returncode != 0:
        fail("uv sync failed.", hint="Run `uv sync --extra dev` manually to see the full error.")
    if not venv_ready():
        fail("uv sync finished but the serving dependencies are not importable.",
             code=EXIT_FAIL)
    ok("Python dependencies ready (uv sync --extra dev)")


def ensure_frontend_deps(assume_yes: bool) -> None:
    if node_modules_ready():
        ok("Frontend dependencies ready (node_modules matches package-lock.json)")
        return
    npm = which_one("npm", "npm.cmd")
    if not npm:
        fail("npm not found but frontend dependencies are missing.", code=EXIT_PREREQ)
    log("  .. installing frontend dependencies (npm ci)…")
    cmd = [npm, "ci"] if (FRONTEND / "package-lock.json").is_file() else [npm, "install"]
    result = sh(cmd, cwd=FRONTEND)
    if result.returncode != 0:
        fail("npm install failed.", hint="Run it manually in frontend/ to see the error.")
    if (FRONTEND / "package-lock.json").is_file():
        stamp = FRONTEND / "node_modules" / ".dw-install-stamp"
        stamp.write_text((FRONTEND / "package-lock.json").read_text())
    ok("Frontend dependencies ready")


RUSTFS_PULL_MB = 60  # compressed image size is on the order of tens of MB


def ensure_rustfs(assume_yes: bool) -> None:
    docker = docker_cmd()
    env = read_env_file()
    if sh([docker, "image", "inspect", RUSTFS_IMAGE], capture=True).returncode != 0:
        if not confirm(
            f"Pull the {RUSTFS_IMAGE} image (~{RUSTFS_PULL_MB} MB) now?",
            assume_yes=assume_yes,
        ):
            print("Skipping — RustFS will be pulled on first `dw start`.", flush=True)
            return
        result = sh([docker, "compose", "pull", "rustfs"], cwd=ROOT)
        if result.returncode != 0:
            fail("Docker pull of the RustFS image failed.", hint=result.stderr.strip() or None)
        ok("RustFS image available")

    if not http_ok(f"{s3_endpoint_url(env)}/health"):
        log("  .. starting RustFS via docker compose…")
        result = sh([docker, "compose", "up", "-d", "rustfs"], cwd=ROOT, capture=True)
        if result.returncode != 0:
            fail("Docker RustFS startup failed.", hint=result.stderr.strip() or None)
        deadline = time.monotonic() + S3_HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            if http_ok(f"{s3_endpoint_url(env)}/health"):
                break
            time.sleep(1)
        else:
            fail(f"RustFS did not become reachable at {s3_endpoint_url(env)} "
                 f"within {S3_HEALTH_TIMEOUT}s.", code=EXIT_FAIL)
    ok(f"RustFS running ({s3_endpoint_url(env)}, console http://localhost:9001)")

    # bucket + object roundtrip + presign verification through the project's
    # own S3Storage (uses the venv's boto3; presigns against
    # S3_PUBLIC_ENDPOINT exactly like the serving stack does).
    probe = r"""
import sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(r"{root}")))
from backend.app.storage.backends import S3Storage

backend = S3Storage()
backend.ensure_bucket()
probe = Path(r"{root}") / "data" / ".dw-probe.txt"
probe.parent.mkdir(parents=True, exist_ok=True)
probe.write_bytes(b"dw-setup-probe")
backend.upload_file(probe, ".dw-probe.txt")
roundtrip = Path(r"{root}") / "data" / ".dw-probe.back"
backend.download_file(".dw-probe.txt", roundtrip)
assert roundtrip.read_bytes() == b"dw-setup-probe"
url = backend.presigned_get(".dw-probe.txt")
assert url, "presigned URL was None"
status = urllib.request.urlopen(url, timeout=5).status
assert status == 200, f"presigned fetch returned {status}"
backend.delete_prefix(".dw-probe")
probe.unlink(missing_ok=True); roundtrip.unlink(missing_ok=True)
print("PROBE-OK")
""".replace("{root}", str(ROOT))
    assert VENV_PYTHON.is_file()
    result = sh([str(VENV_PYTHON), "-c", probe], capture=True, project_env=True)
    if "PROBE-OK" not in result.stdout:
        fail("RustFS verification failed (bucket/upload/presign).",
             hint=(result.stderr or result.stdout).strip()[-800:] or None)
    ok("depthwizard bucket ready (create/upload/download/presign verified)")


def ensure_model_assets(assume_yes: bool) -> None:
    """Checkpoints + backbone. Reports sizes honestly before downloading."""
    env = read_env_file()
    ckpts = checkpoint_paths(env)

    if ckpts["serving"] and ckpts["serving"].is_file():
        ok(f"Serving checkpoint ready: {ckpts['serving'].relative_to(ROOT)}")
    else:
        warn(f"Serving checkpoint missing: {ckpts['serving']} "
             "(CalibrationNet runs will fail until one is trained or provided; "
             "RDAH is the default backend and unaffected)")

    if ckpts["rdah"].is_file():
        ok(f"RDAH checkpoint ready: {ckpts['rdah'].relative_to(ROOT)}")
    elif ckpts["rdah_explicit"]:
        warn(f"DW_CKPT_RDAH points at a missing file: {ckpts['rdah']} "
             "(RDAH jobs will fail; no auto-download for explicit overrides)")
    else:
        if not confirm(
            "Download the released RDAH-Net checkpoint (~65 MB, figshare, "
            "MD5-verified on arrival)?",
            assume_yes=assume_yes,
        ):
            print("Skipping — it will download on the first RDAH processing job.", flush=True)
        else:
            probe = (
                "import sys; from pathlib import Path\n"
                f"sys.path.insert(0, {str(ROOT)!r})\n"
                "from depthwizard.rdah import ensure_rdah_checkpoint\n"
                f"print(ensure_rdah_checkpoint({str(ckpts['rdah'])!r}))\n"
            )
            result = sh([str(VENV_PYTHON), "-c", probe], capture=True, project_env=True)
            if result.returncode != 0:
                warn(f"RDAH checkpoint download failed — it will retry on first use. "
                     f"({(result.stderr or '').strip()[-200:]})")
            else:
                ok("RDAH checkpoint ready (MD5-verified)")

    present, bb_path = backbone_cache_status(env)
    if present:
        ok(f"Depth Anything V2 cache present ({backbone_id(env)})")
    elif (env.get("DW_NO_LIVE") or os.environ.get("DW_NO_LIVE")) == "1":
        ok("Live DAv2 disabled (DW_NO_LIVE=1) — no model download needed")
    elif not confirm(
        f"Download Depth Anything V2 weights ({backbone_id(env)}, 0.3-1.3 GB, "
        f"cached under {bb_path})?",
        assume_yes=assume_yes,
        default=False,
    ):
        print("Skipping — weights will download on the first live inference "
              "(or run ./dw setup again).", flush=True)
    else:
        probe = (
            "import sys; sys.path.insert(0, " + repr(str(ROOT)) + ")\n"
            "from depthwizard.backbone import get_backbone\n"
            "bb = get_backbone(model_id=" + repr(backbone_id(env)) + ", device='cpu')\n"
            "bb.load()\n"
            "print('BB-OK')\n"
        )
        log("  .. downloading DAv2 weights (this can take a while)…")
        result = sh([str(VENV_PYTHON), "-c", probe], project_env=True)
        if result.returncode == 0:
            ok("Depth Anything V2 weights ready")
        else:
            warn("DAv2 download did not complete — it will retry on first use.")


def ensure_native(assume_yes: bool) -> None:
    if native_extension_path():
        ok(f"Native acceleration ready ({native_extension_path().relative_to(ROOT)})")
        return
    cc = which_one("cc", "gcc", "clang", "cl")
    if not cc:
        print("Native acceleration unavailable. Using pure-Python fallback.",
              flush=True)
        warn("no C/C++ compiler found — install one to enable the fast path")
        return
    # pybind11 is a build-time dep of native/; scripts/start.py installs it
    # on demand. Nothing to do here except report — start.py owns the build.
    ok("Native acceleration will be built by scripts/start.py on `dw start` "
       "(pure-Python fallback until then)")


def cmd_setup(args: list) -> int:
    assume_yes = "--yes" in args or "-y" in args
    section("DepthWizard Setup")
    ensure_python()
    ensure_uv(assume_yes)
    ensure_node_npm(assume_yes)
    # .env MUST exist before compose validation: docker-compose.yml declares
    # `env_file: .env` for the backend, so compose config fails without it.
    ensure_env_file(assume_yes)
    ensure_docker(assume_yes)
    ensure_python_deps(assume_yes)
    ensure_frontend_deps(assume_yes)
    ensure_rustfs(assume_yes)
    ensure_model_assets(assume_yes)
    ensure_native(assume_yes)

    print("\nSetup complete.\n", flush=True)
    print("Run:\n    ./dw start", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------
# dw start — preflight, then hand over to scripts/start.py
# ---------------------------------------------------------------------------

def cmd_start(args: list) -> int:
    assume_yes = "--yes" in args or "-y" in args
    env = read_env_file()
    bport = backend_port(env)
    fport = frontend_port(env)

    # duplicate-server guard: detect an already-running stack
    if backend_healthy(bport) or backend_healthy(8000):
        port = bport if backend_healthy(bport) else 8000
        print(f"Backend is already running on http://localhost:{port} — "
              "not launching a duplicate.", flush=True)
        print("Use ./dw status for URLs, ./dw stop to stop it.", flush=True)
        return EXIT_OK

    # lightweight preflight (cheap checks only; setup owns the heavy work)
    if not (ROOT / ".env").is_file():
        cmd_setup(["--yes"] if assume_yes else [])
        print("", flush=True)
    if not venv_ready() or not node_modules_ready():
        print("Prerequisites missing (Python env / frontend deps).", flush=True)
        if not assume_yes:
            if not confirm("Run ./dw setup now?", assume_yes=False):
                sys.exit(EXIT_CANCELLED)
        cmd_setup(["--yes"] if assume_yes else [])
        print("", flush=True)

    env = read_env_file()
    if (env.get("STORAGE_BACKEND") or "s3").lower() in ("s3", "minio"):
        docker = docker_cmd()
        if not docker or sh([docker, "info"], capture=True).returncode != 0:
            fail("Docker daemon is not running — RustFS object storage needs it.",
                 code=EXIT_PREREQ,
                 hint="Start Docker (Docker Desktop / `sudo systemctl start docker`) "
                      "and re-run ./dw start.")
        if not http_ok(f"{s3_endpoint_url(env)}/health"):
            log("RustFS not reachable — starting it via docker compose…")
            result = sh([docker, "compose", "up", "-d", "rustfs"], cwd=ROOT, capture=True)
            if result.returncode != 0:
                fail("RustFS startup failed.", hint=result.stderr.strip() or None)
            deadline = time.monotonic() + S3_HEALTH_TIMEOUT
            while time.monotonic() < deadline and not http_ok(f"{s3_endpoint_url(env)}/health"):
                time.sleep(1)
        ok(f"RustFS ready ({s3_endpoint_url(env)})")

    # hand over: process replacement so Ctrl+C reaches the dev servers
    if not VENV_PYTHON.is_file():
        fail(".venv interpreter missing — run ./dw setup.", code=EXIT_PREREQ)
    print(f"\nHanding over to scripts/start.py (frontend http://localhost:{fport}, "
          f"backend http://localhost:{bport}) — Ctrl+C stops everything.\n", flush=True)
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(ROOT / "scripts" / "start.py")])


# ---------------------------------------------------------------------------
# dw status
# ---------------------------------------------------------------------------

def cmd_status(args: list) -> int:
    env = read_env_file()
    results = doctor_checks()
    ckpts = checkpoint_paths(env)

    def line(label: str, state: bool, note: str = "") -> None:
        state_txt = "READY" if state else "MISSING"
        print(f"{label:20s} {state_txt:8s} {note}", flush=True)

    section("DepthWizard Status")
    line("Python", results["python"][0], results["python"][1])
    line("uv", results["uv"][0])
    line("Node", results["node"][0])
    line("npm", results["npm"][0])
    line("Docker", results["docker"][0] and results["docker_daemon"][0],
         "daemon running" if results["docker_daemon"][0] else "daemon not running")

    rustfs = results["rustfs_endpoint"][0]
    print(f"{'RustFS':20s} {'RUNNING' if rustfs else 'STOPPED':8s} {results['rustfs_endpoint'][1]}")

    database_url = env.get("DATABASE_URL")
    if database_url and database_url.startswith("sqlite"):
        line("Database", True, "SQLite (local file, created on start)")
    elif database_url:
        line("Database", True, "external database configured")
    else:
        line("Database", False, "DATABASE_URL not configured")

    bport = backend_port(env)
    compose_backend = compose_running("backend")
    backend_up = backend_healthy(bport) or (compose_backend and backend_healthy(8000))
    bport_eff = bport if backend_healthy(bport) else (8000 if backend_healthy(8000) else bport)
    print(f"{'Backend':20s} {'RUNNING' if backend_up else 'STOPPED':8s} "
          f"http://localhost:{bport_eff}")

    fport = frontend_port(env)
    frontend_up = http_ok(f"http://localhost:{fport}", timeout=2) or compose_running("frontend")
    print(f"{'Frontend':20s} {'RUNNING' if frontend_up else 'STOPPED':8s} "
          f"http://localhost:{fport}")

    if rustfs and VENV_PYTHON.is_file():
        probe = (
            "import sys; sys.path.insert(0, " + repr(str(ROOT)) + ")\n"
            "from backend.app.storage.backends import S3Storage\n"
            "from botocore.exceptions import ClientError\n"
            "b = S3Storage()\n"
            "try:\n"
            "    b._get_client().head_bucket(Bucket=" + repr(env.get('S3_BUCKET') or 'depthwizard') + ")\n"
            "    print('BUCKET-OK')\n"
            "except ClientError:\n"
            "    print('BUCKET-MISSING')\n"
        )
        result = sh([str(VENV_PYTHON), "-c", probe], capture=True, project_env=True)
        bucket_ok = "BUCKET-OK" in result.stdout
        line("Storage bucket", bucket_ok, env.get("S3_BUCKET") or "depthwizard")
    else:
        line("Storage bucket", False, "RustFS not reachable")

    line("Serving checkpoint", bool(ckpts["serving"] and ckpts["serving"].is_file()),
          str(ckpts["serving"]))
    line("RDAH checkpoint", ckpts["rdah"].is_file(), str(ckpts["rdah"]))
    native = native_extension_path()
    line("Native extension", native is not None,
         "READY" if native else "FALLBACK (pure Python)")

    print("", flush=True)
    print("Frontend:\n  http://localhost:" + str(frontend_port(env)), flush=True)
    print(f"Backend:\n  http://localhost:{bport_eff}", flush=True)
    print(f"API docs:\n  http://localhost:{bport_eff}/docs", flush=True)
    print("RustFS (console):\n  http://localhost:9001", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------
# dw test
# ---------------------------------------------------------------------------

def rustfs_roundtrip_test() -> list:
    """bucket exists / write / read / delete / presign — through the
    project's own S3Storage."""
    probe = r"""
import sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(r"{root}")))
from backend.app.storage.backends import S3Storage
b = S3Storage()
b.ensure_bucket()
p = Path(r"{root}") / "data" / ".dw-test.txt"
p.write_bytes(b"dw-test")
b.upload_file(p, ".dw-test.txt")
out = Path(r"{root}") / "data" / ".dw-test.out"
b.download_file(".dw-test.txt", out)
assert out.read_bytes() == b"dw-test"
assert b.exists(".dw-test.txt")
url = b.presigned_get(".dw-test.txt")
assert url and urllib.request.urlopen(url, timeout=5).status == 200
b.delete_prefix(".dw-test")
p.unlink(); out.unlink()
print("ROUNDTRIP-OK")
""".replace("{root}", str(ROOT))
    result = sh([str(VENV_PYTHON), "-c", probe], capture=True, project_env=True)
    return [("RustFS roundtrip (bucket/write/read/delete/presign)",
             "ROUNDTRIP-OK" in result.stdout,
             (result.stderr or result.stdout).strip()[-300:])]


def cmd_test(args: list) -> int:
    full = "--full" in args
    assume_yes = "--yes" in args or "-y" in args
    env = read_env_file()
    failures = []
    checks: list = []

    section("DepthWizard Test")

    # -- configuration
    if not (ROOT / ".env").is_file():
        checks.append((".env present", False, "run ./dw setup"))
    elif not env.get("DATABASE_URL"):
        checks.append(("env loads / DATABASE_URL", False,
                       "DATABASE_URL missing — run ./dw setup"))
    else:
        checks.append(("env loads / DATABASE_URL", True,
                       env["DATABASE_URL"].split("@")[-1] if "@" in env["DATABASE_URL"]
                       else env["DATABASE_URL"]))
    checks.append(("storage configured",
                   (env.get("STORAGE_BACKEND") or "s3").lower() in ("s3", "minio", "local"),
                   f"STORAGE_BACKEND={env.get('STORAGE_BACKEND', 's3')}"))

    # -- RustFS (needs the venv's boto3)
    if venv_ready() and http_ok(f"{s3_endpoint_url(env)}/health"):
        checks.extend(rustfs_roundtrip_test())
    else:
        checks.append(("RustFS roundtrip", False, "RustFS unreachable or venv missing"))

    for name, passed, note in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" — {note}" if note else ""),
              flush=True)
        if not passed:
            failures.append(name)

    # -- backend / frontend (only meaningful when running)
    bport = backend_port(env)
    backend_up = backend_healthy(bport) or backend_healthy(8000)
    checks_live = []
    if backend_up:
        port = bport if backend_healthy(bport) else 8000
        checks_live.append((f"backend health endpoint (: {port})",
                            backend_healthy(port), "GET /api/v1/health -> 200"))
    else:
        checks_live.append(("backend health endpoint", None,
                            "backend not running — start with ./dw start"))
    fport = frontend_port(env)
    frontend_up = http_ok(f"http://localhost:{fport}", timeout=2) or compose_running("frontend")
    checks_live.append(("frontend reachable",
                        frontend_up if frontend_up is not None else None,
                        f"http://localhost:{fport}" if frontend_up
                        else "vite not running — start with ./dw start"))
    for name, passed, note in checks_live:
        if passed is None:
            print(f"  [SKIP] {name} — {note}", flush=True)
        else:
            print(f"  [{'PASS' if passed else 'FAIL'}] {name} — {note}", flush=True)
            if not passed:
                failures.append(name)

    # -- python test suites
    if not venv_ready():
        print("  [FAIL] python test suite — .venv missing (run ./dw setup)", flush=True)
        return EXIT_TESTS
    if VENV_PYTHON.is_file() and not _pytest_available():
        sh([str(which_one("uv") or "uv"), "pip", "install", "--python",
            str(VENV_PYTHON), "pytest"], capture=True)

    log("\n  .. running backend_tests (uv-managed venv)…")
    result = sh([str(VENV_PYTHON), "-m", "pytest", "backend_tests", "-q"], cwd=ROOT)
    checks.append(("pytest backend_tests", result.returncode == 0, ""))
    if result.returncode != 0:
        failures.append("pytest backend_tests")

    if full:
        log("  .. running model_tests (--full)…")
        result = sh([str(VENV_PYTHON), "-m", "pytest", "model_tests", "-q"], cwd=ROOT)
        checks.append(("pytest model_tests", result.returncode == 0, ""))
        if result.returncode != 0:
            failures.append("pytest model_tests")
        log("  .. running end-to-end inference on the demo tile (--full)…")
        demo = ROOT / "demo_DC_02_26.png"
        if not demo.is_file():
            checks.append(("end-to-end inference", False, "demo_DC_02_26.png missing"))
            failures.append("end-to-end inference")
        else:
            result = sh([str(VENV_PYTHON), "model.py", "infer", "--input", str(demo),
                         "--out", str(ROOT / "scratch" / "dw_full_test"),
                         "--json-out", str(ROOT / "scratch" / "dw_full_test" / "payload.json")],
                        cwd=ROOT)
            payload_ok = (ROOT / "scratch" / "dw_full_test" / "payload.json").is_file()
            checks.append(("end-to-end inference (demo tile)",
                           result.returncode == 0 and payload_ok,
                           "scratch/dw_full_test/payload.json"))
            if not (result.returncode == 0 and payload_ok):
                failures.append("end-to-end inference")

    print("", flush=True)
    if failures:
        bad(f"Test result: FAILED ({', '.join(failures)})")
        return EXIT_TESTS
    ok("Test result: ALL PASS")
    return EXIT_OK


def _pytest_available() -> bool:
    result = sh([str(VENV_PYTHON), "-c", "import pytest"], capture=True)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# dw logs
# ---------------------------------------------------------------------------

def cmd_logs(args: list) -> int:
    target = args[0] if args and not args[0].startswith("-") else "all"
    docker = docker_cmd()

    def compose_logs(service: str) -> None:
        if docker and compose_running(service):
            sh([docker, "compose", "logs", "--tail", "100", "-f", service], cwd=ROOT)
        elif docker:
            result = sh([docker, "compose", "ps", "-a", "--format", "{{.Service}}"],
                        cwd=ROOT, capture=True)
            if service in result.stdout.split():
                sh([docker, "compose", "logs", "--tail", "100", service], cwd=ROOT)
            else:
                print(f"{service}: not a running container — "
                      f"host-side dev servers stream logs to the `dw start` terminal.",
                      flush=True)

    if target in ("all", "rustfs"):
        if target == "rustfs" or compose_running("rustfs"):
            print("--- rustfs (docker) ---", flush=True)
            result = sh([docker, "compose", "logs", "--tail", "50", "rustfs"],
                        cwd=ROOT, capture=True)
            print(result.stdout, flush=True)
    if target in ("all", "backend"):
        if compose_running("backend"):
            print("--- backend (docker) ---", flush=True)
            sh([docker, "compose", "logs", "--tail", "100", "-f", "backend"], cwd=ROOT)
        else:
            print("--- backend ---\nHost-side dev backend logs stream to the "
                  "`dw start` terminal (it runs in the foreground, no log files).",
                  flush=True)
    if target in ("all", "frontend"):
        if compose_running("frontend"):
            print("--- frontend (docker) ---", flush=True)
            sh([docker, "compose", "logs", "--tail", "100", "-f", "frontend"], cwd=ROOT)
        else:
            print("--- frontend ---\nHost-side dev frontend logs stream to the "
                  "`dw start` terminal.", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------------------
# dw stop / restart / clean
# ---------------------------------------------------------------------------

def cmd_stop(args: list) -> int:
    section("DepthWizard Stop")
    pids = load_child_pids()
    stopped = 0
    for pid in pids:
        if not pid_alive(pid):
            continue
        try:
            if os.name == "nt":
                sh(["taskkill", "/PID", str(pid), "/T", "/F"], capture=True)
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            stopped += 1
        except OSError:
            continue
    if stopped:
        ok(f"stopped {stopped} host-side dev process(es) recorded in {PID_FILE.name}")
    else:
        ok("no host-side dev processes running")
    PID_FILE.unlink(missing_ok=True)

    docker = docker_cmd()
    if docker:
        for service in ("backend", "frontend"):
            if compose_running(service):
                sh([docker, "compose", "stop", service], cwd=ROOT, capture=True)
                ok(f"stopped docker compose service: {service}")
        if compose_running("rustfs"):
            print("RustFS container left running (objects persist in the "
                  "rustfs_data volume) — stop it with: docker compose stop rustfs",
                  flush=True)
    print("\nNothing was deleted: rustfs objects, database, outputs, model "
          "caches, .venv and node_modules are all preserved.", flush=True)
    return EXIT_OK


def cmd_restart(args: list) -> int:
    cmd_stop([])
    print("", flush=True)
    return cmd_start(args)


def cmd_clean(args: list) -> int:
    assume_yes = "--yes" in args
    levels = [a for a in args if a.startswith("--") and a != "--yes"]
    if not levels:
        print(
            "Usage: dw clean --cache | --deps | --data | --all [--yes]\n"
            "  --cache  pytest caches, native build artifacts (rebuildable)\n"
            "  --deps   .venv and frontend/node_modules (reinstalled by dw setup)\n"
            "  --data   local scene artifacts and the local SQLite dev database\n"
            "  --all    all of the above\n"
            "Never touched: RustFS objects, checkpoints, model weights.",
            flush=True,
        )
        return EXIT_OK

    def rmtree(path: Path, description: str) -> None:
        if not path.exists():
            print(f"  (nothing to remove: {description})", flush=True)
            return
        print(f"  removing {description}: {path}", flush=True)
        shutil.rmtree(path, ignore_errors=True)

    do_cache = "--cache" in levels or "--all" in levels
    do_deps = "--deps" in levels or "--all" in levels
    do_data = "--data" in levels or "--all" in levels

    if do_cache:
        for cache in ROOT.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        rmtree(ROOT / ".pytest_cache", "pytest cache")
        rmtree(ROOT / "native" / "build", "native build artifacts")
        ok("caches cleaned")
    if do_deps:
        print("\nThis deletes .venv and frontend/node_modules — `dw setup` "
              "reinstalls both (downloads ~2-4 GB from package caches).", flush=True)
        if not confirm_destructive("Delete installed dependencies?"):
            print("Cancelled — nothing deleted.", flush=True)
            return EXIT_CANCELLED
        rmtree(VENV, ".venv")
        rmtree(FRONTEND / "node_modules", "frontend/node_modules")
        ok("dependencies cleaned")
    if do_data:
        print("\nThis deletes LOCAL scene artifacts and the local SQLite dev "
              "database. RustFS objects, checkpoints and model weights are "
              "NOT touched.", flush=True)
        if not confirm_destructive("Delete local dev data?"):
            print("Cancelled — nothing deleted.", flush=True)
            return EXIT_CANCELLED
        rmtree(ROOT / "data" / "process", "scene process dirs")
        rmtree(ROOT / "data" / "output", "scene outputs")
        for db in (ROOT / "data").glob("*.db"):
            db.unlink()
            print(f"  removing {db.name}", flush=True)
        ok("local dev data cleaned")
    if do_cache and "--all" in levels:
        # model cache is a re-download, so it lives behind its own confirmation
        pass
    return EXIT_OK


# ---------------------------------------------------------------------------
# help / entry
# ---------------------------------------------------------------------------

HELP = """DepthWizard master CLI

Usage: ./dw <command> [options]

  doctor          diagnose prerequisites, config, storage, models (read-only)
  setup [--yes]   install/configure/verify everything needed to run
  start [--yes]   preflight checks, then launch the stack (scripts/start.py)
  test [--full]   config + RustFS roundtrip + backend_tests (--full adds
                  model_tests and a real inference on the demo tile)
  status          component status table + URLs
  logs [target]   rustfs | backend | frontend | all
  stop            stop dev processes (deletes nothing)
  restart         stop + start
  clean           --cache | --deps | --data | --all  (destructive: confirmed)
  help            this message

Typical first run:
  ./dw setup
  ./dw start

Then open:
  frontend        http://localhost:5173
  backend         http://localhost:8010 (docs: /docs)
  RustFS console  http://localhost:9001

Exit codes: 0 ok · 1 failure · 2 invalid config · 3 missing prerequisite ·
4 test failure · 5 cancelled
"""

COMMANDS = {
    "doctor": cmd_doctor,
    "setup": cmd_setup,
    "start": cmd_start,
    "test": cmd_test,
    "status": cmd_status,
    "logs": cmd_logs,
    "stop": cmd_stop,
    "restart": cmd_restart,
    "clean": cmd_clean,
    "help": None,
    "--help": None,
    "-h": None,
}


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("help", "--help", "-h"):
        print(HELP, flush=True)
        return EXIT_OK
    command = args[0].lower()
    handler = COMMANDS.get(command)
    if handler is None:
        print(f"Unknown command: {command}\n", flush=True)
        print(HELP, flush=True)
        return EXIT_FAIL
    try:
        return handler(args[1:])
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        return EXIT_CANCELLED


if __name__ == "__main__":
    sys.exit(main())
