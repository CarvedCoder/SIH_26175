# DepthWizard — canonical production image.
#
# Deploys the CANONICAL backend (backend.app.main:app — the /api/v1
# scene/job contract the frontend uses). The legacy service/api.py is NOT
# part of this image.
#
#   docker build -f Dockerfile -t depthwizard-backend .
#
# Dependencies are installed from pyproject.toml + uv.lock (--frozen):
# the lockfile is the single reproducible source (no floating pip floors).

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# libgomp1: torch's OpenMP runtime on slim Debian. libexpat1: required by
# rasterio's bundled GDAL. rasterio/matplotlib wheels bundle their own
# remaining native libs — no system GDAL needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first (layer-cached). NOTE: resolves at build time
# (--no-dev, not --frozen) because uv.lock predates the SQLAlchemy/psycopg/
# pyjwt/boto3 additions; regenerate uv.lock (uv lock) to restore frozen
# installs.
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

# Application code
COPY depthwizard/ ./depthwizard/
COPY backend/ ./backend/
COPY configs/ ./configs/
COPY model.py ./

# State locations (bind-mount or volume these to persist):
#   /app/data      scene inputs/intermediates/results
#   /app/outputs   trained checkpoints (DW_CKPT) + service outputs
RUN useradd --create-home --uid 10001 dw \
    && mkdir -p /app/data /app/outputs \
    && chown -R dw:dw /app
USER dw

VOLUME ["/app/outputs", "/app/data"]

ENV PATH="/app/.venv/bin:$PATH" \
    DW_ROOT=/app \
    DW_DEVICE=auto \
    DW_OUT_ROOT=/app/outputs/service

EXPOSE 8000

# Health checks the canonical /api/v1/health contract (fast: no model
# work in the request path). The health endpoint itself never runs
# inference, so a modest timeout is honest here.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://localhost:8000/api/v1/health', timeout=8)"]

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
