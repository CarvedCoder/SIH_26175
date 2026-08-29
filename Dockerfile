# DepthWizard backend — Python inference service.
# Built from the MONOREPO root:
#   docker build -f Dockerfile -t depthwizard-backend .

FROM python:3.12-slim AS base

# GDAL/rasterio system deps (rasterio wheels ship their own libgdal since 1.3,
# but keep these for safety across wheel variants).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev gdal-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY depthwizard/ ./depthwizard/
COPY service/ ./service/
COPY configs/ ./configs/
COPY main.py ./

# Outputs (per-request DSM products) live here; mount a volume to keep them.
# Your trained checkpoint is expected at
#   /app/outputs/calib_net/<tag>/best.pt
# (mount the repo's ./outputs over /app/outputs — see docker-compose.yml —
#  or set DW_CKPT to wherever the checkpoint lives inside the container).
VOLUME ["/app/outputs"]

EXPOSE 8000
ENV DW_ROOT=/app \
    DW_CKPT=/app/outputs/calib_net/rgb_cos/best.pt \
    DW_OUT_ROOT=/app/outputs/service

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
