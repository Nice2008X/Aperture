# Single image for all of Aperture: builds the frontend (apps/web) and
# bundles it into the GPU backend (apps/api), which serves both the API
# and the built UI itself (see the WEB_DIST static mount at the end of
# apps/api/src/aperture_api/main.py) — one image, one process, one port.
# No nginx, no reverse proxy, no cross-origin requests to reason about.
#
# Needs an actual NVIDIA GPU at runtime:
#   docker run --gpus all -p 8000:8000 -v $(pwd)/data:/app/data <image>
# (or docker-compose.yml, which wires up the GPU + volume for you). The
# host needs the NVIDIA Container Toolkit installed either way —
# https://github.com/NVIDIA/nvidia-container-toolkit.

# ---- frontend build stage ----
FROM node:22-alpine AS frontend
WORKDIR /app

# Copying the whole workspace before `npm ci` (rather than just the root
# package.json) is deliberate: this is an npm-workspaces monorepo, so npm
# needs every package's package.json present to link the workspace
# correctly. That trades away some Docker layer caching (a source-only
# change also invalidates the npm ci layer) for a Dockerfile that keeps
# working as packages are added or removed, which happens often here.
COPY . .
RUN npm ci

# Vite inlines VITE_* variables into the built bundle at build time, not
# runtime. Empty string is correct here (not just the default): the
# frontend and API are the same origin now (same process, same port), so
# every request is a plain relative /api/... or /data/... fetch — see
# setApiBase in apps/web/src/main.tsx.
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

# ---- backend runtime stage ----
# Full CUDA "devel" (not "runtime") on purpose — gpt-oss-style MXFP4
# checkpoints load through triton-compiled kernels (see the `kernels`
# dependency in apps/api/pyproject.toml), which can shell out to
# ptxas/nvcc from the CUDA toolkit at first use; the slimmer runtime image
# doesn't ship those. CUDA 12.8 matches the cu128 wheel index below —
# bump both together.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Ubuntu 24.04's python3 is already 3.12 (matches apps/api/pyproject.toml's
# requires-python), so no deadsnakes PPA needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Preserve the monorepo's on-disk layout, not just apps/api in isolation:
# apps/api/src/aperture_api/paths.py derives the repo root (and so
# data/models/ and the frontend's dist/ below) from its own file location
# via `Path(__file__).resolve().parents[4]`, which only lands in the right
# place if apps/api keeps its real position four directories below the
# root.
WORKDIR /app
COPY . .
COPY --from=frontend /app/apps/web/dist ./apps/web/dist

# A venv here is mostly to keep apt's system Python's PEP 668
# "externally-managed-environment" guard out of the way, not for
# isolation (this container runs nothing else).
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --extra-index-url (rather than --index-url) is deliberate: it adds
# PyTorch's CUDA wheel index as a *second* source alongside regular PyPI,
# rather than replacing PyPI outright, so every other dependency below
# still resolves normally. Pick the index matching the CUDA base image
# above if you change it (see https://pytorch.org for the current list).
RUN pip install --no-cache-dir -e ./apps/api --extra-index-url https://download.pytorch.org/whl/cu128

# data/models/ — downloaded checkpoints — is bind-mounted at runtime
# (docker-compose.yml) so it survives container recreation; this just
# makes sure the directory exists before the app's first write.
RUN mkdir -p /app/data/models

EXPOSE 8000
WORKDIR /app/apps/api
CMD ["uvicorn", "aperture_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
