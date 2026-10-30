# Multi-stage build.
#
# The split exists to keep compilers, headers and pip's build caches out of the
# final image: those are needed to install the wheels and never needed to run
# them. The runtime stage carries onnxruntime rather than PyTorch, which is the
# single biggest saving -- the torch wheel alone is larger than this whole image.

# The base is pinned by digest, not by tag. A tag is a moving target: the same
# `docker build` on two days can produce two different images and nobody can
# see why from the repository. With the digest, a base-image change is a
# visible commit. Refresh it deliberately with
#   docker buildx imagetools inspect python:3.11-slim-bookworm
# and paste the index digest here.
ARG BASE_IMAGE=python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd

# ---------- build stage ----------
FROM ${BASE_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Dependencies are copied and installed before the source so that editing code
# does not invalidate the dependency layer.
COPY requirements-serve.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements-serve.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN /opt/venv/bin/pip install --no-deps .

# Tests, caches and compiled artefacts have no business in the runtime image.
RUN find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && find /opt/venv -name 'tests' -type d -prune -exec rm -rf {} + \
    && find /opt/venv -name '*.pyc' -delete

# ---------- runtime stage ----------
FROM ${BASE_IMAGE} AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MEDEXTRACT_MODEL_DIR=/app/artifacts/onnx \
    MEDEXTRACT_EMBEDDING_ONNX_DIR=/app/artifacts/embedder \
    MEDEXTRACT_BACKEND=onnx \
    # transformers checks for torch/tf at import time; tell it not to bother.
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HUB_OFFLINE=1

# curl is here for the HEALTHCHECK below and nothing else.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 medextract

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Model artefacts are mounted or baked in at deploy time; the directory exists
# so a bind mount lands somewhere sensible and /health can report it missing.
RUN mkdir -p /app/artifacts && chown -R medextract:medextract /app

USER medextract

EXPOSE 8000

# Hits the real endpoint rather than just checking the port, so a process that
# is up but has no model loaded is still reported as running-but-degraded by
# /health rather than passing silently.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "medextract.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
