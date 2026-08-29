# syntax=docker/dockerfile:1

###############################################################################
# Stage 1 — builder: resolve and install the API-only dependency set
###############################################################################
FROM python:3.13-slim-bookworm AS builder

# uv as a static binary; no need to pip-install it into the image
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

WORKDIR /build

# Installs [project.dependencies] only — the api/ service. The `ingestion`
# and `evaluations` extras (torch + CUDA, transformers, hdbscan, ...) add
# ~5 GB and are never imported by api/, so they stay out of the image.
# fastembed runs on onnxruntime (CPU).
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev --no-default-groups

# Drop test suites and static archives that ship inside some wheels.
# __pycache__ is kept on purpose: UV_COMPILE_BYTECODE=1 precompiled it.
RUN find /opt/venv -type d -name "tests" -prune -exec rm -rf {} + \
    && find /opt/venv -type f -name "*.a" -delete

###############################################################################
# Stage 2 — runtime
###############################################################################
FROM python:3.13-slim-bookworm AS runtime

# FASTEMBED_CACHE_PATH / HF_HOME: fastembed and huggingface_hub download ONNX
# weights on first import. Mount a volume at /cache so restarts do not
# re-download ~2.5 GB of models.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/cache/fastembed \
    HF_HOME=/cache/huggingface

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /cache/fastembed /cache/huggingface \
    && chown -R app:app /cache

COPY --from=builder --chown=root:root /opt/venv /opt/venv

# api/ uses flat intra-package imports (`from routers import ...`),
# so the app directory itself must be the working directory.
WORKDIR /app
COPY --chown=app:app api/ ./

USER app

EXPOSE 8000
VOLUME ["/cache"]

# Long start period: the first boot instantiates the dense, sparse and
# ColBERT models at import time, downloading them if the cache is cold.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/', timeout=4)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
