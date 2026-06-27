# syntax=docker/dockerfile:1.7
# =============================================================================
# Adit-Agent — multi-stage production image
#
#   * builder stage  : installs dependencies into an isolated virtualenv
#   * runtime stage  : copies only the venv + app, runs as non-root user
#
# Build:  docker build -t adit-agent:latest .
# Run:    docker run --env-file .env -v "$PWD/data:/app/data" adit-agent:latest
# =============================================================================

# ----------------------------------------------------------------------------
# Stage 1 — builder
# ----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Build toolchain needed to compile wheels (chromadb, lxml, etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /app
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ----------------------------------------------------------------------------
# Stage 2 — runtime
# ----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Runtime-only OS deps. ffmpeg powers the audio/video multimodal pipelines.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libxml2 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for least-privilege execution.
RUN groupadd --gid 1000 adit && \
    useradd --uid 1000 --gid adit --create-home adit

# Bring in the pre-built virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=adit:adit . .

# Persisted state (db, uploads, vector store, logs) lives here.
RUN mkdir -p /app/data && chown -R adit:adit /app/data

USER adit

# Lightweight liveness probe — verifies config & imports load cleanly.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "from app.config import get_settings; get_settings()" || exit 1

# TODO: when exposing a webhook/health HTTP endpoint, EXPOSE its port here.

ENTRYPOINT ["python", "-m", "app.main"]
