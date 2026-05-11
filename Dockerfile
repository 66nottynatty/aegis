# Multi-stage Dockerfile for Aegis Content Security Microservice

# ─── Builder Stage ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir build && \
    python -m build --wheel --outdir /build/dist

# ─── Runtime Stage ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="Aegis Team <aegis@example.com>"
LABEL org.opencontainers.image.description="Aegis Content Security Microservice"
LABEL org.opencontainers.image.source="https://github.com/modelNorth/aegis"

# System dependencies for content processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user
RUN groupadd -r aegis && useradd -r -g aegis aegis

# Install wheel from builder
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Pre-download models for offline mode
ENV HF_HOME=/app/.cache
RUN mkdir -p /app/.cache && chown -R aegis:aegis /app/.cache

# Copy scripts for model downloading
COPY scripts/download_models.py /app/scripts/download_models.py
RUN chown -R aegis:aegis /app/scripts

# Switch to non-root user to download models to the correct cache directory
USER aegis
RUN python /app/scripts/download_models.py

ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${AEGIS_API_PORT:-8000}/health || exit 1

EXPOSE 8000

ENV AEGIS_API_HOST=0.0.0.0
ENV AEGIS_API_PORT=8000

CMD ["aegis", "serve", "--host", "0.0.0.0", "--port", "8000"]
