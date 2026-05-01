# syntax=docker/dockerfile:1.7

# ---------- builder ----------------------------------------------------------
# Install dependencies into an isolated venv. Building deps in a separate
# stage keeps the final image free of pip caches, build toolchains, and
# the wheel cache used to speed up resolution.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv

# Build deps a few transitive C extensions (onnxruntime, grpcio) need at
# install time. They are not present in the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy and install the package itself.
COPY pyproject.toml README.md ./
COPY rag ./rag
RUN pip install --no-deps .

# ---------- runtime ----------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Run as a non-root user. ``app`` has no shell and no home directory —
# enough to run the API, not enough to be a useful exploit target.
RUN groupadd --system app \
    && useradd --system --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app rag ./rag
COPY --chown=app:app evals ./evals

USER app

EXPOSE 8000

# Healthcheck hits the API's /health endpoint. ``--start-period`` gives
# uvicorn a moment to bind before counting failures.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
