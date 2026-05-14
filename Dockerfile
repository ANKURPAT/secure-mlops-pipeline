# syntax=docker/dockerfile:1
# ------------------------------------------------------------
# Stage 1: dependency installation
# We install dependencies in a separate stage so the final image
# doesn't need pip, wheel, or build tools.
# ------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only what's needed to build wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip --no-cache-dir \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Stage 2: runtime image
# ------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user — defence in depth.
# Even though Podman is rootless at the host level, the process
# inside the container should not run as uid 0.
RUN groupadd --gid 1001 appgroup \
 && useradd --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only application source — everything else is excluded by .dockerignore
COPY app/ ./app/
COPY ml/ ./ml/
COPY requirements.txt .

# Ownership — appuser needs to read app files
RUN chown -R appuser:appgroup /app

USER appuser

# Document the port — podman run -p 8000:8000 maps this
EXPOSE 8000

# Explicit host binding required inside a container.
# MLflow URI comes in via env var at runtime — no hardcoded server address.
ENV MLFLOW_TRACKING_URI=http://host.containers.internal:5000 \
    MODEL_NAME=fraud-detector \
    MODEL_ALIAS=staging \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["/usr/local/bin/uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]