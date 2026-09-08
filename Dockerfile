FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libcairo2 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home --user-group --shell /usr/sbin/nologin appuser \
    && install -d -o appuser -g appuser /home/appuser/.cache

WORKDIR /app
COPY requirements-cpu.lock.txt ./
RUN python -m pip install --no-cache-dir --require-hashes -r requirements-cpu.lock.txt

# Application code stays root-owned; only the existing model cache is writable.
COPY app/ ./app/
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XDG_CACHE_HOME=/home/appuser/.cache
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-proxy-headers"]
