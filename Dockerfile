FROM python:3.14-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6

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
