# BackRemove

Background removal API with a fast withoutBG v10 backend and an opt-in
BiRefNet quality backend.

Repository: [sellaro-net/backremove](https://github.com/sellaro-net/backremove).

Nach dem Repository-Transfer verwenden zukünftige reguläre Publisher
`ghcr.io/sellaro-net/backremove`. Das bestehende öffentliche Paket
`ghcr.io/tentoxa/backremove` bleibt davon getrennt; der Transfer verschiebt
keine Images. Bestehende Deployment-Pins bleiben bis zum gesonderten manuellen
Imagewechsel unverändert. Paket-Zugriffe können sich durch den Transfer ändern.

## Native setup (Windows + NVIDIA GPU)

Requires Python 3.12 and a current NVIDIA driver. Double-click
`start-gpu.bat`; it installs/checks the Pascal-compatible CUDA 11, cuDNN 8,
ONNX Runtime, and PyTorch runtimes in `.venv`, caches the pinned model weights,
and starts the API. A system-wide CUDA toolkit is not required.

```bat
start-gpu.bat
```

The first setup downloads both model weights. Confirm that both backends are
preloaded before readiness and that the serialized GPU queue is active:

```powershell
Invoke-RestMethod http://localhost:8080/health
# status: ok
# models.fast.loaded: true
# models.quality.loaded: true
# gpu_queue.capacity: 8
```

`INFERENCE_DEVICE` accepts `cuda`, `cpu`, or `auto` (the default). Explicit
`cuda` fails at startup instead of silently running on the CPU.

## Native CPU setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:INFERENCE_DEVICE = "cpu"
$env:API_KEY = "<strong random key>"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --no-proxy-headers
```

The CPU setup exposes only the `fast` backend. `quality` requires the native
GPU setup.

## Docker

```bash
docker compose up --build
```

Compose reads `API_KEY` from the local `.env` file and stops with an error if
the value is missing.

The Docker image exposes only the `fast` backend. The BiRefNet quality runtime
is installed by `setup-gpu.ps1` for the native NVIDIA deployment.

## Usage

Fast removal is the default:

```bash
curl -X POST "http://localhost:8080/remove-bg?model=fast" \
  -H "X-API-Key: <value from .env>" \
  -F "file=@photo.jpg" \
  --output no-bg.png
```

Retry a difficult image with BiRefNet:

```bash
curl -X POST "http://localhost:8080/remove-bg?model=quality" \
  -H "X-API-Key: <value from .env>" \
  -F "file=@photo.jpg" \
  --output no-bg-quality.png
```

Responses expose `X-Model-Used`, `X-Admission-Time-Ms`, `X-Decode-Time-Ms`,
`X-Queue-Time-Ms`, `X-Inference-Time-Ms`, and `X-Encode-Time-Ms`. Admission
measures waiting before decode; queue time measures waiting from decoded input
to model start. Supported formats: JPEG, PNG, WebP, GIF, AVIF, SVG.
Uploads are limited to 20 MB and 40 million decoded pixels; malformed image
data is rejected before inference.
PNG output uses compression level 3 to favor encode latency over minimum
response size.

## GPU admission control

A bounded pipeline decodes uploads before GPU admission and encodes PNG output
after inference. One GPU actor serializes only the model call across both
models, so the next request can use the GPU while the previous result is being
encoded. Two result slots bound concurrent RGBA and PNG buffers without
removing that overlap. A full or expired queue returns `503` with `Retry-After`;
an inference response deadline returns `504`. Queue waiting and model execution
have separate deadlines whose sum stays below the reverse-proxy budget.

When the quality backend is enabled, startup warms both models before the API
reports ready. This removes first-request latency and prevents a cold model load
from consuming the queue deadline.

| Variable | Default | Purpose |
|----------|---------|---------|
| `GPU_QUEUE_CAPACITY` | `8` | Maximum buffered requests, excluding the active job |
| `GPU_QUEUE_TIMEOUT` | `10` | Seconds a request may wait to start |
| `INFERENCE_TIMEOUT` | `75` | Seconds a running job may take before its response expires |
| `PROXY_REQUEST_BUDGET` | `90` | Upper bound for queue plus inference deadlines |

## Auth

`API_KEY` is required. The application refuses to start when it is missing or
empty. `start-gpu.bat` reads it from the git-ignored `.env` file, and Docker
Compose passes the same value from `.env` into the container.

Send the value in the `X-API-Key` header. `/health` remains public for health
checks.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Health check |
| POST | `/remove-bg?model=fast\|quality` | `X-API-Key` | Remove background, returns PNG |
