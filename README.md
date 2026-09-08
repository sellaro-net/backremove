# BackRemove

Background removal API with a fast withoutBG v10 backend and an opt-in
BiRefNet quality backend.

Part of [Sellaro](https://github.com/sellaro-net/sellaro), maintained independently
from the web/mobile application and [ja3proxy](https://github.com/sellaro-net/ja3proxy).

| Runtime | Models | Local endpoint |
|---------|--------|----------------|
| Docker / native CPU | `fast` (withoutBG v10) | Docker: `127.0.0.1:8000`; native: `127.0.0.1:8080` |
| Windows + NVIDIA | `fast` and `quality` (BiRefNet) | `localhost:8080` |

Published image: `ghcr.io/sellaro-net/backremove`. Releases carry a full
`sha-<commit>` tag, `latest`, SBOM and build provenance. Use the published
`@sha256:…` digest for deployments and keep the previous digest for rollback.
CI does not deploy the service or delete old package versions.

The former `ghcr.io/tentoxa/backremove` package is separate: transferring the
repository does not move images. Existing deployment pins remain unchanged until
an operator explicitly switches them.

## Native setup (Windows + NVIDIA GPU)

Requires Python 3.12 and a current NVIDIA driver.

Create a git-ignored `.env` beside `start-gpu.bat` first:

```dotenv
API_KEY=replace-with-a-long-random-secret
```

Double-click `start-gpu.bat`; it installs/checks PyTorch 2.14.0 with CUDA 12.6,
cuDNN 9 and ONNX Runtime 1.26.0 in `.venv`, caches the pinned model weights,
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

### GPU compatibility is a versioned contract

The CUDA 12.6 PyTorch wheel includes `sm_61` kernels for the GTX 1060.
ONNX Runtime GPU stays on 1.26.0 because 1.27+ wheels require CUDA 13, which
does not support Pascal. Both inference engines use the CUDA/cuDNN libraries
shipped with PyTorch; separate CUDA 11 packages and manual DLL scanning are
no longer required.

This combination was exercised on a GTX 1060 6 GB with NVIDIA driver 582.28.
Do not replace the CUDA index or ONNX Runtime GPU pin with an unqualified
latest version. An upgrade must pass an actual GPU kernel and both model paths,
not merely `torch.cuda.is_available()` or a CPU-only test.

## Native CPU setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-cpu.lock.txt
$env:INFERENCE_DEVICE = "cpu"
$env:API_KEY = "<strong random key>"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8080 --no-proxy-headers
```

The minimal CPU setup exposes only `fast`. The supported native GPU setup
installs the additional quality dependencies and preloads both models.

## Docker

```bash
docker compose up --build
```

Compose reads `API_KEY` from the local `.env` file and stops with an error if
the value is missing.

The Compose endpoint is `http://127.0.0.1:8000`, not port 8080. It is deliberately
bound to loopback; remote access belongs behind an authenticated TLS reverse
proxy or an explicitly configured private network.

The process runs as UID 1000, with a read-only root filesystem, no Linux
capabilities and no privilege escalation. Only the named `model-cache` volume
and bounded `/tmp` are writable. Keep the Compose project name and volume when
upgrading; `docker compose down --volumes` deletes downloaded model weights.
The logical service name stays `backremove`; no fixed `container_name` is needed.

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

## Development and verification

Python 3.12 is the supported interpreter. Dependency ownership:

| File | Purpose |
|------|---------|
| `requirements.txt` | Direct CPU API dependencies |
| `requirements-cpu.lock.txt` | Universal Python 3.12 resolution with hashes; used by Docker and CI |
| `requirements-quality.txt` | Shared BiRefNet dependencies, including Transformers |
| `requirements-gpu.txt` | Quality dependencies plus the Pascal-compatible CUDA 12 / cuDNN 9 stack |
| `requirements-dev.txt` | Pinned lock-generation and advisory tools |

Run in a Python 3.12 virtual environment:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --require-hashes -r requirements-cpu.lock.txt
python -m pip check
python -m unittest discover -s tests -v
python -m pip_audit --strict --disable-pip --require-hashes -r requirements-cpu.lock.txt
```

After editing CPU requirements, regenerate and commit the lock:

```bash
python -m uv --system-certs pip compile requirements.txt \
  --python-version 3.12 --universal --generate-hashes --no-header \
  --output-file requirements-cpu.lock.txt
```

The required `build` PR check validates workflows, lock consistency, advisories
and regression tests. It also builds the real Docker image without publishing,
starts the actual Compose service and checks readiness, non-root execution,
authentication and unsupported uploads. A separate step loads the pinned
BiRefNet model and performs a real CPU inference using the supported PyTorch
version; this catches Transformers/model-code incompatibilities without needing
a GPU runner. CUDA execution remains a native NVIDIA validation.

Only a successful `main` push can publish an image. All external GitHub Actions
are commit-pinned; Dependabot proposes workflow, Python and Docker updates.
