# BackRemove

**Built with DINOv3.**

Standalone native Rust background-removal API. Fast uses the pinned withoutBG
v10 ONNX model; the optional Quality model is a full BiRefNet ONNX export.
HTTP, admission, scheduling, decoding, preprocessing, inference orchestration and
PNG output are native. The shipped server contains **no Python, PyTorch,
Transformers, withoutbg SDK or CUDA development toolkit**.

The canonical repository is [sellaro-net/backremove](https://github.com/sellaro-net/backremove).
Part of [Sellaro](https://github.com/sellaro-net/sellaro), BackRemove is maintained
independently from the web/mobile application and
[ja3proxy](https://github.com/sellaro-net/ja3proxy).

The native Rust release line starts at **v2.0.0**. The latest Python implementation,
through commit `61d07f6243b338b76d393ee569a7249a23fab6a8`, is retained on
[`legacy-python`](https://github.com/sellaro-net/backremove/tree/legacy-python).
Use that branch for the former Python application and its setup/dependency
instructions; it is not a runtime fallback for the native release.

## Run a native Windows release

Runtime prerequisites: Windows x64, the Microsoft Visual C++ 2015–2022 x64
Redistributable, and, for CUDA, a suitable NVIDIA driver. The reference GPU is a
GTX 1060 6 GB; its validated driver was 582.28. The private CUDA 12.8.1/cuDNN
9.8.0.87 libraries are included in the prepared release; a global CUDA install is
not required. Keep `dav1d.dll` beside `backremove.exe` and keep the complete
`artifacts/` and `licenses/` directories with the executable.

Set a strong `API_KEY` in the environment or a local `.env`. `.env.example` lists
the supported keys; setup never writes an existing `.env`. From the repository
with a built `dist/`, or from inside the release directory:

```bat
start-gpu.bat
```

The launcher runs only the executable. It does not install packages, download
models or start a Python fallback. CUDA is explicit and Quality is enabled;
initialization failures stop startup instead of silently falling back to CPU.
Both enabled sessions are loaded and warmed before readiness.

The default native port is **8585**:

```powershell
Invoke-RestMethod http://127.0.0.1:8585/health
.\dist\backremove.exe --version
.\dist\backremove.exe --healthcheck
```

`--healthcheck` exits nonzero when the configured local service is not healthy.
Use a separate `PORT` for development checks rather than disturbing another
running process. Shutdown is Ctrl+C/SIGTERM with a bounded grace period; setup
and artifact preparation never launch or stop a service.

## Build on Windows

Build prerequisites, not runtime dependencies:

- Rust **1.95.0**, Cargo and Git.
- Visual Studio 2022 C++ x64 build tools and the Windows SDK.
- Python **3.12 x64** for isolated model preparation and build inventory tooling
  (reference: 3.12.9); the default setup downloads its pinned inputs.
- Sufficient disk space for source wheels, both models, FP32 intermediate graphs,
  native dependencies and private output copies. These inputs occupy several GB.

From a fresh output directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup-gpu.ps1
```

This creates `.build-tools/python`, downloads hash-locked tool wheels, prepares
`artifacts/windows-cuda/manifest.json`, builds dav1d **1.5.3** from vcpkg commit
`296b89248ad13be09e9324488eb4f74dd322d26c`, and runs
`cargo build --release --locked`. It writes a native `dist/` with a build/license
inventory. Existing artifact packs and `dist/` are not silently overwritten.
Move an old release aside before producing a new one.

The local `vendor/dav1d-sys` patch corrects two swapped field widths in upstream
0.8.3's sequence-header ABI. Its build script compiles C static assertions
against the actual dav1d headers for sizes, alignments and field offsets.
An incompatible binding/header pair fails the build, before native decoding.
The regression suite includes an actual AVIF with an alpha plane.

If the exact CUDA/cuDNN versions are already installed, use them as **read-only,
hash-checked input sources**, not as runtime search paths:

```powershell
.\setup-gpu.ps1 `
  -CudaSource 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin' `
  -CudnnSource 'C:/Program Files/NVIDIA/CUDNN/v9.8/bin/12.8'
```

Every selected DLL must match the locked fingerprint; mismatches fail closed.
Without these options, preparation downloads the exact official NVIDIA
Redistributables listed in `tools/sources.lock.json`. `-PrepareOnly` omits the
native build. With an already prepared pack, `-SkipArtifacts` builds and packages
only. `-ToolPython <python.exe>` can explicitly reuse a compatible installed
build-tool interpreter without modifying it; the Quality exporter checks its
pinned dependency versions and isolates all mutable model/module caches.

For a Windows CPU release:

```powershell
.\setup-gpu.ps1 -Target windows-cpu
# Run from the resulting release directory, not with the CUDA-only launcher:
Set-Location .\dist
$env:INFERENCE_DEVICE = 'cpu'
$env:QUALITY_MODEL_ENABLED = '0'
$env:API_KEY = '<strong random key>'
.\backremove.exe
```

CPU packs contain only Fast. Requesting Quality returns `503` with
`code: model_unavailable`; it is never replaced with Fast.

### Separate download and offline preparation

The preparation command is independent of Cargo and does not execute inference:

```powershell
# Populate the isolated tool environment and all required source inputs:
.\setup-gpu.ps1 -FetchOnly

# Prepare using only cached/pinned inputs; no network is permitted:
.\.build-tools\python\Scripts\python.exe -B tools\prepare.py `
  --target windows-cuda --offline

# A separate native build after preparation:
.\setup-gpu.ps1 -SkipArtifacts
```

`tools/prepare.py --target windows-cpu|linux-cpu` also prepares CPU packs, including
on a different host: it extracts native files without loading that target's
runtime. Linux CPU preparation needs only the hash-locked
`tools/requirements-prepare.txt` environment, not Torch. `--cache`, `--hf-cache`,
`--output`, `--cuda-source` and `--cudnn-source` select explicit local locations.
The default Hugging Face cache is read-only input; weights are copied privately.
`--fetch-only` needs only Python's standard library. `--offline` rejects missing
inputs, corrupt cached files, hash mismatches and unverified alternatives.
Downloads have exact byte/hash checks, HTTPS-only redirects and time limits.

The full native `-Offline` build additionally requires the previously populated
vcpkg bootstrap/download cache and Cargo cache. `cargo fetch --locked` populates
Cargo; an ordinary native build populates vcpkg. A Python/model download cache
alone does not constitute a complete offline compiler environment.

## Artifact and precision contract

`tools/sources.lock.json` pins source URLs, revisions, sizes and real hashes.
The resulting `manifest.json` version 1 binds models, runtime files, load order,
fonts and source/build provenance. The server verifies loaded artifact hashes
before initializing ONNX Runtime. It never discovers a remote model revision,
executes remote Python code or downloads a missing artifact at startup.

| Component | Pinned input / preparation |
|---|---|
| Fast | `withoutbg/withoutbg-openweights-onnx@a34f90ae16bcc5238afeedfa96251d097ecdc2fd`; canonical SHA-256 `29930e48e9d5ecc56d6486c53c35a4c1470566c2a3359fa180b08c8d3c34ef0f` plus checked sidecar |
| Fast CUDA | Exactly one `FusedConv(Sigmoid)` → `Conv` + `Sigmoid`; derived SHA-256 `53dd40f06c8be541c71651b02635ea9a0515b02fb3d604bdc273284b896333e6` |
| Quality | `ZhengPeng7/BiRefNet@e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`; safetensors SHA-256 `9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154` |
| ORT | Official **1.26.0** wheels; outer SHA-256 plus selected-file `RECORD` checks; only native loader files and license/SBOM material enter server packs |
| Windows CUDA | Official `onnxruntime_gpu-1.26.0-cp312-cp312-win_amd64.whl`, SHA-256 `5f49c44689894650990e4c8a857d2edafc276fbd79bba57ceb224bd18d25d491`; CUDA 12.8.1 components and cuDNN 9.8.0.87 individually pinned |
| AVIF | dav1d **1.5.3**, source archive SHA-512 pinned by vcpkg; no automatic dav1d-sys source fallback |
| SVG fonts | Noto Sans Regular/Bold, Noto Serif Regular, Noto Sans Mono Regular from `fonts-noto-{core,mono}_20201225-1_all.deb`; exact TTF hashes and OFL-1.1 copyright/license included |

Quality preparation is **FP32 static opset-19 export → offline ORT CPU
`ORT_ENABLE_BASIC` folding → the pinned ORT Cast-aware FP16 converter**. The
20 standard-domain `DeformConv` nodes are retained. Every converted float tensor
is checked against actual IEEE binary16 rounding; the converter's arbitrary
small/large-value clamps are undone. Full ONNX type checking and the proven
folded graph/tensor counts are required. The rejected onnxconverter-common and
unfolded FP16 paths are not release alternatives. FP32 intermediates and their
provenance remain in the ignored build cache, not in the server release.

The reference folded FP16 SHA-256 is
`97b162897b0e131cb2aecd6ae3369b7b75138ef61b2f7b186d605a3568bf0756`.
Preparation records the produced hash and whether it is byte-identical to that
reference; serialized graph metadata may differ. A changed hash is **not** an
automatic quality or GPU release approval. Native target-specific warmup,
operator placement, image-quality and load checks remain required.

Private CUDA dependencies are preloaded in the verified order. ORT itself
initializes its provider DLL; `onnxruntime_providers_cuda.dll` is not directly
preloaded. The known incompatible official C++ GPU ZIP is not a replacement for
the pinned wheel on the GTX 1060. CUDA sessions disable CPU fallback, parallel
model calls, memory patterns and TF32, and use the verified bounded-workspace
options. `BACKREMOVE_ORT_PROFILE_DIR` optionally captures startup profiles and
checks CUDA operator placement; use a private writable directory for release
verification, not normal traffic logging.

Probability validation permits exactly one FP32 ULP above 1: ORT 1.26's
MLAS FMA3 sigmoid produced `0x3f800001` in the native CPU JPEG check.
The reference-compatible PNG conversion saturates this endpoint to alpha 255.
Larger excursions, negative values and nonfinite output remain errors;
there is no arbitrary epsilon, extra sigmoid or mask renormalization.

## Local Linux CPU Docker build

There is no dependency on a pre-existing published BackRemove Rust image:

```bash
docker build --platform linux/amd64 -t backremove-native:local-cpu .
# Optional local operation, after setting API_KEY in the local environment/.env:
docker compose up --build
```

Compose has the concrete local build context `.` and image tag
`backremove-native:local-cpu`, binds `127.0.0.1:8000`, and does not mount a mutable
model cache. The multi-stage Dockerfile prepares pinned CPU model/runtime
artifacts, builds dav1d 1.5.3 from a hash-checked source archive, and compiles the
application with locked Cargo dependencies and the pinned Rust toolchain.
Python is used only in build stages for artifact preparation and native build
tooling. The final image contains native libraries, the executable, Fast,
explicit fonts and license/provenance records; **no Python**. It runs as UID/GID
10001 with root-owned, read-only release files. Compose adds a read-only
filesystem, dropped capabilities, no-new-privileges, memory/PID limits and a
bounded temporary filesystem. Health checks use the native `--healthcheck`
command. Linux system dependencies are glibc, libstdc++ and the packaged dav1d
library; the runtime image also supplies libgomp.

Compose requires a nonempty `API_KEY` and keeps the logical service name
`backremove`, without a fixed container name. Its loopback endpoint is deliberate:
remote access belongs behind an authenticated TLS reverse proxy or an explicitly
configured private network. Forwarded client IPs are accepted only from
`TRUSTED_PROXIES`; do not broaden that allowlist to untrusted clients.

The base image manifests and model/native source inputs are pinned, but Debian
package repositories are not snapshot-locked. A fresh build is therefore **not
guaranteed byte-for-byte reproducible**. Recorded hashes and package inventories
identify the produced release; they do not replace target-runtime verification.

### Native release publication

CI builds and exercises the real Linux CPU image through the Compose service,
including a complete SVG → Fast → RGBA-PNG request and unavailable Quality.
It stores build evidence. On a published GitHub release, the release workflow
publishes the **exact tested OCI image**, retaining its SBOM and build provenance,
to `ghcr.io/sellaro-net/backremove:<release-tag>-native-cpu`; it does not rebuild
the image for publication. For release `v2.0.0`, that reference is:

```text
ghcr.io/sellaro-net/backremove:v2.0.0-native-cpu
```

This describes the workflow's release contract, not confirmation that the image
is already available. Windows CUDA builds are packaged separately as native
release assets with their private runtime libraries; the Linux image is CPU-only.
Branch/PR builds do not publish images. The native workflow does **not** create
or update `latest`, deploy a service, or delete old package versions.

For an operator-managed deployment, use the published `@sha256:…` digest and keep
the previous digest for rollback. Existing image/deployment pins remain unchanged
until an operator explicitly switches them. The former
`ghcr.io/tentoxa/backremove` package is separate: moving the repository to the
Sellaro organization does not move its images or repoint either package's
existing `latest` tag.

## HTTP contract

```bash
curl --fail-with-body "http://127.0.0.1:8585/remove-bg?model=fast" \
  -H "X-API-Key: <your key>" \
  -H "X-BackRemove-Work-Class: foreground" \
  -F "file=@photo.jpg" --output no-bg.png
```

Use `model=quality` for Quality with a CUDA pack and
`QUALITY_MODEL_ENABLED=1`. Supported input formats: JPEG, PNG, WebP, GIF, AVIF,
static SVG. One multipart `file`, maximum **20 MiB** plus **64 KiB** multipart
overhead, **40,000,000** decoded pixels and **32 MiB** PNG output. Animated
inputs use the first frame. Input alpha is not multiplied into the predicted
mask; output preserves source RGB with straight alpha. SVG cannot load network,
filesystem or external-font resources; embedded raster content is budgeted and
fonts come only from the checked pack.

Successful responses are `image/png` with
`Content-Disposition: attachment; filename=no-bg.png`, `X-Model-Used`,
`X-Admission-Time-Ms`, `X-Decode-Time-Ms`, `X-Queue-Time-Ms`,
`X-Inference-Time-Ms` and `X-Encode-Time-Ms`. Inference timing is the model
processing step, not a claim of isolated CUDA kernel time.

| Endpoint | Authentication | Result |
|---|---|---|
| `GET /health` | Public | Readiness, model/provider and scheduler state |
| `POST /remove-bg?model=fast\|quality` | `X-API-Key` | RGBA PNG; default model is Fast |

Errors contain a stable `code` and a German `detail`:

| HTTP | Code | Meaning |
|---|---|---|
| 400 | `invalid_image` | Image cannot be decoded safely |
| 401 / 429 | `unauthorized` / `rate_limited` | Missing/incorrect key or blocked authentication attempts |
| 408 | `upload_timeout` | Upload did not finish before its deadline |
| 413 | `payload_too_large` | Input byte/pixel limit exceeded |
| 415 | `unsupported_media_type` | Unsupported input media type |
| 422 | `invalid_request` | Invalid request/model selection |
| 503 | `busy` | Capacity exhausted or queued deadline expired; `Retry-After: 2` |
| 503 | `model_unavailable` | Requested model is not available; not a queue retry signal |
| 504 | `deadline_exceeded` | Processing/encoding response deadline expired |
| 500 | `internal_error` | Generic internal failure; no internal diagnostics in the response |

Authentication happens before body processing. Five failed attempts within
60 seconds trigger a 900-second block. Client-IP forwarding headers are trusted
only for peers explicitly configured by `TRUSTED_PROXIES`. `/health` does not
require a key. Do not log API keys, image bytes or output masks.

## Scheduling, deadlines and configuration

A process owns one inference worker and both enabled resident sessions. Admission
is bounded across upload, queue, preparation, inference, encoding and response
ownership: eight waiting jobs and nine total by default. Decode/preparation is
serialized with one prepared slot; two encoders can overlap the next inference.
Input-byte and weighted image-memory limits apply before large allocation.
Foreground/background queues use a 3:1 selection ratio when both contain work.
`X-BackRemove-Work-Class` defaults to `foreground`; `background` is available to
authenticated callers.

Fast has one **9-second** deadline and Quality one **29-second** deadline from
request acceptance, including upload and queue. There is no second hidden
75-second inference budget. A cancelled/expired queued job does not start;
a running native call retains its resources and exclusive inference ownership
until it actually returns. A response timeout cannot launch overlapping CUDA
work. A hung native call is visible in health and may require a supervisor to
terminate the process after its configured shutdown grace period.

These deadlines cover upload through the encoded response, not the client's
network download. Response bytes retain their admission and memory reservations
until the final transport consumer releases them.

An early rejection sends the response and shuts down TCP writes before draining
late raw input with a fixed buffer. Draining ends on peer close/error, after one
second, or at the multipart byte limit. This prevents ordinary split uploads
from losing a `503` to an immediate TCP reset without reading rejected HTTP
bodies into the application. Peers continuing beyond either bound can still see
a reset; HTTP clients can use `Expect: 100-continue` for large uploads.

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | required | Constant-time authenticated request key |
| `HOST` / `PORT` | `0.0.0.0` / `8585` | Native listener; Docker explicitly uses 8000 |
| `INFERENCE_DEVICE` | `auto` | `auto`, `cpu` or `cuda`; pack/device must agree |
| `QUALITY_MODEL_ENABLED` | `0` | Enable resident Quality; requires CUDA pack |
| `ARTIFACT_MANIFEST` | platform/device pack | Local checked manifest path |
| `CORS_ORIGINS` | empty | Explicit comma-separated origin allowlist |
| `TRUSTED_PROXIES` | empty | Explicit comma-separated IP/CIDR allowlist |
| `QUEUE_CAPACITY` | `8` | Waiting capacity; total admission is capacity + 1 |
| `INPUT_BUDGET_MB` | `256` | Encoded-input budget in MiB |
| `MEMORY_BUDGET_MB` | `1024` | Image-stage reservation budget in MiB; not total process RAM/VRAM |
| `FAST_TIMEOUT` / `QUALITY_TIMEOUT` | `9` / `29` | Upload-to-encoded-response seconds |
| `SHUTDOWN_GRACE` | `30` | Graceful drain seconds |
| `AUTH_MAX_ENTRIES` | `10000` | Bound on tracked authentication state |

Remove obsolete `GPU_QUEUE_TIMEOUT`, `INFERENCE_TIMEOUT` and
`PROXY_REQUEST_BUDGET` settings when writing a native configuration.
`GPU_QUEUE_CAPACITY` is replaced by `QUEUE_CAPACITY`; no two competing admission
controllers or separate queue/inference budgets are used.

## Provenance and redistribution

The generated pack includes `provenance/sources.lock.json`, export/type/rounding
records and a file inventory. Native releases add `release-inventory.json` with
source-file hashes, Cargo.lock checksums, compiler identity, native build files,
package license expressions and copied license notices. Docker additionally
records its build/runtime OS package inventories. No source/model hash is a
substitute for checking the actual target image/runtime behavior.

Redistribution requires review of the combined inventory: withoutBG's Apache
license **and DINOv3 agreement/attribution**, BiRefNet MIT terms, ORT MIT and
third-party notices, NVIDIA CUDA/cuDNN redistribution terms, dav1d BSD-2-Clause,
Noto OFL-1.1 and the Cargo/system dependency licenses. Full source notices are
retained in `licenses/`. Build tooling records licenses; it does not provide
legal approval. Preserve the prominent **Built with DINOv3** attribution when
publishing a product or documentation that incorporates these models.
