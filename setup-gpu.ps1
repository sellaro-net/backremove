$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python 3.12 virtual environment."
    }
}

& $python -m pip install --upgrade pip setuptools
if ($LASTEXITCODE -ne 0) { throw "Could not update Python packaging tools." }
& $python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0 torchvision==0.29.0
if ($LASTEXITCODE -ne 0) { throw "Could not install the PyTorch CUDA 12.6 runtime." }

& $python -m pip install -r requirements-gpu.txt
if ($LASTEXITCODE -ne 0) { throw "Could not install GPU requirements." }

# withoutbg depends on the CPU distribution, which shares its Python package
# with onnxruntime-gpu. On a fresh install, activate the GPU wheel last.
& $python -c "import onnxruntime as ort; providers = ort.get_available_providers(); raise SystemExit(0 if ort.__version__ == '1.26.0' and 'CUDAExecutionProvider' in providers else 1)"
if ($LASTEXITCODE -ne 0) {
    & $python -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.26.0
    if ($LASTEXITCODE -ne 0) { throw "Could not activate ONNX Runtime GPU." }
}

& $python -c "import onnxruntime as ort; providers = ort.get_available_providers(); print('ONNX Runtime providers:', providers); raise SystemExit(0 if ort.__version__ == '1.26.0' and 'CUDAExecutionProvider' in providers else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "CUDAExecutionProvider is unavailable. Check the NVIDIA driver."
}

& $python -c "import torch; print('PyTorch CUDA:', torch.version.cuda, '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'); assert torch.version.cuda == '12.6' and torch.cuda.is_available(), 'CUDA 12.6 is unavailable'; value = torch.ones(1, device='cuda'); torch.cuda.synchronize(); assert value.item() == 1"
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch CUDA 12.6 cannot execute on this GPU. Check the NVIDIA driver and hardware support."
}

& $python -m app.prefetch_models
if ($LASTEXITCODE -ne 0) {
    throw "Could not cache the pinned BiRefNet quality model."
}
