[CmdletBinding()]
param(
    [ValidateSet("windows-cuda", "windows-cpu")][string]$Target = "windows-cuda",
    [string]$ToolPython,
    [string]$CudaSource,
    [string]$CudnnSource,
    [switch]$Offline,
    [switch]$FetchOnly,
    [switch]$PrepareOnly,
    [switch]$SkipArtifacts
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

# This is a developer/build entry point. start-gpu.bat never invokes it.
# Nothing is installed globally and the existing .env is never written.
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments, [int]$TimeoutSeconds = 3600)
    $quoted = foreach ($argument in $Arguments) {
        '"' + ([regex]::Replace($argument, '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"'
    }
    # Keep the original native process handle. Windows PowerShell's
    # Start-Process -PassThru can lose ExitCode for a short-lived child.
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo.FileName = $Program
    $process.StartInfo.Arguments = $quoted -join " "
    $process.StartInfo.WorkingDirectory = $PSScriptRoot
    $process.StartInfo.UseShellExecute = $false
    if (-not $process.Start()) { throw "Build-Werkzeug konnte nicht gestartet werden: $Program" }
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        & taskkill.exe /PID $process.Id /T /F | Out-Null
        throw "Zeitbudget fuer Build-Werkzeug ueberschritten: $Program"
    }
    if ($process.ExitCode -ne 0) { throw "Build-Werkzeug fehlgeschlagen ($($process.ExitCode)): $Program" }
}

$cache = Join-Path $PSScriptRoot ".artifacts-cache"
$buildTools = Join-Path $PSScriptRoot ".build-tools"
New-Item -ItemType Directory -Force $cache, $buildTools | Out-Null
if (-not $ToolPython) {
    $toolEnvironment = Join-Path $buildTools "python"
    $ToolPython = Join-Path $toolEnvironment "Scripts\python.exe"
    if (-not (Test-Path $ToolPython)) {
        Invoke-Checked "py.exe" @("-3.12", "-m", "venv", $toolEnvironment) 120
    }
    $requirements = Join-Path $PSScriptRoot "tools\requirements-prepare.txt"
    if ($Target -eq "windows-cuda") { $requirements = Join-Path $PSScriptRoot "tools\requirements-export-windows.txt" }
    $wheels = Join-Path $cache "wheels"
    New-Item -ItemType Directory -Force $wheels | Out-Null
    if (-not $Offline) {
        Invoke-Checked $ToolPython @("-m", "pip", "download", "--disable-pip-version-check", "--timeout", "30", "--retries", "2", "--require-hashes", "--only-binary=:all:", "--dest", $wheels, "-r", $requirements)
    }
    Invoke-Checked $ToolPython @("-m", "pip", "install", "--disable-pip-version-check", "--no-index", "--find-links", $wheels, "--require-hashes", "-r", $requirements)
}
$ToolPython = (Resolve-Path $ToolPython).Path
$pack = Join-Path $PSScriptRoot "artifacts\$Target"
if (-not $SkipArtifacts) {
    $prepareArgs = @("-B", (Join-Path $PSScriptRoot "tools\prepare.py"), "--target", $Target)
    if ($Offline) { $prepareArgs += "--offline" }
    if ($FetchOnly) { $prepareArgs += "--fetch-only" }
    if ($CudaSource) { $prepareArgs += @("--cuda-source", $CudaSource) }
    if ($CudnnSource) { $prepareArgs += @("--cudnn-source", $CudnnSource) }
    Invoke-Checked $ToolPython $prepareArgs 14400
}
if ($FetchOnly -or $PrepareOnly) { return }
if (-not (Test-Path (Join-Path $pack "manifest.json"))) { throw "Artefaktpaket fehlt: $pack" }

# dav1d 1.5.3 is pinned by this vcpkg commit, including archive SHA512 and
# all native build helper versions. Never use dav1d-sys's 1.5.0 auto-builder.
$vcpkg = Join-Path $buildTools "vcpkg"
$revision = "296b89248ad13be09e9324488eb4f74dd322d26c"
if (-not (Test-Path (Join-Path $vcpkg ".git"))) {
    if ($Offline) { throw "Offline-Build benoetigt den vorbereiteten vcpkg-Checkout." }
    New-Item -ItemType Directory -Force $vcpkg | Out-Null
    Invoke-Checked "git.exe" @("-C", $vcpkg, "init") 60
    Invoke-Checked "git.exe" @("-C", $vcpkg, "remote", "add", "origin", "https://github.com/microsoft/vcpkg.git") 60
    Invoke-Checked "git.exe" @("-C", $vcpkg, "fetch", "--depth", "1", "origin", $revision) 600
    Invoke-Checked "git.exe" @("-C", $vcpkg, "checkout", "--detach", "FETCH_HEAD") 120
}
$actualRevision = & git.exe -C $vcpkg rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actualRevision.Trim() -ne $revision) { throw "vcpkg-Quellstand stimmt nicht mit dem Pin ueberein." }
$vcpkgExe = Join-Path $vcpkg "vcpkg.exe"
if (-not (Test-Path $vcpkgExe)) {
    if ($Offline) { throw "Offline-Build benoetigt das bereits vorbereitete vcpkg.exe." }
    Invoke-Checked "powershell.exe" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $vcpkg "scripts\bootstrap.ps1"), "-disableMetrics") 600
}
$installed = Join-Path $PSScriptRoot "build\native\installed"
$nativeArgs = @("install", "dav1d:x64-windows", "--classic", "--disable-metrics", "--x-install-root=$installed")
if ($Offline) { $nativeArgs += "--x-no-downloads" }
Invoke-Checked $vcpkgExe $nativeArgs 3600
$native = Join-Path $installed "x64-windows"
$env:SYSTEM_DEPS_DAV1D_NO_PKG_CONFIG = "1"
$env:SYSTEM_DEPS_DAV1D_LIB = "dav1d"
$env:SYSTEM_DEPS_DAV1D_SEARCH_NATIVE = Join-Path $native "lib"
$env:SYSTEM_DEPS_DAV1D_INCLUDE = Join-Path $native "include"
$env:SYSTEM_DEPS_DAV1D_BUILD_INTERNAL = "never"
if (-not $Offline) { Invoke-Checked "cargo.exe" @("fetch", "--locked") 600 }
$cargoArgs = @("build", "--release", "--locked")
if ($Offline) { $cargoArgs += "--offline" }
Invoke-Checked "cargo.exe" $cargoArgs 3600

$dist = Join-Path $PSScriptRoot "dist"
if (Test-Path $dist) { throw "dist existiert bereits. Fuer einen neuen Build einen frischen Ausgabeordner bereitstellen; vorhandenes Release bleibt unveraendert." }
New-Item -ItemType Directory $dist | Out-Null
Copy-Item (Join-Path $PSScriptRoot "target\release\backremove.exe") (Join-Path $dist "backremove.exe")
Copy-Item (Join-Path $native "bin\dav1d.dll") (Join-Path $dist "dav1d.dll")
New-Item -ItemType Directory (Join-Path $dist "artifacts"), (Join-Path $dist "licenses\dav1d") -Force | Out-Null
Copy-Item $pack (Join-Path $dist "artifacts\$Target") -Recurse
Copy-Item (Join-Path $native "share\dav1d\copyright") (Join-Path $dist "licenses\dav1d\COPYING.txt")
Copy-Item (Join-Path $PSScriptRoot "start-gpu.bat") (Join-Path $dist "start-gpu.bat")
Copy-Item (Join-Path $PSScriptRoot ".env.example") (Join-Path $dist ".env.example")
Invoke-Checked $ToolPython @("-B", (Join-Path $PSScriptRoot "tools\build_inventory.py"), "--output", $dist, "--dav1d-prefix", $native) 600
Write-Host "Natives Release erstellt: $dist"
Write-Host "Kein Dienst wurde gestartet. Laufzeit: MSVC x64 Redistributable; fuer CUDA zusaetzlich geeigneter NVIDIA-Treiber. Kein Python/CUDA-Toolkit erforderlich."
