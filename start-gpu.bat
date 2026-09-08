@echo off
setlocal
cd /d "%~dp0"
set "RELEASE_ROOT=%~dp0"
if not exist "%RELEASE_ROOT%backremove.exe" set "RELEASE_ROOT=%~dp0dist\"
if not exist "%RELEASE_ROOT%backremove.exe" (
    echo [BackRemove] Natives Release fehlt. setup-gpu.ps1 ist der getrennte Entwickler-Build, kein Laufzeit-Setup.
    exit /b 1
)
if not exist "%RELEASE_ROOT%dav1d.dll" (
    echo [BackRemove] Das native Release ist unvollstaendig: dav1d.dll fehlt.
    exit /b 1
)
if not defined ARTIFACT_MANIFEST set "ARTIFACT_MANIFEST=%RELEASE_ROOT%artifacts\windows-cuda\manifest.json"
if not exist "%ARTIFACT_MANIFEST%" (
    echo [BackRemove] Das gepruefte CUDA-Artefaktpaket fehlt.
    exit /b 1
)
set "INFERENCE_DEVICE=cuda"
set "QUALITY_MODEL_ENABLED=1"
echo [BackRemove] Starte nativen Rust-Dienst. Beenden mit Strg+C.
echo [BackRemove] API_KEY wird vom Dienst aus der Umgebung oder der lokalen .env gelesen.
"%RELEASE_ROOT%backremove.exe"
exit /b %ERRORLEVEL%
