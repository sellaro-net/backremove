"""Prepare private, hash-checked native artifacts; never imported by the server.

Python 3.12 is build tooling only. `--fetch-only` needs only its standard library.
The normal command uses the locked ONNX tool environment and does not run inference.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
LOCK_PATH = TOOLS / "sources.lock.json"
LOCK = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
CHUNK = 4 * 1024 * 1024
PRELOAD = (
    "cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cufft64_11.dll",
    "nvrtc-builtins64_128.dll", "nvrtc64_120_0.dll", "cudnn64_9.dll", "cudnn_ops64_9.dll",
    "cudnn_adv64_9.dll", "cudnn_cnn64_9.dll", "cudnn_graph64_9.dll",
    "cudnn_engines_precompiled64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_heuristic64_9.dll",
)
LINUX_PRELOAD = (
    "libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12", "libnvJitLink.so.12",
    "libcufft.so.11", "libcurand.so.10", "libnvrtc-builtins.so.12.8", "libnvrtc.so.12",
    "libcudnn.so.9", "libcudnn_graph.so.9", "libcudnn_ops.so.9", "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9", "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9", "libcudnn_heuristic.so.9",
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, pin):
    if not path.is_file() or path.stat().st_size != pin["size"] or sha256(path) != pin["sha256"]:
        raise ValueError(f"Integritätsprüfung fehlgeschlagen: {path}")
    return path


def record(path, base):
    return {"path": path.relative_to(base).as_posix(), "sha256": sha256(path), "size": path.stat().st_size}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def copy_private(source, destination, pin=None):
    if pin:
        verify(source, pin)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if pin:
        verify(destination, pin)
    return destination


class HttpsRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise ValueError("Unsichere Download-Weiterleitung")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(pin, args, local=None):
    destination = args.cache / pin["filename"]
    if destination.exists():
        return verify(destination, pin)
    if local is not None and local.is_file():
        temporary = destination.with_name(destination.name + ".copy")
        copy_private(local, temporary, pin)
        temporary.replace(destination)
        return destination
    if args.offline:
        raise FileNotFoundError(f"Offline-Quelle fehlt: {destination}")
    if urllib.parse.urlparse(pin["url"]).scheme != "https":
        raise ValueError("Quellen benötigen HTTPS")
    print(f"Lade {pin['filename']} ({pin['size']} Bytes)", flush=True)
    opener = urllib.request.build_opener(HttpsRedirects())
    started = time.monotonic()
    request = urllib.request.Request(pin["url"], headers={"User-Agent": "BackRemove-offline-preparation/1"})
    temporary = destination.with_name(destination.name + ".download")
    total = 0
    try:
        with opener.open(request, timeout=30) as response, temporary.open("wb") as output:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != pin["size"]:
                raise ValueError("Downloadgröße stimmt nicht mit dem Quellenmanifest überein")
            while block := response.read1(CHUNK):
                total += len(block)
                if total > pin["size"] or time.monotonic() - started > args.download_timeout:
                    raise ValueError("Download überschreitet Größen- oder Zeitbudget")
                output.write(block)
        verify(temporary, pin)
        temporary.replace(destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def wheel_records(wheel):
    names = [name for name in wheel.namelist() if name.endswith(".dist-info/RECORD")]
    if len(names) != 1:
        raise ValueError("Wheel benötigt genau ein RECORD")
    rows = list(csv.reader(io.StringIO(wheel.read(names[0]).decode("utf-8"))))
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("Doppelte Wheel-RECORD-Pfade")
    return {row[0]: row[1:] for row in rows}


def wheel_extract(wheel, records, member, destination):
    expected, size = records[member]
    if not expected.startswith("sha256=") or not size:
        raise ValueError(f"Wheel-Datei ohne SHA-256-RECORD: {member}")
    digest = base64.urlsafe_b64decode(expected.split("=", 1)[1] + "==").hex()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wheel.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, CHUNK)
    return verify(destination, {"sha256": digest, "size": int(size)})


def package_runtime(wheel_path, target, output):
    native = ["onnxruntime.dll", "onnxruntime_providers_shared.dll"]
    if target == "windows-cuda":
        native.append("onnxruntime_providers_cuda.dll")
    elif target.startswith("linux-"):
        native = ["libonnxruntime.so.1.26.0", "libonnxruntime_providers_shared.so"]
        if target == "linux-cuda":
            native.append("libonnxruntime_providers_cuda.so")
    with zipfile.ZipFile(wheel_path) as wheel:
        records = wheel_records(wheel)
        for name in native:
            output_name = "libonnxruntime.so" if name.startswith("libonnxruntime.so.") else name
            wheel_extract(wheel, records, "onnxruntime/capi/" + name, output / "runtime" / output_name)
        licenses = [name for name in records if name.startswith("onnxruntime/") and
                    (Path(name).name in ("LICENSE", "ThirdPartyNotices.txt") or "sbom" in name.lower())]
        if not {"onnxruntime/LICENSE", "onnxruntime/ThirdPartyNotices.txt"}.issubset(licenses):
            raise ValueError("ORT-Lizenzinventar unvollständig")
        for name in licenses:
            wheel_extract(wheel, records, name, output / "licenses" / "onnxruntime" / Path(name).name)
    return "runtime/libonnxruntime.so" if target.startswith("linux-") else "runtime/onnxruntime.dll"


def extract_cpu_tool_wheel(wheel_path, destination):
    # This directory remains in the ignored build cache, never in an artifact pack.
    if destination.exists():
        shutil.rmtree(destination)
    with zipfile.ZipFile(wheel_path) as wheel:
        records = wheel_records(wheel)
        for member, fields in records.items():
            if not fields[0]:
                continue  # RECORD itself is protected by the pinned outer-wheel hash.
            relative = Path(member)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member:
                raise ValueError("Ungültiger Wheel-Pfad")
            wheel_extract(wheel, records, member, destination / relative)


def linux_cuda_inputs(args):
    pins = LOCK["cuda_files_linux"]
    native = args.cache / "native-linux-cuda"
    if native.is_symlink():
        raise ValueError(f"Privater CUDA-Cache darf kein Symlink sein: {native}")
    paths = {}
    pending = {}
    for name, pin in pins.items():
        cached = native / name
        if cached.is_symlink():
            raise ValueError(f"Private CUDA-Datei darf kein Symlink sein: {cached}")
        source_dir = args.cudnn_source if pin["component"] == "cudnn-linux" else args.cuda_source
        if source_dir is not None:
            candidates = [source_dir / name, source_dir / "lib64" / name]
            matches = [path for path in candidates if path.exists() or path.is_symlink()]
            if len(matches) != 1:
                raise ValueError(f"CUDA-Quelle fehlt oder ist mehrdeutig: {source_dir}/{name}")
            # Only the explicitly supplied, hash-checked source may contain symlinks.
            source = verify(matches[0].resolve(strict=True), pin)
            paths[name] = verify(cached, pin) if cached.exists() else copy_private(source, cached, pin)
        elif cached.exists():
            paths[name] = verify(cached, pin)
        else:
            pending.setdefault(pin["component"], {})[pin["member"]] = (name, pin)
    for component, members in pending.items():
        archive_path = fetch(LOCK["sources"][component], args)
        found = set()
        # One forward pass per compressed archive; never extract a tree or follow
        # archive links. The lock names the versioned regular file, not its SONAME link.
        with tarfile.open(archive_path, mode="r|xz") as archive:
            for member in archive:
                if member.name not in members:
                    continue
                name, pin = members[member.name]
                if member.name in found or not member.isfile() or member.size != pin["size"]:
                    raise ValueError(f"Ungültige CUDA-Archivdatei: {member.name}")
                found.add(member.name)
                cached = native / name
                cached.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, cached.open("xb") as destination:
                    shutil.copyfileobj(source, destination, CHUNK)
                paths[name] = verify(cached, pin)
        if found != members.keys():
            raise ValueError(f"CUDA-Archivdateien fehlen: {sorted(members.keys() - found)}")
    return paths


def cuda_inputs(args):
    if args.target == "linux-cuda":
        return linux_cuda_inputs(args)
    paths = {}
    archives = {}
    for name, pin in LOCK["cuda_files"].items():
        cached = args.cache / "native" / name
        source_dir = args.cudnn_source if pin["component"] == "cudnn" else args.cuda_source
        if cached.exists():
            paths[name] = verify(cached, pin)
            continue
        if source_dir is not None:
            paths[name] = copy_private(source_dir / name, cached, pin)
            continue
        component = pin["component"]
        if component not in archives:
            archives[component] = fetch(LOCK["sources"][component], args)
        with zipfile.ZipFile(archives[component]) as archive:
            matches = [entry for entry in archive.namelist() if Path(entry).name == name and "/bin/" in entry]
            if len(matches) != 1:
                raise ValueError(f"Redistributable-Datei fehlt oder mehrdeutig: {name}")
            cached.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(matches[0]) as source, cached.open("wb") as destination:
                shutil.copyfileobj(source, destination, CHUNK)
            paths[name] = verify(cached, pin)
    return paths


def deb_payload(path):
    with path.open("rb") as stream:
        if stream.read(8) != b"!<arch>\n":
            raise ValueError("Ungültiges Debian-Fontarchiv")
        while header := stream.read(60):
            if len(header) != 60 or header[58:] != b"`\n":
                raise ValueError("Ungültiger ar-Header")
            size = int(header[48:58])
            name = header[:16].decode("ascii").strip().rstrip("/")
            if name.startswith("data.tar."):
                if size > 32 * 1024 * 1024:
                    raise ValueError("Fontarchiv überschreitet Größenbudget")
                return stream.read(size)
            stream.seek(size + size % 2, 1)
    raise ValueError("Fontarchiv enthält keine Daten")


def package_fonts(inputs, output):
    for component in ("fonts-core", "fonts-mono"):
        with tarfile.open(fileobj=io.BytesIO(deb_payload(inputs[component])), mode="r:*") as archive:
            for name, pin in LOCK["font_files"].items():
                if pin["component"] != component:
                    continue
                member = archive.getmember(pin["member"])
                if not member.isfile() or member.size != pin["size"]:
                    raise ValueError("Ungültige Fontdatei")
                destination = output / "fonts" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, CHUNK)
                verify(destination, pin)
            license_member = "./usr/share/doc/fonts-noto-" + component.removeprefix("fonts-") + "/copyright"
            license_data = archive.extractfile(license_member).read(128 * 1024)
            if not all(clause in license_data for clause in (
                b"License: OFL-1.1\n", b"SIL Open Font License",
                b"PERMISSION & CONDITIONS", b"TERMINATION", b"DISCLAIMER",
            )):
                raise ValueError("Vollständige OFL-1.1-Lizenz im Fontpaket fehlt")
            destination = output / "licenses" / (component + "-copyright.txt")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(license_data)


def package_licenses(inputs, output, target):
    with zipfile.ZipFile(inputs["withoutbg-licenses"]) as wheel:
        records = wheel_records(wheel)
        for name in ("LICENSE", "LICENSE-DINOv3", "NOTICE"):
            member = "withoutbg-1.0.6.dist-info/licenses/" + name
            wheel_extract(wheel, records, member, output / "licenses" / "withoutbg" / name)
    copy_private(inputs["fast-card"], output / "licenses" / "fast-model-card.txt")
    if target.endswith("-cuda"):
        copy_private(inputs["quality-card"], output / "licenses" / "quality-model-card.txt")
        pin = LOCK["birefnet_license"]
        source = TOOLS / pin["path"]
        if sha256(source) != pin["sha256"]:
            raise ValueError("BiRefNet-Lizenz verändert")
        copy_private(source, output / "licenses" / source.name)
        provenance = "provenance-linux.json" if target == "linux-cuda" else "provenance.json"
        for pin in json.loads((TOOLS / "licenses" / provenance).read_text()).values():
            source = TOOLS / pin["path"]
            if sha256(source) != pin["sha256"]:
                raise ValueError("NVIDIA-Lizenz verändert")
            copy_private(source, output / "licenses" / source.name)


def prepare_fast(source, sidecar_path, destination, cuda):
    import onnx
    if onnx.__version__ != "1.22.0":
        raise ValueError("Fast-Ableitung benötigt onnx==1.22.0")
    sidecar = json.loads(sidecar_path.read_text())
    expected = {"canvas_size": 448, "opset_version": 18, "precision": "fp32", "model_version": "10.0.0",
                "input_name": "rgb", "output_name": "alpha", "input_shape": [1, 3, 448, 448],
                "output_shape": [1, 1, 448, 448], "sha256": LOCK["sources"]["fast"]["sha256"]}
    if any(sidecar.get(key) != value for key, value in expected.items()):
        raise ValueError("Fast-Sidecar weicht vom freigegebenen Modellvertrag ab")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cuda:
        copy_private(source, destination, LOCK["sources"]["fast"])
        return sidecar
    model = onnx.load(str(source))
    nodes = []
    defused = 0
    for node in model.graph.node:
        activation = next((attr.s.decode() for attr in node.attribute if attr.name == "activation"), None)
        if node.op_type != "FusedConv" or activation != "Sigmoid":
            nodes.append(node)
            continue
        attributes = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute if attr.name != "activation"}
        intermediate = f"{node.output[0]}_pre_sigmoid"
        nodes.extend([
            onnx.helper.make_node("Conv", node.input, [intermediate], f"{node.name}_conv", **attributes),
            onnx.helper.make_node("Sigmoid", [intermediate], node.output, f"{node.name}_sigmoid"),
        ])
        defused += 1
    if defused != 1:
        raise ValueError("Fast benötigt genau eine Sigmoid-Defusion")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))
    if sha256(destination) != LOCK["fast_derived_sha256"]:
        raise ValueError("Fast-Ableitung ist nicht byteidentisch zum geprüften Artefakt")
    return sidecar


def quality_stage(stage, source, destination, args, extra=()):
    report_path = destination.with_suffix(".provenance.json")
    if destination.exists():
        if not report_path.is_file():
            raise ValueError(f"Unvollständiger Offline-Build: {destination}; privaten Work-Ordner entfernen")
        report = json.loads(report_path.read_text())
        if report["status"] != "ok" or report["artifact"]["sha256"] != sha256(destination) or report["script"]["sha256"] != sha256(TOOLS / "export_quality.py"):
            raise ValueError(f"Veralteter oder beschädigter Offline-Build: {destination}")
        return
    command = [str(args.tool_python), "-B", str(TOOLS / "export_quality.py"), stage,
               "--source", str(source), "--output", str(destination), "--target", args.target, *map(str, extra)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    subprocess.run(command, check=True, timeout=args.export_timeout, env=environment)


def package_quality(inputs, output, args):
    build_id = hashlib.sha256((sha256(LOCK_PATH) + sha256(TOOLS / "export_quality.py")).encode()).hexdigest()[:16]
    work = args.cache / "work"
    if args.target == "linux-cuda":
        work /= args.target
    work /= build_id
    snapshot = work / LOCK["quality_revision"]
    for name in ("config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors"):
        copy_private(inputs["quality-" + name], snapshot / name, LOCK["sources"]["quality-" + name])
    ort_root = work / "cpu-wheel"
    platform = args.target.removesuffix("-cuda")
    extract_cpu_tool_wheel(inputs[f"ort-{platform}-cpu"], ort_root)
    converter_path = work / "ort-float16.py"
    with zipfile.ZipFile(inputs["ort-" + args.target]) as wheel:
        wheel_extract(wheel, wheel_records(wheel), "onnxruntime/transformers/float16.py", converter_path)
    fp32 = work / "quality-fp32.onnx"
    folded = work / "quality-basic-fp32.onnx"
    fp16 = work / "quality-fp16.onnx"
    quality_stage("export", snapshot, fp32, args)
    quality_stage("fold", fp32, folded, args, ("--ort-python-root", ort_root))
    quality_stage("half", folded, fp16, args, ("--converter", converter_path))
    copy_private(fp16, output / "models" / "quality.onnx")
    reports = {}
    for stage, path in (("fp32", fp32), ("folded_fp32", folded), ("fp16", fp16)):
        reports[stage] = json.loads(path.with_suffix(".provenance.json").read_text())
        copy_private(path.with_suffix(".provenance.json"), output / "provenance" / (stage + ".json"))
    copy_private(fp16.with_suffix(".rounding.json"), output / "provenance" / "fp16-rounding.json")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=["windows-cuda", "windows-cpu", "linux-cpu", "linux-cuda"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / ".artifacts-cache")
    parser.add_argument("--hf-cache", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--cuda-source", type=Path, help="Geprüfte Kopierquelle: CUDA-12.8.1-bin (Windows), Präfix oder lib64 (Linux)")
    parser.add_argument("--cudnn-source", type=Path, help="Geprüfte Kopierquelle: cuDNN-9.8.0.87-CUDA12-bin (Windows), Präfix oder lib64 (Linux)")
    parser.add_argument("--tool-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--download-timeout", type=int, default=1800)
    parser.add_argument("--export-timeout", type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.download_timeout <= 3600 or not 1 <= args.export_timeout <= 7200:
        parser.error("Zeitbudgets liegen außerhalb der erlaubten Grenzen")
    if args.target == "windows-cuda" and sys.platform != "win32" and not args.fetch_only:
        parser.error("Der geprüfte Quality-Export benötigt Windows x64 mit den gepinnten Build-Wheels")
    if args.target == "linux-cuda" and sys.platform != "linux" and not args.fetch_only:
        parser.error("Linux-Quality-Export benötigt Linux x64 mit den gepinnten Build-Wheels")
    args.cache = args.cache.resolve()
    if args.target == "linux-cuda":
        for source in (args.cuda_source, args.cudnn_source):
            if source is not None and args.cache.is_relative_to(source.resolve()):
                parser.error("Der private Cache darf nicht innerhalb einer schreibgeschützten CUDA-Quelle liegen")
    args.cache.mkdir(parents=True, exist_ok=True)
    # Resolving a Linux venv's python symlink discards its isolated site-packages.
    args.tool_python = args.tool_python.absolute() if args.target == "linux-cuda" else args.tool_python.resolve()
    keys = ["fast", "fast-sidecar", "fast-card", "fonts-core", "fonts-mono", "withoutbg-licenses", "ort-" + args.target]
    if args.target.endswith("-cuda"):
        platform = args.target.removesuffix("-cuda")
        keys += ["quality-card", f"ort-{platform}-cpu"] + ["quality-" + name for name in
                 ("config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors")]
    inputs = {}
    for key in keys:
        pin = LOCK["sources"][key]
        local = args.hf_cache / pin["hf_path"] if "hf_path" in pin else None
        inputs[key] = fetch(pin, args, local)
    cuda = cuda_inputs(args) if args.target.endswith("-cuda") else {}
    cuda_pins = LOCK["cuda_files_linux" if args.target == "linux-cuda" else "cuda_files"]
    preload = LINUX_PRELOAD if args.target == "linux-cuda" else PRELOAD
    if args.fetch_only:
        print(json.dumps({"status": "sources-ready", "target": args.target, "cache": str(args.cache)}))
        return
    destination = (args.output or ROOT / "artifacts" / args.target).resolve()
    if destination.exists():
        raise FileExistsError(f"Ausgabe existiert bereits und wird nicht verändert: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + args.target + "-", dir=destination.parent))
    try:
        library = package_runtime(inputs["ort-" + args.target], args.target, staging)
        for name, path in cuda.items():
            copy_private(path, staging / "runtime/cuda" / name, cuda_pins[name])
        package_fonts(inputs, staging)
        package_licenses(inputs, staging, args.target)
        sidecar = prepare_fast(inputs["fast"], inputs["fast-sidecar"], staging / "models/fast.onnx", bool(cuda))
        copy_private(inputs["fast-sidecar"], staging / "models/fast.onnx.json")
        quality_report = package_quality(inputs, staging, args) if cuda else None
        manifest = {"version": 1,
                    "runtime": {"version": "1.26.0", "provider": "cuda" if cuda else "cpu", "library": library,
                                "files": [record(path, staging) for path in sorted((staging / "runtime").rglob("*")) if path.is_file()],
                                "preload": ["runtime/cuda/" + name for name in preload] if cuda else []},
                    "fast": {**record(staging / "models/fast.onnx", staging), "input_name": sidecar["input_name"],
                             "output_name": sidecar["output_name"], "input_size": sidecar["canvas_size"], "precision": sidecar["precision"]},
                    "quality": {**record(staging / "models/quality.onnx", staging), "input_name": "input", "output_name": "mask",
                                "input_size": 1024, "precision": "fp16"} if cuda else None,
                    "fonts": [record(path, staging) for path in sorted((staging / "fonts").iterdir())],
                    "provenance": {"target": args.target, "source_lock_sha256": sha256(LOCK_PATH),
                                   "preparation_script_sha256": sha256(Path(__file__)), "sources": {key: LOCK["sources"][key] for key in keys},
                                   "cuda": {"release": "12.8.1", "cudnn": "9.8.0.87", "files": cuda_pins,
                                            "sources": {key: LOCK["sources"][key] for key in sorted({pin["component"] for pin in cuda_pins.values()})}} if cuda else None,
                                   "model_revisions": {"fast": LOCK["fast_revision"], "quality": LOCK["quality_revision"] if cuda else None},
                                   "quality_build": quality_report, "fast_sigmoid_defusions": 1 if cuda else 0,
                                   "attribution": "Built with DINOv3", "source_evidence": LOCK["evidence"],
                                   "release_validation": "not-run-by-preparation; native target smoke and release gates required"}}
        copy_private(LOCK_PATH, staging / "provenance/sources.lock.json")
        inventory = [record(path, staging) for path in sorted(staging.rglob("*")) if path.is_file()]
        manifest["provenance"]["inventory"] = inventory
        dump(staging / "manifest.json", manifest)
        staging.rename(destination)
        print(json.dumps({"status": "prepared", "manifest": str(destination / "manifest.json"),
                          "manifest_sha256": sha256(destination / "manifest.json")}), flush=True)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError, zipfile.BadZipFile, tarfile.TarError) as error:
        print(f"Artefaktvorbereitung fehlgeschlagen: {error}", file=sys.stderr)
        raise SystemExit(1)
