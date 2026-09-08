"""Record native build inputs, Cargo licenses and hashes in a release directory.

Build tooling only. Run after cargo build; no formatter, linter or test is invoked.
The inventory is evidence, not legal approval of redistribution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True, timeout=180).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dav1d-prefix", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(command("cargo", "metadata", "--locked", "--offline", "--format-version", "1"))
    lock_path = ROOT / "Cargo.lock"
    lock = tomllib.loads(lock_path.read_text())
    checksums = {(p["name"], p["version"], p.get("source")): p.get("checksum") for p in lock["package"]}
    packages = []
    for package in sorted(metadata["packages"], key=lambda p: (p["name"], p["version"])):
        if package["id"] == metadata["resolve"]["root"]:
            continue
        if not package.get("license") and not package.get("license_file"):
            raise ValueError(f"Cargo-Paket ohne Lizenznachweis: {package['name']} {package['version']}")
        source_root = Path(package["manifest_path"]).parent
        license_files = set()
        for pattern in ("LICENSE*", "LICENCE*", "COPYING*", "NOTICE*", "license*", "licence*"):
            license_files.update(path for path in source_root.glob(pattern) if path.is_file())
        if package.get("license_file"):
            license_files.add(source_root / package["license_file"])
        copied = []
        for source in sorted(license_files):
            destination = output / "licenses/rust" / (package["name"] + "-" + package["version"]) / (source.name + ".txt")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied.append({"path": destination.relative_to(output).as_posix(), "sha256": digest(destination)})
        packages.append({"name": package["name"], "version": package["version"], "source": package["source"],
                         "repository": package.get("repository"), "license_expression": package.get("license"),
                         "cargo_checksum": checksums.get((package["name"], package["version"], package["source"])),
                         "license_files": copied})
    source_inputs = []
    for path in sorted((ROOT / "src").rglob("*")):
        if path.is_file():
            source_inputs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
    for path in sorted((ROOT / "vendor").rglob("*")):
        if path.is_file():
            source_inputs.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
    for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "Dockerfile", "tools/sources.lock.json",
                 "tools/prepare.py", "tools/export_quality.py", "tools/requirements-prepare.txt", "tools/requirements-export-windows.txt",
                 "tools/build_inventory.py", "setup-gpu.ps1", "start-gpu.bat", ".env.example", "docker-compose.yml"):
        path = ROOT / name
        if path.exists():
            source_inputs.append({"path": name, "sha256": digest(path)})
    dav1d_files = [{"path": path.relative_to(args.dav1d_prefix).as_posix(), "sha256": digest(path)}
                   for path in sorted(args.dav1d_prefix.rglob("*")) if path.is_file()]
    revision = os.environ.get("SOURCE_REVISION")
    if revision is None:
        if (ROOT / ".git").exists():
            revision = command("git", "rev-parse", "HEAD")
        else:
            revision = "unavailable; source file hashes are authoritative"
    inventory = {"version": 1, "scope": "native release and build dependencies", "rustc": command("rustc", "--version", "--verbose"),
                 "cargo": command("cargo", "--version"), "source_revision": revision,
                 "source_inputs": source_inputs, "cargo_packages": packages,
                 "dav1d": {"version": "1.5.3", "source": json.loads((ROOT / "tools/sources.lock.json").read_text())["dav1d"], "build_files": dav1d_files},
                 "attribution": "Built with DINOv3", "legal_review": "required before redistribution",
                 "files": [{"path": path.relative_to(output).as_posix(), "sha256": digest(path), "size": path.stat().st_size}
                           for path in sorted(output.rglob("*")) if path.is_file() and path.name != "release-inventory.json"]}
    (output / "release-inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Build- und Lizenzinventar: {output / 'release-inventory.json'}")


if __name__ == "__main__":
    main()
