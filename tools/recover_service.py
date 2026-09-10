"""Host-side recovery for the native CUDA Compose service; never runs in containers.

Docker handles process exits. This handles sustained unhealthy state and reattaches
cloudflared after its network-namespace owner restarts. Stopped/paused services stay
stopped/paused. Run serially from the supplied systemd user timer.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


DOCKER = ("docker", "--host", "unix:///var/run/docker.sock")
STATE_FORMAT = ('{"id":"{{.Id}}","running":{{.State.Running}},'
                '"paused":{{.State.Paused}},"restarting":{{.State.Restarting}},'
                '"health":"{{if .State.Health}}{{.State.Health.Status}}{{end}}",'
                '"started":"{{.State.StartedAt}}",'
                '"network":"{{.HostConfig.NetworkMode}}"}')


def command(*args, timeout=150):
    # Deployment files, not an interactive worktree's variables/context, are authoritative.
    environment = {"HOME": str(Path.home()), "PATH": os.defpath}
    return subprocess.check_output(args, text=True, timeout=timeout, env=environment).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    release = args.release.resolve(strict=True)
    compose = (*DOCKER, "compose", "--project-name", "backremove",
               "--project-directory", str(release), "--env-file", str(release / "deploy.env"),
               "--file", str(release / "docker-compose.cuda.yml"))
    os.umask(0o077)
    args.state.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.state.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        history = json.loads(args.state.read_text()) if args.state.exists() else {}

        def state(service):
            ids = command(*compose, "ps", "--all", "--quiet", service, timeout=30).splitlines()
            if not ids:
                return None
            if len(ids) != 1:
                raise RuntimeError(f"Expected one {service} container; refusing recovery")
            return json.loads(command(*DOCKER, "inspect", "--format", STATE_FORMAT, ids[0], timeout=30))

        def active(container):
            return container and container["running"] and not container["paused"] and not container["restarting"]

        def reserve_recovery(service):
            now = time.time()
            recent = [stamp for stamp in history.get(service, []) if now - stamp < 600]
            if len(recent) >= 3:
                raise RuntimeError(f"{service}: three recovery attempts in ten minutes; inspect logs")
            history[service] = [*recent, now]
            temporary = args.state.with_suffix(".tmp")
            temporary.write_text(json.dumps(history) + "\n")
            temporary.replace(args.state)

        app = state("backremove")
        if not active(app):
            return
        if app["health"] == "unhealthy":
            reserve_recovery("backremove")
            print("Recovering unhealthy BackRemove; waiting for native readiness", flush=True)
            command(*compose, "restart", "--no-deps", "backremove")
            command(*compose, "up", "--detach", "--no-deps", "--wait", "--wait-timeout", "120", "backremove")
            app = state("backremove")
        if not active(app) or app["health"] != "healthy":
            return
        tunnel = state("cloudflared")
        if not active(tunnel):
            return
        restarted = (datetime.fromisoformat(tunnel["started"]) < datetime.fromisoformat(app["started"]))
        if tunnel["health"] == "unhealthy" or restarted or tunnel["network"] != "container:" + app["id"]:
            reserve_recovery("cloudflared")
            print("Reattaching cloudflared to the healthy native service", flush=True)
            command(*compose, "up", "--detach", "--no-deps", "--force-recreate",
                    "--wait", "--wait-timeout", "90", "cloudflared")


if __name__ == "__main__":
    main()
