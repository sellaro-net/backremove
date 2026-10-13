"""Host-side recovery for the native CUDA Compose service; never runs in containers.

Docker restarts crashed processes. This timer job handles what Docker does not:
sustained unhealthy state, a cloudflared left in the network namespace of an
earlier BackRemove start, and every partial state an interrupted recovery can
leave behind (a leftover container from an aborted recreate, a stopped or
missing tunnel, BackRemove stopped in the middle of its restart). Each run
converges from whatever it finds, so a run killed at any point is completed by
the next one.

BackRemove stays stopped or paused when an operator stopped or paused it;
`backremove-service stop` also stops this timer. cloudflared is a stateless
sidecar: while BackRemove runs healthy, the tunnel is kept running and attached
to it. Run serially from the supplied systemd user timer.
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
STATE_FORMAT = ('{"id":"{{.Id}}","name":"{{.Name}}","running":{{.State.Running}},'
                '"paused":{{.State.Paused}},"restarting":{{.State.Restarting}},'
                # A container that never started (an aborted recreate's leftover) has no Health key.
                '"health":"{{with index .State "Health"}}{{.Status}}{{end}}",'
                '"started":"{{.State.StartedAt}}",'
                '"network":"{{.HostConfig.NetworkMode}}"}')
APP = "backremove"
TUNNEL = "cloudflared"
ATTEMPTS = 3
WINDOW_SECONDS = 600
APP_WAIT_SECONDS = 120
TUNNEL_WAIT_SECONDS = 90


def command(*args, timeout=150):
    # Deployment files, not an interactive worktree's variables/context, are authoritative.
    environment = {"HOME": str(Path.home()), "PATH": os.defpath}
    return subprocess.check_output(args, text=True, timeout=timeout, env=environment).strip()


class Stack:
    """The release's Compose project; the only code that talks to Docker."""

    def __init__(self, release, project=APP):
        self.compose = (*DOCKER, "compose", "--project-name", project,
                        "--project-directory", str(release), "--env-file", str(release / "deploy.env"),
                        "--file", str(release / "docker-compose.cuda.yml"))

    def containers(self, service):
        ids = command(*self.compose, "ps", "--all", "--quiet", service, timeout=30).split()
        if not ids:
            return []
        lines = command(*DOCKER, "inspect", "--format", STATE_FORMAT, *ids, timeout=30).splitlines()
        return [json.loads(line) for line in lines]

    def remove(self, containers):
        # Stateless here: an aborted recreate's leftover or a tunnel bound to a dead namespace.
        command(*DOCKER, "rm", "--force", *(container["id"] for container in containers), timeout=60)

    def restart(self, service):
        command(*self.compose, "restart", "--no-deps", service)

    def up(self, service, wait_seconds):
        command(*self.compose, "up", "--detach", "--no-deps", "--wait",
                "--wait-timeout", str(wait_seconds), service, timeout=wait_seconds + 60)


class Journal:
    """Unsuccessful recovery attempts per service and recovery work not finished yet.

    An attempt counts until the service is healthy again, so the cap backs off a
    persistent failure without delaying the next independent one.
    """

    def __init__(self, path, clock=time.time):
        self.path = path
        self.clock = clock
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def reserve(self, service):
        now = self.clock()
        recent = [stamp for stamp in self.data.get(service, []) if now - stamp < WINDOW_SECONDS]
        if len(recent) >= ATTEMPTS:
            raise RuntimeError(f"{service}: {ATTEMPTS} unsuccessful recovery attempts in "
                               f"{WINDOW_SECONDS // 60} minutes; retrying after the window, inspect logs")
        self.data[service] = [*recent, now]
        self._save()

    def pending(self, service):
        return service in self.data.get("pending", {})

    def begin(self, service):
        self.data.setdefault("pending", {})[service] = self.clock()
        self._save()

    def succeeded(self, service):
        """The service is healthy: forget its attempts and any unfinished recovery."""
        changed = bool(self.data.pop(service, None))
        if self.pending(service):
            del self.data["pending"][service]
            changed = True
        if changed:
            self._save()

    def _save(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data) + "\n")
        temporary.replace(self.path)


def started(container):
    return datetime.fromisoformat(container["started"])


def running_app(stack, journal, log):
    """BackRemove's single running container, or None when it should stay stopped."""
    containers = stack.containers(APP)
    running = [container for container in containers if container["running"]]
    if len(running) > 1:
        raise RuntimeError(f"{len(running)} running {APP} containers; refusing recovery")
    if running and len(containers) > 1:
        leftovers = [container for container in containers if not container["running"]]
        log(f"Removing {len(leftovers)} leftover {APP} container(s) of an interrupted recreate")
        stack.remove(leftovers)
    if running:
        return running[0]
    if not journal.pending(APP):
        return None  # Stopped by an operator, or never started: not ours to start.
    journal.reserve(APP)
    log(f"Finishing an interrupted {APP} recovery; waiting for native readiness")
    if len(containers) > 1:
        stack.remove(containers)
    stack.up(APP, APP_WAIT_SECONDS)
    running = [container for container in stack.containers(APP) if container["running"]]
    return running[0] if len(running) == 1 else None


def tunnel_problem(tunnels, app):
    """Why the tunnel needs reattaching, or None when it serves the current BackRemove start."""
    if not tunnels:
        return "no tunnel container"
    if len(tunnels) > 1:
        return f"{len(tunnels)} tunnel containers, leftovers of an interrupted recreate"
    tunnel = tunnels[0]
    if not tunnel["running"]:
        return "tunnel is not running"
    if tunnel["network"] != "container:" + app["id"] or started(tunnel) < started(app):
        return f"tunnel is attached to an earlier {APP} start"
    if tunnel["health"] == "unhealthy":
        return "tunnel is unhealthy"
    return None


def recover(stack, journal, log=print):
    app = running_app(stack, journal, log)
    if app is None or app["paused"] or app["restarting"]:
        return  # Operator state, or Docker's restart policy is already acting.
    if app["health"] == "unhealthy":
        journal.reserve(APP)
        log(f"Recovering unhealthy {APP}; waiting for native readiness")
        journal.begin(APP)
        stack.restart(APP)
        stack.up(APP, APP_WAIT_SECONDS)
        app = running_app(stack, journal, log)
        if app is None:
            return
    if app["health"] != "healthy":
        return  # Still starting; the next run looks again.
    journal.succeeded(APP)

    tunnels = stack.containers(TUNNEL)
    if len(tunnels) == 1:
        tunnel = tunnels[0]
        if tunnel["paused"]:
            return  # Operator pause.
        if tunnel["restarting"] and tunnel["network"] == "container:" + app["id"]:
            return  # Docker's restart policy rejoins the current namespace by itself.
    problem = tunnel_problem(tunnels, app)
    if problem is None:
        if tunnels[0]["health"] == "healthy":
            journal.succeeded(TUNNEL)
        return
    journal.reserve(TUNNEL)
    log(f"Reattaching {TUNNEL} to the healthy native service: {problem}")
    if tunnels:
        stack.remove(tunnels)
    stack.up(TUNNEL, TUNNEL_WAIT_SECONDS)  # Waits for the healthcheck; raises otherwise.
    journal.succeeded(TUNNEL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    release = args.release.resolve(strict=True)
    os.umask(0o077)
    args.state.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.state.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        recover(Stack(release), Journal(args.state), log=lambda line: print(line, flush=True))


if __name__ == "__main__":
    main()
