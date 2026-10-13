import copy
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location(
    "recover_service", Path(__file__).resolve().parents[1] / "tools" / "recover_service.py")
recover_service = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recover_service)

APP = recover_service.APP
TUNNEL = recover_service.TUNNEL
NEVER = "0001-01-01T00:00:00Z"


def at(second):
    return f"2026-10-13T15:{second // 60:02d}:{second % 60:02d}.123456789Z"


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


class FakeStack:
    """Docker state of one Compose project; mutations follow Docker/Compose semantics."""

    def __init__(self, app=None, tunnels=()):
        self.second = 100
        self.next_id = 0
        self.services = {APP: [], TUNNEL: list(tunnels)}
        if app is not None:
            self.services[APP].append(app)
        self.actions = []
        self.fail = None  # (action, service): Docker did its part, then the run was killed.
        self.tunnel_health = "healthy"

    def tick(self):
        self.second += 1
        return at(self.second)

    def container(self, service, **state):
        self.next_id += 1
        base = {"id": f"{service}-{self.next_id}", "name": f"/backremove-{service}-1", "running": True,
                "paused": False, "restarting": False, "health": "healthy", "started": self.tick(),
                "network": "bridge"}
        return {**base, **state}

    def app(self):
        return self.services[APP][0]

    def containers(self, service):
        return copy.deepcopy(self.services[service])

    def remove(self, containers):
        ids = {container["id"] for container in containers}
        self.actions.append(("remove", sorted(ids)))
        for service in self.services:
            self.services[service] = [c for c in self.services[service] if c["id"] not in ids]

    def restart(self, service):
        self.actions.append(("restart", service))
        container = self.services[service][0]
        container.update(running=False, health="")
        self._maybe_fail("restart", service)
        container.update(running=True, health="starting", started=self.tick())

    def up(self, service, wait_seconds):
        self.actions.append(("up", service))
        self._maybe_fail("up", service)
        if not self.services[service]:
            network = "container:" + self.app()["id"] if service == TUNNEL else "bridge"
            health = self.tunnel_health if service == TUNNEL else "healthy"
            self.services[service].append(self.container(service, network=network, health=health))
        else:
            self.services[service][0].update(running=True, health="healthy", started=self.tick())
        if self.services[service][0]["health"] != "healthy":
            # `compose up --wait` exits non-zero when the healthcheck does not pass.
            raise subprocess.CalledProcessError(1, ["docker", "compose", "up", "--wait", service])

    def _maybe_fail(self, action, service):
        if self.fail == (action, service):
            self.fail = None
            raise subprocess.TimeoutExpired(["docker", "compose", action, service], 1)


class RecoverTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.state = Path(directory.name) / "recovery.json"
        self.clock = Clock()

    def run_recovery(self, stack):
        recover_service.recover(stack, recover_service.Journal(self.state, self.clock), log=lambda _: None)

    def healthy_stack(self):
        stack = FakeStack()
        stack.services[APP].append(stack.container(APP))
        stack.services[TUNNEL].append(stack.container(TUNNEL, network="container:" + stack.app()["id"]))
        return stack

    def assert_tunnel_serves_app(self, stack):
        tunnels = stack.services[TUNNEL]
        self.assertEqual(len(tunnels), 1)
        self.assertTrue(tunnels[0]["running"])
        self.assertEqual(tunnels[0]["network"], "container:" + stack.app()["id"])
        self.assertGreater(tunnels[0]["started"], stack.app()["started"])

    def test_aborted_recreate_leftover_and_stopped_tunnel_converge(self):
        # The 2026-10-13 incident: the recreate was killed after stopping the tunnel and
        # creating its replacement under a temporary name.
        stack = self.healthy_stack()
        old = stack.services[TUNNEL][0]
        old.update(running=False, health="unhealthy")
        stack.services[TUNNEL].append(stack.container(
            TUNNEL, name="/45ca83e84336_backremove-cloudflared-1", running=False, health="",
            started=NEVER, network="container:" + stack.app()["id"]))

        self.run_recovery(stack)

        self.assert_tunnel_serves_app(stack)
        self.assertNotIn(("restart", APP), stack.actions)

    def test_healthy_attached_tunnel_is_left_alone(self):
        stack = self.healthy_stack()

        self.run_recovery(stack)

        self.assertEqual(stack.actions, [])
        self.assertFalse(self.state.exists())

    def test_missing_tunnel_is_created(self):
        stack = self.healthy_stack()
        stack.services[TUNNEL] = []

        self.run_recovery(stack)

        self.assert_tunnel_serves_app(stack)

    def test_tunnel_from_an_earlier_app_start_is_reattached(self):
        stack = self.healthy_stack()
        stack.app()["started"] = stack.tick()

        self.run_recovery(stack)

        self.assert_tunnel_serves_app(stack)

    def test_restarting_tunnel_bound_to_a_replaced_app_is_reattached(self):
        stack = self.healthy_stack()
        stack.services[TUNNEL][0].update(running=False, restarting=True, network="container:gone")

        self.run_recovery(stack)

        self.assert_tunnel_serves_app(stack)

    def test_operator_pause_and_dockers_own_restart_are_respected(self):
        for state in ({"paused": True}, {"restarting": True, "running": False}):
            with self.subTest(state=state):
                stack = self.healthy_stack()
                stack.services[TUNNEL][0].update(state)

                self.run_recovery(stack)

                self.assertEqual(stack.actions, [])

    def test_operator_stopped_app_and_tunnel_stay_stopped(self):
        stack = self.healthy_stack()
        stack.app().update(running=False, health="")
        stack.services[TUNNEL][0].update(running=False)

        self.run_recovery(stack)

        self.assertEqual(stack.actions, [])

    def test_app_stopped_by_an_interrupted_restart_is_finished_next_run(self):
        stack = self.healthy_stack()
        stack.app()["health"] = "unhealthy"
        stack.fail = ("restart", APP)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_recovery(stack)
        self.assertFalse(stack.app()["running"])

        self.run_recovery(stack)

        self.assertTrue(stack.app()["running"])
        self.assertEqual(stack.app()["health"], "healthy")
        self.assert_tunnel_serves_app(stack)

        # Finished work is forgotten: a later operator stop is respected again.
        stack.actions.clear()
        stack.app().update(running=False, health="")
        self.run_recovery(stack)
        self.assertEqual(stack.actions, [])

    def test_interrupted_tunnel_reattach_is_finished_next_run(self):
        stack = self.healthy_stack()
        stack.services[TUNNEL][0]["health"] = "unhealthy"
        stack.fail = ("up", TUNNEL)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_recovery(stack)
        self.assertEqual(stack.services[TUNNEL], [])

        self.run_recovery(stack)

        self.assert_tunnel_serves_app(stack)

    def test_leftover_app_container_is_removed_without_touching_the_running_one(self):
        stack = self.healthy_stack()
        running_id = stack.app()["id"]
        stack.services[APP].append(stack.container(APP, name="/9f3e_backremove-backremove-1",
                                                   running=False, health="", started=NEVER))

        self.run_recovery(stack)

        self.assertEqual([c["id"] for c in stack.services[APP]], [running_id])
        self.assertNotIn(("restart", APP), stack.actions)

    def test_two_running_app_containers_are_refused(self):
        stack = self.healthy_stack()
        stack.services[APP].append(stack.container(APP))

        with self.assertRaisesRegex(RuntimeError, "running backremove containers"):
            self.run_recovery(stack)
        self.assertEqual(stack.actions, [])

    def test_failed_attempts_back_off_and_resume_after_the_window(self):
        stack = self.healthy_stack()
        stack.services[TUNNEL][0]["health"] = "unhealthy"
        stack.tunnel_health = "unhealthy"
        for _ in range(recover_service.ATTEMPTS):
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_recovery(stack)
        stack.actions.clear()
        with self.assertRaisesRegex(RuntimeError, "unsuccessful recovery attempts"):
            self.run_recovery(stack)
        self.assertEqual(stack.actions, [])

        self.clock.now += recover_service.WINDOW_SECONDS
        stack.tunnel_health = "healthy"
        self.run_recovery(stack)

        self.assert_tunnel_serves_app(stack)
        self.assertEqual(stack.services[TUNNEL][0]["health"], "healthy")

    def test_successful_recoveries_never_exhaust_the_cap(self):
        # A flapping network restarts BackRemove again and again; each reattach that
        # works must not use up the budget the next outage needs.
        stack = self.healthy_stack()
        for _ in range(recover_service.ATTEMPTS + 2):
            stack.app()["started"] = stack.tick()
            self.run_recovery(stack)
            self.assert_tunnel_serves_app(stack)

    def test_journal_written_by_the_previous_version_is_read(self):
        self.state.write_text('{"backremove": [], "cloudflared": [%f, %f, %f]}\n'
                              % ((self.clock.now - 1,) * 3))
        stack = self.healthy_stack()
        stack.services[TUNNEL][0]["health"] = "unhealthy"

        with self.assertRaisesRegex(RuntimeError, "recovery attempts"):
            self.run_recovery(stack)


if __name__ == "__main__":
    unittest.main()
