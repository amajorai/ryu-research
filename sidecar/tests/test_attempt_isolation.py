"""Regression tests for approved-attempt isolation.

* An attempt is bound to the immutable ``workspace_tree`` it was approved for:
  writes made while the run is pending are rejected without consuming the
  proposal, and writes racing a running attempt cannot change what executes.
* Agent-authored experiment code runs inside the OS sandbox and cannot read
  Ryu's data directory (Core's node bearer), the engine token, or the campaign
  database, and its output therefore cannot carry them back to the caller.
* Even with network, an attempt cannot listen on a loopback port (to capture a
  bearer that a trusted client sends there) and none of its processes outlive
  the attempt.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ryu_research import campaigns, sandbox, workspaces

SIDECAR_DIR = Path(__file__).resolve().parents[1]
SOURCE_EXPERIMENTS_DIR = SIDECAR_DIR / "experiments"
SANDBOX_SUPPORTED = sandbox.supported()


def _free_loopback_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
    finally:
        probe.close()


def _kill_quietly(pid: int) -> None:
    """Test cleanup only: stop a process a failing (unfixed) sandbox leaked."""
    try:
        os.kill(pid, 9)
    except OSError:
        pass


def _process_running(pid: int) -> bool:
    """True while ``pid`` exists and is not a zombie."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return False
    return fields[0] != "Z"


class _ResearchFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        # Mirror the production layout: <RYU_DIR>/plugin-data/@ryu+research/workspaces.
        self.ryu_dir = self.root / "ryu"
        self.workspaces_root = self.ryu_dir / "plugin-data" / "@ryu+research" / "workspaces"
        self.workspaces_root.mkdir(parents=True)
        self.experiments_directory = self.root / "experiment-kinds"
        shutil.copytree(SOURCE_EXPERIMENTS_DIR, self.experiments_directory)
        self.environment = patch.dict(
            os.environ,
            {
                "RYU_DIR": str(self.ryu_dir),
                "RESEARCH_WORKSPACES": str(self.workspaces_root),
                "RESEARCH_EXPERIMENTS": str(self.experiments_directory),
            },
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary_directory.cleanup()

    def _proposal(self, campaign_id: str) -> dict:
        history = campaigns.history_projection(campaign_id)
        return campaigns.create_proposal(
            campaign_id,
            history_digest=history["history_digest"],
            kind=history["required_proposal_kind"],
        )["proposal"]


class WorkspaceTreeBindingTests(_ResearchFixture):
    def test_write_while_run_is_pending_is_rejected_without_consuming_proposal(self) -> None:
        initialized = workspaces.init_workspace("toy")
        campaign_id = initialized["campaign_id"]
        approved = workspaces.write_file(
            campaign_id, "train.py", "print('val_bpb = 1.5')\n"
        )
        proposal = self._proposal(campaign_id)

        # While the operator is still deciding, a second writer swaps the code.
        workspaces.write_file(
            campaign_id, "train.py", "print('val_bpb = 0.0001')  # swapped\n"
        )
        with self.assertRaises(workspaces.WorkspaceStateMismatchError):
            workspaces.run_experiment(
                campaign_id,
                hypothesis="Run the approved variant",
                proposal_id=proposal["id"],
                workspace_tree=approved["workspace_tree"],
            )

        detail = campaigns.get_campaign(campaign_id)["campaign"]
        self.assertEqual(0, detail["attempt_count"])
        self.assertEqual(proposal["id"], detail["current_proposal_id"])

    def test_run_without_a_tree_token_is_refused(self) -> None:
        campaign_id = workspaces.init_workspace("toy")["campaign_id"]
        proposal = self._proposal(campaign_id)
        with self.assertRaisesRegex(ValueError, "workspace_tree"):
            workspaces.run_experiment(
                campaign_id, hypothesis="no token", proposal_id=proposal["id"]
            )
        self.assertEqual(
            proposal["id"],
            campaigns.get_campaign(campaign_id)["campaign"]["current_proposal_id"],
        )

    def test_write_racing_a_running_attempt_cannot_change_what_executes(self) -> None:
        campaign_id = workspaces.init_workspace("toy")["campaign_id"]
        approved = workspaces.write_file(
            campaign_id,
            "train.py",
            "import os\nprint('cwd=' + os.getcwd())\nprint('val_bpb = 1.25')\n",
        )
        proposal = self._proposal(campaign_id)
        race_errors: list[BaseException] = []
        original_export = workspaces._export_tree

        def export_after_racing_write(dest: Path, tree: str, target: Path) -> None:
            def racer() -> None:
                try:
                    workspaces.write_file(
                        campaign_id, "train.py", "print('val_bpb = 0.0001')\n"
                    )
                except BaseException as exc:  # noqa: BLE001 - recorded for assertion
                    race_errors.append(exc)

            thread = threading.Thread(target=racer)
            thread.start()
            thread.join(timeout=10)
            original_export(dest, tree, target)

        with patch.object(workspaces, "_export_tree", export_after_racing_write):
            result = workspaces.run_experiment(
                campaign_id,
                hypothesis="Bound tree executes",
                proposal_id=proposal["id"],
                workspace_tree=approved["workspace_tree"],
            )

        self.assertEqual(1, len(race_errors))
        self.assertIsInstance(race_errors[0], campaigns.CampaignBusyError)
        self.assertEqual("ok", result["status"])
        self.assertEqual(1.25, result["score"])
        self.assertEqual(approved["workspace_tree"], result["workspace_tree"])
        # The attempt executed from a private checkout, not the mutable workspace.
        workspace_path = str(workspaces.workspace_dir(campaign_id).resolve())
        cwd_lines = [
            line for line in result["logs_tail"].splitlines() if line.startswith("cwd=")
        ]
        self.assertEqual(1, len(cwd_lines))
        self.assertNotIn(workspace_path, cwd_lines[0])
        attempts = campaigns.get_campaign(campaign_id, include_full=True)["campaign"][
            "attempts"
        ]
        self.assertEqual(approved["workspace_tree"], attempts[0]["workspace_tree"])

    def test_unavailable_sandbox_refuses_before_consuming_the_proposal(self) -> None:
        initialized = workspaces.init_workspace("toy")
        campaign_id = initialized["campaign_id"]
        proposal = self._proposal(campaign_id)
        with patch.object(
            sandbox,
            "ensure_available",
            side_effect=sandbox.SandboxUnavailableError("no confinement here"),
        ), self.assertRaises(sandbox.SandboxUnavailableError):
            workspaces.run_experiment(
                campaign_id,
                hypothesis="must not run unconfined",
                proposal_id=proposal["id"],
                workspace_tree=initialized["workspace_tree"],
            )
        detail = campaigns.get_campaign(campaign_id)["campaign"]
        self.assertEqual(0, detail["attempt_count"])
        self.assertEqual(proposal["id"], detail["current_proposal_id"])


# machine -> (AUDIT_ARCH, socket, bind, listen, setsid, setpgid, connect), taken
# from the kernel syscall tables independently of sandbox.py.
_ARCH_SYSCALLS = {
    "x86_64": (0xC000003E, 41, 49, 50, 112, 109, 42),
    "aarch64": (0xC00000B7, 198, 200, 201, 157, 154, 203),
}


def _run_bpf(program: list[tuple[int, int, int, int]], data: dict[int, int]) -> int:
    """Minimal classic-BPF evaluator for the launcher's seccomp program."""
    accumulator = 0
    pc = 0
    while True:
        code, jt, jf, k = program[pc]
        if code == 0x20:  # ld [k]
            accumulator = data[k]
            pc += 1
        elif code in (0x15, 0x35):  # jeq / jge
            hit = accumulator == k if code == 0x15 else accumulator >= k
            pc += 1 + (jt if hit else jf)
        elif code == 0x06:  # ret
            return k
        else:
            raise AssertionError(f"unexpected BPF opcode {code:#x}")


class SeccompProgramTests(unittest.TestCase):
    ALLOW = 0x7FFF0000
    EACCES = 0x00050000 | 13
    EPERM = 0x00050000 | 1
    ENOSYS = 0x00050000 | 38

    def _decide(self, machine: str, network: bool, *, arch: int, nr: int, family: int = 0) -> int:
        program = sandbox.seccomp_program(network, machine)
        return _run_bpf(program, {0: nr, 4: arch, 16: family})

    def test_socket_families_are_filtered_on_every_supported_arch(self) -> None:
        for machine, numbers in _ARCH_SYSCALLS.items():
            arch, socket_nr, bind_nr, listen_nr, setsid_nr, setpgid_nr, connect_nr = numbers
            for network in (False, True):
                with self.subTest(machine=machine, network=network):
                    expected_inet = self.ALLOW if network else self.EACCES
                    cases = [
                        (socket_nr, 1, self.EACCES),  # AF_UNIX: always refused
                        (socket_nr, 16, self.ALLOW),  # AF_NETLINK
                        (socket_nr, 2, expected_inet),  # AF_INET
                        (socket_nr, 10, expected_inet),  # AF_INET6
                        (socket_nr, 40, self.EACCES),  # AF_VSOCK
                        (425, 0, self.ENOSYS),  # io_uring_setup
                        # Never a listening socket, even with network: no
                        # loopback port squatting.
                        (bind_nr, 2, self.EACCES),
                        (listen_nr, 2, self.EACCES),
                        # No descendant may leave the attempt's process group.
                        (setsid_nr, 0, self.EPERM),
                        (setpgid_nr, 0, self.EPERM),
                        (connect_nr, 2, self.ALLOW),  # outgoing connections
                        (0, 0, self.ALLOW),  # an ordinary syscall
                    ]
                    for nr, family, expected in cases:
                        self.assertEqual(
                            expected,
                            self._decide(machine, network, arch=arch, nr=nr, family=family),
                            (nr, family),
                        )
                    # A foreign syscall ABI (here i386 socketcall) cannot
                    # sidestep the rules.
                    self.assertEqual(
                        self.EPERM, self._decide(machine, network, arch=0x40000003, nr=102)
                    )
        x32_socket = 0x40000000 | 41
        self.assertEqual(
            self.EPERM, self._decide("x86_64", True, arch=0xC000003E, nr=x32_socket, family=1)
        )

    def test_unsupported_architectures_refuse(self) -> None:
        with self.assertRaises(sandbox.SandboxUnavailableError):
            sandbox.seccomp_program(False, "riscv64")


@unittest.skipUnless(SANDBOX_SUPPORTED, "OS confinement is unavailable on this host")
class ExperimentSandboxTests(_ResearchFixture):
    SECRET = "node-bearer-sentinel-7f3a9c"
    ENGINE_SECRET = "engine-token-sentinel-41b2"

    def test_authored_code_cannot_read_ryu_data_or_engine_token(self) -> None:
        node_token = self.ryu_dir / "node-auth.token"
        node_token.write_text(self.SECRET, encoding="utf-8")
        node_token.chmod(0o600)
        engine_token = self.workspaces_root / ".engine-token"
        engine_token.write_text(self.ENGINE_SECRET, encoding="utf-8")
        engine_token.chmod(0o600)

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self.addCleanup(listener.close)
        port = listener.getsockname()[1]
        # Stands in for the session bus / keyring and Core's loopback relays.
        unix_path = self.root / "service.sock"
        unix_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        unix_listener.bind(str(unix_path))
        unix_listener.listen(1)
        self.addCleanup(unix_listener.close)

        campaign_id = workspaces.init_workspace("toy")["campaign_id"]
        workspace = workspaces.workspace_dir(campaign_id)
        targets = [
            str(node_token),
            str(engine_token),
            str(self.workspaces_root / campaigns.DATABASE_FILE),
            # The relative path the original exploit used from the workspace.
            str(workspace / ".." / ".." / ".." / ".." / "node-auth.token"),
        ]
        program = (
            "import socket\n"
            f"targets = {targets!r}\n"
            "for path in targets:\n"
            "    try:\n"
            "        with open(path, 'rb') as stream:\n"
            "            print('LEAK ' + stream.read().decode('latin-1'))\n"
            "    except OSError as exc:\n"
            "        print('DENIED ' + type(exc).__name__)\n"
            "import os\n"
            "try:\n"
            "    with open(f'/proc/{os.getppid()}/environ', 'rb') as stream:\n"
            "        print('PROC LEAK ' + stream.read().decode('latin-1'))\n"
            "except OSError as exc:\n"
            "    print('PROC DENIED ' + type(exc).__name__)\n"
            "try:\n"
            f"    socket.create_connection(('127.0.0.1', {port}), timeout=2).close()\n"
            "    print('NET OPEN')\n"
            "except OSError as exc:\n"
            "    print('NET DENIED ' + type(exc).__name__)\n"
            "try:\n"
            "    unix = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            f"    unix.connect({str(unix_path)!r})\n"
            "    print('UNIX OPEN')\n"
            "except OSError as exc:\n"
            "    print('UNIX DENIED ' + type(exc).__name__)\n"
            "print('val_bpb = 1.0')\n"
        )
        written = workspaces.write_file(campaign_id, "train.py", program)
        proposal = self._proposal(campaign_id)
        result = workspaces.run_experiment(
            campaign_id,
            hypothesis="Try to read host secrets",
            proposal_id=proposal["id"],
            workspace_tree=written["workspace_tree"],
        )

        logs = result["logs_tail"]
        self.assertEqual("ok", result["status"], logs)
        self.assertNotIn(self.SECRET, logs)
        self.assertNotIn(self.ENGINE_SECRET, logs)
        self.assertNotIn("LEAK", logs)
        if sys.platform.startswith("linux"):
            # The engine's (and Core's) environment is not readable via /proc.
            self.assertIn("PROC DENIED", logs)
        denied = [line for line in logs.splitlines() if line == "DENIED PermissionError"]
        self.assertEqual(len(targets), len(denied), logs)
        # Enforced by the seccomp socket filter on every Landlock ABI version.
        self.assertIn("NET DENIED PermissionError", logs)
        self.assertIn("UNIX DENIED PermissionError", logs)

        stored = campaigns.get_campaign(campaign_id, include_full=True)["campaign"]
        self.assertNotIn(self.SECRET, repr(stored))

    def test_policy_never_grants_an_ancestor_of_protected_data(self) -> None:
        with self.assertRaises(sandbox.SandboxUnavailableError):
            sandbox.build_policy(
                self.root,
                [sys.executable],
                workspaces_root=self.workspaces_root,
            )
        run_root = Path(tempfile.mkdtemp(prefix="ryu-research-run-test-"))
        self.addCleanup(shutil.rmtree, run_root, True)
        with self.assertRaises(sandbox.SandboxUnavailableError):
            sandbox.build_policy(
                run_root,
                [sys.executable],
                workspaces_root=self.workspaces_root,
                declared_paths=[str(self.ryu_dir.parent)],
            )

    def test_policy_never_makes_shared_or_unconfined_locations_writable(self) -> None:
        home = self.root / "home"
        home.mkdir()
        run_root = Path(tempfile.mkdtemp(prefix="ryu-research-run-test-"))
        self.addCleanup(shutil.rmtree, run_root, True)
        runtime = sandbox.workspace_runtime_dir(self.workspaces_root, "ws-1")
        with patch.dict(os.environ, {"HOME": str(home)}):
            policy = sandbox.build_policy(
                run_root,
                [sys.executable],
                workspaces_root=self.workspaces_root,
                runtime_dir=runtime,
                network=True,
            )
            self.assertEqual(
                [str(run_root.resolve()), str(runtime.resolve())], policy["write"]
            )
            self.assertNotIn("/dev/shm", policy["devices"])
            # Declared paths can no longer name the shared uv cache/tool dirs.
            for declared in ("~/.cache/uv", "~/.local/share/uv", "~", str(self.ryu_dir)):
                with self.assertRaises(sandbox.SandboxUnavailableError, msg=declared):
                    sandbox.build_policy(
                        run_root,
                        [sys.executable],
                        workspaces_root=self.workspaces_root,
                        runtime_dir=runtime,
                        declared_paths=[declared],
                    )
            # The runtime dir must be Research-private, never a shared dir.
            for wrong in (
                home / ".local" / "share" / "uv",
                self.workspaces_root / "ws-1",
                self.workspaces_root.parent / "engine",
            ):
                with self.assertRaises(sandbox.SandboxUnavailableError, msg=str(wrong)):
                    sandbox.build_policy(
                        run_root,
                        [sys.executable],
                        workspaces_root=self.workspaces_root,
                        runtime_dir=wrong,
                    )
            # A run dir may not overlap the engine's own code or interpreter.
            with self.assertRaises(sandbox.SandboxUnavailableError):
                sandbox.build_policy(
                    Path(sandbox.__file__).resolve().parent,
                    [sys.executable],
                    workspaces_root=self.workspaces_root,
                )

    def test_declared_paths_are_read_only(self) -> None:
        outside = tempfile.TemporaryDirectory(prefix="ryu-research-dataset-")
        self.addCleanup(outside.cleanup)
        dataset = Path(outside.name) / "data.txt"
        dataset.write_text("dataset-contents", encoding="utf-8")
        kind_root = self.experiments_directory / "dataset"
        kind_root.mkdir()
        (kind_root / "experiment.toml").write_text(
            (SOURCE_EXPERIMENTS_DIR / "toy" / "experiment.toml").read_text(encoding="utf-8")
            + f'sandbox_paths = ["{outside.name}"]\n',
            encoding="utf-8",
        )
        (kind_root / "train.py").write_text("print('val_bpb = 1.0')\n", encoding="utf-8")
        campaign_id = workspaces.init_workspace("dataset")["campaign_id"]
        program = (
            f"print('READ ' + open({str(dataset)!r}).read())\n"
            "try:\n"
            f"    open({str(dataset)!r}, 'w').write('tampered')\n"
            "    print('WROTE')\n"
            "except OSError as exc:\n"
            "    print('WRITE DENIED ' + type(exc).__name__)\n"
            "print('val_bpb = 1.0')\n"
        )
        written = workspaces.write_file(campaign_id, "train.py", program)
        result = workspaces.run_experiment(
            campaign_id,
            hypothesis="Read a declared dataset",
            proposal_id=self._proposal(campaign_id)["id"],
            workspace_tree=written["workspace_tree"],
        )
        logs = result["logs_tail"]
        self.assertIn("READ dataset-contents", logs)
        self.assertIn("WRITE DENIED PermissionError", logs)
        self.assertEqual("dataset-contents", dataset.read_text(encoding="utf-8"))

    def test_next_attempt_setup_never_follows_symlinks_an_attempt_planted(self) -> None:
        outside_directory = tempfile.TemporaryDirectory(prefix="ryu-research-outside-")
        self.addCleanup(outside_directory.cleanup)
        outside = Path(outside_directory.name)
        campaign_id = workspaces.init_workspace("toy")["campaign_id"]
        # Agent code rewires its own runtime dir so that the engine's unconfined
        # setup of the next attempt would create directories outside it.
        program = (
            "import os, shutil\n"
            "runtime = os.path.dirname(os.environ['HOME'])\n"
            "shutil.rmtree(os.path.join(runtime, 'cache'))\n"
            f"os.symlink({str(outside)!r}, os.path.join(runtime, 'cache'))\n"
            "shutil.rmtree(os.path.join(runtime, 'data', 'uv'))\n"
            f"os.symlink({str(outside)!r}, os.path.join(runtime, 'data', 'uv'))\n"
            "print('PLANTED')\n"
            "print('val_bpb = 1.0')\n"
        )
        written = workspaces.write_file(campaign_id, "train.py", program)
        result = workspaces.run_experiment(
            campaign_id,
            hypothesis="Plant symlinks in the runtime dir",
            proposal_id=self._proposal(campaign_id)["id"],
            workspace_tree=written["workspace_tree"],
        )
        self.assertIn("PLANTED", result["logs_tail"])
        runtime = sandbox.workspace_runtime_dir(self.workspaces_root, campaign_id)
        self.assertTrue((runtime / "cache").is_symlink())

        # The next attempt's setup (the production seam) runs unconfined.
        run_root = Path(tempfile.mkdtemp(prefix="ryu-research-run-test-"))
        self.addCleanup(shutil.rmtree, run_root, True)
        config = workspaces._experiment_for_workspace(workspaces.workspace_dir(campaign_id))
        assert config is not None
        _, environment = workspaces._confine_attempt(
            config, campaign_id, run_root, [sys.executable, "train.py"]
        )

        self.assertEqual([], list(outside.iterdir()))
        for name in ("cache", "data/uv", "cache/uv", "cache/pip", "data/uv/tools"):
            path = runtime / name
            self.assertFalse(path.is_symlink(), name)
            self.assertTrue(path.is_dir(), name)
        self.assertEqual(str(runtime / "cache" / "uv"), environment["UV_CACHE_DIR"])

        # A non-directory left where a runtime dir belongs is replaced, too.
        shutil.rmtree(runtime / "state")
        (runtime / "state").write_text("not a dir", encoding="utf-8")
        sandbox.prepare_runtime_dir(runtime)
        self.assertTrue((runtime / "state").is_dir())
        self.assertEqual([], list(outside.iterdir()))


def _layer_probe(policy: dict, layer: str, connect_port: int, squat_port: int) -> str:
    """Apply one confinement layer alone in a child interpreter, then probe."""
    code = (
        "import json, os, socket, sys\n"
        f"sys.path.insert(0, {str(SIDECAR_DIR)!r})\n"
        "from ryu_research import sandbox\n"
        f"policy = json.loads({json.dumps(policy)!r})\n"
        f"sandbox.apply_{layer}(policy)\n"
        "try:\n"
        f"    socket.create_connection(('127.0.0.1', {connect_port}), timeout=2).close()\n"
        "    print('CONNECT OK')\n"
        "except OSError as exc:\n"
        "    print('CONNECT DENIED ' + type(exc).__name__)\n"
        "server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "try:\n"
        f"    server.bind(('127.0.0.1', {squat_port}))\n"
        "    print('BIND OK')\n"
        "except OSError as exc:\n"
        "    print('BIND DENIED ' + type(exc).__name__)\n"
        "try:\n"
        "    os.setsid()\n"
        "    print('SETSID OK')\n"
        "except OSError as exc:\n"
        "    print('SETSID DENIED ' + type(exc).__name__)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return completed.stdout + completed.stderr


@unittest.skipUnless(SANDBOX_SUPPORTED, "OS confinement is unavailable on this host")
class NetworkConfinementLayerTests(_ResearchFixture):
    """Each layer refuses a listening socket on its own with network on."""

    def _policy(self) -> dict:
        run_root = Path(tempfile.mkdtemp(prefix="ryu-research-run-test-"))
        self.addCleanup(shutil.rmtree, run_root, True)
        policy = sandbox.build_policy(
            run_root,
            [sys.executable],
            workspaces_root=self.workspaces_root,
            network=True,
        )
        policy["read"].append(str(SIDECAR_DIR))
        return policy

    def _listener_port(self) -> int:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        self.addCleanup(listener.close)
        return listener.getsockname()[1]

    @unittest.skipUnless(sandbox.landlock_abi() >= 4, "Landlock networking needs ABI 4")
    def test_landlock_alone_denies_tcp_bind_but_allows_connect(self) -> None:
        output = _layer_probe(
            self._policy(), "landlock", self._listener_port(), _free_loopback_port()
        )
        self.assertIn("CONNECT OK", output)
        self.assertIn("BIND DENIED PermissionError", output)

    def test_seccomp_alone_denies_bind_and_setsid_but_allows_connect(self) -> None:
        output = _layer_probe(
            self._policy(), "seccomp", self._listener_port(), _free_loopback_port()
        )
        self.assertIn("CONNECT OK", output)
        self.assertIn("BIND DENIED PermissionError", output)
        self.assertIn("SETSID DENIED PermissionError", output)


@unittest.skipUnless(SANDBOX_SUPPORTED, "OS confinement is unavailable on this host")
class NanochatConfinementTests(_ResearchFixture):
    """Runs the bundled nanochat experiment's real policy (network on) through
    ``run_experiment``. Only ``uv run`` is swapped for this interpreter so the
    agent-authored probe executes without a GPU or a uv install."""

    SECRET = "node-bearer-sentinel-nanochat-93d1"
    BENIGN = "BENIGN-browser-use"

    def setUp(self) -> None:
        super().setUp()
        self.home = self.root / "home"
        self.home.mkdir()
        self.home_environment = patch.dict(os.environ, {"HOME": str(self.home)})
        self.home_environment.start()

    def tearDown(self) -> None:
        # Unwind in reverse order so the fixture's environment is restored last.
        self.home_environment.stop()
        super().tearDown()

    def test_agent_code_cannot_plant_code_that_runs_unconfined_later(self) -> None:
        node_token = self.ryu_dir / "node-auth.token"
        node_token.write_text(self.SECRET, encoding="utf-8")
        node_token.chmod(0o600)
        # The user's shared uv locations, which Core's uvx-launched MCP/ACP
        # servers execute from without any sandbox.
        shared_cache = self.home / ".cache" / "uv"
        shared_cache.mkdir(parents=True)
        tool_bin = self.home / ".local" / "share" / "uv" / "tools" / "browser-use" / "bin"
        tool_bin.mkdir(parents=True)
        tool = tool_bin / "browser-use"
        tool.write_text(f"#!/bin/sh\necho {self.BENIGN}\n", encoding="utf-8")
        tool.chmod(0o755)
        user_bin = self.home / ".local" / "bin"
        user_bin.mkdir(parents=True)

        initialized = workspaces.init_workspace("nanochat")
        campaign_id = initialized["campaign_id"]
        config = workspaces._experiment_for_workspace(workspaces.workspace_dir(campaign_id))
        assert config is not None
        self.assertTrue(config.sandbox_network)
        self.assertEqual([], config.sandbox_paths)
        hooks = workspaces.workspace_dir(campaign_id) / ".git" / "hooks"
        hooks.mkdir(parents=True, exist_ok=True)
        engine_plant = Path(sandbox.__file__).resolve().parent / "planted_by_attempt.py"
        self.addCleanup(engine_plant.unlink, True)

        planted = [
            shared_cache / "planted",
            tool,
            user_bin / "uv",
            self.home / ".local" / "share" / "uv" / "python" / "planted",
            hooks / "pre-commit",
            engine_plant,
            self.workspaces_root / "planted",
        ]
        program = (
            "import os, socket, stat\n"
            f"for path in {[str(path) for path in planted]!r}:\n"
            "    try:\n"
            "        os.makedirs(os.path.dirname(path), exist_ok=True)\n"
            "        with open(path, 'w') as stream:\n"
            f"            stream.write('#!/bin/sh\\ncat {node_token}\\n')\n"
            "        os.chmod(path, 0o755)\n"
            "        print('WROTE ' + path)\n"
            "    except OSError as exc:\n"
            "        print('WRITE DENIED ' + type(exc).__name__)\n"
            "try:\n"
            f"    print('LEAK ' + open({str(node_token)!r}).read())\n"
            "except OSError as exc:\n"
            "    print('TOKEN DENIED ' + type(exc).__name__)\n"
            "private = os.path.join(os.environ['UV_CACHE_DIR'], 'private-ok')\n"
            "with open(private, 'w') as stream:\n"
            "    stream.write('ok')\n"
            "print('PRIVATE ' + os.environ['UV_CACHE_DIR'])\n"
            "print('HOME ' + os.environ['HOME'])\n"
            "try:\n"
            "    socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close()\n"
            "    print('UNIX OPEN')\n"
            "except OSError as exc:\n"
            "    print('UNIX DENIED ' + type(exc).__name__)\n"
            "print('val_bpb = 1.0')\n"
        )
        written = workspaces.write_file(campaign_id, "train.py", program)
        with patch.object(
            workspaces, "_experiment_argv", return_value=[sys.executable, "train.py"]
        ):
            result = workspaces.run_experiment(
                campaign_id,
                hypothesis="Plant code in shared tool dirs",
                proposal_id=self._proposal(campaign_id)["id"],
                workspace_tree=written["workspace_tree"],
            )

        logs = result["logs_tail"]
        self.assertEqual("ok", result["status"], logs)
        self.assertNotIn("WROTE", logs)
        self.assertNotIn("LEAK", logs)
        self.assertEqual(
            len(planted),
            logs.splitlines().count("WRITE DENIED PermissionError"),
            logs,
        )
        self.assertIn("TOKEN DENIED PermissionError", logs)
        self.assertIn("UNIX DENIED PermissionError", logs)

        runtime = sandbox.workspace_runtime_dir(self.workspaces_root, campaign_id)
        self.assertIn(f"PRIVATE {runtime.resolve() / 'cache' / 'uv'}", logs)
        self.assertIn(f"HOME {runtime.resolve() / 'home'}", logs)
        self.assertTrue((runtime / "cache" / "uv" / "private-ok").is_file())
        self.assertFalse(runtime.resolve().is_relative_to(self.home.resolve()))

        # Nothing was planted, and the shared tool an unconfined consumer runs
        # later is untouched.
        for path in planted:
            if path != tool:
                self.assertFalse(path.exists(), str(path))
        self.assertEqual([], list(shared_cache.iterdir()))
        later = subprocess.run([str(tool)], capture_output=True, text=True, check=True)
        self.assertEqual(self.BENIGN, later.stdout.strip())
        self.assertNotIn(self.SECRET, later.stdout)

    def _run_nanochat_program(self, program: str, hypothesis: str) -> tuple[str, dict]:
        campaign_id = workspaces.init_workspace("nanochat")["campaign_id"]
        written = workspaces.write_file(campaign_id, "train.py", program)
        with patch.object(
            workspaces, "_experiment_argv", return_value=[sys.executable, "train.py"]
        ):
            result = workspaces.run_experiment(
                campaign_id,
                hypothesis=hypothesis,
                proposal_id=self._proposal(campaign_id)["id"],
                workspace_tree=written["workspace_tree"],
            )
        return campaign_id, result

    def test_network_attempt_cannot_squat_a_loopback_port_to_capture_a_bearer(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        self.addCleanup(listener.close)
        outbound_port = listener.getsockname()[1]
        # Stands in for the Research engine's (or a stopped sidecar's) port that
        # a trusted, unconfined client later sends its bearer to.
        squat_port = _free_loopback_port()
        program = (
            "import os, socket, time\n"
            "try:\n"
            f"    socket.create_connection(('127.0.0.1', {outbound_port}), timeout=2).close()\n"
            "    print('CONNECT OK')\n"
            "except OSError as exc:\n"
            "    print('CONNECT DENIED ' + type(exc).__name__)\n"
            "server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
            "try:\n"
            f"    server.bind(('127.0.0.1', {squat_port}))\n"
            "    print('BIND OK')\n"
            "except OSError as exc:\n"
            "    print('BIND DENIED ' + type(exc).__name__)\n"
            "try:\n"
            "    server.listen(8)\n"
            "    print('LISTEN OK')\n"
            "except OSError as exc:\n"
            "    print('LISTEN DENIED ' + type(exc).__name__)\n"
            "udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
            "try:\n"
            "    udp.bind(('127.0.0.1', 0))\n"
            "    print('UDP BIND OK')\n"
            "except OSError as exc:\n"
            "    print('UDP BIND DENIED ' + type(exc).__name__)\n"
            "if os.fork() == 0:\n"
            "    try:\n"
            "        os.setsid()\n"
            "    except OSError:\n"
            "        pass\n"
            "    os.close(1)\n"
            "    os.close(2)\n"
            "    capture = os.path.join(os.environ['HOME'], 'captured')\n"
            "    server.settimeout(0.5)\n"
            "    deadline = time.time() + 30\n"
            "    while time.time() < deadline:\n"
            "        try:\n"
            "            conn, _ = server.accept()\n"
            "        except OSError:\n"
            "            time.sleep(0.1)\n"
            "            continue\n"
            "        with open(capture, 'wb') as stream:\n"
            "            stream.write(conn.recv(4096))\n"
            "        conn.sendall(b'HTTP/1.1 200 OK\\r\\nContent-Length: 2\\r\\n\\r\\n{}')\n"
            "        conn.close()\n"
            "    os._exit(0)\n"
            "print('val_bpb = 1.0')\n"
        )
        campaign_id, result = self._run_nanochat_program(program, "Squat a loopback port")
        logs = result["logs_tail"]
        self.assertEqual("ok", result["status"], logs)

        # After the attempt, a trusted client sends its bearer to that port.
        time.sleep(0.3)
        delivered = False
        try:
            client = socket.create_connection(("127.0.0.1", squat_port), timeout=2)
        except ConnectionRefusedError:
            pass
        else:
            with client:
                client.sendall(
                    b"GET /experiments HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                    b"Authorization: Bearer ENGINE-TOKEN-SENTINEL\r\n\r\n"
                )
                client.recv(64)
                delivered = True
        time.sleep(0.3)
        runtime = sandbox.workspace_runtime_dir(self.workspaces_root, campaign_id)
        captured = runtime / "home" / "captured"
        self.assertFalse(
            captured.exists(),
            captured.read_bytes() if captured.exists() else b"",
        )
        self.assertFalse(delivered, "a process from the attempt accepted the bearer")
        # Outgoing network still works for the experiment that declared it, but
        # no socket could be bound or put into listening state.
        self.assertIn("CONNECT OK", logs)
        self.assertIn("BIND DENIED PermissionError", logs)
        self.assertIn("LISTEN DENIED PermissionError", logs)
        self.assertIn("UDP BIND DENIED PermissionError", logs)
        self.assertNotIn("BIND OK", logs)
        self.assertNotIn("LISTEN OK", logs)

    def test_attempt_processes_cannot_escape_the_group_and_outlive_the_attempt(self) -> None:
        program = (
            "import os, time\n"
            "home = os.environ['HOME']\n"
            "report = os.path.join(home, 'child')\n"
            "if os.fork() == 0:\n"
            "    lines = [str(os.getpid())]\n"
            "    for name, call in (('SETSID', os.setsid), ('SETPGID', lambda: os.setpgid(0, 0))):\n"
            "        try:\n"
            "            call()\n"
            "            lines.append(name + ' OK')\n"
            "        except OSError as exc:\n"
            "            lines.append(name + ' DENIED ' + type(exc).__name__)\n"
            "    with open(report + '.tmp', 'w') as stream:\n"
            "        stream.write('\\n'.join(lines))\n"
            "    os.replace(report + '.tmp', report)\n"
            "    os.close(1)\n"
            "    os.close(2)\n"
            "    deadline = time.time() + 30\n"
            "    while time.time() < deadline:\n"
            "        with open(os.path.join(home, 'heartbeat'), 'a') as stream:\n"
            "            stream.write('x')\n"
            "        time.sleep(0.05)\n"
            "    os._exit(0)\n"
            "while not os.path.exists(report):\n"
            "    time.sleep(0.01)\n"
            "print(open(report).read())\n"
            "print('val_bpb = 1.0')\n"
        )
        campaign_id, result = self._run_nanochat_program(program, "Outlive the attempt")
        logs = result["logs_tail"]
        self.assertEqual("ok", result["status"], logs)

        home = sandbox.workspace_runtime_dir(self.workspaces_root, campaign_id) / "home"
        child_pid = int((home / "child").read_text(encoding="utf-8").splitlines()[0])
        self.addCleanup(_kill_quietly, child_pid)
        heartbeat = home / "heartbeat"
        before = heartbeat.stat().st_size if heartbeat.exists() else 0
        time.sleep(1.0)
        after = heartbeat.stat().st_size if heartbeat.exists() else 0
        self.assertEqual(before, after, "attempt child kept writing after the attempt")
        self.assertFalse(_process_running(child_pid), "attempt child outlived the attempt")
        self.assertIn("SETSID DENIED PermissionError", logs)
        self.assertIn("SETPGID DENIED PermissionError", logs)


if __name__ == "__main__":
    unittest.main()
